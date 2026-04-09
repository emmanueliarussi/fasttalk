#!/usr/bin/env python3
import argparse
import csv
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

# Ensure repo root is on sys.path so local packages resolve when run as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flame_model.FLAME import FLAMEModel



def _load_mask_indices(mask_path: str, preferred_keys: Iterable[str]) -> Tuple[np.ndarray, str]:
    with open(mask_path, "rb") as f:
        try:
            masks = pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            masks = pickle.load(f, encoding="latin1")

    if isinstance(masks, dict):
        for key in preferred_keys:
            if key in masks:
                return np.asarray(masks[key]).astype(np.int64), key

        # Fuzzy match: pick keys that look like forehead or brow regions.
        candidates = []
        for key in masks.keys():
            key_lower = str(key).lower()
            if "forehead" in key_lower or "brow" in key_lower or "eyebrow" in key_lower:
                candidates.append(key)

        if len(candidates) == 1:
            key = candidates[0]
            return np.asarray(masks[key]).astype(np.int64), str(key)

        if len(candidates) > 1:
            # Union all candidates if multiple exist.
            idx_list = [np.asarray(masks[k]).astype(np.int64) for k in candidates]
            return np.unique(np.concatenate(idx_list)), "+".join([str(k) for k in candidates])

    raise ValueError(
        "Could not resolve a forehead mask from FLAME masks. "
        "Provide --mask-key with an explicit key."
    )


def _find_default_mask_path() -> str:
    candidates = [
        "/mnt/fasttalk/flame/assets/FLAME_masks.pkl",
        "/mnt/fasttalk/flame_model/assets/FLAME_masks.pkl",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("FLAME mask asset not found in expected locations.")


def _load_npz_params(npz_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)

    if "exp" in data:
        exp = data["exp"].reshape(-1, 50)
    elif "expression_params" in data:
        exp = data["expression_params"].reshape(-1, 50)
    else:
        raise KeyError("Missing expression params (exp/expression_params) in npz.")

    gpose = None
    jaw = None
    if "pose" in data:
        pose = data["pose"]
        if pose.ndim == 2 and pose.shape[1] == 6:
            gpose = pose[:, :3]
            jaw = pose[:, 3:6]
        else:
            gpose = pose.reshape(-1, 3)
    if "gpose" in data:
        gpose = data["gpose"].reshape(-1, 3)
    if "jaw" in data:
        jaw = data["jaw"].reshape(-1, 3)

    if gpose is None:
        gpose = np.zeros((exp.shape[0], 3), dtype=np.float32)
    if jaw is None:
        jaw = np.zeros((exp.shape[0], 3), dtype=np.float32)

    min_len = min(exp.shape[0], jaw.shape[0])
    if gpose is not None:
        min_len = min(min_len, gpose.shape[0])
    if min_len < exp.shape[0]:
        exp = exp[:min_len]
    if min_len < jaw.shape[0]:
        jaw = jaw[:min_len]
    if gpose is not None and min_len < gpose.shape[0]:
        gpose = gpose[:min_len]

    return exp, gpose, jaw


def _compute_forehead_motion(
    flame: FLAMEModel,
    exp: np.ndarray,
    jaw: np.ndarray,
    forehead_idx: np.ndarray,
    device: torch.device,
    chunk_size: int,
) -> Dict[str, float]:
    t = exp.shape[0]
    if t < 2:
        return {"mean_speed": 0.0, "p95_speed": 0.0, "max_speed": 0.0}

    speeds: List[float] = []

    with torch.no_grad():
        for start in range(0, t, chunk_size):
            end = min(start + chunk_size, t)
            exp_chunk = torch.from_numpy(exp[start:end]).to(device)
            jaw_chunk = torch.from_numpy(jaw[start:end]).to(device)

            shape_params = torch.zeros((end - start, 300), device=device)
            gpose = torch.zeros((end - start, 3), device=device)

            # Ignore global pose by zeroing gpose.
            pose = torch.cat([gpose, jaw_chunk], dim=-1)

            eye_pose = torch.zeros((end - start, 6), device=device)
            verts, _ = flame(
                shape_params=shape_params,
                expression_params=exp_chunk,
                pose_params=pose,
                eye_pose_params=eye_pose,
            )

            # Remove per-frame translation to avoid global drift.
            verts = verts - verts.mean(dim=1, keepdim=True)

            region = verts[:, forehead_idx, :]
            region_center = region.mean(dim=1)
            if start == 0:
                prev_center = region_center[0]
                region_center = region_center[1:]
            else:
                prev_center = prev_tail

            if region_center.shape[0] > 0:
                diffs = region_center - prev_center
                if diffs.ndim == 1:
                    diffs = diffs.unsqueeze(0)
                frame_speeds = torch.linalg.norm(diffs, dim=1).detach().cpu().numpy()
                speeds.extend(frame_speeds.tolist())
                prev_tail = region_center[-1]
            else:
                prev_tail = prev_center

    if len(speeds) == 0:
        return {"mean_speed": 0.0, "p95_speed": 0.0, "max_speed": 0.0}

    speeds_np = np.asarray(speeds, dtype=np.float64)
    return {
        "mean_speed": float(np.mean(speeds_np)),
        "p95_speed": float(np.percentile(speeds_np, 95)),
        "max_speed": float(np.max(speeds_np)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan FLAME npz for forehead motion spikes.")
    parser.add_argument("--input-dir", default="/mnt/Datasets/ARTalk_data_subset/npz")
    parser.add_argument("--suffix", default="_synth.npz")
    parser.add_argument("--output-log", default="/mnt/Datasets/ARTalk_data_subset/forehead_motion_log.csv")
    parser.add_argument("--mask-path", default=None)
    parser.add_argument("--mask-key", default=None)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument("--render-top", type=int, default=5)
    parser.add_argument("--render-dir", default="demo/video")
    parser.add_argument("--render-fps", type=int, default=25)
    parser.add_argument("--render-batch-size", type=int, default=64)
    parser.add_argument("--add-audio", action="store_true")
    parser.add_argument("--audio-dir", default=None)

    args = parser.parse_args()

    mask_path = args.mask_path or _find_default_mask_path()

    preferred_keys = [
        args.mask_key,
        "forehead",
        "forehead_region",
        "brow",
        "eyebrow",
        "left_eyebrow",
        "right_eyebrow",
    ]
    preferred_keys = [k for k in preferred_keys if k]

    forehead_idx, mask_key = _load_mask_indices(mask_path, preferred_keys)

    device = torch.device(args.device)
    flame = FLAMEModel(n_shape=300, n_exp=50).to(device)
    flame.eval()

    entries = []
    npz_files = []
    for root, _, files in os.walk(args.input_dir):
        for fname in files:
            if fname.endswith(args.suffix):
                npz_files.append(os.path.join(root, fname))

    file_iter = tqdm(npz_files, desc="Scanning npz", unit="file") if tqdm else npz_files

    for npz_path in file_iter:
            try:
                exp, _, jaw = _load_npz_params(npz_path)
            except Exception as exc:
                entries.append({
                    "file": npz_path,
                    "frames": 0,
                    "mask_key": mask_key,
                    "mask_count": int(forehead_idx.shape[0]),
                    "mean_speed": "",
                    "p95_speed": "",
                    "max_speed": "",
                    "error": f"load_error: {exc}",
                })
                continue

            if exp.shape[0] < args.min_frames:
                entries.append({
                    "file": npz_path,
                    "frames": int(exp.shape[0]),
                    "mask_key": mask_key,
                    "mask_count": int(forehead_idx.shape[0]),
                    "mean_speed": "",
                    "p95_speed": "",
                    "max_speed": "",
                    "error": "too_few_frames",
                })
                continue

            metrics = _compute_forehead_motion(
                flame=flame,
                exp=exp,
                jaw=jaw,
                forehead_idx=forehead_idx,
                device=device,
                chunk_size=args.chunk_size,
            )

            entries.append({
                "file": npz_path,
                "frames": int(exp.shape[0]),
                "mask_key": mask_key,
                "mask_count": int(forehead_idx.shape[0]),
                "mean_speed": metrics["mean_speed"],
                "p95_speed": metrics["p95_speed"],
                "max_speed": metrics["max_speed"],
                "error": "",
            })

    os.makedirs(os.path.dirname(args.output_log), exist_ok=True)
    with open(args.output_log, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "frames",
                "mask_key",
                "mask_count",
                "mean_speed",
                "p95_speed",
                "max_speed",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(entries)

    scored = [
        e for e in entries
        if isinstance(e.get("p95_speed"), (int, float))
    ]
    scored.sort(key=lambda e: e["p95_speed"], reverse=True)
    top_n = scored[:5]
    if top_n:
        print("Top 5 forehead motion (p95_speed):")
        for rank, item in enumerate(top_n, start=1):
            print(
                f"{rank}. {item['file']} | p95={item['p95_speed']:.6f} | "
                f"mean={item['mean_speed']:.6f} | max={item['max_speed']:.6f}"
            )

    if args.render_top > 0 and top_n:
        from renderer.renderer import Renderer
        import matplotlib.animation as animation
        import matplotlib.pyplot as plt

        os.makedirs(args.render_dir, exist_ok=True)
        renderer = Renderer(render_full_head=True).to(device)
        renderer.eval()

        def get_vertices_from_blendshapes(exp_arr, gpose_arr, jaw_arr):
            exp_tensor = torch.from_numpy(exp_arr).float().to(device)
            gpose_tensor = torch.from_numpy(gpose_arr).float().to(device)
            jaw_tensor = torch.from_numpy(jaw_arr).float().to(device)

            shape_params = torch.zeros(exp_tensor.shape[0], 300, device=device)
            eye_pose = torch.zeros(exp_tensor.shape[0], 6, device=device)
            pose = torch.cat([gpose_tensor, jaw_tensor], dim=-1)

            verts, _ = flame(
                shape_params=shape_params,
                expression_params=exp_tensor,
                pose_params=pose,
                eye_pose_params=eye_pose,
            )
            return verts.detach()

        def add_audio_to_video(video_path: str, audio_path: str) -> str:
            if not os.path.exists(audio_path):
                print(f"Audio file not found: {audio_path}")
                return video_path

            output_with_audio = video_path.replace(".mp4", "_with_audio.mp4")
            cmd = (
                f'ffmpeg -y -i "{video_path}" -i "{audio_path}" '
                f'-c:v copy -c:a aac -strict experimental "{output_with_audio}"'
            )
            os.system(cmd)
            return output_with_audio

        def render_sequence(npz_path: str, output_path: str) -> str:
            exp_arr, gpose_arr, jaw_arr = _load_npz_params(npz_path)
            verts = get_vertices_from_blendshapes(exp_arr, gpose_arr, jaw_arr)

            frames_cpu = []
            with torch.no_grad():
                for start in range(0, verts.shape[0], args.render_batch_size):
                    end = min(start + args.render_batch_size, verts.shape[0])
                    verts_chunk = verts[start:end]
                    cam = torch.tensor([5, 0, 0], dtype=torch.float32).unsqueeze(0).to(device)
                    cam = cam.expand(verts_chunk.shape[0], -1)
                    frames = renderer.forward(verts_chunk, cam)["rendered_img"].detach().cpu()
                    frames_cpu.append(frames)
            frames = torch.cat(frames_cpu, dim=0)

            def update(frame_idx, seq, axes):
                frame = seq[frame_idx].detach().cpu().numpy().transpose(1, 2, 0)
                axes.clear()
                axes.imshow((frame * 255).astype(np.uint8))
                axes.axis("off")

            fig, ax = plt.subplots(figsize=(5, 5))
            ani = animation.FuncAnimation(
                fig,
                update,
                frames=frames.shape[0],
                fargs=(frames, ax),
                interval=100,
            )
            ani.save(output_path, writer="ffmpeg", fps=args.render_fps)
            plt.close(fig)
            return output_path

        render_count = min(args.render_top, len(top_n))
        if args.add_audio:
            if args.audio_dir:
                audio_dir = args.audio_dir
            else:
                audio_dir = os.path.join(os.path.dirname(args.input_dir.rstrip("/")), "wav")

        for item in top_n[:render_count]:
            base_name = os.path.splitext(os.path.basename(item["file"]))[0]
            output_path = os.path.join(args.render_dir, f"{base_name}_forehead_motion.mp4")
            print(f"Rendering {output_path} ...")
            video_path = render_sequence(item["file"], output_path)
            if args.add_audio:
                audio_path = os.path.join(audio_dir, f"{base_name}.wav")
                add_audio_to_video(video_path, audio_path)


if __name__ == "__main__":
    main()
