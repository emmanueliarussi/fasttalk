import argparse
import os
import sys
import subprocess
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch3d.transforms import matrix_to_euler_angles

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flame_model.FLAME import FLAMEModel
from renderer.renderer import Renderer

DEFAULT_OUTPUT_DIR = "/mnt/fasttalk/demo/video"


def load_sequence(npz_path, device):
    data = np.load(npz_path)
    keys = set(data.files)
    print("npz keys:", sorted(keys))

    if "exp" in keys:
        exp = data["exp"]
    elif "expression_params" in keys:
        exp = data["expression_params"]
    else:
        raise KeyError("Missing expression parameters in npz (exp/expression_params)")

    if "pose" in keys:
        gpose = data["pose"]
    elif "pose_params" in keys:
        gpose = data["pose_params"]
    elif "gpose" in keys:
        gpose = data["gpose"]
    else:
        gpose = np.zeros((exp.shape[0], 3), dtype=np.float32)

    if "jaw" in keys:
        jaw = data["jaw"]
    elif "jaw_params" in keys:
        jaw = data["jaw_params"]
    else:
        jaw = np.zeros((exp.shape[0], 3), dtype=np.float32)

    if "eyelids" in keys:
        eyelids = data["eyelids"]
    else:
        eyelids = np.ones((exp.shape[0], 2), dtype=np.float32)

    exp = exp.reshape(-1, 50)
    gpose = gpose.reshape(-1, 3)
    jaw = jaw.reshape(-1, 3)
    eyelids = eyelids.reshape(-1, 2)

    print("exp shape:", exp.shape)
    print("gpose shape:", gpose.shape)
    print("jaw shape:", jaw.shape)
    print("eyelids shape:", eyelids.shape)

    if gpose.shape[0] == exp.shape[0] * 2:
        print("gpose is 2x exp length; downsampling gpose by 2")
        gpose = gpose[::2]
    elif exp.shape[0] == gpose.shape[0] * 2:
        print("exp is 2x gpose length; downsampling exp/jaw/eyelids by 2")
        exp = exp[::2]
        jaw = jaw[::2]
        eyelids = eyelids[::2]

    if not (exp.shape[0] == gpose.shape[0] == jaw.shape[0] == eyelids.shape[0]):
        raise ValueError(
            "Sequence length mismatch after adjustment: "
            f"exp={exp.shape[0]} gpose={gpose.shape[0]} jaw={jaw.shape[0]} eyelids={eyelids.shape[0]}"
        )

    max_frames = 500
    if exp.shape[0] > max_frames:
        print(f"Trimming to first {max_frames} frames")
        exp = exp[:max_frames]
        gpose = gpose[:max_frames]
        jaw = jaw[:max_frames]
        eyelids = eyelids[:max_frames]

    return (
        torch.from_numpy(exp).float().to(device),
        torch.from_numpy(gpose).float().to(device),
        torch.from_numpy(jaw).float().to(device),
        torch.from_numpy(eyelids).float().to(device),
    )


def get_vertices_from_blendshapes(flame_model, expr, gpose, jaw, device):

    target_shape_tensor = torch.zeros(expr.shape[0], 300, device=device)
    identity = matrix_to_euler_angles(
        torch.cat([torch.eye(3, device=device)[None]], dim=0),
        "XYZ",
    )
    eye_r = identity.clone().squeeze()
    eye_l = identity.clone().squeeze()
    eyes = torch.cat([eye_r, eye_l], dim=0).expand(expr.shape[0], -1)
    pose = torch.cat([gpose, jaw], dim=-1)

    vertices, _ = flame_model.forward(
        shape_params=target_shape_tensor,
        expression_params=expr,
        pose_params=pose,
        eye_pose_params=eyes,
    )
    return vertices


def render_sequence(expr, gpose, jaw, eyelids, output_path, device):
    flame_model = FLAMEModel(n_shape=300, n_exp=50).to(device)
    renderer = Renderer(render_full_head=True).to(device)

    vertices = get_vertices_from_blendshapes(flame_model, expr, gpose, jaw, device)
    cam = torch.tensor([5, 0, 0], dtype=torch.float32).unsqueeze(0).to(vertices.device)
    cam = cam.expand(vertices.shape[0], -1)
    frames = renderer.forward(vertices, cam)["rendered_img"]

    def update(frame_idx, pr_seq, axes):
        frame = pr_seq[frame_idx].detach().cpu().numpy().transpose(1, 2, 0)
        scaled = (frame * 255).astype(np.uint8)
        axes.clear()
        axes.imshow(scaled)
        axes.axis("off")

    fig, ax = plt.subplots(figsize=(5, 5))
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=frames.shape[0],
        fargs=(frames, ax),
        interval=100,
    )
    fps = 25
    anim.save(output_path, writer="ffmpeg", fps=fps)
    plt.close(fig)
    duration_sec = frames.shape[0] / float(fps)
    return duration_sec


def build_output_path(npz_path, wav_path, output_dir):
    base = Path(wav_path).stem
    return str(Path(output_dir) / f"{base}.mp4")


def main():
    parser = argparse.ArgumentParser(description="Render a single npz sequence.")
    parser.add_argument("--npz", required=True, help="Path to input .npz file")
    parser.add_argument("--wav", required=True, help="Path to input .wav file (used for naming output)")
    parser.add_argument("--out_dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for mp4")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    expr, gpose, jaw, eyelids = load_sequence(args.npz, device)

    os.makedirs(args.out_dir, exist_ok=True)
    output_path = build_output_path(args.npz, args.wav, args.out_dir)
    duration_sec = render_sequence(expr, gpose, jaw, eyelids, output_path, device)
    print(f"Saved render to {output_path}")

    if args.wav and os.path.exists(args.wav):
        output_with_audio = str(Path(output_path).with_name(f"{Path(output_path).stem}_with_audio.mp4"))
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            output_path,
            "-i",
            args.wav,
            "-t",
            f"{duration_sec:.3f}",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-strict",
            "experimental",
            output_with_audio,
        ]
        subprocess.run(cmd, check=False)
        print(f"Saved render with audio to {output_with_audio}")
    else:
        print(f"Audio file not found: {args.wav}")


if __name__ == "__main__":
    main()
