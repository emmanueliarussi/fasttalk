import os
import pickle
import sys
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch3d.transforms import matrix_to_euler_angles


PROJECT_ROOT = Path("/mnt/fasttalk")
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from flame_model.FLAME import FLAMEModel
from renderer.renderer import Renderer


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NEUTRAL_STYLE_PATH = "/mnt/fasttalk/demo/styles/style_2.npz"
EXPRESSIVE_STYLE_PATH = "/mnt/Datasets/expressive_ft/synthetic_dataset/npz/video_012.npz"
MASK_PATH = "/mnt/fasttalk/flame_model/assets/FLAME_masks.pkl"
PREVIEW_DIR = "/mnt/fasttalk/demo/style_previews"
SAVE_DIR = "/mnt/fasttalk/demo/new_styles"


def to_numpy(array_like):
	if isinstance(array_like, torch.Tensor):
		return array_like.detach().cpu().numpy()
	return np.asarray(array_like)


def to_index_list(mask):
	array = np.asarray(mask)
	if array.dtype == bool:
		return np.where(array)[0]
	return array.astype(int).ravel()


def load_base_styles():
	neutral_style = torch.load(NEUTRAL_STYLE_PATH, map_location=DEVICE)

	expressive_npz = np.load(EXPRESSIVE_STYLE_PATH)
	exp_values = expressive_npz["exp"].reshape(-1, 50)
	global_pose_values = expressive_npz["pose"].reshape(-1, 3)
	jaw_values = expressive_npz["jaw"].reshape(-1, 3)
	eyelid_values = np.ones((exp_values.shape[0], 2), dtype=np.float32)

	expressive_concat = np.concatenate([exp_values, global_pose_values, jaw_values, eyelid_values], axis=1)
	expressive_tensor = torch.from_numpy(expressive_concat).float().unsqueeze(0).to(DEVICE)

	neutral = to_numpy(neutral_style)
	expressive = to_numpy(expressive_tensor)

	if neutral.ndim == 3 and neutral.shape[0] == 1:
		neutral = neutral[0]
	if expressive.ndim == 3 and expressive.shape[0] == 1:
		expressive = expressive[0]

	if neutral.ndim > 2:
		neutral = neutral.reshape(neutral.shape[0], -1)
	if expressive.ndim > 2:
		expressive = expressive.reshape(expressive.shape[0], -1)

	print("neutral:", neutral.shape, "expressive:", expressive.shape)
	return neutral, expressive


def discover_expression_indices(device):
	with open(MASK_PATH, "rb") as handle:
		masks = pickle.load(handle, encoding="latin1")

	def find_mask_key(keywords):
		for key in masks.keys():
			key_lower = key.lower()
			if any(word in key_lower for word in keywords):
				return key
		return None

	mouth_key = find_mask_key(["lip", "mouth"])
	brow_key = find_mask_key(["brow", "eyebrow"])
	if brow_key is None:
		brow_key = find_mask_key(["forehead", "eye_region", "eye region"])

	print("mask keys:", list(masks.keys()))
	print("mouth_key:", mouth_key, "brow_key:", brow_key)

	if mouth_key and brow_key:
		mouth_verts = to_index_list(masks[mouth_key])
		brow_verts = to_index_list(masks[brow_key])
		flame_model = FLAMEModel(n_shape=300, n_exp=50).to(device)
		expression_dirs = flame_model.shapedirs[:, :, -50:]

		mouth_energy = []
		brow_energy = []
		for component_index in range(expression_dirs.shape[-1]):
			component = expression_dirs[:, :, component_index]
			mouth_energy.append(component[mouth_verts].norm(dim=1).mean().item())
			brow_energy.append(component[brow_verts].norm(dim=1).mean().item())

		mouth_energy = np.array(mouth_energy)
		brow_energy = np.array(brow_energy)
		ratio = mouth_energy / (brow_energy + 1e-8)

		mouth_expression_indices = np.argsort(ratio)[-10:]
		brow_expression_indices = np.argsort(ratio)[:10]
	else:
		mouth_expression_indices = np.array([], dtype=int)
		brow_expression_indices = np.array([], dtype=int)

	print("mouth_exp_idx:", mouth_expression_indices)
	print("brow_exp_idx:", brow_expression_indices)
	return mouth_expression_indices, brow_expression_indices, masks, mouth_key, brow_key


def vertices_from_sequence(flame_model, blendshape_sequence):
	expr = blendshape_sequence[:, :50]
	gpose = blendshape_sequence[:, 50:53]
	jaw = blendshape_sequence[:, 53:56]

	target_shape_tensor = torch.zeros(expr.shape[0], 300, device=blendshape_sequence.device)
	identity_euler = matrix_to_euler_angles(torch.cat([torch.eye(3, device=blendshape_sequence.device)[None]], dim=0), "XYZ")
	eye_right = identity_euler.clone().squeeze()
	eye_left = identity_euler.clone().squeeze()
	eyes = torch.cat([eye_right, eye_left], dim=0).expand(expr.shape[0], -1)
	pose = torch.cat([gpose, jaw], dim=-1)

	vertices, _ = flame_model.forward(
		shape_params=target_shape_tensor,
		expression_params=expr,
		pose_params=pose,
		eye_pose_params=eyes,
	)
	return vertices


def build_micro_style(
	neutral_aligned,
	expressive_aligned,
	mouth_expression_indices,
	brow_expression_indices,
	flame_model,
	masks,
	mouth_key,
	brow_key,
):
	num_frames, num_dims = neutral_aligned.shape
	_ = num_frames

	all_indices = np.arange(num_dims)
	pose_jaw_indices = np.arange(50, min(56, num_dims))
	expression_indices = np.arange(min(50, num_dims))

	if len(mouth_expression_indices):
		mouth_expression_indices = np.array(
			[idx for idx in mouth_expression_indices if 0 <= idx < min(50, num_dims)],
			dtype=int,
		)
		if len(mouth_expression_indices) > 4:
			mouth_expression_indices = mouth_expression_indices[-4:]
	else:
		mouth_expression_indices = np.array([0, 4, 11], dtype=int)
		mouth_expression_indices = mouth_expression_indices[mouth_expression_indices < min(50, num_dims)]

	if len(brow_expression_indices):
		brow_expression_indices = np.array(
			[idx for idx in brow_expression_indices if 0 <= idx < min(50, num_dims)],
			dtype=int,
		)
	else:
		brow_expression_indices = np.setdiff1d(expression_indices, mouth_expression_indices)[:10]

	protected_indices = np.unique(np.concatenate([mouth_expression_indices, pose_jaw_indices]))
	inject_indices = np.setdiff1d(all_indices, protected_indices)

	print("Protected dims:", protected_indices.tolist())
	print("Inject dims count:", len(inject_indices))

	neutral_tensor = torch.from_numpy(neutral_aligned).float().to(DEVICE)
	expressive_tensor = torch.from_numpy(expressive_aligned).float().to(DEVICE)

	with torch.no_grad():
		neutral_vertices = vertices_from_sequence(flame_model, neutral_tensor)
		expressive_vertices = vertices_from_sequence(flame_model, expressive_tensor)

	num_verts = neutral_vertices.shape[1]
	all_verts = np.arange(num_verts)

	lip_verts = np.array([], dtype=int)
	if mouth_key is not None and mouth_key in masks:
		lip_verts = to_index_list(masks[mouth_key]).astype(int)
	elif "lips" in masks:
		lip_verts = to_index_list(masks["lips"]).astype(int)

	lip_verts = lip_verts[(lip_verts >= 0) & (lip_verts < num_verts)]
	if "face" in masks:
		face_verts = to_index_list(masks["face"]).astype(int)
		face_verts = face_verts[(face_verts >= 0) & (face_verts < num_verts)]
		nonlip_verts = np.setdiff1d(face_verts, lip_verts)
	else:
		nonlip_verts = np.setdiff1d(all_verts, lip_verts)

	brow_verts = np.array([], dtype=int)
	if brow_key is not None and brow_key in masks:
		brow_verts = to_index_list(masks[brow_key]).astype(int)
	brow_verts = brow_verts[(brow_verts >= 0) & (brow_verts < num_verts)]
	if len(brow_verts) == 0:
		brow_verts = nonlip_verts

	if len(nonlip_verts) == 0:
		nonlip_verts = all_verts

	lip_idx_t = torch.as_tensor(lip_verts, device=DEVICE, dtype=torch.long)
	nonlip_idx_t = torch.as_tensor(nonlip_verts, device=DEVICE, dtype=torch.long)
	brow_idx_t = torch.as_tensor(brow_verts, device=DEVICE, dtype=torch.long)
	inject_idx_t = torch.as_tensor(inject_indices, device=DEVICE, dtype=torch.long)
	brow_dim_idx_t = torch.as_tensor(brow_expression_indices, device=DEVICE, dtype=torch.long)

	# Start slightly toward expressive so the optimizer does not stay too close to neutral
	init_candidate = neutral_tensor[:, inject_idx_t] + 0.35 * (
		expressive_tensor[:, inject_idx_t] - neutral_tensor[:, inject_idx_t]
	)
	opt_var = torch.nn.Parameter(init_candidate.clone())
	optimizer = torch.optim.Adam([opt_var], lr=0.03)

	with torch.no_grad():
		base = neutral_tensor[:, inject_idx_t]
		target = expressive_tensor[:, inject_idx_t]
		delta = torch.abs(target - base)
		lower = torch.minimum(base, target) - 4.0 * delta
		upper = torch.maximum(base, target) + 4.0 * delta

	iterations = 500
	brow_pop = 3.0
	dyn_pop = 2.3
	for step in range(iterations):
		optimizer.zero_grad()

		candidate = neutral_tensor.clone()
		candidate[:, inject_idx_t] = opt_var
		candidate[:, protected_indices] = neutral_tensor[:, protected_indices]

		candidate_vertices = vertices_from_sequence(flame_model, candidate)

		if lip_idx_t.numel() > 0:
			lip_loss = torch.nn.functional.smooth_l1_loss(
				candidate_vertices[:, lip_idx_t],
				neutral_vertices[:, lip_idx_t],
			)
		else:
			lip_loss = torch.zeros([], device=DEVICE)

		nonlip_target = neutral_vertices[:, nonlip_idx_t] + 1.6 * (
			expressive_vertices[:, nonlip_idx_t] - neutral_vertices[:, nonlip_idx_t]
		)
		nonlip_loss = torch.nn.functional.smooth_l1_loss(
			candidate_vertices[:, nonlip_idx_t],
			nonlip_target,
		)

		brow_target = neutral_vertices[:, brow_idx_t] + brow_pop * (
			expressive_vertices[:, brow_idx_t] - neutral_vertices[:, brow_idx_t]
		)
		brow_loss = torch.nn.functional.smooth_l1_loss(
			candidate_vertices[:, brow_idx_t],
			brow_target,
		)

		if candidate_vertices.shape[0] > 1:
			dyn_candidate = candidate_vertices[1:, brow_idx_t] - candidate_vertices[:-1, brow_idx_t]
			dyn_expressive = expressive_vertices[1:, brow_idx_t] - expressive_vertices[:-1, brow_idx_t]
			dyn_target = dyn_pop * dyn_expressive
			dyn_loss = torch.nn.functional.smooth_l1_loss(dyn_candidate, dyn_target)
		else:
			dyn_loss = torch.zeros([], device=DEVICE)

		if candidate.shape[0] > 2:
			acc = candidate[2:, inject_idx_t] - 2 * candidate[1:-1, inject_idx_t] + candidate[:-2, inject_idx_t]
			temp_loss = (acc.pow(2)).mean()
		else:
			temp_loss = torch.zeros([], device=DEVICE)

		if brow_dim_idx_t.numel() > 0:
			candidate_brow = candidate[:, brow_dim_idx_t]
			brow_param_target = neutral_tensor[:, brow_dim_idx_t] + 2.8 * (
				expressive_tensor[:, brow_dim_idx_t] - neutral_tensor[:, brow_dim_idx_t]
			)
			brow_param_loss = torch.nn.functional.smooth_l1_loss(candidate_brow, brow_param_target)
		else:
			brow_param_loss = torch.zeros([], device=DEVICE)

		reg_loss = torch.nn.functional.mse_loss(candidate[:, inject_idx_t], neutral_tensor[:, inject_idx_t])

		loss = (
			80.0 * lip_loss
			+ 2.0 * nonlip_loss
			+ 40.0 * brow_loss
			+ 25.0 * dyn_loss
			+ 10.0 * brow_param_loss
			+ 0.10 * temp_loss
			+ 0.001 * reg_loss
		)
		loss.backward()
		torch.nn.utils.clip_grad_norm_([opt_var], max_norm=2.0)
		optimizer.step()

		with torch.no_grad():
			opt_var.clamp_(lower, upper)

		if step in {0, 49, 99, 149, 249, 349, 449, 499}:
			print(
				f"iter={step + 1}/{iterations}",
				f"total={loss.item():.4f}",
				f"lip={lip_loss.item():.4f}",
				f"nonlip={nonlip_loss.item():.4f}",
				f"brow={brow_loss.item():.4f}",
				f"dyn={dyn_loss.item():.4f}",
			)

	with torch.no_grad():
		final_candidate = neutral_tensor.clone()
		final_candidate[:, inject_idx_t] = opt_var
		final_candidate[:, protected_indices] = neutral_tensor[:, protected_indices]
		if brow_dim_idx_t.numel() > 0:
			final_candidate[:, brow_dim_idx_t] = neutral_tensor[:, brow_dim_idx_t] + 1.25 * (
				final_candidate[:, brow_dim_idx_t] - neutral_tensor[:, brow_dim_idx_t]
			)
		micro_style = final_candidate.detach().cpu().numpy()

	print("micro_style:", micro_style.shape)
	return micro_style


def get_vertices_from_blendshapes(flame_model, expr, gpose, jaw, eyelids):
	expr_tensor = expr.to(DEVICE)
	gpose_tensor = gpose.to(DEVICE)
	jaw_tensor = jaw.to(DEVICE)

	target_shape_tensor = torch.zeros(expr_tensor.shape[0], 300).expand(expr_tensor.shape[0], -1).to(DEVICE)
	identity_euler = matrix_to_euler_angles(torch.cat([torch.eye(3)[None]], dim=0), "XYZ").to(DEVICE)
	eye_right = identity_euler.clone().to(DEVICE).squeeze()
	eye_left = identity_euler.clone().to(DEVICE).squeeze()
	eyes = torch.cat([eye_right, eye_left], dim=0).expand(expr_tensor.shape[0], -1).to(DEVICE)
	pose = torch.cat([gpose_tensor, jaw_tensor], dim=-1).to(DEVICE)

	flame_output, _ = flame_model.forward(
		shape_params=target_shape_tensor,
		expression_params=expr_tensor,
		pose_params=pose,
		eye_pose_params=eyes,
	)
	return flame_output.detach()


def render_style(renderer, flame_model, blendshape_sequence, name):
	expr_pred = blendshape_sequence[:, :50]
	gpose_pred = blendshape_sequence[:, 50:53]
	jaw_pred = blendshape_sequence[:, 53:56]
	eyelids_pred = blendshape_sequence[:, 56:]

	vertices = get_vertices_from_blendshapes(flame_model, expr_pred, gpose_pred, jaw_pred, eyelids_pred)
	camera = torch.tensor([5, 0, 0], dtype=torch.float32).unsqueeze(0).to(vertices.device)
	camera = camera.expand(vertices.shape[0], -1)
	rendered_frames = renderer.forward(vertices, camera)["rendered_img"]

	os.makedirs(PREVIEW_DIR, exist_ok=True)
	video_file = f"{PREVIEW_DIR}/{name}.mp4"

	def update(frame_index, sequence, axis):
		frame = sequence[frame_index].detach().cpu().numpy().transpose(1, 2, 0)
		image = (frame * 255).astype(np.uint8)
		axis.clear()
		axis.imshow(image)
		axis.axis("off")

	figure, axis = plt.subplots(figsize=(5, 5))
	animation_handle = animation.FuncAnimation(
		figure,
		update,
		frames=rendered_frames.shape[0],
		fargs=(rendered_frames, axis),
		interval=100,
	)
	animation_handle.save(video_file, writer="ffmpeg", fps=25)
	plt.close(figure)
	print(f"Saved render to {video_file}")


def main():
	neutral, expressive = load_base_styles()
	mouth_expression_indices, brow_expression_indices, masks, mouth_key, brow_key = discover_expression_indices(DEVICE)
	flame_model = FLAMEModel(n_shape=300, n_exp=50).to(DEVICE)

	min_len = min(neutral.shape[0], expressive.shape[0])
	neutral_aligned = neutral[:min_len]
	expressive_aligned = expressive[:min_len]

	print("neutral_aligned:", neutral_aligned.shape)
	print("expressive_aligned:", expressive_aligned.shape)

	micro_style = build_micro_style(
		neutral_aligned,
		expressive_aligned,
		mouth_expression_indices,
		brow_expression_indices,
		flame_model,
		masks,
		mouth_key,
		brow_key,
	)

	renderer = Renderer(render_full_head=True).to(DEVICE)

	neutral_tensor = torch.from_numpy(neutral_aligned).float().to(DEVICE)
	expressive_tensor = torch.from_numpy(expressive_aligned).float().to(DEVICE)
	micro_style_tensor = torch.from_numpy(micro_style).float().to(DEVICE)

	#render_style(renderer, flame_model, neutral_tensor, name="neutral_style")
	#render_style(renderer, flame_model, expressive_tensor, name="expressive_style")
	render_style(renderer, flame_model, micro_style_tensor, name="micro_style")

	os.makedirs(SAVE_DIR, exist_ok=True)
	micro_style_out = torch.from_numpy(micro_style).unsqueeze(0).to(DEVICE)
	save_path = f"{SAVE_DIR}/micro_style.npz"
	torch.save(micro_style_out, save_path)
	print(f"Saved: {save_path}")


if __name__ == "__main__":
	main()
