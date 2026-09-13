"""
Ground-truth 3D point-flow tracks from RoboTwin simulator state.

Unlike scripts/generate_training_data.py (CoTracker + depth reprojection, an
approximate/drifting 2D-tracking-based method), this script exploits that
RoboTwin is a simulator: every visible surface point belongs to some rigid
link, and that link's exact world-frame trajectory is available by replaying
the episode. A point's future position is reconstructed by rigidly
transforming its clip-start local coordinate through the owning link's
recorded pose — no tracking, no drift.

=== Usage ===

  python scripts/generate_gt_tracks.py \
      --task_name stack_bowls_three \
      --task_config demo_clean \
      --num_episodes 100 \
      --output_file ../JAMB/data/tracks/stack_bowls_three-demo_clean-100-tracks_flat_gt.hdf5 \
      --raw_data_root ../RoboTwin/data \
      --extract_pi3_features --extract_dino_features --extract_vae_features

Must run from an environment that can launch RoboTwin's SAPIEN sim (same
conda env / CUDA_VISIBLE_DEVICES as collect_data.sh) since it replays each
episode's cached dense joint path to recover ground-truth object poses.

=== Output HDF5 structure (flat, same convention as generate_training_data.py) ===

  dataset.hdf5
  ├── image                           (T, H, W, 3)   uint8
  ├── image_patch_centre_3d_position  (T, N, 3)       float32
  ├── robot_state                     (T, 16)         float32  (EEF pose, both arms)
  ├── action                          (T, 16)         float32  (EEF pose, next-state)
  ├── 3d_track                        (T, N, H, 3)    float32
  ├── track_valid_mask                (T, N, H)       bool     (constant across H for this ground-truth
  │                                                              replay, so also serves as valid_point_mask
  │                                                              — see track_valid_mask.any(axis=-1) —
  │                                                              which is therefore NOT stored separately)
  ├── normals_3d                      (T, N, 3)       float32  [optional, --use_normal_estimation]
  ├── pi3_feature                     (T, 1, N, 1024) float32  [optional, --extract_pi3_features]
  ├── dino_features                   (T, 1, N, 1024) float32  [optional, --extract_dino_features;
  │                                                              dinov2_vitl14_reg despite the generic name,
  │                                                              matching the key TrackDataset expects]
  ├── vae_latent                      (T, 4, H/8, W/8) float32  [optional, --extract_vae_features;
  │                                                              raw (unscaled) SDXL-VAE mean latent of
  │                                                              the native RGB image, no channel swap]
  ├── vae_patch_centers_3d_position   (T, M, 3)        float32  [optional, --extract_vae_features;
  │                                                              M = (H/8/2)*(W/8/2) VAE-token pixel
  │                                                              centers (patch_size=16, native image),
  │                                                              unprojected via depth+intrinsics the
  │                                                              same way as image_patch_centre_3d_position
  │                                                              — NOT resampled from that N-patch field]
  ├── vae_valid_mask                  (T, M)          bool     [optional, --extract_vae_features;
  │                                                              depth validity at each VAE token center]
  ├── episode_ends                    (num_episodes,) int64

  N = number of 14px patch centers on the round-to-nearest-14 RESIZED image
      (e.g. 391 for a native 240x320 frame -> 238x322) — this matches Pi3's
      and DINOv2's own internal resize exactly, so image_patch_centre_3d_position
      /3d_track line up 1:1 with pi3_feature/dino_features patch tokens.
  M = number of VAE 2x2-patchify tokens on the NATIVE (unresized) image's
      8x-downsampled latent grid (e.g. 300 = 15x20 for a native 240x320
      frame -> 30x40 latent) — a deliberately different grid from N/H/W
      above, matching scripts/add_vae_to_tracks.py's convention.
  H = action_horizon
"""

import os
import sys
import argparse
import importlib
import numpy as np
import cv2
import h5py
import torch
import yaml
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")  # avoids spurious
# BlockingIOError on this shared filesystem (same workaround already used by
# scripts/visualize_2d_tracks.py and the JAMB codebase's verify_track_data.py)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROBOTWIN_ROOT = PROJECT_ROOT / "RoboTwin"
JAMB_ROOT = PROJECT_ROOT / "JAMB"

# Any relative CLI paths (--output_file, --raw_data_root) are resolved
# against this, before the chdir below moves us into RoboTwin's root.
_LAUNCH_CWD = os.getcwd()

sys.path.insert(0, str(ROBOTWIN_ROOT))
sys.path.insert(0, str(JAMB_ROOT))
sys.path.insert(0, str(JAMB_ROOT / "thirdparty"))
# envs/__init__.py reads paths (e.g. "./assets/objects/objaverse/list.json")
# relative to the CWD at import time, so this must run before importing envs
# regardless of where this script was launched from.
os.chdir(ROBOTWIN_ROOT)

from envs._GLOBAL_CONFIGS import CONFIGS_PATH  # noqa: E402


# ---------------------------------------------------------------------------
# Reused helpers (same math/convention as scripts/generate_training_data.py)
# ---------------------------------------------------------------------------
def compute_patch_centers(img_h: int, img_w: int, patch_size: int = 14):
    ph = img_h // patch_size
    pw = img_w // patch_size
    cy = (np.arange(ph) + 0.5) * patch_size
    cx = (np.arange(pw) + 0.5) * patch_size
    grid_x, grid_y = np.meshgrid(cx, cy)
    return np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)  # (N, 2)


def unproject_pixels_to_3d(pixel_xy: np.ndarray, depth_map: np.ndarray, intrinsic: np.ndarray,
                            depth_scale: float = 1000.0):
    N = pixel_xy.shape[0]
    x, y = pixel_xy[:, 0], pixel_xy[:, 1]
    H, W = depth_map.shape
    x_clamped = np.clip(x, 0, W - 1)
    y_clamped = np.clip(y, 0, H - 1)
    map_x = x_clamped.astype(np.float32).reshape(1, N)
    map_y = y_clamped.astype(np.float32).reshape(1, N)
    depth_sampled = cv2.remap(depth_map.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR).reshape(N)
    depth_metres = depth_sampled / depth_scale
    valid = (depth_metres > 0) & np.isfinite(depth_metres)

    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    z = depth_metres
    x_3d = (x_clamped - cx) * z / fx
    y_3d = (y_clamped - cy) * z / fy
    points_3d = np.stack([x_3d, y_3d, z], axis=-1)
    points_3d[~valid] = 0.0
    return points_3d, valid


def unproject_integer_pixels_to_3d(pixel_xy: np.ndarray, depth_map: np.ndarray, intrinsic: np.ndarray,
                                    depth_scale: float = 1000.0):
    """Same formula/convention as unproject_pixels_to_3d, but for pixel_xy
    guaranteed to already be exact integers — direct array indexing instead
    of cv2.remap bilinear sampling, since there's no sub-pixel position to
    interpolate (see VAE token centers in extract_vae_features)."""
    x = np.clip(pixel_xy[:, 0].astype(np.int64), 0, depth_map.shape[1] - 1)
    y = np.clip(pixel_xy[:, 1].astype(np.int64), 0, depth_map.shape[0] - 1)
    depth_metres = depth_map[y, x].astype(np.float32) / depth_scale
    valid = (depth_metres > 0) & np.isfinite(depth_metres)

    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    z = depth_metres
    x_3d = (x - cx) * z / fx
    y_3d = (y - cy) * z / fy
    points_3d = np.stack([x_3d, y_3d, z], axis=-1)
    points_3d[~valid] = 0.0
    return points_3d, valid


def estimate_normals(points_3d: np.ndarray, radius: float = 0.05, max_nn: int = 30,
                      camera_location: np.ndarray = np.zeros(3)):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d.astype(np.float64))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    pcd.orient_normals_towards_camera_location(camera_location)
    return np.asarray(pcd.normals).astype(np.float32)


def resize_target_hw(h_orig: int, w_orig: int, patch_size: int = 14):
    """Round-to-nearest-multiple-of-patch_size, matching both Pi3's and
    DINOv2's own internal resize convention, so their native patch grids line
    up exactly with compute_patch_centers(*resize_target_hw(...))."""
    target_h = max(round(h_orig / patch_size) * patch_size, patch_size)
    target_w = max(round(w_orig / patch_size) * patch_size, patch_size)
    return target_h, target_w


# ---------------------------------------------------------------------------
# Pi3 feature extraction (verbatim from scripts/generate_training_data.py)
# ---------------------------------------------------------------------------
def load_pi3_model(model_path: str, device: torch.device):
    from pi3.models.pi3 import Pi3

    return Pi3.from_pretrained(model_path).to(device).eval()


def extract_pi3_features(model, images_rgb: list, device: torch.device):
    """Returns (features (T, N, 1024), (target_h, target_w))."""
    B = len(images_rgb)
    h_orig, w_orig = images_rgb[0].shape[:2]
    target_h, target_w = resize_target_hw(h_orig, w_orig)

    imgs = []
    for img in images_rgb:
        img_resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        img_t = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        imgs.append(img_t)

    imgs_tensor = torch.stack(imgs, dim=0).unsqueeze(1).to(device)  # (B, 1, 3, H, W)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    all_features = []
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=dtype):
            for b in range(B):
                inp = imgs_tensor[b:b + 1]  # (1, 1, 3, H, W)
                inp = (inp - model.image_mean) / model.image_std
                flat = inp.reshape(1, 3, target_h, target_w)
                hidden = model.encoder(flat, is_training=True)
                if isinstance(hidden, dict):
                    hidden = hidden["x_norm_patchtokens"]
                hidden, pos = model.decode(hidden, 1, target_h, target_w)
                point_hidden = model.point_decoder(hidden, xpos=pos)
                feats = point_hidden[:, model.patch_start_idx:].float()  # (1, N, 1024)
                all_features.append(feats.squeeze(0))

    features = torch.stack(all_features, dim=0).cpu().numpy()  # (B, N, 1024)
    return features, (target_h, target_w)


# ---------------------------------------------------------------------------
# DINOv2 (with register tokens) feature extraction
# ---------------------------------------------------------------------------
def load_dinov2_reg_model(device: torch.device):
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg")
    return model.to(device).eval()


def extract_dinov2_reg_features(model, images_rgb: list, device: torch.device, batch_size: int = 32):
    """Returns (features (T, N, 1024), (target_h, target_w)) — same resize
    convention as extract_pi3_features, so patch grids align."""
    h_orig, w_orig = images_rgb[0].shape[:2]
    target_h, target_w = resize_target_hw(h_orig, w_orig)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    all_features = []
    with torch.no_grad():
        for start in range(0, len(images_rgb), batch_size):
            chunk = images_rgb[start:start + batch_size]
            imgs = []
            for img in chunk:
                img_resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                imgs.append(torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0)
            batch = torch.stack(imgs, dim=0).to(device)
            batch = (batch - mean) / std
            feats = model.forward_features(batch)["x_norm_patchtokens"]  # (b, N, 1024)
            all_features.append(feats.cpu())

    features = torch.cat(all_features, dim=0).numpy()  # (T, N, 1024)
    return features, (target_h, target_w)


# ---------------------------------------------------------------------------
# SDXL-VAE latent extraction (same convention as scripts/add_vae_to_tracks.py)
# ---------------------------------------------------------------------------
def load_vae_model(model_path: str, device: torch.device):
    # diffusers 0.38 + torch 2.4.1: AutoencoderKL import crashes on a flash-attn-3
    # custom-op schema. We never use that op (VAE enc/dec are pure conv). No-op
    # the two registrations before importing diffusers.
    import torch.library as _tl

    def _noop(op, fn=None, *a, **k):
        def wrap(f):
            return f
        return wrap if fn is None else fn

    _tl.custom_op = _noop
    _tl.register_fake = _noop
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(model_path).to(device).eval()
    return vae.to(torch.float32)


def extract_vae_features(vae_model, images_rgb: list, depth_maps, intrinsic: np.ndarray, extrinsic: np.ndarray,
                          device: torch.device, batch_size: int = 32):
    """Returns (vae_latent (T, 4, lat_h, lat_w), vae_patch_centers (T, M, 3), vae_valid (T, M), (lat_h, lat_w)).

    Deliberately operates on the NATIVE image resolution (SDXL-VAE's own 8x
    downsample), not the Pi3/DINOv2 round-to-nearest-14 target_h/target_w —
    matching scripts/add_vae_to_tracks.py's convention. vae_patch_centers
    lives on the coarser 2x2-patchify token grid (row-major): its pixel
    centers are computed and unprojected via depth + intrinsics the SAME
    way the main N-patch grid is (2D pixel center -> unproject -> 3D),
    rather than by resampling the already-unprojected N-patch 3D field —
    that avoids blending 3D positions across depth discontinuities (a
    silhouette edge would otherwise produce a "phantom" point floating
    between foreground and background). Because these centers are defined
    directly on the native image at patch_size=16 (2 latent cells x 8px
    downsample, which evenly divides the native H/W given the asserts
    below), they land on exact integer native pixels — unlike the
    Pi3/DINOv2 grid, whose round-to-nearest-14 resize makes ITS centers
    land at sub-pixel native coordinates — so depth is read by direct
    indexing (unproject_integer_pixels_to_3d), not bilinear cv2.remap.
    images_rgb is already true RGB (same arrays passed to Pi3/DINOv2 above)
    — no channel swap is applied here.
    """
    T = len(images_rgb)
    h_img, w_img = images_rgb[0].shape[:2]
    assert h_img % 8 == 0 and w_img % 8 == 0, f"VAE needs image dims divisible by 8, got {h_img}x{w_img}"
    lat_h, lat_w = h_img // 8, w_img // 8
    assert lat_h % 2 == 0 and lat_w % 2 == 0, f"VAE 2x2 token patchify needs even latent dims, got {lat_h}x{lat_w}"
    vae_grid_h, vae_grid_w = lat_h // 2, lat_w // 2
    M = vae_grid_h * vae_grid_w

    patch_centers_px = compute_patch_centers(h_img, w_img, patch_size=16)  # (M, 2), exact-integer native pixels
    R = extrinsic[:, :3]
    t_vec = extrinsic[:, 3]

    all_centers = np.zeros((T, M, 3), dtype=np.float32)
    all_valid = np.zeros((T, M), dtype=bool)
    for t in range(T):
        cam_3d, valid = unproject_integer_pixels_to_3d(patch_centers_px, depth_maps[t], intrinsic)
        all_centers[t] = np.matmul(cam_3d - t_vec, R)
        all_valid[t] = valid

    all_latents = np.zeros((T, 4, lat_h, lat_w), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            chunk = np.stack(images_rgb[start:end], axis=0)  # (b, H, W, 3) uint8, true RGB
            x = torch.from_numpy(chunk).to(device).float() / 255.0
            x = x.permute(0, 3, 1, 2)  # (b, 3, H, W) — channel order unchanged (RGB in, RGB out)
            x = x * 2.0 - 1.0  # SDXL VAE expects inputs in [-1, 1]
            lat = vae_model.encode(x).latent_dist.mean  # (b, 4, lat_h, lat_w), raw/unscaled
            all_latents[start:end] = lat.float().cpu().numpy()

    return all_latents, all_centers, all_valid, (lat_h, lat_w)


def load_robotwin_episode(episode_path: str, camera_name: str = "head_camera"):
    """Returns (images[true RGB, uint8], depth_maps[mm], eef_poses(T,16), intrinsic(3,3), extrinsic(3,4))."""
    images, depth_maps = [], []
    with h5py.File(episode_path, "r") as f:
        endpose = f["endpose"]
        left_pose = endpose["left_endpose"][()]
        left_grip = endpose["left_gripper"][()].reshape(-1, 1)
        right_pose = endpose["right_endpose"][()]
        right_grip = endpose["right_gripper"][()].reshape(-1, 1)
        eef_poses = np.concatenate([left_pose, left_grip, right_pose, right_grip], axis=-1)

        cam_data = f["/observation"][camera_name]
        intrinsic = cam_data["intrinsic_cv"][0]
        extrinsic = cam_data["extrinsic_cv"][0]

        rgb_data = cam_data["rgb"][()]
        for j in range(len(rgb_data)):
            # RoboTwin encodes true-RGB arrays via cv2.imencode without a BGR
            # swap, so IMREAD_COLOR decoding hands the same true-RGB values
            # straight back (verified against envs/utils/pkl2hdf5.py).
            images.append(cv2.imdecode(np.frombuffer(rgb_data[j], np.uint8), cv2.IMREAD_COLOR))

        depth_data = cam_data["depth"][()]
        for j in range(len(depth_data)):
            depth_maps.append(depth_data[j])

    return images, depth_maps, eef_poses, intrinsic, extrinsic


# ---------------------------------------------------------------------------
# RoboTwin task/env construction (mirrors RoboTwin/script/collect_data.py's main())
# ---------------------------------------------------------------------------
def build_task_and_args(task_name: str, task_config: str, raw_data_root: str):
    envs_module = importlib.import_module(f"envs.{task_name}")
    env_class = getattr(envs_module, task_name)
    task = env_class()

    config_path = os.path.join(CONFIGS_PATH, f"{task_config}.yml")
    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    args["task_name"] = task_name

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(name):
        robot_file = embodiment_types[name]["file_path"]
        if robot_file is None:
            raise ValueError(f"missing embodiment files for {name}")
        return robot_file

    def get_embodiment_config(robot_file):
        with open(os.path.join(robot_file, "config.yml"), "r", encoding="utf-8") as cf:
            return yaml.load(cf.read(), Loader=yaml.FullLoader)

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("number of embodiment config parameters should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    args["embodiment_name"] = (str(embodiment_type[0]) if len(embodiment_type) == 1 else
                                f"{embodiment_type[0]}+{embodiment_type[1]}")
    args["task_config"] = task_config
    args["save_path"] = os.path.join(str(raw_data_root), task_name, task_config)

    # Headless ground-truth replay: no live viewer, replay cached path (no
    # re-planning), no pkl/hdf5 disk writes (we override _take_picture below).
    args["render_freq"] = 0
    args["need_plan"] = False
    args["save_data"] = False

    return task, args


BACKGROUND_NAMES = {"table", "wall", "ground"}


def get_gripper_joint_names(args):
    """
    Joint names (not link names) that drive the gripper fingers, from the
    embodiment config's `gripper_name` (a "base" joint plus its "mimic"
    joints) — distinct from `arm_joints_name`'s 6 arm joints per side.
    Link-name suffixes (e.g. "_link7") aren't reliable across embodiments
    (the wrist camera mount is also a direct child of the last arm link),
    so links are matched via their driving joint's name instead.
    """
    names = set()
    for cfg in (args["left_embodiment_config"], args["right_embodiment_config"]):
        for entry in cfg.get("gripper_name", []):
            names.add(entry["base"])
            for mimic in entry.get("mimic", []):
                names.add(mimic[0])
    return names


def classify_actors(env, gripper_joint_names):
    """
    robot_ids: per_scene_id set for robot links that are part of the arm
    proper (base, arm links, wheels, camera mounts, ...) — excluded from
    tracking, since their pose is already exactly known via robot_state/action.
    object_owners: dict[per_scene_id -> pose-bearing sapien object] for every
    task object, PLUS the gripper finger links (identified via
    gripper_joint_names) — tracked like any other "visible link" per the
    paper, since only the arm itself (not the gripper) is filtered out.
    """
    robot_articulations = {env.robot.left_entity, env.robot.right_entity}

    robot_ids = set()
    gripper_links = {}
    for art in robot_articulations:
        for link in art.get_links():
            pid = link.entity.per_scene_id
            joint = link.joint
            if joint is not None and joint.name in gripper_joint_names:
                gripper_links[pid] = link
            else:
                robot_ids.add(pid)

    object_owners = dict(gripper_links)
    for actor in env.scene.get_all_actors():
        if actor.name in BACKGROUND_NAMES:
            continue
        object_owners[actor.per_scene_id] = actor

    for art in env.scene.get_all_articulations():
        if art in robot_articulations:
            continue
        for link in art.get_links():
            object_owners[link.entity.per_scene_id] = link

    assert object_owners, "No trackable (non-arm, non-background) objects found in the scene."
    return robot_ids, object_owners


def get_world_transform(obj):
    pose = obj.get_entity_pose() if hasattr(obj, "get_entity_pose") else obj.get_pose()
    return pose.to_transformation_matrix()


# ---------------------------------------------------------------------------
# Ground-truth replay: capture per-frame segmentation + object poses
# ---------------------------------------------------------------------------
def replay_episode_ground_truth(task, args, episode_idx: int, seed: int, camera_name: str = "head_camera"):
    task.setup_demo(now_ep_num=episode_idx, seed=seed, **args)
    gripper_joint_names = get_gripper_joint_names(args)
    robot_ids, object_owners = classify_actors(task, gripper_joint_names)

    capture = {"seg": [], "poses": {pid: [] for pid in object_owners}}

    def _gt_take_picture():
        task._update_render()
        task.cameras.update_picture()
        cam_idx = task.cameras.static_camera_name.index(camera_name)
        cam = task.cameras.static_camera_list[cam_idx]
        capture["seg"].append(cam.get_picture("Segmentation")[..., 1].copy())
        for pid, owner in object_owners.items():
            capture["poses"][pid].append(get_world_transform(owner))
        task.FRAME_IDX += 1
        if task.FRAME_IDX % 50 == 0:
            print(f"    replay frame {task.FRAME_IDX}", end="\r")

    task._take_picture = _gt_take_picture

    traj_data = task.load_tran_data(episode_idx)
    args["left_joint_path"] = traj_data["left_joint_path"]
    args["right_joint_path"] = traj_data["right_joint_path"]
    task.set_path_lst(args)

    task.play_once()
    task.close_env()

    seg_stack = np.stack(capture["seg"], axis=0)  # (T, H, W) uint32
    pose_stack = {pid: np.stack(p, axis=0) for pid, p in capture["poses"].items()}  # each (T, 4, 4)
    return seg_stack, pose_stack, robot_ids


# ---------------------------------------------------------------------------
# Core pipeline: turn one episode's replay capture into flat track arrays
# ---------------------------------------------------------------------------
def build_episode_tracks(images, depth_maps, eef_poses, intrinsic, extrinsic,
                          seg_stack, pose_stack, robot_ids,
                          action_horizon: int, use_normal_estimation: bool = False,
                          pi3_model=None, dinov2_model=None, vae_model=None, device: torch.device = None):
    T_total = len(images)
    assert seg_stack.shape[0] == T_total, (
        f"replay produced {seg_stack.shape[0]} frames but the original episode has {T_total} — "
        "ground-truth replay is not aligned with the saved episode (check save_freq/task_config match)."
    )
    H_action = action_horizon
    h_img, w_img = images[0].shape[:2]
    # Patch grid is defined on the round-to-nearest-14 RESIZED resolution
    # (matching Pi3's/DINOv2's own internal resize), not the native image, so
    # that image_patch_centre_3d_position/3d_track line up 1:1 with
    # pi3_feature/dino_features patch tokens when those are extracted below.
    target_h, target_w = resize_target_hw(h_img, w_img)
    patch_centers = compute_patch_centers(target_h, target_w, patch_size=14)  # (N, 2), resized-grid coords
    N = patch_centers.shape[0]
    scale_x, scale_y = w_img / target_w, h_img / target_h
    patch_centers_orig = patch_centers * [scale_x, scale_y]  # (N, 2), native-resolution pixel coords

    R = extrinsic[:, :3]
    t_vec = extrinsic[:, 3]

    all_cam_3d = np.zeros((T_total, N, 3), dtype=np.float32)
    all_depth_valid = np.zeros((T_total, N), dtype=bool)
    for t in range(T_total):
        pts, valid = unproject_pixels_to_3d(patch_centers_orig, depth_maps[t], intrinsic)
        all_cam_3d[t] = pts
        all_depth_valid[t] = valid
    all_world_3d = np.matmul(all_cam_3d - t_vec, R).astype(np.float32)  # (T, N, 3)

    px = np.clip(np.round(patch_centers_orig[:, 0]).astype(int), 0, w_img - 1)
    py = np.clip(np.round(patch_centers_orig[:, 1]).astype(int), 0, h_img - 1)
    seg_at_patches = seg_stack[:, py, px]  # (T, N)

    owner_ids = np.array(list(pose_stack.keys()))
    id_to_row = {pid: i for i, pid in enumerate(owner_ids)}
    owner_transforms = np.stack([pose_stack[pid] for pid in owner_ids], axis=0)  # (num_owners, T, 4, 4)
    owner_transforms_inv = np.linalg.inv(owner_transforms)

    is_object = np.isin(seg_at_patches, owner_ids)
    valid_point_mask = all_depth_valid & is_object  # (T, N)

    owner_row_of = np.zeros((T_total, N), dtype=np.int64)
    for pid, row in id_to_row.items():
        owner_row_of[seg_at_patches == pid] = row

    n_robot = np.isin(seg_at_patches, np.array(list(robot_ids)) if robot_ids else np.array([])).sum()
    n_object = is_object.sum()
    n_total = T_total * N
    print(f"    patch-point classification: object={n_object} ({100 * n_object / n_total:.1f}%)  "
          f"robot={n_robot} ({100 * n_robot / n_total:.1f}%)  "
          f"background/other={n_total - n_object - n_robot} "
          f"({100 * (n_total - n_object - n_robot) / n_total:.1f}%)")

    track = np.zeros((T_total, N, H_action, 3), dtype=np.float32)
    track_valid = np.zeros((T_total, N, H_action), dtype=bool)

    for t in range(T_total):
        valid_idx = np.where(valid_point_mask[t])[0]
        if valid_idx.size == 0:
            continue
        rows = owner_row_of[t, valid_idx]
        p_world_anchor = all_world_3d[t, valid_idx]  # (n_valid, 3)
        ones = np.ones((valid_idx.size, 1), dtype=np.float32)
        p_world_anchor_h = np.concatenate([p_world_anchor, ones], axis=-1)
        p_local = np.einsum("nij,nj->ni", owner_transforms_inv[rows, t], p_world_anchor_h)[:, :3]
        p_local_h = np.concatenate([p_local, ones], axis=-1)  # reused across all h

        for h in range(H_action):
            future_t = min(t + h + 1, T_total - 1)
            p_world_future = np.einsum("nij,nj->ni", owner_transforms[rows, future_t], p_local_h)[:, :3]
            track[t, valid_idx, h, :] = p_world_future - p_world_anchor
            track_valid[t, valid_idx, h] = True

    all_images = np.stack(images, axis=0)  # (T, H, W, 3) uint8, true RGB

    result = {
        "image": all_images,
        "image_patch_centre_3d_position": all_world_3d,
        # valid_point_mask is deliberately NOT stored: for this ground-truth
        # replay (unlike generate_training_data.py's CoTracker path, where
        # DBSCAN outlier removal can make it a stricter subset), it is
        # exactly track_valid_mask[..., h] for every h — see the h-loop
        # below, which sets track_valid uniformly from valid_point_mask with
        # no further per-h filtering. Consumers should derive it via
        # track_valid_mask.any(axis=-1) (equivalently [..., 0]) if needed.
        "robot_state": eef_poses.astype(np.float32),
        "action": np.concatenate([eef_poses[1:], eef_poses[-1:]], axis=0).astype(np.float32),
        "3d_track": track,
        "track_valid_mask": track_valid,
    }

    if pi3_model is not None:
        print("    extracting Pi3 features...")
        pi3_feat, (pi3_h, pi3_w) = extract_pi3_features(pi3_model, images, device)
        assert (pi3_h, pi3_w) == (target_h, target_w), (
            f"Pi3 resized to {(pi3_h, pi3_w)} but patch grid assumed {(target_h, target_w)}"
        )
        result["pi3_feature"] = np.expand_dims(pi3_feat.astype(np.float32), axis=1)  # (T, 1, N, 1024)

    if dinov2_model is not None:
        print("    extracting DINOv2-reg features...")
        dino_feat, (dino_h, dino_w) = extract_dinov2_reg_features(dinov2_model, images, device)
        assert (dino_h, dino_w) == (target_h, target_w), (
            f"DINOv2 resized to {(dino_h, dino_w)} but patch grid assumed {(target_h, target_w)}"
        )
        result["dino_features"] = np.expand_dims(dino_feat.astype(np.float32), axis=1)  # (T, 1, N, 1024)

    if vae_model is not None:
        print("    extracting VAE latents...")
        vae_latent, vae_patch_centers, vae_valid, _ = extract_vae_features(
            vae_model, images, depth_maps, intrinsic, extrinsic, device
        )
        result["vae_latent"] = vae_latent
        result["vae_patch_centers_3d_position"] = vae_patch_centers
        result["vae_valid_mask"] = vae_valid

    if use_normal_estimation:
        all_normals = np.zeros((T_total, N, 3), dtype=np.float32)
        for t in range(T_total):
            valid_idx = np.where(valid_point_mask[t] & (np.linalg.norm(all_cam_3d[t], axis=-1) > 1e-6))[0]
            if valid_idx.size >= 3:
                normals_cam = estimate_normals(all_cam_3d[t, valid_idx])
                all_normals[t, valid_idx] = normals_cam @ R
        result["normals_3d"] = all_normals

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate ground-truth (simulator-exact) 3D point-flow track training data from RoboTwin demos."
    )
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--task_config", type=str, default="demo_clean")
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--raw_data_root", type=str, default=str(ROBOTWIN_ROOT / "data"))
    parser.add_argument("--camera_name", type=str, default="head_camera")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--action_horizon", type=int, default=16)
    parser.add_argument("--use_normal_estimation", action="store_true")
    parser.add_argument("--normal_radius", type=float, default=0.05)
    parser.add_argument("--normal_max_nn", type=int, default=30)
    parser.add_argument("--start_episode", type=int, default=0)
    parser.add_argument("--extract_pi3_features", action="store_true",
                         help="Also extract Pi3 patch features per frame and store as 'pi3_feature'")
    parser.add_argument("--extract_dino_features", action="store_true",
                         help="Also extract DINOv2-reg (dinov2_vitl14_reg) patch features per frame "
                              "and store as 'dino_features'")
    parser.add_argument("--extract_vae_features", action="store_true",
                         help="Also extract SDXL-VAE latents per frame and store as 'vae_latent' "
                              "(+ 'vae_patch_centers_3d_position' and 'vae_valid_mask' for its token grid)")
    parser.add_argument("--pi3_model_path", type=str, default=str(JAMB_ROOT / "pretrained" / "Pi3"))
    parser.add_argument("--vae_model_path", type=str, default=str(JAMB_ROOT / "pretrained" / "sdxl-vae"))
    parser.add_argument("--gpu_id", type=int, default=0,
                         help="CUDA device for Pi3/DINOv2/VAE inference (only used if any --extract_*_features is set)")
    args = parser.parse_args()
    # Resolve relative paths against the caller's original CWD, not
    # ROBOTWIN_ROOT (which is where the process chdir's to on import).
    args.output_file = os.path.join(_LAUNCH_CWD, args.output_file)
    args.raw_data_root = os.path.join(_LAUNCH_CWD, args.raw_data_root)
    return args


def main():
    args_cli = parse_args()

    device = None
    pi3_model = None
    dinov2_model = None
    vae_model = None
    if args_cli.extract_pi3_features or args_cli.extract_dino_features or args_cli.extract_vae_features:
        device = torch.device(f"cuda:{args_cli.gpu_id}" if torch.cuda.is_available() else "cpu")
    if args_cli.extract_pi3_features:
        print("Loading Pi3 model...")
        pi3_model = load_pi3_model(args_cli.pi3_model_path, device)
        print("Pi3 model loaded.")
    if args_cli.extract_dino_features:
        print("Loading DINOv2-reg model...")
        dinov2_model = load_dinov2_reg_model(device)
        print("DINOv2-reg model loaded.")
    if args_cli.extract_vae_features:
        print("Loading SDXL-VAE model...")
        vae_model = load_vae_model(args_cli.vae_model_path, device)
        print("SDXL-VAE model loaded.")

    task, task_args = build_task_and_args(args_cli.task_name, args_cli.task_config, args_cli.raw_data_root)

    task_dir = os.path.join(args_cli.raw_data_root, args_cli.task_name, args_cli.task_config)
    raw_dir = os.path.join(task_dir, "data")
    seed_path = os.path.join(task_dir, "seed.txt")
    if not os.path.isdir(raw_dir):
        print(f"Error: raw data directory not found: {raw_dir}")
        sys.exit(1)
    with open(seed_path, "r", encoding="utf-8") as f:
        seed_list = [int(s) for s in f.read().split()]

    available_episodes = sorted(
        int(fn[len("episode"):-len(".hdf5")])
        for fn in os.listdir(raw_dir) if fn.startswith("episode") and fn.endswith(".hdf5")
    )
    if not available_episodes:
        print(f"Error: no episode HDF5 files found in {raw_dir}")
        sys.exit(1)
    end_episode = min(args_cli.start_episode + args_cli.num_episodes, len(available_episodes))
    print(f"Found {len(available_episodes)} episodes; processing [{args_cli.start_episode}, {end_episode})")

    os.makedirs(os.path.dirname(os.path.abspath(args_cli.output_file)), exist_ok=True)

    # --- Auto-resume: detect existing partial output and skip done episodes ---
    ds = {}
    current_T = 0
    n_written = 0
    resume_from = args_cli.start_episode
    hdf5_mode = "w"

    if os.path.isfile(args_cli.output_file):
        try:
            with h5py.File(args_cli.output_file, "r") as hf:
                if "episode_ends" in hf and len(hf["episode_ends"]) > 0:
                    ep_ends = hf["episode_ends"][()]
                    n_written = len(ep_ends)
                    current_T = int(ep_ends[-1])
                    resume_from = args_cli.start_episode + n_written
                    hdf5_mode = "a"  # append to existing file
                    print(f"[RESUME] Found existing output with {n_written} episodes "
                          f"({current_T} frames). Resuming from episode index {resume_from}.")
                else:
                    print("[RESUME] Existing output file is empty. Starting fresh.")
        except Exception as e:
            print(f"[RESUME] Could not read existing output ({e}). Starting fresh.")

    if resume_from >= end_episode:
        print(f"[RESUME] All {end_episode - args_cli.start_episode} episodes already written. Nothing to do.")
        return

    with h5py.File(args_cli.output_file, hdf5_mode) as hdf5_file:
        # Recover dataset references when resuming into an existing file
        if hdf5_mode == "a" and n_written > 0:
            for k in hdf5_file:
                ds[k] = hdf5_file[k]

        for ep_idx in range(resume_from, end_episode):
            ep_num = ep_idx + 1 - args_cli.start_episode
            ep_total = end_episode - args_cli.start_episode
            print(f"\n=== Episode {ep_num}/{ep_total} "
                  f"(episode{ep_idx}) ===")
            ep_path = os.path.join(raw_dir, f"episode{ep_idx}.hdf5")
            images, depth_maps, eef_poses, intrinsic, extrinsic = load_robotwin_episode(
                ep_path, camera_name=args_cli.camera_name
            )

            traj_path = os.path.join(task_dir, "_traj_data", f"episode{ep_idx}.pkl")
            if not os.path.exists(traj_path):
                raise FileNotFoundError(
                    f"{traj_path} not found — ground-truth replay requires the cached dense joint "
                    "path from the original collection run; it must have been cleaned up."
                )

            seg_stack, pose_stack, robot_ids = replay_episode_ground_truth(
                task, task_args, ep_idx, seed_list[ep_idx], camera_name=args_cli.camera_name
            )

            data = build_episode_tracks(
                images, depth_maps, eef_poses, intrinsic, extrinsic,
                seg_stack, pose_stack, robot_ids,
                action_horizon=args_cli.action_horizon,
                use_normal_estimation=args_cli.use_normal_estimation,
                pi3_model=pi3_model, dinov2_model=dinov2_model, vae_model=vae_model, device=device,
            )

            if not ds:
                for k, v in data.items():
                    ds[k] = hdf5_file.create_dataset(
                        k, shape=(0,) + v.shape[1:], maxshape=(None,) + v.shape[1:],
                        dtype=v.dtype, chunks=True, compression="gzip", compression_opts=4,
                    )
                ds["episode_ends"] = hdf5_file.create_dataset(
                    "episode_ends", shape=(0,), maxshape=(None,), dtype=np.int64
                )

            T_ep = data["robot_state"].shape[0]
            for k, v in data.items():
                ds[k].resize(current_T + T_ep, axis=0)
                ds[k][current_T:] = v
            current_T += T_ep
            n_written += 1
            ds["episode_ends"].resize(n_written, axis=0)
            ds["episode_ends"][n_written - 1] = current_T
            hdf5_file.flush()  # persist each episode immediately for crash safety

            print(f"  Episode {ep_idx}: {T_ep} frames appended. Total frames so far: {current_T}.")

    print(f"\nSaved ground-truth track dataset to: {args_cli.output_file}")
    print(f"Use `python scripts/verify_track_data.py --tracks_file {args_cli.output_file} ...` to verify.")


if __name__ == "__main__":
    main()
