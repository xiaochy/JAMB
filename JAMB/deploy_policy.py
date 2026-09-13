import sys
import os
import torch
import numpy as np
from termcolor import cprint

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, "thirdparty"))
# Pi3 (thirdparty/pi3) is only needed for use_pi3_features=True, which this
# method doesn't use (it takes DINOv2 features) -- imported lazily where
# actually constructed so this module loads without that submodule present.

# Fixed resolution for point map visualization: matches this method's patch grid (17×23) × patch_size 14
_PI3_VIS_H = 17 * 14  # 238
_PI3_VIS_W = 23 * 14  # 322

import cv2
from jamb_policy.policy.jamb import JAMBPolicy
from jamb_policy.model.vision.dinov2_encoder import DINOV2


PRETRAINED_ROOT = os.environ.get("GAP_PRETRAINED_ROOT", "pretrained")
DEFAULT_PI3_PATH = os.path.join(PRETRAINED_ROOT, "Pi3")
DEFAULT_DINOV2_WEIGHTS = os.path.join(
    PRETRAINED_ROOT,
    "dinov2_vitl14_reg4_pretrain.pth",
)


def compute_patch_centers_world(depth_mm, intrinsic_cv, extrinsic_cv,
                                patch_size=14):
    """
    Compute world-frame 3D positions of the 17x23 patch-grid centers from a
    live sim depth map. Replicates the training-data math exactly
    (generate_training_data.py): centers on the resized 238x322 grid ->
    scaled to original pixel coords -> bilinear depth sample (mm -> m) ->
    unproject with intrinsics -> world frame via P_w = R^T (P_c - t).

    Args:
        depth_mm: [H, W] depth in millimetres (RoboTwin obs convention)
        intrinsic_cv: [3, 3]
        extrinsic_cv: [3, 4] world-to-camera [R | t]

    Returns:
        centers_world: [N, 3] float32; invalid-depth points are zeros
    """
    H, W = depth_mm.shape
    th = max(round(H / patch_size) * patch_size, patch_size)
    tw = max(round(W / patch_size) * patch_size, patch_size)
    ph, pw = th // patch_size, tw // patch_size

    cy = (np.arange(ph) + 0.5) * patch_size
    cx = (np.arange(pw) + 0.5) * patch_size
    gx, gy = np.meshgrid(cx, cy)
    centers = np.stack([gx.ravel(), gy.ravel()], axis=-1)  # (N, 2) resized px
    centers[:, 0] *= W / tw
    centers[:, 1] *= H / th

    N = centers.shape[0]
    map_x = np.clip(centers[:, 0], 0, W - 1).astype(np.float32).reshape(1, N)
    map_y = np.clip(centers[:, 1], 0, H - 1).astype(np.float32).reshape(1, N)
    depth_sampled = cv2.remap(depth_mm.astype(np.float32), map_x, map_y,
                              cv2.INTER_LINEAR).reshape(N)
    depth_m = depth_sampled / 1000.0
    valid = (depth_m > 0) & np.isfinite(depth_m)

    fx, fy = intrinsic_cv[0, 0], intrinsic_cv[1, 1]
    cx0, cy0 = intrinsic_cv[0, 2], intrinsic_cv[1, 2]
    x_cam = (centers[:, 0] - cx0) * depth_m / fx
    y_cam = (centers[:, 1] - cy0) * depth_m / fy
    pts_cam = np.stack([x_cam, y_cam, depth_m], axis=-1)  # (N, 3)

    R = extrinsic_cv[:, :3]
    t = extrinsic_cv[:, 3]
    pts_world = (pts_cam - t) @ R  # = R^T (P_c - t) for row vectors

    pts_world[~valid] = 0.0
    return pts_world.astype(np.float32)


def compute_dense_pointcloud_world(depth_mm, rgb, intrinsic_cv, extrinsic_cv, stride=4):
    """
    Dense (strided, not per-patch) colored point cloud in world frame, for
    3D track visualization backgrounds (see visualize_track_3d.py) -- same
    unprojection math as compute_patch_centers_world but sampling every
    `stride`-th pixel directly instead of the 17x23 DinoV2 patch grid, so
    the robot/scene geometry is actually recognizable instead of 391 sparse
    anchor points.

    Args:
        depth_mm: [H, W] depth in millimetres
        rgb: [H, W, 3] uint8
        intrinsic_cv: [3, 3]
        extrinsic_cv: [3, 4] world-to-camera [R | t]
        stride: pixel stride (4 -> ~1/16 of pixels)

    Returns:
        pts_world: [M, 3] float32, M = ceil(H/stride)*ceil(W/stride) always
            (invalid-depth points zeroed, not dropped, matching
            compute_patch_centers_world's convention -- keeps M constant
            across calls so callers/consumers can assume a fixed point count)
        colors: [M, 3] uint8 (RGB, matching pts_world)
    """
    H, W = depth_mm.shape
    ys = np.arange(0, H, stride)
    xs = np.arange(0, W, stride)
    gx, gy = np.meshgrid(xs, ys)
    px = gx.ravel().astype(np.float32)
    py = gy.ravel().astype(np.float32)

    depth_m = depth_mm[gy.ravel(), gx.ravel()].astype(np.float32) / 1000.0
    valid = (depth_m > 0) & np.isfinite(depth_m)

    fx, fy = intrinsic_cv[0, 0], intrinsic_cv[1, 1]
    cx0, cy0 = intrinsic_cv[0, 2], intrinsic_cv[1, 2]
    x_cam = (px - cx0) * depth_m / fx
    y_cam = (py - cy0) * depth_m / fy
    pts_cam = np.stack([x_cam, y_cam, depth_m], axis=-1)

    R = extrinsic_cv[:, :3]
    t = extrinsic_cv[:, 3]
    pts_world = (pts_cam - t) @ R

    colors = rgb[gy.ravel(), gx.ravel()].copy()

    pts_world[~valid] = 0.0
    colors[~valid] = 0
    return pts_world.astype(np.float32), colors


def resolve_repo_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(current_dir, path)


class JAMBPolicyWrapper:

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        debug: bool = False,
        swap_rgb_channels: bool = True,
        sample_seed: int = 0,
    ):
        self.swap_rgb_channels = swap_rgb_channels
        self.device = device
        self.debug = debug
        # Fixes the diffusion sampling noise (conditional_sample's initial
        # torch.randn + any stochastic scheduler steps) so eval rollouts are
        # reproducible run-to-run given the same checkpoint + environment
        # seed. Without this the denoising trajectory is drawn unseeded, so
        # re-running the identical eval can silently land on a different
        # success rate purely from sampling noise, not model quality.
        self._sample_generator = torch.Generator(device=device)
        self._sample_generator.manual_seed(sample_seed)

        cprint(f"[JAMB] Initializing deployment policy", "cyan")
        cprint(f"[JAMB] Device: {device}", "cyan")

        # Load checkpoint
        cprint(f"[JAMB] Loading checkpoint from {ckpt_path}", "cyan")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        class SubscriptableNamespace(dict):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                for k, v in self.items():
                    if isinstance(v, dict):
                        self[k] = SubscriptableNamespace(v)
                    elif isinstance(v, list):
                        self[k] = [SubscriptableNamespace(x) if isinstance(x, dict) else x for x in v]
            def __getattr__(self, key):
                try:
                    return self[key]
                except KeyError:
                    raise AttributeError(key)
            def __setattr__(self, key, value):
                self[key] = value
            def __delattr__(self, key):
                del self[key]

        self.cfg = ckpt.get("cfg")
        if isinstance(self.cfg, dict):
            self.cfg = SubscriptableNamespace(self.cfg)

        if self.cfg is None:
            raise ValueError("Checkpoint does not contain 'cfg' key")

        # Get policy configuration
        policy_cfg = self.cfg.policy

        cprint(f"[JAMB] Creating JAMB policy model", "cyan")
        cprint(f"[JAMB] DinoV2: {policy_cfg.get('dino_model_name', 'disabled') if policy_cfg.get('use_dino_features', False) else 'disabled'}", "cyan")
        cprint(f"[JAMB] Horizon: {policy_cfg.horizon}", "cyan")
        cprint(f"[JAMB] N obs steps: {policy_cfg.n_obs_steps}", "cyan")
        cprint(f"[JAMB] N action steps: {policy_cfg.n_action_steps}", "cyan")

        # Create noise scheduler from config
        noise_scheduler_cfg = policy_cfg.noise_scheduler
        target = noise_scheduler_cfg["_target_"]
        module_path, class_name = target.rsplit(".", 1)
        import importlib
        module = importlib.import_module(module_path)
        scheduler_class = getattr(module, class_name)

        # Create scheduler with config parameters
        scheduler_params = {k: v for k, v in noise_scheduler_cfg.items() if k != "_target_"}
        noise_scheduler = scheduler_class(**scheduler_params)
        policy_cfg.noise_scheduler = noise_scheduler
        cprint(f"[JAMB] Created {class_name} noise scheduler", "cyan")

        # Instantiate the track branch's own scheduler the same way as the
        # action noise_scheduler above — the saved cfg only has its raw
        # _target_ dict, not a live diffusers object. TRACK_CLIP_RANGE lets
        # clip_sample_range be overridden for checkpoints predating the
        # track_noise_scheduler fix (clip_sample_range=1.0 silently truncated
        # real moving-patch track magnitude every denoising step, and — since
        # action/track tokens share one self-attention stream in
        # bridge_mode="shared" — that truncated signal could leak into the
        # action branch too), or for any checkpoint when explicitly set.
        track_clip_range = os.environ.get("TRACK_CLIP_RANGE")
        track_noise_scheduler_cfg = policy_cfg.get("track_noise_scheduler")
        if track_noise_scheduler_cfg is not None:
            track_scheduler_params = {k: v for k, v in track_noise_scheduler_cfg.items() if k != "_target_"}
            if track_clip_range is not None:
                track_scheduler_params["clip_sample_range"] = float(track_clip_range)
                cprint(f"[JAMB] Track branch clip_sample_range override: {track_clip_range}", "yellow")
            policy_cfg.track_noise_scheduler = scheduler_class(**track_scheduler_params)
            cprint(f"[JAMB] Created {class_name} track noise scheduler", "cyan")
        elif track_clip_range is not None:
            track_scheduler_params = dict(scheduler_params)
            track_scheduler_params["clip_sample_range"] = float(track_clip_range)
            policy_cfg.track_noise_scheduler = scheduler_class(**track_scheduler_params)
            cprint(f"[JAMB] Track branch clip_sample_range override: {track_clip_range}", "yellow")
        else:
            policy_cfg.track_noise_scheduler = None

        # Create JAMB model
        self.policy_model = JAMBPolicy(
            **policy_cfg
        )

        # Load model weights (use EMA if available)
        if "ema" in ckpt and ckpt["ema"] is not None:
            cprint(f"[JAMB] Loading EMA model weights", "cyan")
            self.policy_model.load_state_dict(ckpt["ema"])
        elif "model" in ckpt:
            cprint(f"[JAMB] Loading model weights", "cyan")
            self.policy_model.load_state_dict(ckpt["model"])
        else:
            raise ValueError("Checkpoint does not contain model weights")

        self.policy_model.to(device)
        self.policy_model.eval()

        # Load normalizer
        if "normalizer" in ckpt:
            class FakedNormalizer:
                def __init__(self, state_dict):
                    self._state_dict = state_dict
                def state_dict(self):
                    return self._state_dict
            normalizer = FakedNormalizer(ckpt["normalizer"])
            self.policy_model.set_normalizer(normalizer)
            cprint(f"[JAMB] Loaded normalizer from checkpoint", "cyan")
        else:
            raise ValueError(
                "Checkpoint does not contain 'normalizer'. "
                "Use checkpoints saved by scripts/train.py or add the training normalizer state to the checkpoint."
            )

        cprint(f"[JAMB] Policy loaded successfully!", "green")


        # Load DinoV2 feature extractor (if enabled)
        self.use_dino = policy_cfg.get('use_dino_features', False)
        if self.use_dino:
            cprint(f"[JAMB] Loading DinoV2 feature extractor...", "cyan")
            dino_repo = resolve_repo_path(policy_cfg.get("dino_repo_dir", "thirdparty/dinov2"))
            dino_weights = os.environ.get(
                "DINOV2_WEIGHTS_PATH",
                policy_cfg.get("dino_weights_path", DEFAULT_DINOV2_WEIGHTS),
            )
            dino_weights = resolve_repo_path(dino_weights)
            self.dino_model = DINOV2(
                model_name=policy_cfg.get("dino_model_name", "dinov2_vitl14_reg"),
                repo_dir=dino_repo,
                weights_path=dino_weights,
                freeze=True,
            ).to(device).eval()
            cprint(f"[JAMB] DinoV2 feature extractor loaded", "green")
        else:
            self.dino_model = None

        # Load Pi3 feature extractor (if enabled)
        self.use_pi3 = policy_cfg.get('use_pi3_features', False)
        if self.use_pi3:
            from pi3.models.pi3 import Pi3
            cprint(f"[JAMB] Loading Pi3 feature extractor...", "cyan")
            pi3_model_name_or_path = os.environ.get(
                "PI3_MODEL_NAME_OR_PATH",
                os.environ.get(
                    "PI3_MODEL_PATH",
                    policy_cfg.get("pi3_model_name_or_path", DEFAULT_PI3_PATH),
                ),
            )
            pi3_model_name_or_path = resolve_repo_path(pi3_model_name_or_path)
            self.pi3_model = Pi3.from_pretrained(pi3_model_name_or_path).to(device).eval()
            cprint(f"[JAMB] Pi3 feature extractor loaded", "green")
        else:
            self.pi3_model = None

        # Store config
        self.n_action_steps = policy_cfg.n_action_steps
        self.state_dim = policy_cfg.state_dim


    def reset(self):
        """Reset policy state between episodes (JAMB is stateless)"""
        if self.debug:
            cprint("[JAMB] Policy reset (stateless)", "cyan")

    def extract_dino_features(self, rgb_image: np.ndarray) -> torch.Tensor:
        """
        Extract DinoV2 features from an RGB image. Resizes to the nearest
        multiple of 14 (240x320 -> 238x322), matching training preprocessing
        and Pi3's patch grid.

        Returns: [1, N_patches, D]
        """
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        img = torch.from_numpy(rgb_image).permute(2, 0, 1).float() / 255.0
        img = img.unsqueeze(0).to(self.device)
        H, W = rgb_image.shape[:2]
        th = max(round(H / 14) * 14, 14)
        tw = max(round(W / 14) * 14, 14)
        if (H, W) != (th, tw):
            img = torch.nn.functional.interpolate(
                img, size=(th, tw), mode='bilinear', align_corners=False)
        img = (img - mean) / std
        with torch.no_grad():
            return self.dino_model(img)

    def extract_pi3_features(self, rgb_image: np.ndarray) -> torch.Tensor:
        """
        Extract Pi3 features from RGB image

        Args:
            rgb_image: numpy array [H, W, 3] in RGB format, range [0, 255]

        Returns:
            features: [1, N_patches, 1024] tensor
        """
        # Convert RGB to tensor
        img_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, 3, H, W]

        # Resize to be divisible by 14 (Pi3 patch size)
        H, W = rgb_image.shape[:2]
        target_h = round(H / 14) * 14
        target_w = round(W / 14) * 14
        target_h = max(target_h, 14)
        target_w = max(target_w, 14)

        if H != target_h or W != target_w:
            img_tensor = torch.nn.functional.interpolate(
                img_tensor.squeeze(0), size=(target_h, target_w), mode='bilinear', align_corners=False
            ).unsqueeze(0)

        # Normalize
        img_tensor = (img_tensor - self.pi3_model.image_mean) / self.pi3_model.image_std

        # Extract features
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=dtype):
                N, C, H_new, W_new = img_tensor.shape[1:]
                imgs_flat = img_tensor.squeeze(0)  # [N, C, H, W]

                # Encode
                hidden = self.pi3_model.encoder(imgs_flat, is_training=True)
                if isinstance(hidden, dict):
                    hidden = hidden["x_norm_patchtokens"]

                # Decode
                hidden, pos = self.pi3_model.decode(hidden, N, H_new, W_new)

                # Extract point features
                point_hidden = self.pi3_model.point_decoder(hidden, xpos=pos)  # [N, hw, 1024]
                point_hidden = point_hidden[:, self.pi3_model.patch_start_idx:].float()  # [N, num_patches, 1024]

        return point_hidden.unsqueeze(0)  # [1, N, num_patches, 1024]

    def decode_pointmap(self, pi3_features: torch.Tensor) -> np.ndarray:
        """Decode pi3 hidden features → XYZ point map via Pi3's point_head.

        Args:
            pi3_features: [1, 391, 1024] tensor on GPU
        Returns:
            xyz: [N, 3] numpy array of 3D points
        """
        with torch.no_grad():
            xyz = self.pi3_model.point_head([pi3_features.float()], (_PI3_VIS_H, _PI3_VIS_W))
        return xyz[0].cpu().float().numpy().reshape(-1, 3)

    def extract_pi3_features_fixed_size(self, rgb_image: np.ndarray) -> torch.Tensor:
        """Extract pi3 features at the fixed visualization resolution (_PI3_VIS_H × _PI3_VIS_W).

        Resizes the image to exactly 238×322 so the output has exactly 391 patches,
        matching pi3_features_pred for direct spatial comparison.

        Returns:
            features: [1, 391, 1024] tensor on GPU
        """
        img_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)  # [1, 3, H, W]
        img_tensor = torch.nn.functional.interpolate(
            img_tensor, size=(_PI3_VIS_H, _PI3_VIS_W), mode='bilinear', align_corners=False
        ).unsqueeze(0)  # [1, 1, 3, 238, 322]
        img_tensor = (img_tensor - self.pi3_model.image_mean) / self.pi3_model.image_std

        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=dtype):
                imgs_flat = img_tensor.squeeze(0)  # [1, 3, 238, 322]
                hidden = self.pi3_model.encoder(imgs_flat, is_training=True)
                if isinstance(hidden, dict):
                    hidden = hidden["x_norm_patchtokens"]
                hidden, pos = self.pi3_model.decode(hidden, 1, _PI3_VIS_H, _PI3_VIS_W)
                point_hidden = self.pi3_model.point_decoder(hidden, xpos=pos)
                point_hidden = point_hidden[:, self.pi3_model.patch_start_idx:].float()
        return point_hidden  # [1, 391, 1024]

    def get_action(self, rgb_image: np.ndarray, state: np.ndarray,
                   patch_centers: np.ndarray = None) -> np.ndarray:
        """
        Get action chunk based on current observation

        Args:
            rgb_image: numpy array [H, W, 3] RGB image
            state: numpy array [16] EEF-pose state
            patch_centers: numpy array [N, 3] world-frame patch-center
                positions from the live depth map (4D RoPE coordinates)

        Returns:
            actions: numpy array [n_action_steps, action_dim] action chunk
            track_pred: [1, N, H, 3] predicted 3D tracks or None
        """
        # Channel-order consistency with training: the feature-generation
        # pipelines ran DinoV2/Pi3 on channel-SWAPPED images (spurious
        # BGR2RGB after imdecode of blobs that already decode to true RGB).
        # Swap the live RGB the same way so eval matches training. Remove
        # this together with the pipelines' cvtColor when regenerating data.
        if self.swap_rgb_channels:
            rgb_image = rgb_image[..., ::-1].copy()

        # Extract Pi3 features (if enabled)
        pi3_features = None
        if self.use_pi3 and self.pi3_model is not None:
            pi3_features = self.extract_pi3_features(rgb_image)  # [1, 1, N_patches, 1024]

        # Preprocess state: [state_dim] -> tensor
        state_tensor = torch.from_numpy(state).float().to(self.device)  # [state_dim]
        state_tensor = state_tensor.unsqueeze(0)  # [1, state_dim]

        obs_dict = {
            "agent_pos": state_tensor,
        }

        if pi3_features is not None:
            obs_dict["pi3_features"] = pi3_features

        if self.use_dino and self.dino_model is not None:
            dino_features = self.extract_dino_features(rgb_image)  # [1, N, D]
            obs_dict["dino_features"] = dino_features.unsqueeze(1)  # [1, 1, N, D]

        if patch_centers is not None:
            obs_dict["patch_centers"] = (
                torch.from_numpy(patch_centers).float().unsqueeze(0).to(self.device)
            )  # [1, N, 3]

        # Predict action chunk
        with torch.no_grad():
            result = self.policy_model.predict_action(obs_dict, generator=self._sample_generator)
            actions = result["action"][0].cpu().numpy()
            track_pred = result.get("track_pred", None)

        if self.debug:
            cprint(f"[JAMB] Action predicted. Shape: {actions.shape}", "cyan")

        return actions, track_pred


# ============================================================================
# RoboTwin Evaluation Interface
# ============================================================================

def encode_obs(observation, task_env=None):
    """
    Extract and format observation for policy input

    Args:
        observation: Raw observation from environment
        task_env: Environment instance for extracting EEF poses

    Returns:
        obs: Dict with 'rgb' and 'state'
    """
    # Extract RGB image from head camera
    rgb = observation["observation"]["head_camera"]["rgb"]  # [H, W, 3]

    if "endpose" in observation and "left_endpose" in observation["endpose"]:
        left_endpose = observation["endpose"]["left_endpose"]
        left_gripper = observation["endpose"]["left_gripper"]
        right_endpose = observation["endpose"]["right_endpose"]
        right_gripper = observation["endpose"]["right_gripper"]
    elif task_env is not None:
        left_endpose = task_env.robot.get_left_ee_pose()
        left_gripper = task_env.robot.get_left_gripper_val()
        right_endpose = task_env.robot.get_right_ee_pose()
        right_gripper = task_env.robot.get_right_gripper_val()
    else:
        # Fallback to joint angles if needed, but the model expects 16D EEF pose
        raise ValueError("observation does not contain endpose and task_env is None")

    state = np.concatenate([left_endpose, [left_gripper], right_endpose, [right_gripper]])

    # World-frame patch centers from the live depth map (4D RoPE coordinates).
    # Requires depth in the obs: eval with a task config that sets depth: true
    # (e.g. demo_clean_depth).
    patch_centers = None
    head_cam = observation["observation"]["head_camera"]
    if "depth" in head_cam:
        patch_centers = compute_patch_centers_world(
            np.asarray(head_cam["depth"], dtype=np.float32),
            np.asarray(head_cam["intrinsic_cv"]),
            np.asarray(head_cam["extrinsic_cv"]),
        )

    return {
        "rgb": rgb,
        "state": state,
        "patch_centers": patch_centers,
    }


def get_model(usr_args):
    """
    Factory function to create policy for evaluation
    Required by RoboTwin evaluation framework

    Args:
        usr_args: Dict with configuration parameters
            - ckpt_path: Path to checkpoint (optional, auto-constructed if not provided)
            - task_name: Task name
            - ckpt_setting: Checkpoint setting
            - expert_data_num: Number of expert demonstrations
            - seed: Random seed
            - checkpoint_num: Checkpoint epoch number
            - device: Device to use (optional)
            - debug: Enable debug mode (optional)

    Returns:
        policy: JAMBPolicyWrapper instance
    """
    # Get current directory for default paths
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Set device
    device = usr_args.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    debug = usr_args.get("debug", False)

    # eval_policy.py's override parser runs eval() on CLI string values, so
    # "True"/"False" (capitalized) parse as real Python bools; anything
    # else falls back to the raw string — handle that defensively too.
    swap_rgb_channels = usr_args.get("swap_rgb_channels", True)
    if isinstance(swap_rgb_channels, str):
        swap_rgb_channels = swap_rgb_channels.strip().lower() not in ("false", "0", "no", "")

    # Construct checkpoint path if not provided
    ckpt_path = usr_args.get("ckpt_path", None)
    if ckpt_path is None:
        # Format: checkpoints/{task_name}_{ckpt_setting}_{expert_data_num}/{checkpoint_num}.ckpt
        ckpt_dir = os.path.join(
            current_dir,
            "checkpoints",
            # f"{usr_args['task_name']}_{usr_args['ckpt_setting']}_{usr_args['expert_data_num']}"
            f"{usr_args['task_name']}_{usr_args['ckpt_setting']}_{usr_args['expert_data_num']}"
        )
        ckpt_path = os.path.join(ckpt_dir, f"{usr_args['checkpoint_num']}.ckpt")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Seeds the diffusion sampling noise so eval rollouts reproduce
    # run-to-run given the same checkpoint (see JAMBPolicyWrapper docstring).
    sample_seed = int(usr_args.get("seed", 0))

    cprint(f"[JAMB] Loading checkpoint: {ckpt_path}", "cyan")
    cprint(f"[JAMB] swap_rgb_channels: {swap_rgb_channels}", "cyan")
    cprint(f"[JAMB] sample_seed: {sample_seed}", "cyan")

    # Create policy
    policy = JAMBPolicyWrapper(
        ckpt_path=ckpt_path,
        device=device,
        debug=debug,
        swap_rgb_channels=swap_rgb_channels,
        sample_seed=sample_seed,
    )

    return policy


def reset_model(model: JAMBPolicyWrapper):
    """
    Reset model state between episodes
    Required by RoboTwin evaluation framework

    Args:
        model: JAMBPolicyWrapper instance
    """
    model.reset()


def eval(TASK_ENV, model: JAMBPolicyWrapper, observation, episode_info=None):
    """
    Evaluation step - execute one action chunk.
    Required by RoboTwin evaluation framework.

    Returns:
        observation: latest observation after executing the action chunk
        track_data: dict with track information
    """
    obs = encode_obs(observation, TASK_ENV)
    actions, track_pred = model.get_action(
        obs["rgb"], obs["state"], patch_centers=obs.get("patch_centers")
    )

    for action in actions:
        # action is 16D EEF pose, tell the env to use IK
        TASK_ENV.take_action(action, action_type='ee')
        observation = TASK_ENV.get_obs()

    track_data = None
    if track_pred is not None:
        track_data = {
            "track_pred": track_pred,
        }

    return observation, track_data
