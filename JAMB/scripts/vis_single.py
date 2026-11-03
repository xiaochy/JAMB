"""
Visualize predicted tracks from a single model vs GT.

Left panel:  GT tracks  (all patches, sorted by GT displacement)
Right panel: Model pred  (all patches, sorted by pred displacement)

Usage:
    cd GAP-track-only-dino
    python scripts/vis_single.py \
        --ckpt  data/outputs/.../checkpoints/50.ckpt \
        --label "pi3dino-uniform" \
        --zarr_path /data/.../tracks.zarr \
        --raw_data_root /data/.../RoboTwin/data \
        --task_name stack_blocks_two \
        --task_config demo_clean_depth \
        --episodes 8 43 64 75 99 \
        --top_k 0 \
        --out_dir data/outputs/.../vis
"""
import argparse
import os
import sys

import cv2
import h5py
import numpy as np
import torch
import zarr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─── geometry helpers ─────────────────────────────────────────────────────────

def _world_to_cam(pts_w, ext):
    return pts_w @ ext[:, :3].T + ext[:, 3]


def _project(pts_cam, K):
    z = np.where(pts_cam[..., 2] == 0, 1e-6, pts_cam[..., 2])
    u = pts_cam[..., 0] * K[0, 0] / z + K[0, 2]
    v = pts_cam[..., 1] * K[1, 1] / z + K[1, 2]
    return np.stack([u, v], axis=-1), z


def _draw_tracks(rgb_bgr, patch_centers, track_disps, K, ext,
                 select_idx, color_start, color_end):
    img = rgb_bgr.copy()
    H_img, W_img = img.shape[:2]
    _, T, _ = track_disps.shape
    anc_uv, _ = _project(_world_to_cam(patch_centers, ext), K)
    for i in select_idx:
        ax, ay = int(anc_uv[i, 0]), int(anc_uv[i, 1])
        if not (0 <= ax < W_img and 0 <= ay < H_img):
            continue
        fut_w = patch_centers[i:i+1] + track_disps[i]
        fut_uv, fut_z = _project(_world_to_cam(fut_w, ext), K)
        prev = (ax, ay)
        for h in range(T):
            if fut_z[h] < 0.01:
                break
            px, py = int(fut_uv[h, 0]), int(fut_uv[h, 1])
            if not (0 <= px < W_img and 0 <= py < H_img):
                break
            r = h / max(T - 1, 1)
            color = (
                int(color_start[0] * (1-r) + color_end[0] * r),
                int(color_start[1] * (1-r) + color_end[1] * r),
                int(color_start[2] * (1-r) + color_end[2] * r),
            )
            cv2.line(img, prev, (px, py), color, 1, cv2.LINE_AA)
            prev = (px, py)
        cv2.circle(img, (ax, ay), 2, (0, 200, 0), -1)
    return img


def _select(disps, top_k, min_movement=0.01):
    """Select patch indices by displacement magnitude (largest first).

    Selects patches whose endpoint displacement exceeds min_movement, then
    returns up to top_k of them.  top_k <= 0 means keep all that pass.
    """
    mag = np.linalg.norm(disps[:, -1, :], axis=-1)
    moving = np.where(mag > min_movement)[0]
    if len(moving) == 0:
        moving = np.argsort(mag)[::-1][:5]  # fallback: at least show top-5
    order = moving[np.argsort(mag[moving])[::-1]]
    if top_k > 0:
        order = order[:top_k]
    return order


def _label(img, text, color=(255, 255, 255)):
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img


def _write_h264(frames_list, out_path, fps):
    import subprocess
    H, W = frames_list[0][0].shape[:2]
    n = len(frames_list)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{W*n}x{H}", "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "fast",
        "-crf", "23", "-pix_fmt", "yuv420p",
        out_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for panels in zip(*frames_list):
        proc.stdin.write(np.concatenate(panels, axis=1).tobytes())
    proc.stdin.close()
    proc.wait()


# ─── model loading / inference ────────────────────────────────────────────────

def load_model(ckpt_path, device, track_clip_range=None):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    import copy
    import hydra
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver("tag_suffix", lambda tag: f"_{tag}" if tag else "", replace=True)
    cfg = OmegaConf.create(ckpt["cfg"])
    if track_clip_range is not None:
        # Checkpoint's saved cfg predates the track_noise_scheduler fix
        # (clip_sample_range=1.0 silently truncated real moving-patch track
        # magnitude). Inject a separately-clipped scheduler for track only.
        OmegaConf.set_struct(cfg, False)
        track_sched = copy.deepcopy(cfg.policy.noise_scheduler)
        track_sched["clip_sample_range"] = track_clip_range
        cfg.policy.track_noise_scheduler = track_sched
    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(ckpt["model"])
    from jamb_policy.model.common.normalizer import LinearNormalizer
    normalizer = LinearNormalizer()
    normalizer.load_state_dict(ckpt["normalizer"])
    policy.set_normalizer(normalizer)
    policy = policy.to(device).eval()
    return policy


@torch.no_grad()
def predict_tracks(policy, dino_feat, agent_pos, patch_centers, device, pi3_feat=None,
                   num_inference_steps=10):
    obs = {
        "agent_pos":     torch.from_numpy(agent_pos).float().unsqueeze(0).to(device),
        "patch_centers": torch.from_numpy(patch_centers).float().unsqueeze(0).to(device),
    }
    if dino_feat is not None:
        t = torch.from_numpy(dino_feat).float()
        if t.ndim == 2:
            t = t.unsqueeze(0)
        obs["dino_features"] = t.unsqueeze(0).to(device)
    if pi3_feat is not None:
        t = torch.from_numpy(pi3_feat).float()
        if t.ndim == 2:
            t = t.unsqueeze(0)
        obs["pi3_features"] = t.unsqueeze(0).to(device)
    # Use full diffusion sampling so action tokens are present in the decoder
    # alongside track queries — matching the training-time setup.
    orig_steps = policy.num_inference_steps
    policy.num_inference_steps = num_inference_steps
    result = policy.predict_action(obs)
    policy.num_inference_steps = orig_steps
    assert "track_pred" in result, "Model did not return track_pred; check aux_task=track"
    return result["track_pred"][0].cpu().numpy()


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",           required=True)
    ap.add_argument("--label",          default="model")
    ap.add_argument("--zarr_path",      required=True)
    ap.add_argument("--raw_data_root",  required=True)
    ap.add_argument("--task_name",      required=True)
    ap.add_argument("--task_config",    required=True)
    ap.add_argument("--episodes",       type=int, nargs="+", default=[0])
    ap.add_argument("--camera",         default="head_camera")
    ap.add_argument("--track_key",      default="gt_3d_track")
    ap.add_argument("--out_dir",        default=None,
                    help="Output dir; defaults to <ckpt_dir>/../vis")
    ap.add_argument("--top_k",          type=int, default=60,
                    help="Top-k patches by pred displacement; 0=all that pass threshold")
    ap.add_argument("--min_movement",   type=float, default=0.01,
                    help="Min GT endpoint displacement (metres) to show a patch (default 0.01)")
    ap.add_argument("--num_inference_steps", type=int, default=100,
                    help="Diffusion denoising steps; default 100 matches rollout")
    ap.add_argument("--fps",            type=int, default=10)
    ap.add_argument("--device",         default="cuda")
    ap.add_argument("--track_clip_range", type=float, default=None,
                     help="override track branch's clip_sample_range (None = use checkpoint's saved value)")
    ap.add_argument("--select_by_gt", action="store_true", default=False,
                     help="select the GT panel's patches by GT's own displacement "
                          "instead of reusing the pred-derived selection — shows "
                          "GT's actually-moving patches, even ones the model missed "
                          "(the two panels may then highlight different patches)")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.out_dir is None:
        ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt))
        args.out_dir = os.path.join(os.path.dirname(ckpt_dir), "vis")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Output dir: {args.out_dir}")

    print(f"Loading model: {args.ckpt}")
    policy = load_model(args.ckpt, device, track_clip_range=args.track_clip_range)

    print(f"Loading zarr: {args.zarr_path}")
    z = zarr.open(args.zarr_path, mode="r")
    ep_ends  = z["meta/episode_ends"][:]
    ep_starts = np.concatenate([[0], ep_ends[:-1]])
    dino     = z["data/dino_features"] if "dino_features" in z["data"] else None
    pi3      = z["data/pi3_features"]  if "pi3_features"  in z["data"] else None
    state    = z["data/state"]
    pc       = z["data/patch_centers"]
    gt_track = z[f"data/{args.track_key}"]

    raw_dir = os.path.join(args.raw_data_root, args.task_name, args.task_config, "data")

    for ep in args.episodes:
        ep_start = int(ep_starts[ep])
        ep_end   = int(ep_ends[ep])
        n_frames = ep_end - ep_start
        print(f"\nEpisode {ep}: frames {ep_start}–{ep_end-1} ({n_frames} frames)")

        raw_path = os.path.join(raw_dir, f"episode{ep}.hdf5")
        raw_f = h5py.File(raw_path, "r", swmr=True)
        cam   = raw_f[f"observation/{args.camera}"]
        K     = cam["intrinsic_cv"][0]
        ext   = cam["extrinsic_cv"][0]

        frames_gt   = []
        frames_pred = []

        for t in range(n_frames):
            idx = ep_start + t
            rgb_bytes = cam["rgb"][t]
            rgb_bgr = cv2.imdecode(np.frombuffer(rgb_bytes, np.uint8), cv2.IMREAD_COLOR)

            d_feat  = dino[idx] if dino is not None else None
            p3_feat = pi3[idx]  if pi3  is not None else None
            s       = state[idx]
            p       = pc[idx]
            gt_t    = gt_track[idx]   # [N, H, 3]

            use_pi3 = getattr(policy, "use_pi3_features", False)
            pred = predict_tracks(policy, d_feat, s, p, device, pi3_feat=(p3_feat if use_pi3 else None),
                                  num_inference_steps=args.num_inference_steps)

            # Select by PRED displacement — both panels show the same patches
            # so we can judge whether the model's predicted movement matches GT.
            # --select_by_gt overrides this for the GT panel only: it selects
            # by GT's OWN displacement, so GT shows what's actually moving
            # (including patches the model failed to predict as moving) —
            # the two panels can then highlight different patches.
            sel_pred = _select(pred, args.top_k, args.min_movement)
            sel_gt = _select(gt_t, args.top_k, args.min_movement) if args.select_by_gt else sel_pred

            frame_gt = _draw_tracks(rgb_bgr, p, gt_t, K, ext, sel_gt,
                                    color_start=(0, 200, 0), color_end=(0, 255, 120))
            frame_pred = _draw_tracks(rgb_bgr, p, pred, K, ext, sel_pred,
                                      color_start=(0, 140, 255), color_end=(0, 0, 220))

            _label(frame_gt,   "GT",         color=(0, 255, 100))
            _label(frame_pred, args.label,   color=(0, 200, 255))

            frames_gt.append(frame_gt)
            frames_pred.append(frame_pred)

        raw_f.close()

        out_path = os.path.join(args.out_dir, f"ep{ep:03d}.mp4")
        _write_h264([frames_gt, frames_pred], out_path, args.fps)
        print(f"  Saved {out_path}")

        png_path = os.path.join(args.out_dir, f"ep{ep:03d}_frame0.png")
        cv2.imwrite(png_path, np.concatenate([frames_gt[0], frames_pred[0]], axis=1))
        print(f"  Saved {png_path}")

    print(f"\nDone. Output: {args.out_dir}")


if __name__ == "__main__":
    main()
