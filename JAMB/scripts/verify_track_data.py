"""
Verify a flat world-frame tracks dataset against its raw episodes.

Checks (per episode):
  1. Overlay video: world-frame patch centers, track endpoints, and both
     arms' EEF positions projected back onto the RGB frames, side-by-side
     with the depth map. If the world-frame math is wrong anywhere
     (unprojection, extrinsics, track accumulation, endpose recording),
     the overlays visibly drift off the scene.
  2. Depth consistency: camera-frame z of reprojected patch centers vs the
     raw depth map sampled at the projected pixel (quantitative stats).
  3. action[t] == robot_state[t+1] (the action-construction definition).

Usage:
    python scripts/verify_track_data.py \
        --tracks_file /data/.../handover_block-demo_clean_depth-100-tracks.hdf5 \
        --raw_data_root /data/.../RoboTwin/data \
        --task_name handover_block --task_config demo_clean_depth \
        --episodes 0 1 50 --out_dir /data/.../data/vis
"""
import argparse
import os

import cv2
import h5py
import numpy as np

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def _reencode_h264(path):
    """cv2's mp4v output doesn't play in many viewers (VSCode preview,
    QuickTime); re-encode to H.264 in place when ffmpeg is available."""
    import shutil, subprocess
    if shutil.which("ffmpeg") is None:
        print(f"[warn] ffmpeg not found; {path} left as mp4v (may not play everywhere)")
        return
    tmp = path + ".h264.mp4"
    r = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", path,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", tmp])
    if r.returncode == 0:
        os.replace(tmp, path)



def world_to_cam(pts_w, extrinsic_cv):
    """P_c = R @ P_w + t for row-vector points (inverse of the dataset's
    P_w = R^T (P_c - t))."""
    R = extrinsic_cv[:, :3]
    t = extrinsic_cv[:, 3]
    return pts_w @ R.T + t


def project(pts_cam, K):
    z = np.where(pts_cam[..., 2] == 0, 1e-6, pts_cam[..., 2])
    u = pts_cam[..., 0] * K[0, 0] / z + K[0, 2]
    v = pts_cam[..., 1] * K[1, 1] / z + K[1, 2]
    return np.stack([u, v], axis=-1), z


def sample_depth(depth_mm, uv):
    H, W = depth_mm.shape
    N = uv.shape[0]
    mx = np.clip(uv[:, 0], 0, W - 1).astype(np.float32).reshape(1, N)
    my = np.clip(uv[:, 1], 0, H - 1).astype(np.float32).reshape(1, N)
    return cv2.remap(depth_mm.astype(np.float32), mx, my, cv2.INTER_LINEAR).reshape(N) / 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks_file", required=True)
    ap.add_argument("--raw_data_root", required=True)
    ap.add_argument("--task_name", required=True)
    ap.add_argument("--task_config", required=True)
    ap.add_argument("--episodes", type=int, nargs="+", default=[0])
    ap.add_argument("--camera", default="head_camera")
    ap.add_argument("--out_dir", default="data/vis")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--track_step", type=int, default=15,
                    help="which future step of the track to draw (0-based)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    f = h5py.File(args.tracks_file, "r", swmr=True)
    episode_ends = f["episode_ends"][:]
    starts = np.concatenate([[0], episode_ends[:-1]])

    for ep in args.episodes:
        s, e = int(starts[ep]), int(episode_ends[ep])
        L = e - s
        centers_w = f["image_patch_centre_3d_position"][s:e]  # [L, N, 3]
        tracks_w = f["3d_track"][s:e]                          # [L, N, H, 3]
        tmask = f["track_valid_mask"][s:e]                     # [L, N, H]
        pmask = f["valid_point_mask"][s:e]                     # [L, N]
        state = f["robot_state"][s:e]                          # [L, 16]
        action = f["action"][s:e]                              # [L, 16]

        raw_path = os.path.join(args.raw_data_root, args.task_name,
                                args.task_config, "data", f"episode{ep}.hdf5")
        rf = h5py.File(raw_path, "r", swmr=True)
        cam = rf["observation"][args.camera]
        K = cam["intrinsic_cv"][0]
        ext = cam["extrinsic_cv"][0]
        rgb_blobs = cam["rgb"]
        depth_all = cam["depth"]

        # ---- Check 3: action[t] == state[t+1] --------------------------
        diff = np.abs(action[:-1] - state[1:]).max(axis=0)  # per-dim max
        print(f"\n=== episode {ep} (frames {s}:{e}, L={L}) ===")
        print(f"[action==next_state] max |action[t]-state[t+1]| per dim:")
        print(f"  pos/quat dims max: {diff.max():.2e}   gripper dims: "
              f"L={diff[7]:.2e} R={diff[15]:.2e}")

        # ---- Check 2: depth consistency --------------------------------
        errs = []
        for t in range(0, L, max(1, L // 20)):
            pc = world_to_cam(centers_w[t], ext)
            uv, z = project(pc, K)
            d = sample_depth(np.asarray(depth_all[t]), uv)
            ok = pmask[t] & (d > 0)
            if ok.sum():
                errs.append(np.abs(z[ok] - d[ok]))
        errs = np.concatenate(errs)
        print(f"[depth consistency] |z_reproj - depth_map|: "
              f"mean={errs.mean()*1000:.1f}mm  p95={np.percentile(errs,95)*1000:.1f}mm  "
              f"max={errs.max()*1000:.1f}mm  (n={len(errs)})")

        # ---- Check 1: overlay video ------------------------------------
        h = args.track_step
        probe = cv2.imdecode(np.frombuffer(rgb_blobs[0], np.uint8), cv2.IMREAD_COLOR)
        H_img, W_img = probe.shape[:2]
        out_path = os.path.join(args.out_dir,
                                f"{args.task_name}_ep{ep}_verify.mp4")
        vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (W_img * 2, H_img))
        for t in range(L):
            # RoboTwin raw blobs were imencode'd from RGB without conversion,
            # so imdecode output is TRUE RGB in memory; convert for BGR video
            img = cv2.imdecode(np.frombuffer(rgb_blobs[t], np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # patch centers (green)
            pc = world_to_cam(centers_w[t], ext)
            uv, _ = project(pc, K)
            for i in range(uv.shape[0]):
                if pmask[t, i]:
                    cv2.circle(img, tuple(uv[i].astype(int)), 1, (0, 255, 0), -1)

            # track endpoints: ALL valid tracks, no magnitude cutoff —
            # shows the raw supervision incl. small drift on static regions
            fut_w = centers_w[t] + tracks_w[t, :, h, :]
            fuv, _ = project(world_to_cam(fut_w, ext), K)
            for i in range(uv.shape[0]):
                if pmask[t, i] and tmask[t, i, h]:
                    cv2.line(img, tuple(uv[i].astype(int)),
                             tuple(fuv[i].astype(int)), (0, 0, 255), 1)

            # EEF positions (left=blue, right=yellow crosses)
            for xyz, col in [(state[t, 0:3], (255, 0, 0)),
                             (state[t, 8:11], (0, 255, 255))]:
                euv, _ = project(world_to_cam(xyz[None], ext), K)
                p = tuple(euv[0].astype(int))
                cv2.drawMarker(img, p, col, cv2.MARKER_CROSS, 14, 2)

            # depth panel with patch grid
            d = np.asarray(depth_all[t], dtype=np.float32)
            dn = np.clip(d / max(d.max(), 1e-6), 0, 1)
            dimg = cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            for i in range(uv.shape[0]):
                if pmask[t, i]:
                    cv2.circle(dimg, tuple(uv[i].astype(int)), 1, (255, 255, 255), -1)

            cv2.putText(img, f"ep{ep} t={t} | green=patch red=RAW track+{h+1} "
                             f"blue/yellow=EEF L/R", (6, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            vw.write(np.hstack([img, dimg]))
        vw.release()
        _reencode_h264(out_path)
        print(f"[video] {out_path}")
        rf.close()
    f.close()


if __name__ == "__main__":
    main()
