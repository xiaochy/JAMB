"""
3D visualization of predicted tracks over an episode, from the per-chunk
trackvis pkl logged by eval_policy.py (TRACK_VIS_LOG_DIR) -- same source
scripts/visualize_eval_tracks.py uses for the 2D video overlay.

v3: the dense colored point cloud (see deploy_policy.py's
compute_dense_pointcloud_world) is now logged and drawn PER CHUNK, not just
once at episode start -- a static background made picked-up/moved objects
look frozen in place even on a successful episode. No axis box/grid/ticks
(pure point-cloud + track view). View bounds are recomputed each frame from
that chunk's own dense point cloud (5th-95th percentile per axis, padded,
forced to a cube so geometry isn't stretched) so the workspace fills the
frame instead of sitting in a fixed, mostly-empty room-sized box.

v2 (kept): fixed camera (no rotation), previous chunk's trajectory held
faintly visible for one extra frame (alpha 0.25) so consecutive chunks'
replanned tracks read as a transition rather than a hard cut -- that
discontinuity is a real property of receding-horizon replanning (each chunk
is a fresh prediction from the current state), not a rendering bug, but a
fade instead of an instant swap is easier to follow. Each chunk is held for
exactly as many output frames as it covered real sim/video frames, so this
video's runtime matches the original episode video's runtime.

Usage:
    python scripts/visualize_track_3d.py \
        --trackvis results/.../trackvis/episode0_trackvis.pkl \
        --out results/.../episode0_track3d.mp4
"""
import argparse
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np


def _tight_cube_bounds(xyz, pad_frac=0.25, min_half=0.1):
    lo = np.percentile(xyz, 5, axis=0)
    hi = np.percentile(xyz, 95, axis=0)
    center = (lo + hi) / 2.0
    half = max(float(np.max(hi - lo)) / 2.0, min_half)
    half *= (1.0 + pad_frac)
    return center, half


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trackvis", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--move_thresh", type=float, default=0.01)
    ap.add_argument("--fps", type=float, default=10.0,
                     help="output video fps -- default 10 matches the raw episode video's recording fps")
    ap.add_argument("--source_fps", type=float, default=10.0,
                     help="fps the chunk n_frames counts were recorded at (raw episode video's fps)")
    ap.add_argument("--elev", type=float, default=15.0)
    ap.add_argument("--azim", type=float, default=-60.0)
    args = ap.parse_args()

    with open(args.trackvis, "rb") as f:
        chunks = pickle.load(f)
    if not chunks:
        print(f"[skip] {args.trackvis}: empty track log")
        return

    all_mag = np.concatenate([
        np.linalg.norm(np.asarray(c["track_pred"])[:, -1, :], axis=-1) for c in chunks
    ])
    mag_ref = max(float(all_mag.max()), args.move_thresh * 3)

    # Dense point cloud is only captured every DENSE_PC_EVERY chunks (see
    # eval_policy.py) to bound export size on long episodes; forward-fill
    # the most recent snapshot for chunks that didn't capture their own.
    has_dense = any("dense_pc_xyz" in c for c in chunks)
    if not has_dense:
        print("[warn] no chunk has dense_pc_xyz (old-format log, or logged before this "
              "field existed) -- falling back to sparse patch_centers as the background")
    else:
        last_xyz, last_rgb = None, None
        for c in chunks:
            if "dense_pc_xyz" in c:
                last_xyz, last_rgb = c["dense_pc_xyz"], c["dense_pc_rgb"]
            else:
                c["dense_pc_xyz"], c["dense_pc_rgb"] = last_xyz, last_rgb

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("autumn")

    def draw_chunk(c, alpha):
        pc = np.asarray(c["patch_centers"], dtype=np.float32)
        disp = np.asarray(c["track_pred"], dtype=np.float32)
        mag = np.linalg.norm(disp[:, -1, :], axis=-1)
        idx = np.where(mag > args.move_thresh)[0]
        for j in idx:
            norm = np.clip((mag[j] - args.move_thresh) / max(mag_ref - args.move_thresh, 1e-6), 0, 1)
            color = cmap(float(norm))
            traj = np.vstack([pc[j], pc[j] + disp[j]])
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=color, linewidth=3.5, alpha=alpha)
            ax.scatter([pc[j, 0]], [pc[j, 1]], [pc[j, 2]], color="black", s=10, alpha=alpha, depthshade=False)

    writer = FFMpegWriter(fps=args.fps)
    with writer.saving(fig, args.out, dpi=120):
        for i, c in enumerate(chunks):
            ax.cla()
            ax.set_axis_off()

            if has_dense:
                dense_xyz = np.asarray(c["dense_pc_xyz"], dtype=np.float32)
                dense_rgb = np.asarray(c["dense_pc_rgb"], dtype=np.float32) / 255.0
                ax.scatter(dense_xyz[:, 0], dense_xyz[:, 1], dense_xyz[:, 2],
                           c=dense_rgb, s=2.0, alpha=0.6, depthshade=False)
                bounds_src = dense_xyz
            else:
                pc = np.asarray(c["patch_centers"], dtype=np.float32)
                ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c="lightgray", s=3, alpha=0.35, depthshade=False)
                bounds_src = pc

            if i > 0:
                draw_chunk(chunks[i - 1], alpha=0.25)
            draw_chunk(c, alpha=1.0)

            center, half = _tight_cube_bounds(bounds_src)
            ax.set_xlim(center[0] - half, center[0] + half)
            ax.set_ylim(center[1] - half, center[1] + half)
            ax.set_zlim(center[2] - half, center[2] + half)
            ax.view_init(elev=args.elev, azim=args.azim)
            # Hold this chunk for as many output frames as it covered real
            # sim frames, converted from source_fps to the output fps, so
            # total runtime matches the raw episode video (not sped up).
            n_hold = max(1, round(c.get("n_frames", 1) * args.fps / args.source_fps))
            for _ in range(n_hold):
                writer.grab_frame()

    plt.close(fig)
    total_frames = sum(max(1, round(c.get("n_frames", 1) * args.fps / args.source_fps)) for c in chunks)
    print(f"[ok] {args.out}: {len(chunks)} chunks, {total_frames} output frames @ {args.fps}fps "
          f"(~{total_frames/args.fps:.1f}s), mag_ref={mag_ref:.4f}m, dense_pc={'yes' if has_dense else 'no'}")


if __name__ == "__main__":
    main()
