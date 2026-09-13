"""
Overlay predicted track trajectories onto RoboTwin eval rollout videos.

Consumes the per-chunk track-prediction logs written by scripts/eval_policy.py
when TRACK_VIS_LOG_DIR is set during eval (episode{N}_trackvis.pkl: a list of
per-action-chunk dicts with track_pred [N,H,3] displacement, patch_centers
[N,3] world-frame, and that chunk's camera intrinsic/extrinsic), and draws
the same threshold+colormap track overlay used by visualize_track_analysis.py
onto the corresponding frame range of the raw episode{N}.mp4, so failure-case
analysis has the model's predicted track visible alongside what it actually
did.

Usage (single episode):
    python scripts/visualize_eval_tracks.py \
        --video results/.../episode0.mp4 \
        --trackvis results/.../trackvis/episode0_trackvis.pkl \
        --out results/.../episode0_overlay.mp4

Usage (batch over a directory -- pairs episodeN.mp4 with
trackvis_dir/episodeN_trackvis.pkl for every N found; if a _rollouts.json
sits next to the episode videos, its per-episode success flag is used to
sort outputs into out_dir/success/ and out_dir/failure/ so failure-case
review doesn't need to scrub through everything):
    python scripts/visualize_eval_tracks.py \
        --episode_dir results/.../seed_0/200 \
        --trackvis_dir results/.../seed_0/200/trackvis \
        --out_dir results/.../seed_0/200/overlay

Color: red->yellow gradient (COLORMAP_AUTUMN) by predicted displacement
magnitude -- low (near move_thresh) = red, high (near mag_ref) = yellow.
Fixed per episode (not per-frame), so the mapping is stable throughout a
video instead of cycling through unrelated colors.
"""
import argparse
import glob
import json
import os
import pickle
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from visualize_track_analysis import _draw_tracks, _reencode_h264  # noqa: E402


def _load_success_map(episode_dir):
    """episode -> bool success, from _rollouts.json next to the episode videos. {} if absent."""
    path = os.path.join(episode_dir, "_rollouts.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    return {int(e["episode"]): bool(e["success"]) for e in data.get("episodes", [])}


def overlay_one(video_path, trackvis_path, out_path, move_thresh=0.01, mag_ref=None,
                 colormap=cv2.COLORMAP_AUTUMN):
    with open(trackvis_path, "rb") as f:
        chunks = pickle.load(f)
    if not chunks:
        print(f"  [skip] {trackvis_path}: empty track log")
        return False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [skip] {video_path}: could not open")
        return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Fixed color scale across the whole episode (not per-chunk / per-frame),
    # so track color is comparable frame-to-frame instead of re-normalizing.
    if mag_ref is None:
        all_mag = []
        for c in chunks:
            disp = np.asarray(c["track_pred"])  # [N, H, 3]
            all_mag.append(np.linalg.norm(disp[:, -1, :], axis=-1))
        all_mag = np.concatenate(all_mag) if all_mag else np.array([move_thresh * 3])
        mag_ref = max(float(all_mag.max()), move_thresh * 3)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    # chunk lookup: frame_idx -> chunk dict (last chunk covers any tail past
    # its logged n_frames, e.g. the final partial chunk before success/timeout)
    chunk_ranges = [(c["start_frame"], c["start_frame"] + max(c["n_frames"], 1), c) for c in chunks]

    frame_idx = 0
    drawn = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        chunk = None
        for start, end, c in chunk_ranges:
            if start <= frame_idx < end:
                chunk = c
                break
        if chunk is None and chunk_ranges:
            chunk = chunk_ranges[-1][2] if frame_idx >= chunk_ranges[-1][0] else chunk_ranges[0][2]

        if chunk is not None:
            patch_centers = np.asarray(chunk["patch_centers"], dtype=np.float32)
            track_pred = np.asarray(chunk["track_pred"], dtype=np.float32)
            K = np.asarray(chunk["intrinsic_cv"], dtype=np.float32)
            ext = np.asarray(chunk["extrinsic_cv"], dtype=np.float32)
            out_frame = _draw_tracks(frame_bgr, patch_centers, track_pred, K, ext,
                                      move_thresh=move_thresh, mag_ref=mag_ref,
                                      colormap=colormap)
            drawn += 1
        else:
            out_frame = frame_bgr
        writer.write(out_frame)
        frame_idx += 1

    cap.release()
    writer.release()
    _reencode_h264(out_path)
    print(f"  [ok] {out_path}: {drawn}/{frame_idx} frames overlaid ({len(chunks)} chunks), mag_ref={mag_ref:.4f}m")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video")
    ap.add_argument("--trackvis")
    ap.add_argument("--out")
    ap.add_argument("--episode_dir")
    ap.add_argument("--trackvis_dir")
    ap.add_argument("--out_dir")
    ap.add_argument("--move_thresh", type=float, default=0.01)
    ap.add_argument("--mag_ref", type=float, default=None)
    args = ap.parse_args()

    if args.video:
        overlay_one(args.video, args.trackvis, args.out,
                    move_thresh=args.move_thresh, mag_ref=args.mag_ref)
        return

    videos = sorted(glob.glob(os.path.join(args.episode_dir, "episode*.mp4")))
    videos = [v for v in videos if not v.endswith("_overlay.mp4")]
    print(f"Found {len(videos)} episode videos in {args.episode_dir}")
    success_map = _load_success_map(args.episode_dir)
    if success_map:
        print(f"  success/failure split from _rollouts.json: "
              f"{sum(success_map.values())} success, {sum(not v for v in success_map.values())} failure")
    else:
        print("  no _rollouts.json found -- outputs go directly under out_dir (no success/failure split)")
    n_ok = 0
    for v in videos:
        m = re.search(r"episode(\d+)\.mp4$", v)
        if not m:
            continue
        ep = m.group(1)
        tv = os.path.join(args.trackvis_dir, f"episode{ep}_trackvis.pkl")
        if not os.path.exists(tv):
            print(f"  [skip] episode{ep}: no trackvis log at {tv}")
            continue
        if int(ep) in success_map:
            subdir = "success" if success_map[int(ep)] else "failure"
            out = os.path.join(args.out_dir, subdir, f"episode{ep}_overlay.mp4")
        else:
            out = os.path.join(args.out_dir, f"episode{ep}_overlay.mp4")
        if overlay_one(v, tv, out, move_thresh=args.move_thresh, mag_ref=args.mag_ref):
            n_ok += 1
    print(f"Done: {n_ok}/{len(videos)} overlaid")


if __name__ == "__main__":
    main()
