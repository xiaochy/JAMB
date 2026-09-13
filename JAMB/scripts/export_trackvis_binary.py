"""
Pack one episode's trackvis pkl (from scripts/eval_policy.py's
TRACK_VIS_LOG_DIR logging) into the binary-in-JSON format webviz/track3d's
viewer expects: {"b64": "<base64 of the packed binary>"}.

Binary layout (little-endian). The dense point cloud is only captured every
DENSE_PC_EVERY chunks (see eval_policy.py) to keep long episodes under the
16MB per-file export budget, so chunks are split into a small snapshot
table (one dense point cloud each) plus a per-chunk index into it -- a
chunk without its own capture reuses the most recent earlier snapshot.
compute_dense_pointcloud_world zeros (never drops) invalid-depth points, so
every snapshot has the same point count (n_dense):

    u32 n_chunks, n_snapshots, n_dense, n_patch, horizon
    snapshots (n_snapshots of):
      f32[n_dense*3]         xyz
      u8[n_dense*3]          rgb
    chunks (n_chunks of):
      u32 n_frames
      u32 snapshot_idx        (index into the snapshots table above)
      f32[n_patch*3]         patch_centers
      f32[n_patch*horizon*3] track_pred

Usage:
    python scripts/export_trackvis_binary.py \
        results/.../trackvis/episode0_trackvis.pkl \
        webviz/track3d/episodes/my_episode.json
"""
import argparse
import base64
import json
import pickle
import struct

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trackvis_pkl")
    ap.add_argument("out_json")
    args = ap.parse_args()

    with open(args.trackvis_pkl, "rb") as f:
        chunks = pickle.load(f)
    if not chunks:
        raise SystemExit(f"{args.trackvis_pkl}: empty trackvis log")
    if not any("dense_pc_xyz" in c for c in chunks):
        raise SystemExit(f"{args.trackvis_pkl}: no chunk has dense_pc_xyz "
                          "(old-format log, or eval_policy.py ran before that field existed)")

    n_chunks = len(chunks)
    n_patch = np.asarray(chunks[0]["patch_centers"]).shape[0]
    horizon = np.asarray(chunks[0]["track_pred"]).shape[1]

    snapshots = []  # list of (xyz, rgb)
    chunk_snapshot_idx = []
    n_dense = None
    for c in chunks:
        if "dense_pc_xyz" in c:
            xyz = np.asarray(c["dense_pc_xyz"], dtype="<f4")
            rgb = np.asarray(c["dense_pc_rgb"], dtype=np.uint8)
            if n_dense is None:
                n_dense = xyz.shape[0]
            elif xyz.shape[0] != n_dense:
                raise SystemExit(f"{args.trackvis_pkl}: dense point count changed mid-episode "
                                  f"({n_dense} -> {xyz.shape[0]}) -- compute_dense_pointcloud_world "
                                  "should always zero-pad to a constant count")
            snapshots.append((xyz, rgb))
            chunk_snapshot_idx.append(len(snapshots) - 1)
        else:
            if not snapshots:
                raise SystemExit(f"{args.trackvis_pkl}: first chunk has no dense_pc_xyz to seed the snapshot table")
            chunk_snapshot_idx.append(len(snapshots) - 1)

    buf = bytearray()
    buf += struct.pack("<IIIII", n_chunks, len(snapshots), n_dense, n_patch, horizon)
    for xyz, rgb in snapshots:
        buf += xyz.tobytes()
        buf += rgb.tobytes()
    for c, snap_idx in zip(chunks, chunk_snapshot_idx):
        buf += struct.pack("<II", int(c["n_frames"]), snap_idx)
        buf += np.asarray(c["patch_centers"], dtype="<f4").tobytes()
        buf += np.asarray(c["track_pred"], dtype="<f4").tobytes()

    b64 = base64.b64encode(bytes(buf)).decode("ascii")
    with open(args.out_json, "w") as f:
        json.dump({"b64": b64}, f)
    print(f"[ok] {args.out_json}: {n_chunks} chunks, {len(snapshots)} dense snapshots "
          f"({n_dense} pts each), {len(buf)} bytes packed ({len(b64)} bytes base64)")


if __name__ == "__main__":
    main()
