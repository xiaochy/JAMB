"""
Convert a flat tracks HDF5 (peilin's generation format) into a JAMB-style
zarr so the track pipeline shares the same storage format, ReplayBuffer
loading path, and tooling as the original JAMB datasets.

Key mapping (hdf5 -> zarr data/):
    robot_state                     -> state
    action                          -> action
    pi3_feature                     -> pi3_features
    image_patch_centre_3d_position  -> patch_centers
    3d_track                        -> 3d_track
    track_valid_mask                -> track_valid_mask
    valid_point_mask                -> valid_point_mask
    dino_features (if present)      -> dino_features
    episode_ends                    -> meta/episode_ends
"""
import argparse
import os
import shutil

import h5py
import numcodecs
import numpy as np
import zarr

KEY_MAP = {
    "robot_state": "state",
    "action": "action",
    "pi3_feature": "pi3_features",
    "image_patch_centre_3d_position": "patch_centers",
    "3d_track": "3d_track",
    "track_valid_mask": "track_valid_mask",
    "valid_point_mask": "valid_point_mask",
    "dino_features": "dino_features",
}


def append_rgb(data, raw_task_dir, episode_ends, compressor, kwargs_extra):
    """Store decoded camera RGB frames aligned to the tracks' per-episode
    lengths. NOTE on colors: RoboTwin saves channel-swapped JPEGs, so
    cv2.imdecode output here is TRUE RGB in memory (double swap cancels) —
    the stored array is genuine RGB. The dino/pi3 features in this zarr are
    still extracted through the legacy (swapped) pipeline; keeping raw RGB
    makes a future true-color feature regeneration possible without the
    raw episodes (docs/color-channel-issue.md)."""
    import cv2
    starts = np.concatenate([[0], episode_ends[:-1]])
    probe = None
    arr = None
    for ep, (s, e) in enumerate(zip(starts, episode_ends)):
        L = int(e - s)
        path = os.path.join(raw_task_dir, "data", f"episode{ep}.hdf5")
        with h5py.File(path, "r") as rf:
            blobs = rf["observation"]["head_camera"]["rgb"]
            assert len(blobs) >= L, \
                f"episode{ep}: raw has {len(blobs)} frames < track length {L}"
            frames = np.stack([
                cv2.imdecode(np.frombuffer(blobs[t], np.uint8), cv2.IMREAD_COLOR)
                for t in range(L)
            ])  # [L, H, W, 3] true RGB (see note above)
        if arr is None:
            probe = frames.shape[1:]
            arr = data.zeros(
                name="rgb",
                shape=(int(episode_ends[-1]), *probe),
                chunks=(8, *probe),
                dtype=np.uint8,
                compressor=compressor,
                **kwargs_extra,
            )
        arr[int(s):int(e)] = frames
    print(f"  raw episodes -> data/rgb: {(int(episode_ends[-1]), *probe)} uint8 (true RGB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf5_path")
    parser.add_argument("zarr_path")
    parser.add_argument("--rows_per_write", type=int, default=512)
    parser.add_argument("--rgb_from", default=None,
                        help="raw task dir (RoboTwin/data/<task>/<config>); "
                             "when set, decoded RGB frames are stored as data/rgb")
    args = parser.parse_args()

    if os.path.exists(args.zarr_path):
        shutil.rmtree(args.zarr_path)

    f = h5py.File(args.hdf5_path, "r")
    compressor = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=1)
    _zarr_v3 = not hasattr(zarr, "Blosc")

    root = zarr.group(args.zarr_path)
    data = root.create_group("data")
    meta = root.create_group("meta")

    kwargs_extra = {"zarr_format": 2} if _zarr_v3 else {}
    ee = f["episode_ends"][:]
    meta_arr = meta.zeros(name="episode_ends", shape=ee.shape, dtype=ee.dtype, **kwargs_extra)
    meta_arr[:] = ee

    for src, dst in KEY_MAP.items():
        if src not in f:
            if src == "dino_features":
                print(f"  (no {src} in source — skipping)")
                continue
            raise KeyError(f"{args.hdf5_path} missing expected key {src}")
        d = f[src]
        # float64 state/action from the sim are stored as float32 like the
        # original JAMB zarrs
        dtype = np.float32 if d.dtype == np.float64 else d.dtype
        chunk0 = min(100, d.shape[0])
        arr = data.zeros(
            name=dst,
            shape=d.shape,
            chunks=(chunk0, *d.shape[1:]),
            dtype=dtype,
            compressor=compressor,
            **kwargs_extra,
        )
        for start in range(0, d.shape[0], args.rows_per_write):
            end = min(start + args.rows_per_write, d.shape[0])
            arr[start:end] = d[start:end].astype(dtype)
        print(f"  {src} -> data/{dst}: {d.shape} {dtype}")

    if args.rgb_from:
        append_rgb(data, args.rgb_from, ee, compressor, kwargs_extra)

    f.close()
    print(f"Done: {args.zarr_path}")


if __name__ == "__main__":
    main()
