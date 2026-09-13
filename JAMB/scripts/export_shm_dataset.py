"""
Export a tracks zarr to a directory of raw .npy files (typically on
/dev/shm tmpfs) so that MULTIPLE concurrent trainings share ONE physical
copy of the dataset via np.load(mmap_mode='r') instead of each process
holding its own ~95GB decompressed copy.

The RAM cost is paid once by the export (tmpfs pages); every training
process then maps the same pages read-only. TrackDataset auto-detects a
directory containing episode_ends.npy; point trainings at it with
TRACKS_SHM_DIR=<dir>.

Usage:
    python scripts/export_shm_dataset.py \
        JAMB/data/tracks/handover_block-demo_clean_depth-100-tracks.zarr \
        /dev/shm/gap_tracks_handover
"""
import argparse
import os
import shutil

import numpy as np
import zarr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zarr_path")
    ap.add_argument("out_dir")
    ap.add_argument("--keys", nargs="+", default=None,
                    help="subset of data keys (default: all)")
    args = ap.parse_args()

    root = zarr.open(args.zarr_path, mode="r")
    data = root["data"]
    keys = args.keys or list(data.keys())

    total = sum(
        int(np.prod(data[k].shape)) * data[k].dtype.itemsize for k in keys
    )
    stat = shutil.disk_usage(os.path.dirname(args.out_dir) or "/")
    print(f"export size: {total/2**30:.1f} GiB; "
          f"target free: {stat.free/2**30:.1f} GiB")
    if total > stat.free:
        raise SystemExit("Not enough space at target — aborting. "
                         "(On /dev/shm this consumes RAM; free memory first.)")

    os.makedirs(args.out_dir, exist_ok=True)
    ee = root["meta"]["episode_ends"][:]
    np.save(os.path.join(args.out_dir, "episode_ends.npy"), ee)

    for k in keys:
        arr = data[k]
        out_path = os.path.join(args.out_dir, f"{k}.npy")
        print(f"  {k}: {arr.shape} {arr.dtype} -> {out_path}")
        out = np.lib.format.open_memmap(
            out_path, mode="w+", dtype=arr.dtype, shape=arr.shape)
        step = max(1, 512)
        for s in range(0, arr.shape[0], step):
            e = min(s + step, arr.shape[0])
            out[s:e] = arr[s:e]
        out.flush()
        del out

    print(f"Done. Launch trainings with TRACKS_SHM_DIR={args.out_dir}")


if __name__ == "__main__":
    main()
