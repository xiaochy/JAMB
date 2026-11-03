"""
Append a `dino_features` dataset (DinoV2 vitl14_reg, 17x23=391 patches) to a
flat tracks HDF5, computed from the SAME raw episodes the tracks were
generated from.

Frame alignment: the flat HDF5 concatenates episodes; episode i contributes
L_i = episode_ends[i] - episode_ends[i-1] frames, which are the FIRST L_i
frames of raw episode i (generate_training_data.py uses frames 0..T-2).
This script verifies L_i <= T_raw_i for every episode before writing —
if that fails, the HDF5 was generated from different raw data and adding
features from these episodes would silently misalign.

Preprocessing matches the training pipeline: resize to nearest multiple of
14 (240x320 -> 238x322, same grid as Pi3), BGR->RGB, ImageNet normalize.
"""
import argparse
import os
import sys

import cv2
import h5py
import numpy as np
import torch

GAP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, GAP_ROOT)
from jamb_policy.model.vision.dinov2_encoder import DINOV2


def decode_episode_frames(raw_path, n_frames, camera="head_camera"):
    """Decode the first n_frames JPEG frames of a raw episode (BGR)."""
    with h5py.File(raw_path, "r") as f:
        blobs = f[f"observation/{camera}/rgb"]
        total = blobs.shape[0]
        assert n_frames <= total, (
            f"{raw_path}: need {n_frames} frames but raw episode has {total} — "
            "the tracks HDF5 was generated from DIFFERENT raw data; aborting "
            "to avoid silent misalignment."
        )
        frames = []
        for j in range(n_frames):
            img = cv2.imdecode(np.frombuffer(blobs[j], np.uint8), cv2.IMREAD_COLOR)
            frames.append(img)
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tracks_hdf5")
    parser.add_argument("--raw_data_root", required=True)
    parser.add_argument("--task_name", required=True)
    parser.add_argument("--task_config", default="demo_clean")
    parser.add_argument("--camera", default="head_camera")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--model_name", default="dinov2_vitl14_reg",
    )
    parser.add_argument(
        "--weights_path",
        default=os.path.join(GAP_ROOT, "pretrained", "dinov2_vitl14_reg4_pretrain.pth"),
    )
    args = parser.parse_args()

    f = h5py.File(args.tracks_hdf5, "a")
    if "dino_features" in f:
        print(f"dino_features already present ({f['dino_features'].shape}); nothing to do.")
        return
    episode_ends = f["episode_ends"][:]
    T_total = int(episode_ends[-1])
    lengths = np.diff(np.concatenate([[0], episode_ends])).astype(int)
    print(f"{len(lengths)} episodes, {T_total} total frames")

    device = torch.device(args.device)
    model = DINOV2(
        model_name=args.model_name,
        freeze=True,
        repo_dir=os.path.join(GAP_ROOT, "thirdparty", "dinov2"),
        weights_path=args.weights_path,
    ).to(device).eval()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    # Probe first frame for target size / patch count
    raw0 = os.path.join(args.raw_data_root, args.task_name, args.task_config,
                        "data", "episode0.hdf5")
    probe = decode_episode_frames(raw0, 1, args.camera)[0]
    H, W = probe.shape[:2]
    th = max(round(H / 14) * 14, 14)
    tw = max(round(W / 14) * 14, 14)
    n_patches = (th // 14) * (tw // 14)
    print(f"resize {H}x{W} -> {th}x{tw}, {n_patches} patches")

    out = f.create_dataset(
        "dino_features",
        shape=(T_total, 1, n_patches, model.embed_dim),
        dtype=np.float32,
        chunks=(16, 1, n_patches, model.embed_dim),
        compression="gzip",
        compression_opts=4,
    )

    write_ptr = 0
    for ep_idx, L in enumerate(lengths):
        raw_path = os.path.join(args.raw_data_root, args.task_name,
                                args.task_config, "data", f"episode{ep_idx}.hdf5")
        frames = decode_episode_frames(raw_path, L, args.camera)
        for start in range(0, L, args.batch_size):
            chunk = frames[start : start + args.batch_size]
            imgs = []
            for img in chunk:
                img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                imgs.append(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)
            batch = torch.stack(imgs).to(device)
            batch = (batch - mean) / std
            with torch.no_grad():
                feats = model(batch)  # [b, n_patches, 1024]
            b = feats.shape[0]
            out[write_ptr : write_ptr + b] = (
                feats.unsqueeze(1).cpu().numpy().astype(np.float32)
            )
            write_ptr += b
        print(f"episode {ep_idx + 1}/{len(lengths)} done ({write_ptr}/{T_total})", flush=True)

    assert write_ptr == T_total, f"wrote {write_ptr}, expected {T_total}"
    f.close()
    print(f"Done: dino_features ({T_total}, 1, {n_patches}, {model.embed_dim}) "
          f"appended to {args.tracks_hdf5}")


if __name__ == "__main__":
    main()
