"""
Evaluate predicted track quality vs ground truth for one or more models.

Track is extracted from the full DDPM denoising loop (predict_action), so
action tokens are present in the decoder — matching training-time behaviour.

Metrics (per-patch, averaged over sampled frames and episodes):
  - endpoint_err   : L2(pred[:,-1,:], gt[:,-1,:])  — final-step error
  - traj_err       : mean over horizon of L2(pred[:,h,:], gt[:,h,:])
  - pred_norm      : ||pred[:,-1,:]||               — magnitude of prediction
  - gt_norm        : ||gt[:,-1,:]||                 — GT magnitude

Reported separately for:
  - moving patches     : GT endpoint norm > gt_move_thresh
  - background patches : GT endpoint norm <= bg_thresh
  - all patches

Usage:
    cd GAP-track-only-dino
    python scripts/eval_track.py \
        --ckpts  <ckpt_a> <ckpt_b> \
        --labels "uniform" "movew-a01" \
        --zarr_path /data/.../tracks.zarr \
        --episodes 8 43 64 75 99 \
        --num_inference_steps 10 \
        --stride 10 \
        --gt_move_thresh 0.01 \
        --bg_thresh 0.005
"""
import argparse
import os
import sys

import numpy as np
import torch
import zarr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


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
        # This checkpoint's saved cfg predates the track_noise_scheduler fix
        # (clip_sample_range=1.0 silently truncated real moving-patch track
        # magnitude). Inject a separately-clipped scheduler for the track
        # branch only; action's scheduler/behavior is untouched.
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
def predict_tracks(policy, dino_feat, agent_pos, patch_centers, device, pi3_feat=None):
    """Run full DDPM denoising so action tokens are in the decoder alongside track tokens."""
    obs = {
        "agent_pos":     torch.from_numpy(agent_pos).float().unsqueeze(0).to(device),
        "patch_centers": torch.from_numpy(patch_centers).float().unsqueeze(0).to(device),
    }
    if dino_feat is not None:
        dino_tensor = torch.from_numpy(dino_feat).float()
        if dino_tensor.ndim == 2:
            dino_tensor = dino_tensor.unsqueeze(0)
        obs["dino_features"] = dino_tensor.unsqueeze(0).to(device)
    if pi3_feat is not None:
        pi3_tensor = torch.from_numpy(pi3_feat).float()
        if pi3_tensor.ndim == 2:
            pi3_tensor = pi3_tensor.unsqueeze(0)
        obs["pi3_features"] = pi3_tensor.unsqueeze(0).to(device)
    result = policy.predict_action(obs)
    if "track_pred" not in result:
        raise RuntimeError("Model did not return track_pred — check aux_task=track")
    return result["track_pred"][0].cpu().numpy()  # [N, H, 3]


# ─── metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(pred, gt, gt_move_thresh, bg_thresh):
    """
    pred, gt: [N, H, 3]
    Returns dict of scalar metrics.
    """
    N, H, _ = pred.shape
    gt_endpoint_norm = np.linalg.norm(gt[:, -1, :], axis=-1)  # [N]

    moving_mask = gt_endpoint_norm > gt_move_thresh
    bg_mask     = gt_endpoint_norm <= bg_thresh

    # per-patch endpoint error
    endpoint_err = np.linalg.norm(pred[:, -1, :] - gt[:, -1, :], axis=-1)  # [N]

    # per-patch trajectory error (mean over horizon)
    traj_err = np.linalg.norm(pred - gt, axis=-1).mean(axis=-1)  # [N]

    # pred magnitude at endpoint
    pred_norm = np.linalg.norm(pred[:, -1, :], axis=-1)  # [N]
    gt_norm   = np.linalg.norm(gt[:, -1, :], axis=-1)    # [N]

    out = {}
    for name, mask in [("moving", moving_mask), ("bg", bg_mask), ("all", np.ones(N, dtype=bool))]:
        if mask.sum() == 0:
            continue
        out[f"{name}_endpoint_err"] = endpoint_err[mask].mean()
        out[f"{name}_traj_err"]     = traj_err[mask].mean()
        out[f"{name}_pred_norm"]    = pred_norm[mask].mean()
        out[f"{name}_gt_norm"]      = gt_norm[mask].mean()
        out[f"{name}_n_patches"]    = int(mask.sum())
    return out


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts",                nargs="+", required=True)
    ap.add_argument("--labels",               nargs="+", default=None)
    ap.add_argument("--zarr_path",            required=True)
    ap.add_argument("--track_key",            default="gt_3d_track")
    ap.add_argument("--episodes",             type=int, nargs="+", default=list(range(5)))
    ap.add_argument("--gt_move_thresh",       type=float, default=0.01)
    ap.add_argument("--bg_thresh",            type=float, default=0.005)
    ap.add_argument("--num_inference_steps",  type=int, default=10,
                    help="DDPM/DDIM steps for action denoising (10 is fast, 100 matches training)")
    ap.add_argument("--stride",               type=int, default=10,
                    help="Evaluate every Nth frame per episode (1=all frames, 10=every 10th)")
    ap.add_argument("--device",               default="cuda")
    ap.add_argument("--track_clip_range",     type=float, default=None,
                    help="override track branch's clip_sample_range (None = use checkpoint's saved value)")
    args = ap.parse_args()

    labels = args.labels or [f"model_{i}" for i in range(len(args.ckpts))]
    assert len(labels) == len(args.ckpts)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Loading zarr: {args.zarr_path}")
    z = zarr.open(args.zarr_path, mode="r")
    ep_ends  = z["meta/episode_ends"][:]
    ep_starts = np.concatenate([[0], ep_ends[:-1]])
    dino     = z["data/dino_features"] if "dino_features" in z["data"] else None
    pi3      = z["data/pi3_features"]  if "pi3_features"  in z["data"] else None
    state    = z["data/state"]
    pc       = z["data/patch_centers"]
    gt_track = z[f"data/{args.track_key}"]

    # accumulators: label → list of per-frame metric dicts
    accum = {lbl: [] for lbl in labels}

    policies = []
    for ckpt_path, lbl in zip(args.ckpts, labels):
        print(f"Loading {lbl}: {ckpt_path}")
        pol = load_model(ckpt_path, device, track_clip_range=args.track_clip_range)
        pol.num_inference_steps = args.num_inference_steps
        policies.append(pol)

    for ep in args.episodes:
        ep_start = int(ep_starts[ep])
        ep_end   = int(ep_ends[ep])
        n_frames = ep_end - ep_start
        frame_indices = list(range(0, n_frames, args.stride))
        print(f"\nEpisode {ep}: {n_frames} frames, sampling {len(frame_indices)} (stride={args.stride})")

        for t in frame_indices:
            idx = ep_start + t
            d_feat = dino[idx]  if dino  is not None else None
            p3_feat = pi3[idx]  if pi3   is not None else None
            s      = state[idx]
            p      = pc[idx]
            gt_t   = gt_track[idx]  # [N, H, 3]

            for policy, lbl in zip(policies, labels):
                use_pi3 = getattr(policy, "use_pi3_features", False)
                pred_t = predict_tracks(policy, d_feat, s, p, device, pi3_feat=(p3_feat if use_pi3 else None))
                m = compute_metrics(pred_t, gt_t, args.gt_move_thresh, args.bg_thresh)
                accum[lbl].append(m)

    # ── aggregate and print ───────────────────────────────────────────────────
    print("\n" + "="*70)
    print(f"Track evaluation  |  episodes={args.episodes}  |  "
          f"move_thresh={args.gt_move_thresh}m  bg_thresh={args.bg_thresh}m  |  "
          f"inference_steps={args.num_inference_steps}  stride={args.stride}")
    print("="*70)

    groups = ["moving", "bg", "all"]
    metrics = ["endpoint_err", "traj_err", "pred_norm", "gt_norm"]

    for grp in groups:
        key_ep  = f"{grp}_endpoint_err"
        key_tr  = f"{grp}_traj_err"
        key_pn  = f"{grp}_pred_norm"
        key_gtn = f"{grp}_gt_norm"
        key_n   = f"{grp}_n_patches"

        print(f"\n── {grp.upper()} patches ──")

        header = f"{'model':<20}  {'#patches':>8}  {'endpoint_err':>12}  {'traj_err':>9}  {'pred_norm':>9}  {'gt_norm':>9}"
        print(header)
        print("-" * len(header))

        for lbl in labels:
            frames = [f for f in accum[lbl] if key_ep in f]
            if not frames:
                print(f"  {lbl}: no data")
                continue
            ep_err  = np.mean([f[key_ep]  for f in frames])
            tr_err  = np.mean([f[key_tr]  for f in frames])
            pn      = np.mean([f[key_pn]  for f in frames])
            gtn     = np.mean([f[key_gtn] for f in frames])
            n_pat   = int(np.mean([f[key_n] for f in frames]))
            print(f"  {lbl:<18}  {n_pat:>8}  {ep_err:>12.4f}m  {tr_err:>9.4f}m  {pn:>9.4f}m  {gtn:>9.4f}m")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
