"""
Temporal-localization diagnostics for track-predicting GAP checkpoints.

Decides whether the compressed track token (Approach B) shows a measurable
temporal failure that would justify upgrading to point-time tokens +
serialized attention (Approach C / advisor doc §15.4 discipline).

Metrics (all offline, no simulator):
  1. per-step ADE:  err(h) = mean ||pred_d[:,h] - gt_d[:,h]||  for h=1..16
     -> late-step blowup vs a reference ckpt = temporal capacity issue
  2. time-shift lag k*: argmin_k mean_h ||pred[:,h] - gt[:,h+k]||
     -> systematic |k*| > 0 = the model knows WHAT but not WHEN
  3. temporal blur ratio: Var_h(pred) / Var_h(gt) (valid, moving points)
     -> << 1 = prediction is a time-average, temporal structure lost
  4. velocity error (doc eq 49): mean ||dpred/dh - dgt/dh||

Works with any ckpt whose policy outputs track_pred [B,N,16,3]:
regression (track head), jointdiff compressed, pointtime.

Usage:
    python scripts/eval_track_timing.py --ckpt <path/to/50.ckpt> \
        --data /dev/shm/gap_tracks_handover [--n_batches 25] [--device cuda:0]
"""
import argparse
import json
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch


def build_policy_from_ckpt(ckpt_path, device):
    """Mirror deploy_policy: model architecture comes from the cfg INSIDE
    the ckpt, so this script works across branches/architectures."""
    import importlib
    from jamb_policy.policy.jamb import JAMBPolicy
    from jamb_policy.model.common.normalizer import LinearNormalizer

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    pol = dict(cfg["policy"]) if isinstance(cfg, dict) else dict(cfg.policy)
    target = pol.pop("_target_", None)
    ns = pol["noise_scheduler"]
    ns = dict(ns) if not isinstance(ns, dict) else dict(ns)
    ns_target = ns.pop("_target_")
    module, clsname = ns_target.rsplit(".", 1)
    scheduler = getattr(importlib.import_module(module), clsname)(**ns)
    pol["noise_scheduler"] = scheduler
    policy = JAMBPolicy(**pol)
    state = ckpt.get("ema") or ckpt["model"]
    policy.load_state_dict(state)
    norm = LinearNormalizer()
    norm.load_state_dict(ckpt["normalizer"])
    policy.set_normalizer(norm)
    return policy.to(device).eval(), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="/dev/shm/gap_tracks_handover")
    ap.add_argument("--n_batches", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_lag", type=int, default=4)
    ap.add_argument("--move_thresh", type=float, default=0.005,
                    help="a point counts as 'moving' if its gt displacement "
                         "exceeds this (m) at any step — blur/lag metrics "
                         "are computed on moving points only")
    ap.add_argument("--out", default=None, help="json output path")
    args = ap.parse_args()

    from torch.utils.data import DataLoader
    from jamb_policy.dataset.track_dataset import TrackDataset

    policy, cfg = build_policy_from_ckpt(args.ckpt, args.device)
    getcfg = (lambda k, d=None: cfg.get(k, d)) if isinstance(cfg, dict) \
        else (lambda k, d=None: getattr(cfg, k, d))
    ds = TrackDataset(
        data_path=args.data, horizon=getcfg("horizon", 16),
        pad_before=getcfg("n_obs_steps", 1) - 1,
        pad_after=getcfg("n_action_steps", 16) - 1,
        seed=0, val_ratio=0.05, max_train_episodes=100,
        task_name="handover_block", use_dino_features=True, aux_task="track")
    val = ds.get_validation_dataset()
    loader = DataLoader(val, batch_size=args.batch_size, shuffle=True,
                        num_workers=2)
    torch.manual_seed(0)

    T = getcfg("horizon", 16)
    sum_ade = np.zeros(T); n_ade = np.zeros(T)
    lag_hist = np.zeros(2 * args.max_lag + 1)
    blur_ratios, vel_errs = [], []

    for bi, batch in enumerate(loader):
        if bi >= args.n_batches:
            break
        obs = {k: v.to(args.device) for k, v in batch["obs"].items()}
        gt = batch["gt_3d_track"].to(args.device)          # [B, N, T, 3]
        mask = batch["track_valid_mask"].to(args.device)   # [B, N, T]
        with torch.no_grad():
            pred = policy.predict_action(obs)["track_pred"]  # [B, N, T, 3] meters
        # ---- 1. per-step ADE (valid entries) ----
        err = (pred - gt).norm(dim=-1)                     # [B, N, T]
        for h in range(T):
            m = mask[..., h]
            if m.any():
                sum_ade[h] += err[..., h][m].sum().item()
                n_ade[h] += m.sum().item()
        # moving-point selection for lag/blur
        moving = (gt.norm(dim=-1).amax(dim=-1) > args.move_thresh) \
                 & mask.all(dim=-1)                        # [B, N]
        if moving.any():
            p = pred[moving]                               # [M, T, 3]
            g = gt[moving]
            # ---- 2. time-shift lag ----
            for k in range(-args.max_lag, args.max_lag + 1):
                if k >= 0:
                    d = (p[:, : T - k] - g[:, k:]).norm(dim=-1).mean(dim=1)
                else:
                    d = (p[:, -k:] - g[:, : T + k]).norm(dim=-1).mean(dim=1)
                if k == -args.max_lag:
                    best = d.clone(); best_k = torch.full_like(d, k)
                else:
                    upd = d < best
                    best = torch.where(upd, d, best)
                    best_k = torch.where(upd, torch.full_like(d, k), best_k)
            for k in range(-args.max_lag, args.max_lag + 1):
                lag_hist[k + args.max_lag] += (best_k == k).sum().item()
            # ---- 3. temporal blur ratio ----
            vp = p.var(dim=1).mean(dim=-1)                 # Var over T, [M]
            vg = g.var(dim=1).mean(dim=-1)
            ok = vg > 1e-8
            if ok.any():
                blur_ratios.append((vp[ok] / vg[ok]).cpu())
            # ---- 4. velocity error ----
            vel_errs.append(((p[:, 1:] - p[:, :-1]) - (g[:, 1:] - g[:, :-1]))
                            .norm(dim=-1).mean().item())

    ade = (sum_ade / np.maximum(n_ade, 1)) * 1000  # mm
    blur = torch.cat(blur_ratios).median().item() if blur_ratios else float("nan")
    lag_p = lag_hist / max(lag_hist.sum(), 1)
    mean_lag = float(np.sum(lag_p * np.arange(-args.max_lag, args.max_lag + 1)))

    print("\n===== temporal-localization diagnostics =====")
    print(f"ckpt: {args.ckpt}")
    print("per-step ADE (mm):", " ".join(f"{v:.1f}" for v in ade))
    print(f"late/early ADE ratio (mean h12-16 / mean h1-4): "
          f"{ade[11:].mean() / max(ade[:4].mean(), 1e-6):.2f}")
    print(f"time-shift lag: mean={mean_lag:+.2f} steps, "
          f"P(k*=0)={lag_p[args.max_lag]:.2f}, dist="
          + " ".join(f"{k:+d}:{p:.2f}" for k, p in
                     zip(range(-args.max_lag, args.max_lag + 1), lag_p) if p > 0.01))
    print(f"temporal blur ratio (median, moving pts): {blur:.3f}  (1.0=full "
          f"temporal structure, <<1 = time-averaged)")
    print(f"velocity error (mm/step): {np.mean(vel_errs)*1000:.2f}")
    print("verdict hints: blur<0.5 or |mean lag|>1 or P(k*=0)<0.5 or "
          "late/early>4 => temporal-localization concern (Phase 4 candidate)")

    if args.out:
        json.dump({"ckpt": args.ckpt, "ade_mm": ade.tolist(),
                   "late_early_ratio": float(ade[11:].mean() / max(ade[:4].mean(), 1e-6)),
                   "mean_lag": mean_lag, "p_lag0": float(lag_p[args.max_lag]),
                   "blur_ratio_median": blur,
                   "vel_err_mm": float(np.mean(vel_errs) * 1000)},
                  open(args.out, "w"), indent=2)
        print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
