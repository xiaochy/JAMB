"""
Track prediction and self-attention visualization for JAMB experiments.

Generates per experiment:
  1. Full episode VIDEO: GT track vs predicted track side-by-side (all frames)
  2. Static images at specified timesteps: GT | Pred | Error heatmap
  3. Self-attention heatmaps at specified timesteps (action→action vs action→track)
  4. Comparison summary chart across experiments

Usage:
    cd GAP-norope
    python scripts/visualize_track_analysis.py \
        --ckpts CKPT1 CKPT2 CKPT3 \
        --labels track_norope tracklambda jointdiff \
        --tracks_file /data/.../tracks/handover_block-...-tracks.hdf5 \
        --raw_data_root /data/.../RoboTwin/data \
        --task_name handover_block --task_config demo_clean_depth \
        --episodes 0 1 --attn_timesteps 40 80 120 \
        --out_dir data/vis/track_analysis/handover_block
"""
import argparse
import io
import json
import os
import sys
import types

import cv2
import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


# ─────────────────────────────────────────────────────────────────────────────
# Attention weight capture
# ─────────────────────────────────────────────────────────────────────────────

def _patch_self_attn_for_weights(policy_model):
    patches = []

    def make_forward_with_weights(orig_module):
        def forward_with_weights(
            self, query, key, value,
            rope=None, q_cos_sin=None, k_cos_sin=None,
            q_rope_mask=None, k_rope_mask=None,
            attn_mask=None, key_padding_mask=None,
        ):
            q = self._split(self.q_proj(query))
            k = self._split(self.k_proj(key))
            v = self._split(self.v_proj(value))
            if rope is not None:
                if q_cos_sin is not None:
                    q = rope.apply(q, q_cos_sin[0], q_cos_sin[1], rope_mask=q_rope_mask)
                if k_cos_sin is not None:
                    k = rope.apply(k, k_cos_sin[0], k_cos_sin[1], rope_mask=k_rope_mask)
            scale = q.shape[-1] ** -0.5
            scores = (q @ k.transpose(-2, -1)) * scale
            if attn_mask is not None:
                scores = scores + attn_mask
            if key_padding_mask is not None:
                kp = key_padding_mask[:, None, None, :]
                bias = torch.zeros_like(kp, dtype=scores.dtype).masked_fill(kp, float("-inf"))
                scores = scores + bias
            weights = scores.softmax(dim=-1)
            self._last_attn_weights = weights.detach().cpu().float()
            out = weights @ v
            B, _, N_q, _ = out.shape
            out = out.transpose(1, 2).reshape(B, N_q, self.nhead * self.head_dim)
            return self.out_proj(out)
        return types.MethodType(forward_with_weights, orig_module)

    for layer in policy_model.transformer_decoder.layers:
        sa = layer.self_attn
        orig = sa.forward
        sa.forward = make_forward_with_weights(sa)
        patches.append((sa, orig))
    return patches


def _unpatch_self_attn(patches):
    for module, orig in patches:
        module.forward = orig


def _collect_attn_weights(policy_model):
    return [getattr(layer.self_attn, "_last_attn_weights", None)
            for layer in policy_model.transformer_decoder.layers
            if hasattr(layer.self_attn, "_last_attn_weights")]


# ─────────────────────────────────────────────────────────────────────────────
# Projection helpers
# ─────────────────────────────────────────────────────────────────────────────

def _world_to_cam(pts_w, ext):
    return pts_w @ ext[:, :3].T + ext[:, 3]


def _project(pts_cam, K):
    z = np.where(pts_cam[..., 2] == 0, 1e-6, pts_cam[..., 2])
    u = pts_cam[..., 0] * K[0, 0] / z + K[0, 2]
    v = pts_cam[..., 1] * K[1, 1] / z + K[1, 2]
    return np.stack([u, v], axis=-1), z


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _draw_tracks(rgb_bgr, patch_centers, track_disps, K, ext,
                 valid_mask=None, top_k=50,
                 color_start=(255, 0, 0), color_end=(0, 0, 255)):
    """Draw top_k most-dynamic track trajectories as colored polylines."""
    img = rgb_bgr.copy()
    H_img, W_img = img.shape[:2]
    N, T, _ = track_disps.shape

    mag = np.linalg.norm(track_disps[:, -1, :], axis=-1)
    top_idx = np.argsort(mag)[-top_k:]

    anc_uv, _ = _project(_world_to_cam(patch_centers, ext), K)

    for i in top_idx:
        if valid_mask is not None and not valid_mask[i]:
            continue
        ax, ay = int(anc_uv[i, 0]), int(anc_uv[i, 1])
        if not (0 <= ax < W_img and 0 <= ay < H_img):
            continue
        fut_w = patch_centers[i:i+1] + track_disps[i]   # [T, 3]
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
                int(color_start[0] * (1 - r) + color_end[0] * r),
                int(color_start[1] * (1 - r) + color_end[1] * r),
                int(color_start[2] * (1 - r) + color_end[2] * r),
            )
            cv2.line(img, prev, (px, py), color, 1, cv2.LINE_AA)
            prev = (px, py)
        cv2.circle(img, (ax, ay), 2, (0, 200, 0), -1)
    return img


def _draw_error_heatmap(rgb_bgr, patch_centers, pred, gt, K, ext, valid_mask=None):
    """Per-patch final-step error as green→red dots."""
    img = rgb_bgr.copy()
    H_img, W_img = img.shape[:2]
    err = np.linalg.norm(pred[:, -1, :] - gt[:, -1, :], axis=-1)
    if valid_mask is not None:
        err = np.where(valid_mask, err, np.nan)
    max_err = np.nanmax(err) if np.any(~np.isnan(err)) else 1e-6
    anc_uv, _ = _project(_world_to_cam(patch_centers, ext), K)
    for i in range(len(patch_centers)):
        if valid_mask is not None and not valid_mask[i]:
            continue
        ax, ay = int(anc_uv[i, 0]), int(anc_uv[i, 1])
        if not (0 <= ax < W_img and 0 <= ay < H_img):
            continue
        r = float(err[i]) / float(max_err + 1e-9)
        cv2.circle(img, (ax, ay), 3, (0, int(255*(1-r)), int(255*r)), -1)
    return img


def _label(img, text, color=(255, 255, 255)):
    cv2.putText(img, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
    return img


def _reencode_h264(path):
    import shutil, subprocess
    if shutil.which("ffmpeg") is None:
        return
    tmp = path + ".h264.mp4"
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path,
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", tmp])
    if r.returncode == 0:
        os.replace(tmp, path)


# ─────────────────────────────────────────────────────────────────────────────
# Attention heatmap
# ─────────────────────────────────────────────────────────────────────────────

def render_attn_heatmap(attn_weights_list, n_action=32, title="Self-Attention"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = []
    for li, w in enumerate(attn_weights_list):
        w_avg = w[0].mean(0).numpy()   # [N_q, N_q]
        n_track = w_avg.shape[0] - n_action
        aa = w_avg[:n_action, :n_action]
        at = w_avg[:n_action, n_action:]
        ta = w_avg[n_action:, :n_action]
        tt = w_avg[n_action:, n_action:]

        fig, axes = plt.subplots(1, 5, figsize=(25, 4))
        fig.suptitle(f"{title} — Layer {li+1}", fontsize=11)

        im = axes[0].imshow(w_avg, aspect="auto", cmap="hot", vmin=0)
        axes[0].axhline(n_action - 0.5, color="cyan", lw=1, ls="--")
        axes[0].axvline(n_action - 0.5, color="cyan", lw=1, ls="--")
        axes[0].set_title("Full attention matrix")
        axes[0].set_xlabel("Key"); axes[0].set_ylabel("Query")
        plt.colorbar(im, ax=axes[0], fraction=0.03)

        im2 = axes[1].imshow(aa, aspect="auto", cmap="hot", vmin=0)
        axes[1].set_title(f"Action→Action  mean={aa.mean():.4f}")
        axes[1].set_xlabel("Key action"); axes[1].set_ylabel("Query action")
        plt.colorbar(im2, ax=axes[1], fraction=0.03)

        im3 = axes[2].imshow(at, aspect="auto", cmap="hot", vmin=0)
        axes[2].set_title(f"Action→Track  mean={at.mean():.4f}  "
                          f"ratio={at.mean()/max(aa.mean(),1e-9):.1f}×")
        axes[2].set_xlabel("Key track"); axes[2].set_ylabel("Query action")
        plt.colorbar(im3, ax=axes[2], fraction=0.03)

        im4 = axes[3].imshow(ta, aspect="auto", cmap="hot", vmin=0)
        axes[3].set_title(f"Track→Action  mean={ta.mean():.4f}  "
                          f"ratio={ta.mean()/max(tt.mean(),1e-9):.1f}×")
        axes[3].set_xlabel("Key action"); axes[3].set_ylabel("Query track")
        plt.colorbar(im4, ax=axes[3], fraction=0.03)

        im5 = axes[4].imshow(tt, aspect="auto", cmap="hot", vmin=0)
        axes[4].set_title(f"Track→Track  mean={tt.mean():.4f}")
        axes[4].set_xlabel("Key track"); axes[4].set_ylabel("Query track")
        plt.colorbar(im5, ax=axes[4], fraction=0.03)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100); buf.seek(0)
        panels.append(cv2.imdecode(np.frombuffer(buf.read(), np.uint8), cv2.IMREAD_COLOR))
        plt.close(fig)
    return np.vstack(panels)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_frame(tracks_f, raw_cam, idx, t, ep_start):
    """Load single-frame obs + GT from HDF5. idx = ep_start + t."""
    K   = raw_cam["intrinsic_cv"][0]
    ext = raw_cam["extrinsic_cv"][0]
    rgb = cv2.imdecode(np.frombuffer(raw_cam["rgb"][t], np.uint8), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    pi3  = tracks_f["pi3_feature"][idx]                           # [1, N, D]
    apos = tracks_f["robot_state"][idx]                           # [16]
    pc   = tracks_f["image_patch_centre_3d_position"][idx]        # [N, 3]
    gt   = tracks_f["3d_track"][idx]                              # [N, H, 3]
    pmask = tracks_f["valid_point_mask"][idx].astype(bool)        # [N]
    tmask = tracks_f["track_valid_mask"][idx].astype(bool)        # [N, H]
    valid = pmask & tmask.all(axis=-1)

    obs = {
        "pi3_features": torch.from_numpy(pi3).float().unsqueeze(0),   # [1, 1, N, D]
        "agent_pos":    torch.from_numpy(apos).float().unsqueeze(0),
        "patch_centers": torch.from_numpy(pc).float().unsqueeze(0).unsqueeze(0),
    }
    return obs, rgb, K, ext, gt, valid, pc


def _predict_track(policy, obs_device):
    """Single-pass track prediction (skips DDIM; valid for regression models)."""
    state_xyz, patch_centers = policy._extract_rope_coords(obs_device)
    nobs = policy.normalizer.normalize(obs_device)
    memory, mem_pos, mem_coords, mem_mask = policy.encode_observations(
        nobs, state_xyz=state_xyz, patch_centers=patch_centers)
    # One forward pass at t=0
    t0 = torch.zeros(1, dtype=torch.long, device=memory.device)
    noise = torch.zeros(1, policy.horizon, policy.action_dim, device=memory.device)
    extra_kwargs = {}
    if getattr(policy, "joint_diffusion", False):
        n_tracks = policy.track_query_embed.shape[0] * policy.track_query_embed.shape[1]
        extra_kwargs["noised_tracks"] = torch.zeros(
            1, n_tracks, policy.horizon, 3, device=memory.device)
    fwd = policy.forward_diffusion(
        noised_actions=noise, timestep=t0,
        memory=memory, memory_pos=mem_pos,
        memory_coords=mem_coords, memory_rope_mask=mem_mask,
        state_xyz=state_xyz, patch_centers=patch_centers,
        **extra_kwargs,
    )
    _, track_raw = fwd  # [1, N, H, 3]
    # Unnormalize
    try:
        track = policy.normalizer["3d_track"].unnormalize(track_raw)[0].cpu().numpy()
    except Exception:
        track = track_raw[0].cpu().numpy()
    return track


# ─────────────────────────────────────────────────────────────────────────────
# Per-experiment run
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(ckpt_path, label, tracks_hdf5, raw_data_root,
                   task_name, task_config, episodes, attn_timesteps,
                   out_dir, device="cuda", camera="head_camera", fps=10, top_k=50):
    from deploy_policy import JAMBPolicyWrapper

    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*60}\nExperiment: {label}\nCheckpoint: {ckpt_path}")

    wrapper = JAMBPolicyWrapper(ckpt_path, device=device)
    policy  = wrapper.policy_model
    policy.eval()

    tracks_f = h5py.File(tracks_hdf5, "r", swmr=True)
    ep_ends  = tracks_f["episode_ends"][:]
    ep_starts = np.concatenate([[0], ep_ends[:-1]])
    raw_dir  = os.path.join(raw_data_root, task_name, task_config, "data")

    summary = {"label": label, "episodes": {}}

    for ep in episodes:
        ep_start = int(ep_starts[ep])
        ep_end   = int(ep_ends[ep])
        ep_len   = ep_end - ep_start
        print(f"\n  Episode {ep}  ({ep_len} frames)")

        raw_f  = h5py.File(os.path.join(raw_dir, f"episode{ep}.hdf5"), "r", swmr=True)
        raw_cam = raw_f["observation"][camera]

        # ── VIDEO: all frames ──────────────────────────────────────────
        # Read first frame to get image size
        probe_rgb = cv2.imdecode(np.frombuffer(raw_cam["rgb"][0], np.uint8), cv2.IMREAD_COLOR)
        H_img, W_img = probe_rgb.shape[:2]
        # Side-by-side: [GT | Pred | Error] → width × 3
        vid_path = os.path.join(out_dir, f"{label}_ep{ep}_track_video.mp4")
        vw = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (W_img * 3, H_img))

        frame_errors = []

        for t in range(ep_len):
            idx = ep_start + t
            obs, rgb, K, ext, gt_track, valid, pc = _load_frame(
                tracks_f, raw_cam, idx, t, ep_start)

            obs_dev = {k: v.to(device) for k, v in obs.items()}
            with torch.no_grad():
                track_pred = _predict_track(policy, obs_dev)

            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            img_gt   = _draw_tracks(rgb_bgr, pc, gt_track,    K, ext, valid, top_k,
                                    color_start=(0,200,0), color_end=(0,255,128))
            img_pred = _draw_tracks(rgb_bgr, pc, track_pred,  K, ext, valid, top_k,
                                    color_start=(255,0,0), color_end=(0,0,255))
            img_err  = _draw_error_heatmap(rgb_bgr, pc, track_pred, gt_track, K, ext, valid)

            # Per-frame mean error
            if valid.any():
                fe = np.linalg.norm(
                    track_pred[valid] - gt_track[valid], axis=-1).mean() * 1000
                frame_errors.append(fe)
            else:
                frame_errors.append(0.0)

            _label(img_gt,   f"GT Track  ep{ep} t={t}")
            _label(img_pred, f"Pred Track  {label}  ep{ep} t={t}")
            _label(img_err,  f"Error map  mean={frame_errors[-1]:.1f}mm")
            vw.write(np.hstack([img_gt, img_pred, img_err]))

            if t % 50 == 0:
                print(f"    frame {t}/{ep_len}  err={frame_errors[-1]:.1f}mm")

        vw.release()
        _reencode_h264(vid_path)
        print(f"  [video] {vid_path}")

        # ── STATIC images + ATTENTION at selected timesteps ────────────
        ep_stats = {"frame_errors_mm": frame_errors, "attn_timesteps": {}}

        for t in attn_timesteps:
            if t >= ep_len:
                continue
            idx = ep_start + t
            obs, rgb, K, ext, gt_track, valid, pc = _load_frame(
                tracks_f, raw_cam, idx, t, ep_start)
            obs_dev = {k: v.to(device) for k, v in obs.items()}

            patches = _patch_self_attn_for_weights(policy)
            with torch.no_grad():
                track_pred = _predict_track(policy, obs_dev)
            attn = _collect_attn_weights(policy)
            _unpatch_self_attn(patches)

            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            img_gt   = _draw_tracks(rgb_bgr, pc, gt_track,   K, ext, valid, top_k,
                                    color_start=(0,200,0), color_end=(0,255,128))
            img_pred = _draw_tracks(rgb_bgr, pc, track_pred, K, ext, valid, top_k,
                                    color_start=(255,0,0), color_end=(0,0,255))
            img_err  = _draw_error_heatmap(rgb_bgr, pc, track_pred, gt_track, K, ext, valid)

            mean_err = frame_errors[t] if t < len(frame_errors) else 0.0
            _label(img_gt,   f"GT  ep{ep} t={t}")
            _label(img_pred, f"Pred {label}  ep{ep} t={t}")
            _label(img_err,  f"Error {mean_err:.1f}mm")

            static_path = os.path.join(out_dir, f"{label}_ep{ep}_t{t}_static.png")
            cv2.imwrite(static_path, np.hstack([img_gt, img_pred, img_err]))
            print(f"  [static] {static_path}")

            if attn:
                n_action = 2 * policy.horizon
                attn_img = render_attn_heatmap(attn, n_action=n_action,
                                               title=f"{label} | ep{ep} t={t}")
                attn_path = os.path.join(out_dir, f"{label}_ep{ep}_t{t}_attn.png")
                cv2.imwrite(attn_path, attn_img)

                attn_stats = []
                for li, w in enumerate(attn):
                    wa = w[0].mean(0).numpy()
                    aa = wa[:n_action, :n_action].mean()
                    at = wa[:n_action, n_action:].mean()
                    ta = wa[n_action:, :n_action].mean()
                    tt = wa[n_action:, n_action:].mean()
                    attn_stats.append({
                        "layer": li + 1,
                        "action_to_action": float(aa),
                        "action_to_track":  float(at),
                        "track_to_action":  float(ta),
                        "track_to_track":   float(tt),
                        "ratio_at_aa": float(at / max(aa, 1e-9)),
                        "ratio_ta_tt": float(ta / max(tt, 1e-9)),
                    })
                ep_stats["attn_timesteps"][t] = {
                    "mean_err_mm": mean_err,
                    "attn": attn_stats,
                }

        raw_f.close()
        summary["episodes"][ep] = ep_stats

    tracks_f.close()

    json_path = os.path.join(out_dir, f"{label}_summary.json")
    with open(json_path, "w") as jf:
        json.dump(summary, jf, indent=2)
    print(f"  [summary] {json_path}")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Error curve + comparison chart
# ─────────────────────────────────────────────────────────────────────────────

def render_error_curves(summaries, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["steelblue", "tomato", "mediumseagreen", "orange", "purple"]
    fig, axes = plt.subplots(1, max(1, len(summaries[0]["episodes"])),
                             figsize=(7 * len(summaries[0]["episodes"]), 4), squeeze=False)

    for ep_idx, ep in enumerate(sorted(summaries[0]["episodes"].keys())):
        ax = axes[0][ep_idx]
        for ci, s in enumerate(summaries):
            errs = s["episodes"].get(ep, {}).get("frame_errors_mm", [])
            if errs:
                ax.plot(errs, label=s["label"], color=colors[ci % len(colors)], lw=1.2)
        ax.set_title(f"Episode {ep}")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Mean track error (mm)")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle("Per-frame track prediction error across experiments", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[error curves] {out_path}")


def render_attn_comparison(summaries, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [s["label"] for s in summaries]
    colors = ["steelblue", "tomato", "mediumseagreen", "orange"]

    def mean_layer1(summary, key):
        vals = []
        for ep_data in summary["episodes"].values():
            for t_data in ep_data.get("attn_timesteps", {}).values():
                if t_data.get("attn"):
                    vals.append(t_data["attn"][0][key])
        return np.mean(vals) if vals else 0.0

    aa    = [mean_layer1(s, "action_to_action") for s in summaries]
    at    = [mean_layer1(s, "action_to_track")  for s in summaries]
    ta    = [mean_layer1(s, "track_to_action")  for s in summaries]
    tt    = [mean_layer1(s, "track_to_track")   for s in summaries]
    ratio_at = [mean_layer1(s, "ratio_at_aa")   for s in summaries]
    ratio_ta = [mean_layer1(s, "ratio_ta_tt")   for s in summaries]

    def mean_err(summary):
        vals = []
        for ep_data in summary["episodes"].values():
            vals.extend(ep_data.get("frame_errors_mm", []))
        return np.mean(vals) if vals else 0.0

    errs = [mean_err(s) for s in summaries]

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    x = np.arange(len(labels))
    w = 0.35

    # Row 1: action-side attention
    axes[0, 0].bar(x - w/2, aa, w, label="Action→Action", color="steelblue")
    axes[0, 0].bar(x + w/2, at, w, label="Action→Track",  color="tomato")
    axes[0, 0].set_xticks(x); axes[0, 0].set_xticklabels(labels, rotation=15, ha="right")
    axes[0, 0].set_ylabel("Mean attention weight (layer 1)")
    axes[0, 0].set_title("Action queries: A→A vs A→Track")
    axes[0, 0].legend()

    axes[0, 1].bar(x, ratio_at, color=[colors[i % len(colors)] for i in range(len(labels))])
    axes[0, 1].set_xticks(x); axes[0, 1].set_xticklabels(labels, rotation=15, ha="right")
    axes[0, 1].set_ylabel("A→Track / A→A ratio")
    axes[0, 1].set_title("Action dilution ratio (A→Track / A→A)")

    axes[0, 2].bar(x, errs, color=[colors[i % len(colors)] for i in range(len(labels))])
    axes[0, 2].set_xticks(x); axes[0, 2].set_xticklabels(labels, rotation=15, ha="right")
    axes[0, 2].set_ylabel("Mean track error (mm)")
    axes[0, 2].set_title("Track prediction error")

    # Row 2: track-side attention
    axes[1, 0].bar(x - w/2, tt, w, label="Track→Track",  color="mediumseagreen")
    axes[1, 0].bar(x + w/2, ta, w, label="Track→Action", color="orange")
    axes[1, 0].set_xticks(x); axes[1, 0].set_xticklabels(labels, rotation=15, ha="right")
    axes[1, 0].set_ylabel("Mean attention weight (layer 1)")
    axes[1, 0].set_title("Track queries: T→T vs T→Action")
    axes[1, 0].legend()

    axes[1, 1].bar(x, ratio_ta, color=[colors[i % len(colors)] for i in range(len(labels))])
    axes[1, 1].set_xticks(x); axes[1, 1].set_xticklabels(labels, rotation=15, ha="right")
    axes[1, 1].set_ylabel("T→Action / T→T ratio")
    axes[1, 1].set_title("Track dilution ratio (T→Action / T→T)")

    # Row 2 col 3: all four quadrants side by side
    w4 = 0.2
    axes[1, 2].bar(x - 1.5*w4, aa, w4, label="A→A",  color="steelblue")
    axes[1, 2].bar(x - 0.5*w4, at, w4, label="A→T",  color="tomato")
    axes[1, 2].bar(x + 0.5*w4, ta, w4, label="T→A",  color="orange")
    axes[1, 2].bar(x + 1.5*w4, tt, w4, label="T→T",  color="mediumseagreen")
    axes[1, 2].set_xticks(x); axes[1, 2].set_xticklabels(labels, rotation=15, ha="right")
    axes[1, 2].set_ylabel("Mean attention weight (layer 1)")
    axes[1, 2].set_title("All four attention quadrants")
    axes[1, 2].legend(fontsize=8)

    plt.suptitle("Experiment comparison", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[attn comparison] {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",  help="Single checkpoint path")
    ap.add_argument("--label", default="exp")
    ap.add_argument("--ckpts",  nargs="+", help="Multiple checkpoints")
    ap.add_argument("--labels", nargs="+")
    ap.add_argument("--tracks_file",   required=True)
    ap.add_argument("--raw_data_root", required=True)
    ap.add_argument("--task_name",     required=True)
    ap.add_argument("--task_config",   required=True)
    ap.add_argument("--episodes",       type=int, nargs="+", default=[0])
    ap.add_argument("--attn_timesteps", type=int, nargs="+", default=[40, 80, 120],
                    help="Frames at which to also save static image + attention heatmap")
    ap.add_argument("--camera", default="head_camera")
    ap.add_argument("--out_dir", default="data/vis/track_analysis")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fps",    type=int, default=10)
    ap.add_argument("--top_k", type=int, default=50,
                    help="Number of most-dynamic patches to draw")
    args = ap.parse_args()

    ckpts  = args.ckpts  if args.ckpts  else [args.ckpt]
    labels = args.labels if args.labels else [args.label]
    assert len(ckpts) == len(labels)

    common = dict(
        tracks_hdf5=args.tracks_file,
        raw_data_root=args.raw_data_root,
        task_name=args.task_name,
        task_config=args.task_config,
        episodes=args.episodes,
        attn_timesteps=args.attn_timesteps,
        camera=args.camera,
        device=args.device,
        fps=args.fps,
        top_k=args.top_k,
    )

    all_summaries = []
    for ckpt, label in zip(ckpts, labels):
        s = run_experiment(ckpt, label,
                           out_dir=os.path.join(args.out_dir, label),
                           **common)
        all_summaries.append(s)

    if len(all_summaries) > 1:
        render_error_curves(all_summaries,
                            os.path.join(args.out_dir, "error_curves.png"))
        render_attn_comparison(all_summaries,
                               os.path.join(args.out_dir, "attn_comparison.png"))


if __name__ == "__main__":
    main()
