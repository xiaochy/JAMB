# JAMB: Joint Action–Motion Diffusion for Bimanual Manipulation

This directory holds the policy implementation for our method. It's built on top of the [GAP](https://github.com/Chongyang-99/GAP) (Action-Geometry Prediction) codebase — see [Acknowledgements](#acknowledgements) — but the architecture and training objective described below are our own modification, not the original GAP paper's PMP (point-map prediction) approach.

> Authors and citation are not finalized yet and are intentionally left out of this README for now.

## What this method does

A bimanual manipulation policy that predicts actions and a 3D point-track auxiliary target *jointly*, in one diffusion process, instead of treating the track as a separate one-shot regression head or a passive visual feature:

- **Per-patch scene/track fusion, not cross-attention.** Each DINOv2 visual patch token is concatenated with its corresponding track query token and projected back down (`track_fusion_proj`), so scene and track information are fused *before* the decoder rather than the track queries cross-attending to a separate visual memory. Action, track, state, and diffusion-timestep tokens are then all denoised together in **one** self-attention stack (`bridge_mode=concat_selfattn`) — no separate cross-attention pass between streams.
- **Track is diffused jointly with actions**, not regressed in one shot. The predicted 3D displacement trajectory (16-step horizon, world frame, per visual patch) is a diffusion variable denoised alongside the action chunk through the same DDIM schedule (`joint_diffusion=true`), rather than a deterministic head bolted onto a frozen action decoder.
- **4D RoPE extends to the action tokens, not just perception.** World-frame (x, y, z) + timestep rotary position embeddings are applied to the visual/state tokens (from depth-derived patch centers, as usual) *and* to the action tokens themselves — via a noise-level–blended interpolation between the current known end-effector pose and the position implied by the currently-denoised action, which avoids a train/inference exposure-bias mismatch that a naive application of RoPE to action tokens would introduce.
- **Movement-weighted track loss.** Patches with less than ~1cm of predicted displacement (i.e. static background) are down-weighted in the track loss so the objective isn't dominated by the (typically large) majority of non-moving scene patches.

We validated each of these choices as an ablation against RoboTwin's 8-task bimanual benchmark (both the clean and hard/domain-randomized settings) — removing 4D RoPE, removing the concat fusion (falling back to cross-attention), or replacing joint diffusion with one-shot track regression each measurably hurt success rate relative to the full model.

## Repository Layout

```text
JAMB/
├── deploy_policy.py               # Policy wrapper used by RoboTwin evaluation
├── deploy_policy.yml               # Default evaluation config
├── train.sh                        # Training entry point
├── eval.sh                         # RoboTwin evaluation entry point
├── jamb_policy/                    # Policy, dataset, diffusion modules, configs
│   └── config/JAMB.yaml            # Main training config (see Configuration below)
├── scripts/
│   ├── add_dinov2_to_tracks.py     # Adds dino_features to a tracks HDF5 that doesn't have them yet
│   ├── convert_tracks_to_zarr.py   # Optional: flat HDF5 -> zarr (TrackDataset reads either directly)
│   ├── train.py                    # Hydra training script
│   ├── eval_policy.py              # Local copy of RoboTwin's eval runner
│   ├── visualize_eval_tracks.py    # 2D track overlay on eval rollout videos
│   ├── visualize_track_3d.py       # 3D track + point cloud rollout video (matplotlib)
│   └── export_trackvis_binary.py   # Packs a trackvis log for webviz/track3d/
├── webviz/track3d/                 # Interactive 3D track/point-cloud viewer (see its own README)
├── thirdparty/
│   └── dinov2/                     # DINOv2 inference code (this method's visual backbone)
└── pretrained/
    └── download_weights.sh         # Downloads external pretrained weights
```

## Prerequisites

This method runs inside a working RoboTwin environment. If you're using the top-level `JAMB` repo (this directory alongside `../RoboTwin`), that's already set up — see the top-level `README.md`. Otherwise, install RoboTwin following its official instructions and verify a standard policy evaluation runs before continuing here.

```bash
export ROBOTWIN_ROOT=/path/to/RoboTwin
```

`ROBOTWIN_ROOT` must point to the complete RoboTwin root directory and contain `script/eval_policy.py`.

## Robot Embodiment

Simulation experiments use RoboTwin's AgileX bimanual embodiment (ALOHA-AgileX).

## Pretrained Weights

This method's visual backbone is DINOv2 (`dinov2_vitl14_reg`):

```bash
bash pretrained/download_weights.sh
```

This downloads `pretrained/dinov2_vitl14_reg4_pretrain.pth`, used by `jamb_policy/config/JAMB.yaml`'s `policy.dino_weights_path`.

Useful overrides:

```bash
export DINOV2_WEIGHTS_URL=...   # if the default fbaipublicfiles.com URL is unreachable
```

## Data Preparation

First collect demonstrations in RoboTwin itself (see `RoboTwin/README.md`) so `${ROBOTWIN_ROOT}/data/<task_name>/<task_config>/data/episode*.hdf5` exists for the task.

Then build the flat tracks HDF5 this method trains on: `scripts/generate_gt_tracks.py` replays each episode's cached joint path in RoboTwin's simulator and reconstructs every visible surface point's exact future 3D position by rigidly transforming it through its owning link's recorded pose (no tracking, no drift — this is a simulator ground truth, not an estimate). Run from the repo root (it expects `RoboTwin/` and `JAMB/` as sibling directories, i.e. this checkout's own layout):

```bash
python scripts/generate_gt_tracks.py \
    --task_name place_dual_shoes \
    --task_config demo_clean \
    --num_episodes 100 \
    --output_file JAMB/data/tracks/place_dual_shoes-demo_clean-100-tracks_flat.hdf5 \
    --raw_data_root RoboTwin/data \
    --extract_dino_features
```

This must run in an environment that can launch RoboTwin's SAPIEN simulator (same as demo collection). `--extract_dino_features` adds the DINOv2 patch features this method's `use_dino_features=true` needs directly into the same HDF5, alongside `3d_track` / `track_valid_mask` (the keys `jamb_policy/config/JAMB.yaml`'s `track_key`/`mask_key` expect) — so no separate feature-extraction pass is needed. The resulting file is what `TrackDataset` (and `train.sh`, below) loads directly; see the script's own docstring for the full HDF5 schema and additional optional fields (`--extract_pi3_features`, `--extract_vae_features`, surface normals).

## Training

```bash
bash train.sh place_dual_shoes demo_clean 100 0 0 32 300 100
```

Arguments: `train.sh <task_name> <task_config> <expert_data_num> <seed> <gpu_id> <batch_size> <num_epochs> <checkpoint_every>`

Checkpoints are saved to `checkpoints/<task_name>_<task_config>_<expert_data_num>/<epoch>.ckpt`. Logging defaults to offline Weights & Biases (`export WANDB_MODE=online` to change it).

## Evaluation on RoboTwin

```bash
export ROBOTWIN_ROOT=/path/to/RoboTwin
bash eval.sh place_dual_shoes demo_clean demo_clean 100 100 0 "0" 100
```

Arguments: `eval.sh <task_name> <task_config> <ckpt_setting> <expert_data_num> <checkpoint_num> <gpu_id> <seeds> [test_num]`

| Argument | Default | Description |
|---|---|---|
| `task_name` | `place_dual_shoes` | RoboTwin task name |
| `task_config` | `demo_clean` | Task configuration (use a `_depth` variant, e.g. `demo_clean_depth`, for hard/domain-randomized eval — 4D RoPE needs depth-derived patch centers) |
| `ckpt_setting` | `demo_clean` | Setting the checkpoint was *trained* under (used to auto-locate it — independent of `task_config`) |
| `expert_data_num` | `100` | Number of expert demonstrations used during training |
| `checkpoint_num` | `300` | Epoch number of the checkpoint to evaluate |
| `gpu_id` | `0` | CUDA device index |
| `seeds` | `"0"` | Space-separated list of seeds, e.g. `"0 1 2"` |
| `test_num` | `100` | Number of evaluation trajectories per seed |

To evaluate a specific checkpoint instead of auto-locating one: `export CKPT_PATH=/path/to/checkpoint.ckpt`.

Results are written to `results/<task_name>/JAMB/<task_config>/<ckpt_setting>/seed_<seed>/<checkpoint_num>/_result.txt` (override with `export RESULTS_ROOT=/path/to/results`).

**Track-prediction visualization:** set `TRACK_VIS_LOG_DIR=/some/dir` before calling `eval.sh` to additionally log each action chunk's predicted track (used by `scripts/visualize_eval_tracks.py` for a 2D video overlay, `scripts/visualize_track_3d.py` for a 3D rollout video, or `scripts/export_trackvis_binary.py` to feed the interactive viewer in `webviz/track3d/`).

## Configuration

Main training config: `jamb_policy/config/JAMB.yaml`. Options specific to this method's architecture:

| Key | Value here | Meaning |
|---|---|---|
| `aux_task` | `track` | Predict a 3D displacement track (vs. `pmp`, the original GAP paper's future point-map feature target, or `none`) |
| `joint_diffusion` | `true` | Track is denoised jointly with actions (vs. a one-shot regression head) |
| `policy.bridge_mode` | `concat_selfattn` | Per-patch scene+track concat fusion, one joint self-attention decoder |
| `policy.use_rope4d` | `true` | 4D RoPE on vision/state tokens *and* action tokens |
| `policy.rope_spatial_scale` | `0.01` | xyz normalization for RoPE (validated via a frequency ablation) |
| `policy.movement_loss_tau` | `0.01` | Displacement threshold (m) below which a patch is treated as static in the track loss |
| `track_key` / `mask_key` | `gt_3d_track` / `track_valid_mask_gt` | zarr keys for the 3D track target / validity mask |

## Notes on RoboTwin Integration

This method treats RoboTwin as an external environment:

- preprocessing reads official RoboTwin HDF5 demonstrations;
- evaluation changes into `ROBOTWIN_ROOT` so RoboTwin task configs and simulator imports resolve normally;
- `scripts/eval_policy.py` in this repository is a local copy that supports repository-local result paths;
- no files inside the official RoboTwin repository need to be edited.

## Acknowledgements

This builds on the [GAP](https://github.com/Chongyang-99/GAP) (Action-Geometry Prediction, CVPR 2026) codebase and its release of a DINOv3/Pi3-based diffusion policy for RoboTwin; our modifications replace its point-map-prediction auxiliary objective and cross-attention fusion with the joint track-diffusion / per-patch concat-fusion / action-token 4D RoPE design described above. We also thank the authors of [Pi3](https://github.com/yyfz/Pi3), [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin), and Xu et al.'s [Diffusion-Based Imaginative Coordination](https://github.com/return-sleep/Diffusion_based_imaginative_Coordination) repository for their open-source codebases.
