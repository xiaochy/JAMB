# JAMB: Joint Action–Motion Diffusion for Bimanual Manipulation

[Chuyang Xiao](https://xiaochy.github.io/)<sup>1,\*</sup>, [Peilin Meng](https://mengpeilin.github.io/)<sup>2,\*</sup>, [David Held](https://davheld.github.io/)<sup>1,†</sup>

<sup>1</sup>Robotics Institute, Carnegie Mellon University &nbsp;&nbsp;
<sup>2</sup>University of Michigan &nbsp;&nbsp;
<sup>\*</sup>Equal contribution &nbsp;&nbsp;
<sup>†</sup>Corresponding author

[**Project page**](https://jam-bimanual.github.io/) &nbsp;|&nbsp;
[**arXiv:2609.25322**](https://arxiv.org/abs/2609.25322)

A bimanual manipulation policy that denoises the action chunk and a 3D point-track
auxiliary target **in a single diffusion process**, evaluated on
[RoboTwin](https://robotwin-platform.github.io/). Built on the
[GAP](https://github.com/Chongyang-99/GAP) codebase.

## Key ideas

- **Joint diffusion, not an auxiliary head.** The 16-step, world-frame per-patch 3D
  displacement track is a diffusion variable denoised alongside the actions through the
  same DDIM schedule (`joint_diffusion=true`) — not a one-shot regression head bolted
  onto a frozen action decoder.
- **Per-patch concat fusion, not cross-attention.** Each DINOv2 patch token is
  concatenated with its track query and projected back down (`track_fusion_proj`), so
  scene and track are fused *before* the decoder; action, track, state and
  diffusion-timestep tokens then share **one** self-attention stack
  (`bridge_mode=concat_selfattn`), with no cross-attention pass between streams.
- **4D RoPE on action tokens too.** World-frame (x, y, z) + time rotary embeddings cover
  the vision/state tokens (from depth-derived patch centers) *and* the action tokens, via
  a noise-level–blended interpolation between the known end-effector pose and the pose
  implied by the currently-denoised action — which avoids the train/inference
  exposure-bias mismatch a naive application would create.
- **Movement-weighted track loss.** Patches with less than ~1 cm of predicted
  displacement (static background) are down-weighted, so the objective is not dominated
  by the typically large majority of non-moving patches.

Each choice is validated as an ablation on RoboTwin's 8 bimanual tasks (clean and
hard/domain-randomized): dropping 4D RoPE, falling back to cross-attention, or replacing
joint diffusion with one-shot track regression each lowers success rate.

## Repository layout

```text
.
├── JAMB/                               # the policy
│   ├── deploy_policy.py / .yml         # wrapper + default config used by RoboTwin eval
│   ├── train.sh / eval.sh              # entry points
│   ├── jamb_policy/                    # policy, dataset, diffusion modules
│   │   └── config/JAMB.yaml            # main training config (see Configuration)
│   ├── scripts/
│   │   ├── train.py                    # Hydra training script
│   │   ├── eval_policy.py              # local copy of RoboTwin's eval runner
│   │   ├── add_dinov2_to_tracks.py     # adds dino_features to a tracks HDF5 lacking them
│   │   ├── convert_tracks_to_zarr.py   # optional flat HDF5 -> zarr (either is readable)
│   │   ├── visualize_eval_tracks.py    # 2D track overlay on rollout videos
│   │   ├── visualize_track_3d.py       # 3D track + point cloud video (matplotlib)
│   │   └── export_trackvis_binary.py   # packs a trackvis log for webviz/track3d/
│   ├── webviz/track3d/                 # interactive 3D track / point-cloud viewer
│   ├── thirdparty/dinov2/              # DINOv2 inference code (visual backbone)
│   └── pretrained/download_weights.sh  # downloads external pretrained weights
├── RoboTwin/                           # simulator, task suite, eval harness
└── scripts/generate_gt_tracks.py       # builds the training data (see Data preparation);
                                        # at the root since it imports from both siblings
```

`JAMB/` and `RoboTwin/` are plain snapshots of their source repos' tracked files, not git
submodules — `git clone` alone is enough, there is no `git submodule update` step. Large
binaries (pretrained weights, datasets, checkpoints) are excluded by design, matching
each component's own `.gitignore`; download them per the steps below.

## Setup

JAMB runs inside a working RoboTwin environment.

1. **Install RoboTwin** per `RoboTwin/README.md` (environment, assets, task/data
   generation). Confirm a standard policy eval runs before continuing — that separates
   simulator problems from anything method-specific.
2. **Install the policy** into that same environment
   (`pip install -r JAMB/requirements-train.txt`), then point it at this checkout's
   RoboTwin:

   ```bash
   export ROBOTWIN_ROOT=/path/to/JAMB-repo/RoboTwin
   ```

   `ROBOTWIN_ROOT` must be a complete RoboTwin root and contain `script/eval_policy.py`.
3. **Download pretrained weights.** The visual backbone is DINOv2 (`dinov2_vitl14_reg`):

   ```bash
   bash JAMB/pretrained/download_weights.sh
   ```

   This writes `JAMB/pretrained/dinov2_vitl14_reg4_pretrain.pth`, used by
   `JAMB.yaml`'s `policy.dino_weights_path`. Set `DINOV2_WEIGHTS_URL=...` if the default
   fbaipublicfiles.com URL is unreachable.

Simulation experiments use RoboTwin's AgileX bimanual embodiment (ALOHA-AgileX).

## Data preparation

First collect demonstrations in RoboTwin itself (see `RoboTwin/README.md`), so that
`${ROBOTWIN_ROOT}/data/<task_name>/<task_config>/data/episode*.hdf5` exists.

Then build the flat tracks HDF5 this method trains on. `scripts/generate_gt_tracks.py`
replays each episode's cached joint path in the simulator and reconstructs every visible
surface point's exact future 3D position by rigidly transforming it through its owning
link's recorded pose — **simulator ground truth, not an estimate**: no tracking, no
drift. Run from the repo root, in an environment that can launch SAPIEN (same as demo
collection):

```bash
python scripts/generate_gt_tracks.py \
    --task_name place_dual_shoes \
    --task_config demo_clean \
    --num_episodes 100 \
    --output_file JAMB/data/tracks/place_dual_shoes-demo_clean-100-tracks_flat.hdf5 \
    --raw_data_root RoboTwin/data \
    --extract_dino_features
```

`--extract_dino_features` writes the DINOv2 patch features required by
`use_dino_features=true` into the same HDF5, next to `3d_track` / `track_valid_mask` (the
keys `JAMB.yaml`'s `track_key`/`mask_key` expect), so no separate feature-extraction pass
is needed. `TrackDataset` and `train.sh` load the resulting file directly. The script's
docstring documents the full HDF5 schema and the optional extras
(`--extract_pi3_features`, `--extract_vae_features`, surface normals).

## Training

```bash
cd JAMB
bash train.sh place_dual_shoes demo_clean 100 0 0 32 300 100
#              <task_name> <task_config> <expert_data_num> <seed> <gpu_id> <batch_size> <num_epochs> <checkpoint_every>
```

Checkpoints land in `JAMB/checkpoints/<task_name>_<task_config>_<expert_data_num>/<epoch>.ckpt`.
Logging defaults to offline Weights & Biases (`export WANDB_MODE=online` to change it).

## Evaluation on RoboTwin

```bash
cd JAMB
export ROBOTWIN_ROOT=/path/to/JAMB-repo/RoboTwin
bash eval.sh place_dual_shoes demo_clean demo_clean 100 100 0 "0" 100
```

| Argument | Default | Description |
|---|---|---|
| `task_name` | `place_dual_shoes` | RoboTwin task name |
| `task_config` | `demo_clean` | Task configuration — use a `_depth` variant (e.g. `demo_clean_depth`) for hard/domain-randomized eval, since 4D RoPE needs depth-derived patch centers |
| `ckpt_setting` | `demo_clean` | Setting the checkpoint was *trained* under, used to auto-locate it (independent of `task_config`) |
| `expert_data_num` | `100` | Number of expert demonstrations used during training |
| `checkpoint_num` | `300` | Epoch of the checkpoint to evaluate |
| `gpu_id` | `0` | CUDA device index |
| `seeds` | `"0"` | Space-separated seed list, e.g. `"0 1 2"` |
| `test_num` | `100` | Evaluation trajectories per seed |

To evaluate a specific file instead of auto-locating one:
`export CKPT_PATH=/path/to/checkpoint.ckpt`.

Results go to
`JAMB/results/<task_name>/JAMB/<task_config>/<ckpt_setting>/seed_<seed>/<checkpoint_num>/_result.txt`
(override with `export RESULTS_ROOT=/path/to/results`).

**Track visualization:** set `TRACK_VIS_LOG_DIR=/some/dir` before `eval.sh` to log each
action chunk's predicted track, then feed it to `scripts/visualize_eval_tracks.py` (2D
video overlay), `scripts/visualize_track_3d.py` (3D rollout video), or
`scripts/export_trackvis_binary.py` (interactive viewer, below).

## Configuration

Main training config: `JAMB/jamb_policy/config/JAMB.yaml`. The options specific to this
method's architecture:

| Key | Value here | Meaning |
|---|---|---|
| `aux_task` | `track` | Predict a 3D displacement track (vs. `pmp`, GAP's future point-map target, or `none`) |
| `joint_diffusion` | `true` | Track is denoised jointly with actions (vs. a one-shot regression head) |
| `policy.bridge_mode` | `concat_selfattn` | Per-patch scene+track concat fusion, one joint self-attention decoder |
| `policy.use_rope4d` | `true` | 4D RoPE on vision/state tokens *and* action tokens |
| `policy.rope_spatial_scale` | `0.01` | xyz normalization for RoPE (validated by a frequency ablation) |
| `policy.movement_loss_tau` | `0.01` | Displacement threshold (m) below which a patch counts as static in the track loss |
| `track_key` / `mask_key` | `gt_3d_track` / `track_valid_mask_gt` | Keys for the 3D track target / validity mask |

## 3D track viewer

`JAMB/webviz/track3d/` is a standalone, no-build static page (three.js) for inspecting a
trained policy's predicted 3D tracks against a colored point cloud of the scene — drag to
orbit, scroll to zoom, scrub an episode frame-by-frame, laid out as a gallery so several
episodes sit side by side. `JAMB/webviz/track3d/README.md` covers how it reads its data
and how to add episodes; `JAMB/scripts/export_trackvis_binary.py` packs a
`TRACK_VIS_LOG_DIR` log into the format it consumes.

## Notes on RoboTwin integration

RoboTwin is treated as an external environment: preprocessing reads official RoboTwin
HDF5 demonstrations; evaluation changes into `ROBOTWIN_ROOT` so task configs and
simulator imports resolve normally; `JAMB/scripts/eval_policy.py` is a local copy of
RoboTwin's runner that supports repository-local result paths. **No files inside the
RoboTwin repository need to be edited.**

## Acknowledgements

Built on the [GAP](https://github.com/Chongyang-99/GAP) (Action-Geometry Prediction,
CVPR 2026) codebase and its DINOv3/Pi3-based diffusion policy for RoboTwin; our
modifications replace its point-map-prediction auxiliary objective and cross-attention
fusion with the joint track-diffusion / per-patch concat-fusion / action-token 4D RoPE
design above. Thanks also to the authors of [Pi3](https://github.com/yyfz/Pi3),
[RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin), and Xu et al.'s
[Diffusion-Based Imaginative Coordination](https://github.com/return-sleep/Diffusion_based_imaginative_Coordination).

## Citation

```bibtex
@misc{xiao2026jamb,
  title         = {JAMB: Joint Action--Motion Diffusion for Bimanual Manipulation},
  author        = {Xiao, Chuyang and Meng, Peilin and Held, David},
  year          = {2026},
  eprint        = {2609.25322},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO}
}
```
