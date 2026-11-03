# JAMB: Joint Action–Motion Diffusion for Bimanual Manipulation

[Chuyang Xiao](https://xiaochy.github.io/)<sup>1,\*</sup>, [Peilin Meng](https://mengpeilin.github.io/)<sup>2,\*</sup>, [David Held](https://davheld.github.io/)<sup>1,†</sup>

<sup>1</sup>Robotics Institute, Carnegie Mellon University &nbsp;&nbsp;
<sup>2</sup>University of Michigan &nbsp;&nbsp;
<sup>\*</sup>Equal contribution &nbsp;&nbsp;
<sup>†</sup>Corresponding author

Accepted at the **CoRL 2026 G2A Workshop**.

**[🌐 Project Website](https://jam-bimanual.github.io/)** • **[📄 arXiv](https://arxiv.org/abs/2609.25322)** • **[📋 PDF](https://arxiv.org/pdf/2609.25322)**

A bimanual manipulation policy that denoises the action chunk and a 3D point-track
auxiliary target **in a single diffusion process**, evaluated on
[RoboTwin](https://robotwin-platform.github.io/). Built on the
[GAP](https://github.com/Chongyang-99/GAP) codebase.

## Key ideas

- **Joint diffusion, not an auxiliary head** — the per-patch 3D displacement track (16
  steps, world frame) is denoised alongside the actions in the same DDIM schedule.
- **Per-patch concat fusion, not cross-attention** — each DINOv2 patch token is
  concatenated with its track query and projected down, then action, track and state
  tokens share one self-attention stack.
- **4D RoPE on the action tokens too**, not just perception — via a noise-blended
  interpolation between the known end-effector pose and the denoised action's, which
  avoids a train/inference exposure-bias mismatch.
- **Movement-weighted track loss** — patches moving less than ~1 cm are down-weighted, so
  static background does not dominate the objective.

Each is validated as an ablation on RoboTwin's 8 bimanual tasks, clean and
hard/domain-randomized.

## Repository layout

```text
JAMB/                          # the policy: train.sh, eval.sh, jamb_policy/, scripts/
RoboTwin/                      # simulator, task suite, eval harness
scripts/generate_gt_tracks.py  # builds the training data (imports from both siblings)
```

`JAMB/` and `RoboTwin/` are plain snapshots, not submodules — `git clone` is enough.
Large binaries (weights, datasets, checkpoints) are excluded; download them in setup.

`main` carries only data collection, training and evaluation. The visualization and
diagnostic tooling lives on the [`full`](https://github.com/xiaochy/JAMB/tree/full)
branch.

## Setup

JAMB runs inside a working RoboTwin environment, and needs no edits to RoboTwin itself.

```bash
# 1. install RoboTwin per RoboTwin/README.md, then this policy on top of it
pip install -r JAMB/requirements-train.txt
export ROBOTWIN_ROOT=/path/to/JAMB-repo/RoboTwin   # must contain script/eval_policy.py

# 2. download the DINOv2 backbone (dinov2_vitl14_reg)
bash JAMB/pretrained/download_weights.sh
```

Verifying a standard RoboTwin policy eval first is worth the time — it separates
simulator problems from method-specific ones. Experiments use RoboTwin's AgileX bimanual
embodiment (ALOHA-AgileX).

## Data preparation

Collect demonstrations in RoboTwin first (see `RoboTwin/README.md`), so that
`${ROBOTWIN_ROOT}/data/<task_name>/<task_config>/data/episode*.hdf5` exists. Then build
the tracks HDF5 this method trains on, from the repo root in an environment that can
launch SAPIEN:

```bash
python scripts/generate_gt_tracks.py \
    --task_name place_dual_shoes --task_config demo_clean --num_episodes 100 \
    --output_file JAMB/data/tracks/place_dual_shoes-demo_clean-100-tracks_flat.hdf5 \
    --raw_data_root RoboTwin/data --extract_dino_features
```

The tracks are **simulator ground truth, not an estimate**: each episode's joint path is
replayed and every visible surface point is transformed through its owning link's
recorded pose, so there is no tracking and no drift. `--extract_dino_features` writes the
DINOv2 patch features into the same file, so no separate feature pass is needed. The
script's docstring has the full HDF5 schema and the optional extras.

## Training

```bash
cd JAMB
bash train.sh place_dual_shoes demo_clean 100 0 0 32 300 100
#    <task> <config> <expert_data_num> <seed> <gpu> <batch_size> <epochs> <ckpt_every>
```

Checkpoints land in `JAMB/checkpoints/<task>_<config>_<expert_data_num>/<epoch>.ckpt`;
logging defaults to offline wandb (`WANDB_MODE=online` to change it).

## Evaluation

```bash
cd JAMB && export ROBOTWIN_ROOT=/path/to/JAMB-repo/RoboTwin
bash eval.sh place_dual_shoes demo_clean demo_clean 100 100 0 "0" 100
#    <task> <config> <ckpt_setting> <expert_data_num> <ckpt_epoch> <gpu> <seeds> <test_num>
```

- `config` — use a `_depth` variant (e.g. `demo_clean_depth`) for the hard,
  domain-randomized setting; 4D RoPE needs depth-derived patch centers.
- `ckpt_setting` — the setting the checkpoint was *trained* under, used to locate it;
  independent of `config`. `CKPT_PATH=...` overrides the lookup.
- `seeds` — space-separated, e.g. `"0 1 2"`.

Results go to `JAMB/results/<task>/JAMB/<config>/<ckpt_setting>/seed_<seed>/<epoch>/_result.txt`
(`RESULTS_ROOT` to relocate). `TRACK_VIS_LOG_DIR=/some/dir` additionally dumps each
action chunk's predicted track; the tooling that renders those logs is on the
[`full`](https://github.com/xiaochy/JAMB/tree/full) branch.

## Configuration

`JAMB/jamb_policy/config/JAMB.yaml`, with the keys specific to this method:

| Key | Value | Meaning |
|---|---|---|
| `aux_task` | `track` | 3D displacement track (vs. `pmp`, GAP's point-map target, or `none`) |
| `joint_diffusion` | `true` | Track denoised jointly with actions, not one-shot regressed |
| `policy.bridge_mode` | `concat_selfattn` | Per-patch concat fusion, one self-attention decoder |
| `policy.use_rope4d` | `true` | 4D RoPE on vision/state *and* action tokens |
| `policy.rope_spatial_scale` | `0.01` | xyz normalization for RoPE |
| `policy.movement_loss_tau` | `0.01` | Displacement (m) below which a patch counts as static |
| `track_key` / `mask_key` | `gt_3d_track` / `track_valid_mask_gt` | Track target / validity mask |

## Acknowledgements

Built on the [GAP](https://github.com/Chongyang-99/GAP) (Action-Geometry Prediction, CVPR
2026) codebase, replacing its point-map auxiliary objective and cross-attention fusion
with the design above. Thanks also to [Pi3](https://github.com/yyfz/Pi3),
[RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) and
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
