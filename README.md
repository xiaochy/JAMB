# JAMB: Joint Action–Motion Diffusion for Bimanual Manipulation

Reference implementation of our method, evaluated on [RoboTwin](https://robotwin-platform.github.io/). This repo bundles both pieces needed to reproduce our results in one place:

- `JAMB/` — our policy implementation
- `RoboTwin/` — the simulator, task suite, and eval harness we evaluate against
- `scripts/generate_gt_tracks.py` — builds this method's training data (simulator ground-truth 3D tracks) from RoboTwin demonstrations; lives here at the repo root, not inside `JAMB/`, since it imports from both `RoboTwin/` and `JAMB/` as sibling directories

`JAMB/` and `RoboTwin/` are each a plain snapshot of their source repo's tracked files at the time this was assembled — not a git submodule, so `git clone` alone is enough; there's no separate `git submodule update` step. See `JAMB/README.md` for the full method description.

## What the method does

Actions and a 3D point-track auxiliary target are predicted *jointly*, in one diffusion process: DINOv2 visual patch tokens are fused per-patch with track query tokens (concatenation + projection, not cross-attention) and denoised together with the action chunk in one self-attention stack; 4D RoPE (world-frame xyz + time) is applied not just to perception but to the action tokens themselves. `JAMB/README.md` has the full breakdown of each design choice and the ablations that motivate them.

## Setup

The method runs on top of a working RoboTwin install, so set up RoboTwin first.

1. **Install RoboTwin.** Follow `RoboTwin/README.md` (environment setup, assets, and the task/data generation pipeline). Confirm a standard policy eval runs before moving on — that isolates simulator problems from anything method-specific.

2. **Install the policy inside that environment.** Follow `JAMB/README.md` for the conda/pip setup, then point it at your RoboTwin checkout:

   ```bash
   cd JAMB
   export ROBOTWIN_ROOT=/path/to/JAMB-repo/RoboTwin
   ```

3. **Download pretrained weights** (DINOv2 — not tracked by git):

   ```bash
   bash pretrained/download_weights.sh
   ```

## Quick start

```bash
# 1. Collect RoboTwin demonstrations for a task (see RoboTwin/README.md), then
#    build this method's training data from them -- run from this repo's root:
python scripts/generate_gt_tracks.py \
    --task_name place_dual_shoes --task_config demo_clean --num_episodes 100 \
    --output_file JAMB/data/tracks/place_dual_shoes-demo_clean-100-tracks_flat.hdf5 \
    --raw_data_root RoboTwin/data --extract_dino_features

cd JAMB
export ROBOTWIN_ROOT=/path/to/JAMB-repo/RoboTwin

# 2. Train
bash train.sh place_dual_shoes demo_clean 100 0 0 32 300 100

# 3. Evaluate the trained checkpoint back in RoboTwin
bash eval.sh place_dual_shoes demo_clean demo_clean 100 100 0 "0" 100
```

`JAMB/README.md` documents every argument, the config options specific to this method's architecture, and how to run the hard (domain-randomized) RoboTwin setting.

## 3D track viewer

`JAMB/webviz/track3d/` is a standalone, no-build static page (three.js) for interactively inspecting a trained policy's predicted 3D tracks against a colored point cloud of the scene — drag to orbit, scroll to zoom, scrub through an episode frame-by-frame, laid out as a gallery so several episodes can sit side by side. See `JAMB/webviz/track3d/README.md` for how it reads its data and how to add more episodes (`JAMB/scripts/export_trackvis_binary.py` packs a `TRACK_VIS_LOG_DIR` trackvis log into the format the page consumes).

Large binary assets (pretrained weights, datasets, trained checkpoints) are intentionally excluded, matching each component's own `.gitignore` — download them per the setup steps above rather than expecting them in this repo.
