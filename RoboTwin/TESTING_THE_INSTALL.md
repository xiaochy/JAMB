# How to verify a RoboTwin install actually works

Three checks, in increasing order of coverage. Run them in order — each one
isolates a different layer, so if a later check fails you already know the
earlier layers are fine.

## 1. Python imports (seconds)

Confirms the conda env's packages are wired up correctly — torch, CUDA libs,
pytorch3d, curobo, sapien, mplib all importable in the same process.

```bash
conda activate RoboTwin
python -c "
import torch; print('torch', torch.__version__, torch.cuda.is_available())
import pytorch3d; print('pytorch3d', pytorch3d.__version__)
import curobo; print('curobo OK')
import sapien, mplib; print('sapien', sapien.__version__, 'mplib OK')
"
```

If this fails, fix the import error before trying anything else — every
later check depends on these.

## 2. Renderer smoke test (~10s)

`script/test_render.py` only spins up a SAPIEN engine + ray-tracing renderer
(Vulkan/OptiX path) with no task, robot, or assets involved. It isolates
GPU/driver/Vulkan problems from task-logic problems.

```bash
python script/test_render.py
```

Expected output: `Render Well` in green. `Render Error` in red means a
GPU/Vulkan/display issue (check `vulkaninfo`, `nvidia-smi`, and that you're
not on a headless box without a virtual display if running with a window).

## 3. End-to-end task collection (1-5 min for one episode)

This is the real integration test: it exercises asset loading, the motion
planner (curobo/mplib), physics simulation, camera rendering, and the HDF5/
video data writer all together — i.e. everything `collect_data.sh` needs in
production.

Don't run a full `task_config/demo_randomized.yml` (default `episode_num: 50`)
just to check things work — it can take a long time. Instead make a throwaway
1-episode config:

```bash
cp task_config/demo_randomized.yml task_config/_smoketest.yml
# then edit _smoketest.yml: episode_num: 1, save_freq: 1
bash collect_data.sh beat_block_hammer _smoketest 0
```

What to look for in the output:
- `Render Well` — renderer initialized
- `simulate data episode 0 success! (seed = N)` — a valid seed was found and
  the bimanual motion plan completed (this is where curobo/mplib actually run)
- `🎬 Video is saved to ./data/beat_block_hammer/_smoketest/video/episode0.mp4`
  — data collection and writing succeeded

It's normal to see one or more `simulate data episode 0 fail! (seed = N)`
lines before a success — the collector searches random seeds until it finds
one where the planned motion succeeds; that's expected behavior, not a bug.

Confirm the output landed on disk:

```bash
ls data/beat_block_hammer/_smoketest/
# expect: data/episode0.hdf5, video/episode0.mp4, instructions/episode0.json,
#         _traj_data/episode0.pkl, seed.txt, scene_info.json
```

Watch `episode0.mp4` to visually confirm the dual-arm robot actually performs
the task (e.g. picks up and uses the hammer) rather than just running without
crashing — a script that "succeeds" but produces a robot flailing in place is
still a bug.

Clean up the throwaway config/data once satisfied:

```bash
rm task_config/_smoketest.yml
rm -rf data/beat_block_hammer/_smoketest
```

## If you only have time for one check

Run #3 with a single episode. It's the only one that actually proves the
embodiment assets, planner, simulator, and renderer all cooperate — #1 and #2
can each pass individually while #3 still fails (e.g. a working torch +
working renderer but a broken curobo planner config, or missing/corrupt
embodiment assets under `assets/embodiments/`).
