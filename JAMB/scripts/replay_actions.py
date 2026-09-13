"""
Replay recorded EEF-pose actions in the simulator to verify the collected
data: rebuild each episode's scene from its collection seed, execute the
recorded 16-D EEF actions via take_action(action_type='ee') (the same IK
path eval uses), and report

  1. tracking error: achieved EEF pose after action[t] vs recorded
     robot_state[t+1] (position / quaternion / gripper)
  2. end-of-episode task success
  3. a side-by-side video: live replay render vs the recorded RGB

Run from ROBOTWIN_ROOT (like eval.sh does):
    cd $ROBOTWIN_ROOT && python $GAP_ROOT/scripts/replay_actions.py \
        --task_name handover_block --task_config demo_clean_depth \
        --episodes 0 1 --action_stride 4 --out_dir /data/.../data/vis
"""
import argparse
import importlib
import os
import sys

import cv2
import h5py
import numpy as np
import yaml

sys.path.insert(0, os.getcwd())
from envs import CONFIGS_PATH  # noqa: E402

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def _reencode_h264(path):
    """cv2's mp4v output doesn't play in many viewers (VSCode preview,
    QuickTime); re-encode to H.264 in place when ffmpeg is available."""
    import shutil, subprocess
    if shutil.which("ffmpeg") is None:
        print(f"[warn] ffmpeg not found; {path} left as mp4v (may not play everywhere)")
        return
    tmp = path + ".h264.mp4"
    r = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", path,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", tmp])
    if r.returncode == 0:
        os.replace(tmp, path)



def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    env_class = getattr(envs_module, task_name)
    return env_class()


def get_embodiment_config(robot_file):
    with open(os.path.join(robot_file, "config.yml"), "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def build_args(task_name, task_config):
    """Mirror scripts/eval_policy.py's config assembly."""
    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    args["task_name"] = task_name
    args["task_config"] = task_config

    embodiment_type = args.get("embodiment")
    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def emb_file(t):
        return _embodiment_types[t]["file_path"]

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)
    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = emb_file(embodiment_type[0])
        args["right_robot_file"] = emb_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    else:
        args["left_robot_file"] = emb_file(embodiment_type[0])
        args["right_robot_file"] = emb_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    args["eval_video_log"] = False
    args["render_freq"] = 0
    args["eval_mode"] = True
    args["need_plan"] = False
    return args


def get_eef_state(task_env):
    lp = task_env.robot.get_left_ee_pose()
    rp = task_env.robot.get_right_ee_pose()
    lg = task_env.robot.get_left_gripper_val()
    rg = task_env.robot.get_right_gripper_val()
    return np.concatenate([lp, [lg], rp, [rg]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_name", required=True)
    ap.add_argument("--task_config", required=True)
    ap.add_argument("--episodes", type=int, nargs="+", default=[0])
    ap.add_argument("--raw_data_root", default="./data")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--action_stride", type=int, default=4,
                    help="execute every k-th recorded action (planner "
                    "interpolates between targets; stride 1 is slow)")
    ap.add_argument("--fps", type=int, default=10)
    args_cli = ap.parse_args()
    os.makedirs(args_cli.out_dir, exist_ok=True)

    data_dir = os.path.join(args_cli.raw_data_root, args_cli.task_name, args_cli.task_config)
    with open(os.path.join(data_dir, "seed.txt")) as f:
        seed_list = [int(x) for x in f.read().split()]

    args = build_args(args_cli.task_name, args_cli.task_config)
    task_env = class_decorator(args_cli.task_name)

    for ep in args_cli.episodes:
        seed = seed_list[ep]
        raw_path = os.path.join(data_dir, "data", f"episode{ep}.hdf5")
        with h5py.File(raw_path, "r", swmr=True) as rf:
            lp = rf["endpose/left_endpose"][:]
            lg = rf["endpose/left_gripper"][:]
            rp = rf["endpose/right_endpose"][:]
            rg = rf["endpose/right_gripper"][:]
            rec_rgb = rf["observation/head_camera/rgb"][:]
        rec_state = np.concatenate(
            [lp, lg[:, None], rp, rg[:, None]], axis=-1
        )  # [T, 16]; action[t] = rec_state[t+1]
        T = rec_state.shape[0]

        print(f"\n=== episode {ep} (seed {seed}, {T} frames) ===")
        task_env.setup_demo(now_ep_num=ep, seed=seed, **args)

        pos_errs, quat_errs, grip_errs = [], [], []
        frames = []
        step_ids = list(range(args_cli.action_stride, T, args_cli.action_stride))
        for t in step_ids:
            target = rec_state[t]  # absolute EEF pose at recorded frame t
            task_env.take_action(target, action_type="ee")

            achieved = get_eef_state(task_env)
            for lo in (0, 8):
                pos_errs.append(np.linalg.norm(achieved[lo:lo+3] - target[lo:lo+3]))
                q_a, q_t = achieved[lo+3:lo+7], target[lo+3:lo+7]
                quat_errs.append(1.0 - abs(float(np.dot(q_a, q_t))))  # sign-invariant
                grip_errs.append(abs(achieved[lo+7] - target[lo+7]))

            obs = task_env.get_obs()
            live = obs["observation"]["head_camera"]["rgb"]
            live = cv2.cvtColor(np.asarray(live, dtype=np.uint8), cv2.COLOR_RGB2BGR)
            # raw blobs decode to TRUE RGB (double swap cancels); to BGR for video
            rec = cv2.imdecode(np.frombuffer(rec_rgb[t], np.uint8), cv2.IMREAD_COLOR)
            rec = cv2.cvtColor(rec, cv2.COLOR_RGB2BGR)
            panel = np.hstack([live, rec])
            cv2.putText(panel, f"t={t} live | recorded", (6, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            frames.append(panel)

            if task_env.eval_success:
                print(f"  success reached at replay step t={t}")
                break

        success = bool(task_env.eval_success)
        task_env.close_env()

        pos_errs, quat_errs, grip_errs = map(np.array, (pos_errs, quat_errs, grip_errs))
        print(f"[tracking] pos err  mean={pos_errs.mean()*1000:.1f}mm "
              f"p95={np.percentile(pos_errs,95)*1000:.1f}mm max={pos_errs.max()*1000:.1f}mm")
        print(f"[tracking] quat err (1-|dot|) mean={quat_errs.mean():.4f} "
              f"max={quat_errs.max():.4f}")
        print(f"[tracking] gripper err mean={grip_errs.mean():.3f} max={grip_errs.max():.3f}")
        print(f"[task] success after replay: {success}")

        if frames:
            hh, ww = frames[0].shape[:2]
            out_path = os.path.join(
                args_cli.out_dir, f"{args_cli.task_name}_ep{ep}_replay.mp4")
            vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 args_cli.fps, (ww, hh))
            for fr in frames:
                vw.write(fr)
            vw.release()
            _reencode_h264(out_path)
            print(f"[video] {out_path}")


if __name__ == "__main__":
    main()
