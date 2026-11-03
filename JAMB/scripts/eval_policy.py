import sys
import os
import json
import subprocess
import pickle

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./script")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib
import argparse
import pdb

from generate_episode_instructions import *

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.abspath("./script") if os.path.isdir("./script") else os.path.dirname(current_file_path)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e

def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    checkpoint_num = usr_args['checkpoint_num']
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    seed = usr_args["seed"]
    save_dir = None
    video_save_dir = None
    video_size = None

    get_model = eval_function_decorator(policy_name, "get_model")

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    result_root = Path(os.environ.get("EVAL_RESULT_ROOT", "eval_result"))
    # save_dir = result_root / task_name / policy_name / task_config / ckpt_setting / current_time
    save_dir = result_root / task_name / policy_name / task_config / ckpt_setting / f"seed_{seed}" / str(checkpoint_num)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args["eval_video_log"]:
        video_save_dir = save_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    # output camera config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args["seed"]

    st_seed = 100000 * (1 + seed)
    suc_nums = []
    test_num = int(usr_args.get("test_num", 100))
    topk = 1

    model = get_model(usr_args)

    wandb_mode = os.environ.get("WANDB_MODE", "offline")
    use_wandb_eval = wandb_mode != "disabled"
    if use_wandb_eval:
        import wandb as wandb_module
        wandb_module.init(
            project="RoboTwin_GAP_eval",
            name=f"eval_{task_name}_{ckpt_setting}_{checkpoint_num}_seed{seed}",
            config=usr_args,
            mode=wandb_mode,
        )
        usr_args["_wandb"] = wandb_module
        args["_wandb"] = wandb_module
    else:
        usr_args["_wandb"] = None
        args["_wandb"] = None

    st_seed, suc_num, episode_records = eval_policy(task_name,
                                                    TASK_ENV,
                                                    args,
                                                    model,
                                                    st_seed,
                                                    test_num=test_num,
                                                    video_size=video_size,
                                                    instruction_type=instruction_type)
    suc_nums.append(suc_num)

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]

    success_eps = [r for r in episode_records if r["success"]]
    failure_eps = [r for r in episode_records if not r["success"]]

    # _result.txt — overall summary + per-episode table
    file_path = os.path.join(save_dir, "_result.txt")
    with open(file_path, "w") as f:
        f.write(f"Timestamp: {current_time}\n\n")
        f.write(f"Instruction Type: {instruction_type}\n\n")
        f.write(f"Success rate: {suc_num}/{test_num} = {suc_num/test_num:.3f}\n\n")
        f.write(f"Success seeds: {[r['seed'] for r in success_eps]}\n")
        f.write(f"Failure seeds: {[r['seed'] for r in failure_eps]}\n\n")
        f.write(f"{'ep':>4}  {'seed':>8}  {'result':>7}  {'steps':>10}  {'stage_score':>11}  failure_reason\n")
        f.write("-" * 70 + "\n")
        for r in episode_records:
            result_str = "SUCCESS" if r["success"] else "FAIL"
            stage_str = f"{r['stage_score']:.2f}" if r["stage_score"] is not None else "n/a"
            reason_str = r["failure_reason"] or ""
            f.write(f"{r['episode']:>4}  {r['seed']:>8}  {result_str:>7}  {r['steps']:>5}/{r['step_lim']:<4}  {stage_str:>11}  {reason_str}\n")

    # _rollouts.json — machine-readable per-episode records
    json_path = os.path.join(save_dir, "_rollouts.json")
    with open(json_path, "w") as f:
        json.dump({
            "task": task_name,
            "config": task_config,
            "checkpoint": checkpoint_num,
            "seed": seed,
            "success_rate": suc_num / test_num,
            "success_count": suc_num,
            "total": test_num,
            "episodes": episode_records,
        }, f, indent=2)

    print(f"Data has been saved to {file_path}")
    print(f"Per-episode rollout log: {json_path}")

    wandb_module = usr_args.get("_wandb")
    if wandb_module is not None:
        wandb_module.finish()
    # return task_reward


def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=100,
                video_size=None,
                instruction_type=None):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []
    episode_records = []

    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval")
    reset_func = eval_function_decorator(policy_name, "reset_model")

    now_seed = st_seed
    task_total_reward = 0
    clear_cache_freq = args["clear_cache_freq"]

    args["eval_mode"] = True

    while succ_seed < test_num:
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError as e:
                # print(" -------------")
                # print("Error: ", e)
                # print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception as e:
                # stack_trace = traceback.format_exc()
                # print(" -------------")
                # print("Error: ", e)
                # print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                print("error occurs !")
                continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
        instruction = np.random.choice(results[0][instruction_type])
        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction

        if TASK_ENV.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        succ = False
        reset_func(model)
        wandb_module = args.get("_wandb")
        VIS_MAX_EPISODES = 3
        chunk_idx = 0

        # Opt-in per-chunk track-prediction logging for offline video overlay
        # (see visualize_eval_tracks.py) -- zero-cost / no-op unless the env
        # var is set, does not touch the normal eval control flow or aux_data.
        track_vis_dir = os.environ.get("TRACK_VIS_LOG_DIR")
        track_vis_log = [] if track_vis_dir else None

        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            observation = TASK_ENV.get_obs()
            chunk_start_frame = TASK_ENV.take_action_cnt
            _, aux_data = eval_func(TASK_ENV, model, observation)

            if track_vis_log is not None and aux_data is not None and aux_data.get("track_pred") is not None:
                head_cam = observation["observation"]["head_camera"]
                if "depth" in head_cam:
                    policy_module = importlib.import_module(args["policy_name"])
                    depth_arr = np.asarray(head_cam["depth"], dtype=np.float32)
                    intrinsic_arr = np.asarray(head_cam["intrinsic_cv"])
                    extrinsic_arr = np.asarray(head_cam["extrinsic_cv"])
                    patch_centers = policy_module.compute_patch_centers_world(
                        depth_arr, intrinsic_arr, extrinsic_arr,
                    )
                    tp = aux_data["track_pred"]
                    tp = tp.detach().cpu().numpy() if hasattr(tp, "detach") else np.asarray(tp)
                    entry = {
                        "chunk_idx": chunk_idx,
                        "start_frame": chunk_start_frame,
                        "n_frames": TASK_ENV.take_action_cnt - chunk_start_frame,
                        "track_pred": tp[0],  # [N, H, 3] displacement from patch_centers
                        "patch_centers": patch_centers,  # [N, 3] world-frame
                        "intrinsic_cv": intrinsic_arr,
                        "extrinsic_cv": extrinsic_arr,
                    }
                    # Dense colored point cloud for 3D track viz backgrounds
                    # (see visualize_track_3d.py) -- logged every DENSE_PC_EVERY
                    # chunks (not every single one -- a long episode times out
                    # at 50 chunks and logging every chunk at stride=2 blows
                    # past the 16MB per-episode export budget), skipping
                    # chunks reuse the most recent snapshot (compute_dense_
                    # pointcloud_world zeros rather than drops invalid points,
                    # so M stays constant and export_trackvis_binary.py can
                    # store one shared-size snapshot table). Skip entirely if
                    # rgb isn't available for any reason.
                    dense_pc_every = int(os.environ.get("DENSE_PC_EVERY", "3"))
                    if chunk_idx % dense_pc_every == 0 and "rgb" in head_cam and hasattr(policy_module, "compute_dense_pointcloud_world"):
                        pc_xyz, pc_rgb = policy_module.compute_dense_pointcloud_world(
                            depth_arr, np.asarray(head_cam["rgb"]), intrinsic_arr, extrinsic_arr, stride=2,
                        )
                        entry["dense_pc_xyz"] = pc_xyz
                        entry["dense_pc_rgb"] = pc_rgb
                    track_vis_log.append(entry)

            if wandb_module is not None and aux_data is not None and TASK_ENV.test_num < VIS_MAX_EPISODES:
                if "pred_xyz" in aux_data:
                    # PMP format (zarr line): predicted vs actual point maps
                    pred_xyz = aux_data["pred_xyz"]
                    actual_xyz = aux_data["actual_xyz"]
                    mse = aux_data["mse"]

                    pred_colored = np.column_stack([pred_xyz, np.full((len(pred_xyz), 3), [0, 0, 255])])
                    actual_colored = np.column_stack([actual_xyz, np.full((len(actual_xyz), 3), [255, 0, 0])])
                    combined = np.vstack([pred_colored, actual_colored]).astype(np.float32)

                    wandb_module.log({
                        "eval/pointmap_comparison": wandb_module.Object3D(combined),
                        "eval/pointmap_mse": mse,
                        "eval/episode": TASK_ENV.test_num,
                        "eval/chunk": chunk_idx,
                    })
                elif aux_data.get("track_pred") is not None:
                    # Track-prediction format: log predicted displacement
                    # magnitude (mean/max over patches at the chunk's final step)
                    tp = aux_data["track_pred"]
                    tp = tp.detach().cpu().numpy() if hasattr(tp, "detach") else np.asarray(tp)
                    final_disp = np.linalg.norm(tp[0, :, -1, :], axis=-1)  # [N]
                    wandb_module.log({
                        "eval/track_final_disp_mean": float(final_disp.mean()),
                        "eval/track_final_disp_max": float(final_disp.max()),
                        "eval/episode": TASK_ENV.test_num,
                        "eval/chunk": chunk_idx,
                    })
            chunk_idx += 1

            if TASK_ENV.eval_success:
                succ = True
                break
        # task_total_reward += TASK_ENV.episode_score
        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

        if track_vis_log is not None:
            os.makedirs(track_vis_dir, exist_ok=True)
            with open(os.path.join(track_vis_dir, f"episode{TASK_ENV.test_num}_trackvis.pkl"), "wb") as f:
                pickle.dump(track_vis_log, f)

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        # Collect per-episode record before environment is closed (state is lost after close_env)
        stage_score = getattr(TASK_ENV, "stage_eval_score", None)
        rec = {
            "episode": TASK_ENV.test_num,
            "seed": now_seed,
            "success": succ,
            "steps": TASK_ENV.take_action_cnt,
            "step_lim": TASK_ENV.step_lim,
            "stage_score": float(stage_score) if stage_score is not None else None,
            "failure_reason": None,
        }
        if not succ:
            if stage_score is not None:
                rec["failure_reason"] = "no_progress" if stage_score == 0.0 else "partial_progress"
            else:
                rec["failure_reason"] = "timeout"
        episode_records.append(rec)

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        if wandb_module is not None:
            wandb_module.log({
                "eval/success": int(succ),
                "eval/success_rate": TASK_ENV.suc / TASK_ENV.test_num,
                "eval/episode": TASK_ENV.test_num,
            })

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )
        # TASK_ENV._take_picture()
        now_seed += 1

    return now_seed, TASK_ENV.suc, episode_records


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()

    main(usr_args)
