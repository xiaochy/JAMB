"""
Locate a JAMB training checkpoint by (task_name, setting, expert_data_num, checkpoint_num).

Searches the Hydra-managed run directories under data/outputs/*/*_JAMB_<task_name>*/
(the layout used since checkpoints were moved into the run's own output folder),
confirming a match via the run's .hydra/overrides.yaml rather than the directory
name alone. Falls back to the legacy flat layout
checkpoints/<task>_<setting>_<num>/<epoch>.ckpt for checkpoints saved before that
change. Prints the resolved absolute checkpoint path to stdout on success.
"""
import argparse
import glob
import os
import sys


def parse_overrides(overrides_path):
    overrides = {}
    with open(overrides_path) as f:
        for line in f:
            line = line.strip().lstrip("-").strip()
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            overrides[key.strip()] = value.strip()
    return overrides


def find_new_style(jamb_root, task_name, setting, expert_data_num, checkpoint_num, run_tag):
    pattern = os.path.join(jamb_root, "data", "outputs", "*", f"*_JAMB_{task_name}*")
    candidates = []
    for run_dir in glob.glob(pattern):
        overrides_path = os.path.join(run_dir, ".hydra", "overrides.yaml")
        ckpt_path = os.path.join(run_dir, "checkpoints", f"{checkpoint_num}.ckpt")
        if not (os.path.isfile(overrides_path) and os.path.isfile(ckpt_path)):
            continue
        overrides = parse_overrides(overrides_path)
        if overrides.get("task_name") != task_name:
            continue
        if overrides.get("setting") != setting:
            continue
        if overrides.get("expert_data_num") != str(expert_data_num):
            continue
        if run_tag is not None and overrides.get("run_tag", "") != run_tag:
            continue
        candidates.append((run_dir, ckpt_path, overrides.get("run_tag", "")))

    if not candidates:
        return None

    # Run dirs are named .../<date>/<time>_..., which sorts chronologically.
    candidates.sort(key=lambda c: c[0])
    if len(candidates) > 1:
        distinct_tags = sorted({tag for _, _, tag in candidates})
        if len(distinct_tags) > 1:
            print(
                f"[find_checkpoint] Warning: {len(candidates)} matching runs with "
                f"different run_tags ({distinct_tags}); using the most recent "
                f"({candidates[-1][0]}). Pass --run-tag to disambiguate.",
                file=sys.stderr,
            )
        else:
            print(
                f"[find_checkpoint] Warning: {len(candidates)} matching runs found; "
                f"using the most recent ({candidates[-1][0]}).",
                file=sys.stderr,
            )
    return candidates[-1][1]


def find_legacy(jamb_root, task_name, setting, expert_data_num, checkpoint_num):
    ckpt_path = os.path.join(
        jamb_root,
        "checkpoints",
        f"{task_name}_{setting}_{expert_data_num}",
        f"{checkpoint_num}.ckpt",
    )
    return ckpt_path if os.path.isfile(ckpt_path) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jamb-root", required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--expert-data-num", required=True)
    parser.add_argument("--checkpoint-num", required=True)
    parser.add_argument("--run-tag", default=None)
    args = parser.parse_args()

    ckpt_path = find_new_style(
        args.jamb_root, args.task_name, args.setting, args.expert_data_num,
        args.checkpoint_num, args.run_tag,
    )
    if ckpt_path is None:
        ckpt_path = find_legacy(
            args.jamb_root, args.task_name, args.setting, args.expert_data_num,
            args.checkpoint_num,
        )

    if ckpt_path is None:
        print(
            "[find_checkpoint] No checkpoint found for "
            f"task_name={args.task_name} setting={args.setting} "
            f"expert_data_num={args.expert_data_num} checkpoint_num={args.checkpoint_num} "
            f"run_tag={args.run_tag}\n"
            f"  searched: data/outputs/*/*_JAMB_{args.task_name}*/checkpoints/{args.checkpoint_num}.ckpt\n"
            f"  and legacy: checkpoints/{args.task_name}_{args.setting}_{args.expert_data_num}/{args.checkpoint_num}.ckpt\n"
            "  Pass CKPT_PATH explicitly if the checkpoint lives elsewhere.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(ckpt_path)


if __name__ == "__main__":
    main()
