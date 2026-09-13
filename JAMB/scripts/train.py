"""
Training script for JAMB.
"""
if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import sys
import hydra
import torch
from omegaconf import OmegaConf
import pathlib

JAMB_ROOT = str(pathlib.Path(__file__).parent.parent)

sys.path.append(JAMB_ROOT)
sys.path.append(os.path.join(JAMB_ROOT, "jamb_policy"))

from torch.utils.data import DataLoader
import copy

import wandb
from tqdm import tqdm
import numpy as np
from termcolor import cprint
import random
from hydra.core.hydra_config import HydraConfig

from jamb_policy.dataset.track_dataset import TrackDataset
from jamb_policy.common.pytorch_util import dict_apply
from jamb_policy.model.diffusion.ema_model import EMAModel
from jamb_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver(
    "tag_suffix", lambda tag: f"_{tag}" if tag else "", replace=True
)


@hydra.main(
    version_base=None,
    config_path="../jamb_policy/config",
    config_name="JAMB",
)
def main(cfg: OmegaConf):
    # Set random seed
    seed = cfg.training.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Get task name and setting
    task_name = cfg.task_name
    setting = cfg.get("setting", "demo_clean")
    expert_data_num = cfg.expert_data_num
    observation_chunk = cfg.observation_chunk
    interval = cfg.interval
    model_3d = cfg.model_3d

    data_root = cfg.get("data_root", os.environ.get("JAMB_DATA_ROOT", "data"))
    if not os.path.isabs(data_root):
        data_root = os.path.join(JAMB_ROOT, data_root)
    # Prefer the JAMB-style zarr (convert_tracks_to_zarr.py); fall back to
    # the flat tracks HDF5 for datasets that were never converted
    data_path = os.path.join(
        data_root, "tracks",
        f"{task_name}-{setting}-{expert_data_num}-tracks.zarr",
    )
    if not os.path.exists(data_path):
        data_path = os.path.join(
            data_root, "tracks",
            f"{task_name}-{setting}-{expert_data_num}-tracks_flat.hdf5",
        )

    # Shared-memory dataset override: a directory of .npy files exported by
    # scripts/export_shm_dataset.py — all concurrent trainings mmap ONE copy
    shm_dir = os.environ.get("TRACKS_SHM_DIR")
    if shm_dir:
        assert os.path.isdir(shm_dir), f"TRACKS_SHM_DIR not found: {shm_dir}"
        data_path = shm_dir

    cprint(f"[JAMB Training]", "cyan", attrs=["bold"])
    cprint(f"  Task: {task_name}", "cyan")
    cprint(f"  Setting: {setting}", "cyan")
    cprint(f"  Expert data: {expert_data_num}", "cyan")
    cprint(f"  Data path: {data_path}", "cyan")
    cprint(f"  Seed: {seed}", "cyan")

    # Create dataset
    val_ratio = cfg.training.get("val_ratio", 0.1)
    dataset = TrackDataset(
        data_path=data_path,
        horizon=cfg.horizon,
        pad_before=cfg.n_obs_steps - 1,
        pad_after=cfg.n_action_steps - 1,
        seed=seed,
        val_ratio=val_ratio,
        max_train_episodes=expert_data_num,
        task_name=task_name,
        use_dino_features=cfg.get("use_dino_features", False),
        use_pi3_features=cfg.get("use_pi3_features", True),
        aux_task=cfg.get("aux_task", "track"),
        track_key=cfg.get("track_key", "3d_track"),
        mask_key=cfg.get("mask_key", "track_valid_mask"),
    )

    # Get normalizer
    normalizer = dataset.get_normalizer()

    # Create dataloader
    train_dataloader = DataLoader(
        dataset,
        batch_size=cfg.dataloader.batch_size,
        num_workers=cfg.dataloader.num_workers,
        shuffle=cfg.dataloader.shuffle,
        pin_memory=cfg.dataloader.pin_memory,
        persistent_workers=cfg.dataloader.persistent_workers,
    )

    val_dataset = dataset.get_validation_dataset()
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.val_dataloader.batch_size,
        num_workers=cfg.val_dataloader.num_workers,
        shuffle=cfg.val_dataloader.shuffle,
        pin_memory=cfg.val_dataloader.pin_memory,
        persistent_workers=cfg.val_dataloader.persistent_workers,
    )

    # Create policy
    cprint("\nCreating JAMB policy...", "green")
    policy = hydra.utils.instantiate(cfg.policy)

    # Set normalizer
    policy.set_normalizer(normalizer)

    # Move to device
    device = torch.device(cfg.training.device)
    policy = policy.to(device)

    # Create EMA model
    ema: EMAModel = None
    if cfg.training.use_ema:
        ema_policy = copy.deepcopy(policy)
        ema = hydra.utils.instantiate(cfg.ema, model=ema_policy)

    # Create optimizer
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=policy.parameters())

    # Create learning rate scheduler
    lr_scheduler = get_scheduler(
        cfg.training.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.training.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * cfg.training.num_epochs) // cfg.training.gradient_accumulate_every,
    )

    # Initialize wandb
    wandb_mode = cfg.logging.get("mode", "online")
    use_wandb = (not cfg.training.debug) and wandb_mode != "disabled"
    if use_wandb:
        run_name = f"{cfg.name}_{task_name}_{setting}_{expert_data_num}"
        run_tag = cfg.get("run_tag", "")
        if run_tag:
            run_name = f"{run_name}_{run_tag}"
        wandb.init(
            project=cfg.logging.project,
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=wandb_mode,
        )
    if cfg.training.debug:
        cfg.training.num_epochs = 100
        cfg.training.max_train_steps = 10
        cfg.training.max_val_steps = 3
        cfg.training.checkpoint_every = 1
        cfg.training.val_every = 1

    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    resume_ckpt = os.environ.get("RESUME_CKPT") or cfg.get("resume_ckpt", None)
    if resume_ckpt:
        cprint(f"\nResuming from checkpoint: {resume_ckpt}", "yellow", attrs=["bold"])
        ckpt = torch.load(resume_ckpt, map_location=device)
        policy.load_state_dict(ckpt["model"])
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
            global_step = ckpt.get("global_step", 0)
        except (ValueError, KeyError) as e:
            cprint(f"  Optimizer/scheduler state skipped ({e}); weights loaded, fresh optimizer", "yellow")
        if ema is not None and "ema" in ckpt:
            ema.averaged_model.load_state_dict(ckpt["ema"])
        start_epoch = ckpt["epoch"] + 1
        cprint(f"Resumed from epoch {ckpt['epoch']+1}, global_step {global_step}", "yellow")

    # Training loop
    cprint("\nStarting training...", "green", attrs=["bold"])

    # bf16 autocast: no GradScaler needed (bf16 has fp32's exponent range),
    # and it's a no-op-safe API back to Ampere (sm_80+), so this doesn't
    # need to be reverted when moving the run to a 3090.
    amp_enabled = cfg.training.get("amp", True) and device.type == "cuda"
    cprint(f"  AMP (bf16): {'on' if amp_enabled else 'off'}", "cyan")

    for epoch in range(start_epoch, cfg.training.num_epochs):
        policy.train()

        epoch_loss = 0.0
        with tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{cfg.training.num_epochs}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                if cfg.training.max_train_steps is not None and batch_idx >= cfg.training.max_train_steps:
                    break
                # Move batch to device
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

                # Forward pass
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                    loss, loss_dict = policy.compute_loss(batch)

                # Backward pass
                loss.backward()

                # Gradient accumulation
                if (batch_idx + 1) % cfg.training.gradient_accumulate_every == 0:
                    # Clip gradients
                    if cfg.training.get("clip_grad_norm", None):
                        torch.nn.utils.clip_grad_norm_(
                            policy.parameters(),
                            cfg.training.clip_grad_norm
                        )

                    # Update parameters
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    # Update EMA
                    if ema is not None:
                        ema.step(policy)

                    global_step += 1

                # Logging
                epoch_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

                # Log to wandb
                if use_wandb and (batch_idx % 10 == 0):
                    log_dict = {
                        "train/loss": loss.item(),
                        "train/lr": lr_scheduler.get_last_lr()[0],
                        "train/epoch": epoch,
                        "train/global_step": global_step,
                    }
                    # Add loss_dict items with train/ prefix
                    for key, value in loss_dict.items():
                        log_dict[f"train/{key}"] = value
                    wandb.log(log_dict, step=global_step)

        # Epoch summary
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        cprint(f"Epoch {epoch+1} - Avg Loss: {avg_epoch_loss:.4f}", "yellow")

        if use_wandb:
            wandb.log({
                "train/epoch_loss": avg_epoch_loss,
                "train/epoch": epoch,
            }, step=global_step)

        # Validation
        if (epoch + 1) % cfg.training.val_every == 0 and len(val_dataloader) > 0:
            policy.eval()
            val_loss_total = 0.0
            val_steps = 0
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_dataloader):
                    if cfg.training.max_val_steps is not None and batch_idx >= cfg.training.max_val_steps:
                        break
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                        loss, loss_dict = policy.compute_loss(batch)
                    val_loss_total += loss.item()
                    val_steps += 1
            avg_val_loss = val_loss_total / val_steps if val_steps > 0 else 0.0
            cprint(f"Epoch {epoch+1} - Val Loss: {avg_val_loss:.4f}", "cyan")
            if use_wandb:
                val_log = {"val/epoch_loss": avg_val_loss, "val/epoch": epoch}
                for key, value in loss_dict.items():
                    val_log[f"val/{key}"] = value
                wandb.log(val_log, step=global_step)

        # Save checkpoint (into this run's hydra output dir, not the repo root)
        run_output_dir = HydraConfig.get().runtime.output_dir
        checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
        if (epoch + 1) % cfg.training.checkpoint_every == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"{epoch+1}.ckpt")
            os.makedirs(checkpoint_dir, exist_ok=True)

            checkpoint = {
                "epoch": epoch,
                "global_step": global_step,
                "model": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "cfg": OmegaConf.to_container(cfg, resolve=True),
                'normalizer': normalizer.state_dict(),
            }

            if ema is not None:
                checkpoint["ema"] = ema.averaged_model.state_dict()

            torch.save(checkpoint, checkpoint_path)
            cprint(f"Saved checkpoint to {checkpoint_path}", "green")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
