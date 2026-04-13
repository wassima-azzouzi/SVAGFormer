#!/usr/bin/env python3
"""
Train SVAGFormer on SVAG-Bench.

Usage:
    python scripts/train.py --config configs/default.yaml

The script handles:
  - Multi-GPU training via torch.distributed (DDP)
  - Gradient clipping, AdamW optimizer, step-based LR decay
  - Checkpoint saving / resuming
  - Periodic validation with SVAGEval metrics
"""

import os
import argparse
import yaml
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Project imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.svagformer import SVAGFormer
from models.losses     import SVAGLoss
from data.dataset      import SVAGDataset, collate_fn
from evaluation.svageval import SVAGEvaluator


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_checkpoint(model, optimizer, epoch, path: str):
    state = {
        "epoch":     epoch,
        "model":     model.state_dict() if not isinstance(model, DDP) else model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state, path)
    log.info(f"Checkpoint saved → {path}")


def load_checkpoint(model, optimizer, path: str, device: torch.device) -> int:
    ckpt = torch.load(path, map_location=device)
    model_to_load = model.module if isinstance(model, DDP) else model
    model_to_load.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    log.info(f"Resumed from {path} (epoch {ckpt['epoch']})")
    return ckpt["epoch"]


# ---------------------------------------------------------------------------
# One epoch of training
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device,
                    epoch: int, grad_clip: float = 1.0) -> float:
    model.train()
    total_loss = 0.0

    for step, (videos, queries, targets) in enumerate(loader):
        videos = videos.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(videos, queries)
        losses  = criterion(outputs, targets)

        loss = losses["loss_total"]
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()

        if step % 50 == 0:
            log.info(
                f"Epoch {epoch} | Step {step}/{len(loader)} | "
                f"loss_total={loss.item():.4f}  "
                f"loss_spatial={losses['loss_spatial'].item():.4f}  "
                f"loss_temporal={losses['loss_temporal'].item():.4f}"
            )

    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, device) -> dict:
    model.eval()
    evaluator = SVAGEvaluator()

    for videos, queries, targets in loader:
        videos = videos.to(device, non_blocking=True)
        outputs = model(videos, queries)

        B = videos.shape[0]
        for b in range(B):
            vid_id = targets[b]["video_id"]
            gt_moment = targets[b]["moment"].tolist()

            # Temporal: convert predicted moment to list of (start, end, score)
            start, end = outputs["moments"][b].tolist()
            tscore     = outputs["tscores"][b].item()
            evaluator.update_temporal(
                pred_moments=[(start, end, tscore)],
                gt_moment=gt_moment,
            )

            # Spatial: collapse frame predictions to tracks (simplified)
            # In full implementation, apply NMS + ID association here
            pred_tracks: dict = {}
            gt_tracks   = targets[b].get("boxes", {})
            evaluator.update_spatial(
                query_id   = f"{vid_id}_{b}",
                pred_tracks= pred_tracks,   # TODO: fill from outputs["bboxes"]
                gt_tracks  = {int(k): v for k, v in gt_tracks.items()},
            )

    return evaluator.compute()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="configs/default.yaml")
    parser.add_argument("--resume",   default="", help="Path to checkpoint")
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Distributed setup
    use_ddp = "LOCAL_RANK" in os.environ
    if use_ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0

    is_main = (local_rank == 0)

    # ----- Model -----
    model = SVAGFormer(cfg["model"]).to(device)
    if use_ddp:
        model = DDP(model, device_ids=[local_rank])

    # ----- Optimizer -----
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"].get("lr", 1e-4),
        weight_decay=cfg["training"].get("weight_decay", 1e-4),
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg["training"].get("lr_step", 40),
        gamma=cfg["training"].get("lr_gamma", 0.1),
    )

    criterion = SVAGLoss(
        w_spatial=cfg["training"].get("w_spatial", 1.0),
        w_temporal=cfg["training"].get("w_temporal", 1.0),
    )

    # ----- Resume -----
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, args.resume, device)

    # ----- Data -----
    data_root  = cfg["data"].get("root", "")
    num_frames = cfg["data"].get("num_frames", 32)
    batch_size = cfg["training"].get("batch_size", 4)
    num_workers= cfg["training"].get("num_workers", 4)

    train_dataset = SVAGDataset(root=data_root, split="train", num_frames=num_frames)
    val_dataset   = SVAGDataset(root=data_root, split="val",   num_frames=num_frames)

    train_sampler = DistributedSampler(train_dataset) if use_ddp else None
    train_loader  = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=train_sampler, shuffle=(train_sampler is None),
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    # ----- Training loop -----
    num_epochs = cfg["training"].get("epochs", 60)
    save_dir   = Path(cfg["training"].get("save_dir", "checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, num_epochs):
        if use_ddp:
            train_sampler.set_epoch(epoch)

        avg_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            grad_clip=cfg["training"].get("grad_clip", 1.0),
        )
        scheduler.step()

        if is_main:
            log.info(f"Epoch {epoch} done | avg_loss={avg_loss:.4f} | lr={scheduler.get_last_lr()}")

            # Validate every N epochs
            if (epoch + 1) % cfg["training"].get("val_every", 5) == 0:
                val_results = validate(model, val_loader, device)
                log.info(f"Val results: {val_results}")

            # Save checkpoint
            save_checkpoint(model, optimizer, epoch + 1,
                            str(save_dir / f"epoch_{epoch+1:04d}.pth"))

    if is_main:
        log.info("Training complete.")


if __name__ == "__main__":
    main()
