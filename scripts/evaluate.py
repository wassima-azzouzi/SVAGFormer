#!/usr/bin/env python3
"""
Evaluate a trained SVAGFormer checkpoint on SVAG-Bench.

Usage:
    python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/epoch_0060.pth
    python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/epoch_0060.pth --split test
"""

import argparse
import json
import logging
import sys
import yaml
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.svagformer    import SVAGFormer
from data.dataset         import SVAGDataset, collate_fn
from evaluation.svageval  import SVAGEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NMS for temporal predictions (greedy)
# ---------------------------------------------------------------------------

def temporal_nms(moments, scores, iou_threshold: float = 0.7):
    """
    Args:
        moments: list of (start, end)
        scores : list of floats
    Returns:
        kept moments and scores
    """
    if not moments:
        return [], []

    order  = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep   = []

    while order:
        i = order.pop(0)
        keep.append(i)
        s1, e1 = moments[i]
        order = [
            j for j in order
            if max(0, min(e1, moments[j][1]) - max(s1, moments[j][0]))
               / max(1e-6, max(e1, moments[j][1]) - min(s1, moments[j][0]))
               < iou_threshold
        ]

    return [moments[i] for i in keep], [scores[i] for i in keep]


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(cfg: dict, checkpoint: str, split: str = "val",
             use_nms: bool = True, nms_threshold: float = 0.7,
             output_path: str = "") -> dict:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = SVAGFormer(cfg["model"]).to(device)
    ckpt  = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    log.info(f"Loaded checkpoint: {checkpoint}")

    # Data
    data_root   = cfg["data"].get("root", "")
    num_frames  = cfg["data"].get("num_frames", 32)
    batch_size  = cfg["eval"].get("batch_size", 4)
    num_workers = cfg["eval"].get("num_workers", 4)

    dataset = SVAGDataset(root=data_root, split=split, num_frames=num_frames)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
    )
    log.info(f"Evaluating on {split} split | {len(dataset)} samples")

    evaluator = SVAGEvaluator()
    all_predictions = []

    for videos, queries, targets in loader:
        videos = videos.to(device, non_blocking=True)
        outputs = model(videos, queries)

        B = videos.shape[0]
        for b in range(B):
            vid_id    = targets[b]["video_id"]
            query     = targets[b]["query"]
            gt_moment = targets[b]["moment"].tolist()

            # ---- Temporal ----
            s, e  = outputs["moments"][b].tolist()
            score = outputs["tscores"][b].sigmoid().item()

            # Apply NMS (only meaningful when running multiple proposals per query)
            pred_moments = [(s, e)]
            pred_scores  = [score]
            if use_nms:
                pred_moments, pred_scores = temporal_nms(
                    pred_moments, pred_scores, nms_threshold
                )

            evaluator.update_temporal(
                pred_moments=[(pm[0], pm[1], ps)
                              for pm, ps in zip(pred_moments, pred_scores)],
                gt_moment=gt_moment,
            )

            # ---- Spatial ----
            # Convert model outputs to track format
            # outputs["bboxes"]: (B, T, Nq, 4)  outputs["scores"]: (B, T, Nq, 1)
            pred_tracks: dict = {}
            T_frames = outputs["bboxes"].shape[1]
            box_scores = outputs["scores"][b].sigmoid().squeeze(-1)  # (T, Nq)
            for t in range(T_frames):
                for q in range(outputs["bboxes"].shape[2]):
                    if box_scores[t, q].item() > cfg["eval"].get("score_thr", 0.3):
                        tid = int(q)  # simplified: each query = one track
                        box = outputs["bboxes"][b, t, q].tolist()
                        if tid not in pred_tracks:
                            pred_tracks[tid] = []
                        pred_tracks[tid].append([t] + box)

            gt_tracks = {int(k): v for k, v in targets[b].get("boxes", {}).items()}
            evaluator.update_spatial(
                query_id    = f"{vid_id}_{b}",
                pred_tracks = pred_tracks,
                gt_tracks   = gt_tracks,
            )

            # Collect for output file
            all_predictions.append({
                "video_id"   : vid_id,
                "query"      : query,
                "pred_start" : s,
                "pred_end"   : e,
                "score"      : score,
                "gt_start"   : gt_moment[0],
                "gt_end"     : gt_moment[1],
            })

    results = evaluator.compute()

    # Pretty print
    log.info("=" * 60)
    log.info(f"{'Metric':<30} {'Value':>10}")
    log.info("-" * 60)
    for k, v in sorted(results.items()):
        log.info(f"{k:<30} {v:>10.4f}")
    log.info("=" * 60)
    log.info(f"m-HIoU (primary): {results['m-HIoU']:.4f}")

    # Save predictions
    if output_path:
        with open(output_path, "w") as f:
            json.dump({"results": results, "predictions": all_predictions}, f, indent=2)
        log.info(f"Predictions saved → {output_path}")

    return results


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split",      default="val", choices=["val", "test"])
    parser.add_argument("--nms",        action="store_true", default=True)
    parser.add_argument("--nms_thr",    type=float, default=0.7)
    parser.add_argument("--output",     default="", help="JSON output path")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    evaluate(cfg, args.checkpoint, args.split, args.nms, args.nms_thr, args.output)


if __name__ == "__main__":
    main()
