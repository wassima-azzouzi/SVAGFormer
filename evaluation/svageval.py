"""
SVAGEval: Evaluation toolkit for the SVAG task.

Implements:
  - HOTA  (Higher Order Tracking Accuracy) for spatial grounding
  - mIoU, R@1/5/10 @0.1/0.3/0.5 for temporal grounding
  - m-HIoU  (competition primary metric = mean of HOTA and mIoU)
  - Majority-voting ID mapping for bridging spatial and temporal evaluation
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def iou_1d(pred_start: float, pred_end: float,
           gt_start: float,   gt_end: float) -> float:
    inter = max(0.0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    return inter / union if union > 0 else 0.0


def iou_2d(boxA: List[float], boxB: List[float]) -> float:
    """Boxes in [x1, y1, x2, y2] format."""
    xA = max(boxA[0], boxB[0]);  yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]);  yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# HOTA computation
# ---------------------------------------------------------------------------

class HOTAEvaluator:
    """
    Per-query HOTA, averaged across queries and localisation thresholds α ∈ [0.05, 0.95, 0.05].

    Usage:
        ev = HOTAEvaluator()
        ev.update(query_id, pred_tracks, gt_tracks)   # per video/query
        results = ev.compute()
    """

    ALPHA_RANGE = np.arange(0.05, 0.96, 0.05)

    def __init__(self):
        self.reset()

    def reset(self):
        self._queries: Dict[str, dict] = {}

    def update(self, query_id: str,
               pred_tracks: Dict[int, List[List[float]]],
               gt_tracks:   Dict[int, List[List[float]]]):
        """
        Args:
            query_id   : unique string identifying this (video, query) pair
            pred_tracks: {track_id: [[frame, x1, y1, x2, y2], ...]}
            gt_tracks  : same format
        """
        self._queries[query_id] = {
            "pred": pred_tracks,
            "gt":   gt_tracks
        }

    def _hota_alpha(self, pred: Dict, gt: Dict, alpha: float) -> Dict[str, float]:
        """Compute DetA, AssA, LocA at a single threshold."""
        # Build frame-level correspondences
        all_frames = set()
        for boxes in list(pred.values()) + list(gt.values()):
            for b in boxes:
                all_frames.add(int(b[0]))

        TP = FP = FN = 0
        assoc_tp = 0
        assoc_total = 0

        # ID mapping via majority vote (simplified)
        id_map: Dict[int, int] = {}   # gt_id → pred_id
        for gt_id, gt_boxes in gt.items():
            match_counts: Dict[int, int] = defaultdict(int)
            for gb in gt_boxes:
                frame = int(gb[0]);  gb_box = gb[1:]
                for pred_id, pb_list in pred.items():
                    for pb in pb_list:
                        if int(pb[0]) == frame:
                            if iou_2d(pb[1:], gb_box) >= alpha:
                                match_counts[pred_id] += 1
            if match_counts:
                id_map[gt_id] = max(match_counts, key=match_counts.get)

        for frame in all_frames:
            gt_in_frame = {gid: [b for b in boxes if int(b[0]) == frame]
                           for gid, boxes in gt.items()}
            pred_in_frame = {pid: [b for b in boxes if int(b[0]) == frame]
                             for pid, boxes in pred.items()}

            matched_gt   = set()
            matched_pred = set()
            for gt_id, gb_list in gt_in_frame.items():
                if not gb_list:
                    continue
                best_pred = id_map.get(gt_id)
                if best_pred is None or best_pred not in pred_in_frame:
                    FN += 1
                    continue
                pb_list = pred_in_frame[best_pred]
                if not pb_list:
                    FN += 1
                    continue
                iou = iou_2d(pb_list[0][1:], gb_list[0][1:])
                if iou >= alpha:
                    TP += 1
                    matched_gt.add(gt_id)
                    matched_pred.add(best_pred)
                    # Association accuracy
                    assoc_tp    += iou
                    assoc_total += 1
                else:
                    FN += 1

            # False positives: predicted but not matched
            for pid in pred_in_frame:
                if pid not in matched_pred and pred_in_frame[pid]:
                    FP += 1

        DetA = TP / max(TP + FP + FN, 1)
        AssA = assoc_tp / max(assoc_total, 1)
        LocA = assoc_tp / max(TP, 1)
        HOTA = np.sqrt(DetA * AssA)

        return {"HOTA": HOTA, "DetA": DetA, "AssA": AssA, "LocA": LocA}

    def compute(self) -> Dict[str, float]:
        per_alpha = []
        DetA_vals = []
        AssA_vals = []
        LocA_vals = []

        for qid, data in self._queries.items():
            q_hota = []
            for alpha in self.ALPHA_RANGE:
                r = self._hota_alpha(data["pred"], data["gt"], alpha)
                q_hota.append(r["HOTA"])
                if np.isclose(alpha, 0.5):
                    DetA_vals.append(r["DetA"])
                    AssA_vals.append(r["AssA"])
                    LocA_vals.append(r["LocA"])
            per_alpha.append(np.mean(q_hota))

        return {
            "HOTA": float(np.mean(per_alpha)),
            "DetA": float(np.mean(DetA_vals)) if DetA_vals else 0.0,
            "AssA": float(np.mean(AssA_vals)) if AssA_vals else 0.0,
            "LocA": float(np.mean(LocA_vals)) if LocA_vals else 0.0,
        }


# ---------------------------------------------------------------------------
# Temporal grounding metrics
# ---------------------------------------------------------------------------

class TemporalGroundingEvaluator:
    """
    Computes Recall@{1,5,10} × IoU@{0.1, 0.3, 0.5}, mAP, and mIoU.

    Usage:
        ev = TemporalGroundingEvaluator()
        ev.update(pred_moments, gt_moment)
        results = ev.compute()
    """

    IOU_THRESHOLDS = [0.1, 0.3, 0.5]
    RECALL_KS      = [1, 5, 10]

    def __init__(self):
        self.reset()

    def reset(self):
        self._records: List[dict] = []

    def update(self, pred_moments: List[Tuple[float, float, float]],
               gt_moment: Tuple[float, float]):
        """
        Args:
            pred_moments: list of (start, end, score) – sorted desc by score
            gt_moment   : (start, end)
        """
        self._records.append({"pred": pred_moments, "gt": gt_moment})

    def compute(self) -> Dict[str, float]:
        metrics: Dict[str, List[float]] = defaultdict(list)
        iou_all: List[float] = []

        for rec in self._records:
            preds  = sorted(rec["pred"], key=lambda x: x[2], reverse=True)
            gs, ge = rec["gt"]

            ious   = [iou_1d(p[0], p[1], gs, ge) for p in preds]
            best   = max(ious) if ious else 0.0
            iou_all.append(best)

            for k in self.RECALL_KS:
                top_k = ious[:k]
                for thr in self.IOU_THRESHOLDS:
                    hit = float(any(v >= thr for v in top_k))
                    metrics[f"R{k}@{thr}"].append(hit)

        # mAP: mean over queries of AP (simplified: area under precision-recall curve)
        ap_list: List[float] = []
        for rec in self._records:
            preds  = sorted(rec["pred"], key=lambda x: x[2], reverse=True)
            gs, ge = rec["gt"]
            hits   = [iou_1d(p[0], p[1], gs, ge) >= 0.5 for p in preds]
            if not hits:
                ap_list.append(0.0)
                continue
            prec_at = []
            n_hit   = 0
            for i, h in enumerate(hits):
                if h:
                    n_hit += 1
                    prec_at.append(n_hit / (i + 1))
            ap_list.append(np.mean(prec_at) if prec_at else 0.0)

        results = {k: float(np.mean(v)) for k, v in metrics.items()}
        results["mAP"]  = float(np.mean(ap_list))
        results["mIoU"] = float(np.mean(iou_all))
        return results


# ---------------------------------------------------------------------------
# Combined SVAG Evaluator
# ---------------------------------------------------------------------------

class SVAGEvaluator:
    """
    Top-level evaluator that combines spatial (HOTA) and temporal metrics
    and computes the competition's primary metric m-HIoU.

    Usage:
        evaluator = SVAGEvaluator()
        evaluator.update_spatial(query_id, pred_tracks, gt_tracks)
        evaluator.update_temporal(pred_moments, gt_moment)
        results = evaluator.compute()
    """

    def __init__(self):
        self.hota_eval     = HOTAEvaluator()
        self.temporal_eval = TemporalGroundingEvaluator()

    def update_spatial(self, query_id, pred_tracks, gt_tracks):
        self.hota_eval.update(query_id, pred_tracks, gt_tracks)

    def update_temporal(self, pred_moments, gt_moment):
        self.temporal_eval.update(pred_moments, gt_moment)

    def compute(self) -> Dict[str, float]:
        spatial  = self.hota_eval.compute()
        temporal = self.temporal_eval.compute()

        m_hiou = (spatial["HOTA"] + temporal["mIoU"]) / 2.0

        return {
            **{f"spatial/{k}": v for k, v in spatial.items()},
            **{f"temporal/{k}": v for k, v in temporal.items()},
            "m-HIoU": m_hiou,
        }
