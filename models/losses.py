"""
Loss functions for SVAGFormer.

Spatial loss  : Hungarian matching + L1 box regression + focal classification loss
Temporal loss : L1 regression on (start, end) + binary cross-entropy on confidence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def box_iou_1d(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """1-D IoU between moment intervals.  Both tensors: (N, 2) [start, end]."""
    inter_start = torch.max(pred[:, 0], gt[:, 0])
    inter_end   = torch.min(pred[:, 1], gt[:, 1])
    inter = (inter_end - inter_start).clamp(min=0)
    union = (pred[:, 1] - pred[:, 0]) + (gt[:, 1] - gt[:, 0]) - inter
    return inter / union.clamp(min=1e-6)


def generalized_box_iou(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Generalised IoU for axis-aligned 2-D boxes.
    pred, gt: (N, 4) in cx-cy-w-h format, already converted to x1y1x2y2 inside.
    """
    def to_xyxy(b):
        x1 = b[..., 0] - b[..., 2] / 2
        y1 = b[..., 1] - b[..., 3] / 2
        x2 = b[..., 0] + b[..., 2] / 2
        y2 = b[..., 1] + b[..., 3] / 2
        return torch.stack([x1, y1, x2, y2], dim=-1)

    pred = to_xyxy(pred)
    gt   = to_xyxy(gt)

    inter_x1 = torch.max(pred[..., 0], gt[..., 0])
    inter_y1 = torch.max(pred[..., 1], gt[..., 1])
    inter_x2 = torch.min(pred[..., 2], gt[..., 2])
    inter_y2 = torch.min(pred[..., 3], gt[..., 3])
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    area_pred = (pred[..., 2] - pred[..., 0]) * (pred[..., 3] - pred[..., 1])
    area_gt   = (gt[..., 2]   - gt[..., 0])   * (gt[..., 3]   - gt[..., 1])
    union = area_pred + area_gt - inter

    iou = inter / union.clamp(1e-6)

    enc_x1 = torch.min(pred[..., 0], gt[..., 0])
    enc_y1 = torch.min(pred[..., 1], gt[..., 1])
    enc_x2 = torch.max(pred[..., 2], gt[..., 2])
    enc_y2 = torch.max(pred[..., 3], gt[..., 3])
    enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)

    giou = iou - (enc_area - union) / enc_area.clamp(1e-6)
    return giou


# ---------------------------------------------------------------------------
# Spatial Loss (Hungarian matching, per-frame)
# ---------------------------------------------------------------------------

class SpatialLoss(nn.Module):
    """
    Bipartite matching loss for referred multi-object detection.

    For each frame, matches predicted queries to ground-truth boxes using the
    Hungarian algorithm, then computes:
      - Focal classification loss
      - L1 box regression loss
      - GIoU box regression loss
    """

    def __init__(self, cost_cls: float = 1.0, cost_l1: float = 5.0,
                 cost_giou: float = 2.0, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.cost_cls  = cost_cls
        self.cost_l1   = cost_l1
        self.cost_giou = cost_giou
        self.alpha = alpha
        self.gamma = gamma

    @torch.no_grad()
    def _match(self, pred_boxes: torch.Tensor, pred_scores: torch.Tensor,
               gt_boxes: torch.Tensor) -> Tuple:
        """Hungarian matching for one frame, one sample."""
        Nq = pred_boxes.shape[0]
        Ng = gt_boxes.shape[0]

        # Classification cost (negative probability of foreground)
        prob = pred_scores.sigmoid().squeeze(-1)              # (Nq,)
        cost_cls = -prob.unsqueeze(1).expand(-1, Ng)         # (Nq, Ng)

        # L1 box cost
        cost_l1 = torch.cdist(pred_boxes, gt_boxes, p=1)    # (Nq, Ng)

        # GIoU cost
        giou = generalized_box_iou(
            pred_boxes.unsqueeze(1).expand(-1, Ng, -1).reshape(-1, 4),
            gt_boxes.unsqueeze(0).expand(Nq, -1, -1).reshape(-1, 4)
        ).reshape(Nq, Ng)
        cost_giou = -giou

        C = (self.cost_cls * cost_cls +
             self.cost_l1  * cost_l1  +
             self.cost_giou * cost_giou)

        row_ind, col_ind = linear_sum_assignment(C.cpu().numpy())
        return (torch.as_tensor(row_ind, dtype=torch.long),
                torch.as_tensor(col_ind, dtype=torch.long))

    def focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = pred.sigmoid()
        ce   = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        pt   = prob * target + (1 - prob) * (1 - target)
        focal_weight = (self.alpha * target + (1 - self.alpha) * (1 - target)) * (1 - pt) ** self.gamma
        return (focal_weight * ce).mean()

    def forward(self, bboxes: torch.Tensor, scores: torch.Tensor,
                targets: list) -> torch.Tensor:
        """
        Args:
            bboxes  : (B, T, Nq, 4)
            scores  : (B, T, Nq, 1)
            targets : list of B dicts with 'boxes' ground truth
        Returns:
            scalar loss
        """
        B, T, Nq, _ = bboxes.shape
        total_loss = bboxes.new_zeros(1)
        n_pairs = 0

        for b in range(B):
            gt_raw = targets[b].get("boxes", {})
            if not gt_raw:
                continue

            # Flatten GT boxes: list of tensors per track
            gt_per_frame: Dict[int, List] = {}
            for tid, frame_boxes in gt_raw.items():
                for fb in frame_boxes:
                    fid = int(fb[0])
                    box = fb[1:]                            # [x, y, w, h]
                    gt_per_frame.setdefault(fid, []).append(box)

            for t in range(T):
                gt_boxes = gt_per_frame.get(t, [])
                if not gt_boxes:
                    continue
                gt_t  = torch.tensor(gt_boxes, device=bboxes.device, dtype=torch.float32)
                pb_t  = bboxes[b, t]                       # (Nq, 4)
                ps_t  = scores[b, t]                       # (Nq, 1)

                row_ind, col_ind = self._match(pb_t, ps_t, gt_t)

                # Classification targets
                cls_target = torch.zeros(Nq, 1, device=bboxes.device)
                cls_target[row_ind] = 1.0

                loss_cls  = self.focal_loss(ps_t, cls_target)
                loss_l1   = F.l1_loss(pb_t[row_ind], gt_t[col_ind])
                loss_giou = (1 - generalized_box_iou(pb_t[row_ind], gt_t[col_ind])).mean()

                total_loss = total_loss + loss_cls + 5.0 * loss_l1 + 2.0 * loss_giou
                n_pairs += 1

        return total_loss / max(n_pairs, 1)


# ---------------------------------------------------------------------------
# Temporal Loss
# ---------------------------------------------------------------------------

class TemporalLoss(nn.Module):
    """
    L1 regression on (start, end) + BCE on confidence.
    Optionally adds 1-D IoU loss for tighter temporal boundaries.
    """

    def __init__(self, lambda_iou: float = 1.0):
        super().__init__()
        self.lambda_iou = lambda_iou

    def forward(self, moments: torch.Tensor, tscores: torch.Tensor,
                targets: list) -> torch.Tensor:
        """
        Args:
            moments : (B, 2) predicted (start, end)
            tscores : (B, 1) confidence logits
            targets : list of dicts with 'moment' (tensor of shape (2,))
        """
        gt_moments = torch.stack([t["moment"] for t in targets]).to(moments.device)

        loss_l1   = F.l1_loss(moments, gt_moments)
        iou       = box_iou_1d(moments, gt_moments)
        loss_iou  = (1 - iou).mean()

        gt_scores = torch.ones(len(targets), 1, device=tscores.device)
        loss_bce  = F.binary_cross_entropy_with_logits(tscores, gt_scores)

        return loss_l1 + self.lambda_iou * loss_iou + 0.1 * loss_bce


# ---------------------------------------------------------------------------
# Combined Loss
# ---------------------------------------------------------------------------

class SVAGLoss(nn.Module):
    def __init__(self, w_spatial: float = 1.0, w_temporal: float = 1.0):
        super().__init__()
        self.spatial_loss  = SpatialLoss()
        self.temporal_loss = TemporalLoss()
        self.w_spatial  = w_spatial
        self.w_temporal = w_temporal

    def forward(self, outputs: dict, targets: list) -> Dict[str, torch.Tensor]:
        ls = self.spatial_loss(outputs["bboxes"], outputs["scores"], targets)
        lt = self.temporal_loss(outputs["moments"], outputs["tscores"], targets)
        total = self.w_spatial * ls + self.w_temporal * lt
        return {"loss_spatial": ls, "loss_temporal": lt, "loss_total": total}


# ---------------------------------------------------------------------------
# Missing import fix
# ---------------------------------------------------------------------------
from typing import Dict, List, Tuple
