# SVAGFormer

**Baseline implementation of SVAGFormer for the SVAG-Bench benchmark.**

> Paper: *SVAG-Bench: A Large-Scale Benchmark for Multi-Instance Spatio-temporal Video Action Grounding* (arXiv:2510.13016)

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Repository Structure](#repository-structure)
4. [Installation](#installation)
5. [Dataset Setup](#dataset-setup)
6. [Training](#training)
7. [Evaluation](#evaluation)
8. [Configuration Reference](#configuration-reference)
9. [Metrics](#metrics)
10. [Citation](#citation)

---

## Overview

SVAGFormer addresses the **Spatio-temporal Video Action Grounding (SVAG)** task, which requires a model to:

1. **Detect** all objects performing a queried action described in natural language.
2. **Track** their spatial positions (bounding boxes) across video frames.
3. **Localize** the temporal interval (start/end frame) during which those actions occur.

The framework is fully **modular**: the spatial and temporal grounding heads are decoupled and can be trained or replaced independently.

---

## Architecture

```
Input video (B, T, C, H, W)  +  Query text (list of strings)
          │                               │
          ▼                               ▼
  ┌──────────────┐               ┌───────────────┐
  │ VisualEncoder│               │  TextEncoder  │
  │ (InternVideo2│               │  (LLaMA-2,    │
  │  frozen)     │               │   frozen)     │
  └──────┬───────┘               └──────┬────────┘
         │  visual_feats (B,T,D)        │ text_feats (B,D)
         │                              │
         ├──────────────────────────────┤
         │                              │
         ▼                              ▼
  ┌─────────────────────┐    ┌──────────────────────┐
  │  Temporal Grounding │    │  Spatial Grounding   │
  │  Head (FlashVTG)    │    │  Head (TempRMOT)     │
  │                     │    │                      │
  │  TemporalFeature    │    │  Object queries      │
  │  Layering (5-scale) │    │  + TransformerDecoder│
  │  + Adaptive Score   │    │  + QueryMemory (GRU) │
  │    Refinement       │    │  + Text gating       │
  └──────────┬──────────┘    └──────────┬───────────┘
             │                          │
             ▼                          ▼
       moments (B,2)           bboxes (B,T,Nq,4)
       tscores (B,1)           scores (B,T,Nq,1)
```

### Key Modules

| Module | File | Description |
|--------|------|-------------|
| `TextEncoder` | `models/svagformer.py` | LLaMA-2 backbone (frozen) with linear projection to hidden dim D |
| `VisualEncoder` | `models/svagformer.py` | InternVideo2 backbone (frozen) producing per-frame features |
| `SpatialGroundingHead` | `models/svagformer.py` | TempRMOT-style transformer decoder with GRU query memory for temporal consistency |
| `TemporalFeatureLayering` | `models/svagformer.py` | 5-level temporal pyramid with self-attention at each scale |
| `AdaptiveScoreRefinement` | `models/svagformer.py` | Cross-scale attention for query-conditioned moment refinement |
| `TemporalGroundingHead` | `models/svagformer.py` | FlashVTG-style temporal head predicting (start, end) and confidence |
| `SVAGLoss` | `models/losses.py` | Hungarian-matched focal + GIoU spatial loss; L1 + IoU temporal loss |
| `SVAGEvaluator` | `evaluation/svageval.py` | HOTA, mIoU, Recall@K, mAP, and m-HIoU |

---

## Repository Structure

```
SVAGFormer/
├── configs/
│   └── default.yaml          # All hyperparameters
├── data/
│   └── dataset.py            # SVAGDataset + DataLoader factory
├── evaluation/
│   └── svageval.py           # SVAGEval: HOTA + temporal metrics
├── models/
│   ├── __init__.py
│   ├── svagformer.py         # Full model definition
│   └── losses.py             # Spatial + temporal losses
├── scripts/
│   ├── train.py              # Training entry point
│   └── evaluate.py           # Evaluation entry point
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-org>/SVAGFormer.git
cd SVAGFormer

# 2. Create a virtual environment (Python 3.10+)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Download spaCy model for verb parsing in annotation tools
python -m spacy download en_core_web_sm
```

> **GPU**: Training requires at least one CUDA GPU. The code falls back to CPU automatically (very slow for full runs).

---

## Dataset Setup

### Directory Layout

Organize SVAG-Bench as follows:

```
<your_dataset_root>/
├── videos/
│   ├── OVIS/
│   │   └── <sequence_name>/
│   │       ├── 000001.jpg
│   │       └── ...
│   ├── MOT17/
│   │   └── <sequence_name>/
│   │       └── ...
│   └── MOT20/
│       └── <sequence_name>/
│           └── ...
└── annotations/
    ├── train.json
    ├── val.json
    └── test.json
```

Each JSON file is a list of records with this schema:

```json
[
  {
    "video_id"   : "OVIS/dogs_001",
    "query"      : "A dog is eating the food",
    "track_ids"  : [3, 7],
    "start_frame": 12,
    "end_frame"  : 48,
    "boxes"      : {
      "3": [[12, 0.10, 0.20, 0.05, 0.10], [13, 0.11, 0.21, 0.05, 0.10]],
      "7": [[14, 0.40, 0.50, 0.06, 0.12]]
    }
  }
]
```

> Box format: `[frame_index, x_center, y_center, width, height]` — all values normalized `[0, 1]`.

### Tell the Code Where Your Data Lives

**Option A** — environment variable (recommended):
```bash
export SVAG_ROOT=/path/to/your/dataset
```

**Option B** — edit `configs/default.yaml`:
```yaml
data:
  root: "/path/to/your/dataset"
```

---

## Training

### Single GPU

```bash
python scripts/train.py --config configs/default.yaml
```

### Multi-GPU (DDP — recommended)

```bash
torchrun --nproc_per_node=4 scripts/train.py --config configs/default.yaml
```

### Resume from Checkpoint

```bash
python scripts/train.py --config configs/default.yaml --resume checkpoints/epoch_0020.pth
```

### Training Logs

The script logs to `stdout` every 50 steps:

```
2025-10-16 10:02:31 | INFO | Epoch 0 | Step 50/430 | loss_total=3.2145  loss_spatial=2.1032  loss_temporal=1.1113
```

Checkpoints are saved to `checkpoints/epoch_XXXX.pth` after every epoch (configurable via `save_dir`).

Validation metrics are printed every `val_every` epochs (default: 5).

---

## Evaluation

### Validate on Val Split

```bash
python scripts/evaluate.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/epoch_0060.pth \
    --split val
```

### Evaluate on Test Split with NMS

```bash
python scripts/evaluate.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/epoch_0060.pth \
    --split test \
    --nms \
    --nms_thr 0.7 \
    --output results/test_predictions.json
```

The script prints a full metrics table and saves predictions to the JSON file if `--output` is specified.

### Example Output

```
============================================================
Metric                          Value
------------------------------------------------------------
m-HIoU                         14.1480
spatial/HOTA                    9.1590
spatial/DetA                    4.0920
spatial/AssA                   27.6980
temporal/mIoU                  19.1370
temporal/R1@0.5                 6.4100
temporal/R5@0.5                21.0400
temporal/mAP@0.5                8.6900
============================================================
```

---

## Configuration Reference

All options live in `configs/default.yaml`:

| Key | Default | Description |
|-----|---------|-------------|
| `data.root` | `""` | Dataset root path (or use `SVAG_ROOT` env var) |
| `data.num_frames` | `32` | Frames sampled per clip |
| `model.hidden_dim` | `256` | Feature dimension D throughout the model |
| `model.text_model` | LLaMA-2-7b | HuggingFace model identifier for text encoder |
| `model.num_queries` | `100` | Number of learnable object queries in spatial head |
| `model.spatial_layers` | `6` | Transformer decoder layers in spatial head |
| `model.tfl_layers` | `5` | Temporal Feature Layering depth |
| `model.memory_length` | `5` | GRU memory look-back for TempRMOT |
| `training.epochs` | `60` | Total training epochs |
| `training.batch_size` | `4` | Per-GPU batch size |
| `training.lr` | `1e-4` | Initial learning rate (AdamW) |
| `training.lr_step` | `40` | Epoch at which LR is multiplied by `lr_gamma` |
| `training.lr_gamma` | `0.1` | LR decay factor |
| `training.grad_clip` | `1.0` | Gradient norm clipping |
| `training.val_every` | `5` | Validation frequency (epochs) |
| `eval.score_thr` | `0.3` | Minimum detection score for spatial predictions |
| `eval.nms_thr` | `0.7` | Temporal NMS IoU threshold |

---

## Metrics

### Spatial — HOTA

**Higher Order Tracking Accuracy** measures joint detection and association quality:

```
HOTA = (1/|A|) Σ_{α∈A} sqrt(DetA_α × AssA_α)
```

- Thresholds A = {0.05, 0.10, …, 0.95}
- ID mapping uses majority-voting at α = 0.5
- Final score averaged across all queries

### Temporal — Recall & mIoU

- **R@K @IoU**: fraction of queries where any of the top-K predictions has IoU ≥ threshold
- **mIoU**: mean IoU between best prediction and ground-truth moment
- **mAP**: mean average precision at IoU ≥ 0.5

### Competition Primary Metric

```
m-HIoU = (HOTA + mIoU) / 2
```

The final leaderboard score on Codabench is the arithmetic mean of m-HIoU across OVIS, MOT17, and MOT20 sub-datasets.

---

## Citation

```bibtex
@article{hannan2025svagbench,
  title   = {SVAG-Bench: A Large-Scale Benchmark for Multi-Instance
             Spatio-temporal Video Action Grounding},
  author  = {Hannan, Tanveer and Wu, Shuaicong and Weber, Mark and
             Shit, Suprosanna and Gu, Jindong and Koner, Rajat and
             O{\v{s}}ep, Aljo{\v{s}}a and Leal-Taix{\'e}, Laura and Seidl, Thomas},
  journal = {arXiv preprint arXiv:2510.13016},
  year    = {2025}
}
```
