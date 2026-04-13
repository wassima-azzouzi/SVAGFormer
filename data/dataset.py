"""
SVAG-Bench Dataset Loader.

Expected directory structure (set SVAG_ROOT in your environment or config):

    <root>/
    ├── videos/
    │   ├── OVIS/
    │   │   └── <sequence_name>/
    │   │       ├── 000001.jpg
    │   │       └── ...
    │   ├── MOT17/
    │   └── MOT20/
    └── annotations/
        ├── train.json
        ├── val.json
        └── test.json

Annotation JSON schema (one entry per record):
{
    "video_id"   : "OVIS/dogs_001",
    "query"      : "A dog is eating the food",
    "track_ids"  : [3, 7],          // ground-truth object IDs
    "start_frame": 12,
    "end_frame"  : 48,
    "boxes"      : {                 // dict[track_id -> list of [frame, x, y, w, h]]
        "3": [[12, 0.1, 0.2, 0.05, 0.1], ...],
        "7": [[14, 0.4, 0.5, 0.06, 0.12], ...]
    }
}
"""

import os
import json
import glob
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def build_transforms(split: str = "train") -> T.Compose:
    if split == "train":
        return T.Compose([
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.2, 0.2, 0.2, 0.05),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# Core Dataset
# ---------------------------------------------------------------------------

class SVAGDataset(Dataset):
    """
    Loads video clips, queries, and ground-truth annotations for the SVAG task.

    Args:
        root       : path to dataset root (leave empty → set via env SVAG_ROOT)
        split      : 'train', 'val', or 'test'
        num_frames : number of frames to sample per clip
        transforms : optional torchvision transform pipeline
    """

    def __init__(self,
                 root: str = "",
                 split: str = "train",
                 num_frames: int = 32,
                 transforms: Optional[T.Compose] = None):

        # ------------------------------------------------------------------ #
        # Root resolution                                                      #
        # ------------------------------------------------------------------ #
        if not root:
            root = os.environ.get("SVAG_ROOT", "")
        if not root:
            raise ValueError(
                "Dataset root is not set.\n"
                "Either pass root=<path> or export SVAG_ROOT=<path> before running."
            )
        self.root = Path(root)

        self.split = split
        self.num_frames = num_frames
        self.transforms = transforms or build_transforms(split)

        ann_path = self.root / "annotations" / f"{split}.json"
        if not ann_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {ann_path}")

        with open(ann_path) as f:
            self.records: List[Dict] = json.load(f)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _load_frames(self, video_id: str,
                     start_frame: int, end_frame: int) -> torch.Tensor:
        """Load and sample frames from disk, returns (T, C, H, W)."""
        frame_dir = self.root / "videos" / video_id
        pattern   = sorted(glob.glob(str(frame_dir / "*.jpg")) +
                           glob.glob(str(frame_dir / "*.png")))

        # Clip to annotated window
        pattern = pattern[start_frame: end_frame + 1]

        # Uniform sampling
        T = self.num_frames
        if len(pattern) == 0:
            # Return blank tensor if frames missing (useful during dry-run)
            return torch.zeros(T, 3, 224, 224)

        indices = torch.linspace(0, len(pattern) - 1, T).long().tolist()
        frames  = []
        for idx in indices:
            img = Image.open(pattern[idx]).convert("RGB")
            frames.append(self.transforms(img))

        return torch.stack(frames)   # (T, C, H, W)

    def _build_target(self, record: Dict, num_frames_total: int) -> Dict:
        """Build a target dict compatible with the loss functions."""
        start_norm = record["start_frame"] / max(num_frames_total - 1, 1)
        end_norm   = record["end_frame"]   / max(num_frames_total - 1, 1)
        moment = torch.tensor([start_norm, end_norm], dtype=torch.float32)

        return {
            "moment"   : moment,                         # (2,)
            "track_ids": record.get("track_ids", []),
            "boxes"    : record.get("boxes", {}),        # raw boxes for eval
            "video_id" : record["video_id"],
            "query"    : record["query"],
        }

    # ------------------------------------------------------------------ #
    # Dataset interface                                                    #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, Dict]:
        record = self.records[idx]

        frames = self._load_frames(
            video_id    = record["video_id"],
            start_frame = record.get("start_frame", 0),
            end_frame   = record.get("end_frame", -1),
        )
        target = self._build_target(record, num_frames_total=frames.shape[0])

        return frames, record["query"], target


# ---------------------------------------------------------------------------
# Collate & DataLoader factory
# ---------------------------------------------------------------------------

def collate_fn(batch):
    """Pads variable-length video clips to the same length in a batch."""
    videos, queries, targets = zip(*batch)

    max_T = max(v.shape[0] for v in videos)
    padded = []
    for v in videos:
        pad = torch.zeros(max_T - v.shape[0], *v.shape[1:])
        padded.append(torch.cat([v, pad], dim=0))

    videos_tensor = torch.stack(padded)   # (B, T, C, H, W)
    return videos_tensor, list(queries), list(targets)


def build_dataloader(root: str = "",
                     split: str = "train",
                     batch_size: int = 4,
                     num_workers: int = 4,
                     num_frames: int = 32) -> DataLoader:
    dataset = SVAGDataset(root=root, split=split, num_frames=num_frames)
    shuffle = (split == "train")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, collate_fn=collate_fn,
                      pin_memory=True)
