"""
Dataset utilities pour SVAGFormer — Conforme à SVAG-Bench.

Gère le chargement et la conversion des datasets OVIS, MOT17, MOT20.

Conversion au format QVHighlights (requis par FlashVTG, Section 5.3):
    Chaque entrée contient:
        - qid: identifiant de la query
        - query: texte de la requête en langage naturel
        - vid: identifiant de la vidéo
        - duration: durée de la vidéo
        - relevant_windows: [[start, end], ...] — segments temporels GT

SVAG-Bench (Section 3):
    - 688 vidéos, 19590 annotated records, 903 unique verbs
    - Sources: MOT17 [4], MOT20 [5], OVIS [22]
    - Requêtes action-centric (pas appearance-based)
    - Moyenne 28.47 queries par vidéo
    - Longueur moyenne des queries: 9.58 mots
"""
import os
import json
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset


class SVAGDataset(Dataset):
    """
    Dataset générique pour SVAG-Bench.

    Chaque échantillon contient:
        - video_id: identifiant de la vidéo
        - frames: liste de frames (chargées à la demande)
        - queries: liste de requêtes texte avec annotations
        - tracks: annotations spatiales (bbox par frame par track_id)
        - segments: annotations temporelles (start, end par query)
    """

    def __init__(self, root, dataset_name="mot17", split="train", transform=None):
        self.root = root
        self.dataset_name = dataset_name
        self.split = split
        self.transform = transform
        self.samples = self._load_annotations()

    def _load_annotations(self):
        """Charge les annotations SVAG-Bench."""
        ann_path = os.path.join(self.root, "annotations")
        samples = []

        if os.path.exists(ann_path):
            for f in sorted(os.listdir(ann_path)):
                if f.endswith(".json"):
                    with open(os.path.join(ann_path, f)) as fp:
                        data = json.load(fp)
                        # Chaque annotation SVAG contient:
                        # - video_id, query, track_ids, temporal_segment, spatial_boxes
                        samples.append(data)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Charger les frames vidéo si nécessaire
        if "video_path" in sample:
            video_path = os.path.join(self.root, sample["video_path"])
            if os.path.exists(video_path):
                frames = self._load_frames(video_path)
                sample["frames"] = frames

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _load_frames(self, video_path):
        """Charge les frames depuis un fichier vidéo ou un dossier d'images."""
        frames = []
        if os.path.isdir(video_path):
            # Dossier d'images (format MOT)
            for img_name in sorted(os.listdir(video_path)):
                if img_name.endswith((".jpg", ".png")):
                    img = cv2.imread(os.path.join(video_path, img_name))
                    if img is not None:
                        frames.append(img)
        else:
            # Fichier vidéo
            cap = cv2.VideoCapture(video_path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
        return frames


class MOT17Dataset(SVAGDataset):
    """
    Dataset MOT17 pour SVAG-Bench [4].
    Scènes urbaines avec piétons, vidéos longues (centaines à milliers de frames).
    """
    def __init__(self, root="data/mot17", **kwargs):
        super().__init__(root, dataset_name="mot17", **kwargs)


class MOT20Dataset(SVAGDataset):
    """
    Dataset MOT20 pour SVAG-Bench [5].
    Scènes denses avec beaucoup d'objets, vidéos très longues.
    """
    def __init__(self, root="data/mot20", **kwargs):
        super().__init__(root, dataset_name="mot20", **kwargs)


class OVISDataset(SVAGDataset):
    """
    Dataset OVIS pour SVAG-Bench [22].
    Vidéo Instance Segmentation avec occlusion, 25 catégories.
    Vidéos plus courtes mais avec occlusion complexe.
    """
    def __init__(self, root="data/ovis", **kwargs):
        super().__init__(root, dataset_name="ovis", **kwargs)


def convert_to_qvhighlights_format(data_root, output_dir, dataset_name):
    """
    Convertit les annotations SVAG-Bench au format QVHighlights.

    Requis par FlashVTG (Section 5.3):
        "we convert our data into QVHighlights format"

    Format QVHighlights (jsonl):
        {"qid": 0, "query": "...", "vid": "video_id",
         "duration": 150.0, "relevant_windows": [[10.0, 25.0]]}
    """
    ann_path = os.path.join(data_root, "annotations")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"Annotations non trouvées: {ann_path}")

    output_file = os.path.join(output_dir, f"{dataset_name}_qvhighlights.jsonl")
    qid = 0

    with open(output_file, "w") as out_f:
        for f in sorted(os.listdir(ann_path)):
            if not f.endswith(".json"):
                continue

            with open(os.path.join(ann_path, f)) as fp:
                data = json.load(fp)

            # Extraire les informations SVAG et convertir au format QVH
            video_id = data.get("video_id", f.replace(".json", ""))
            duration = data.get("duration", 0.0)

            # Chaque query dans l'annotation
            queries = data.get("queries", [data]) if "queries" in data else [data]
            for q_data in queries:
                query_text = q_data.get("query", "")
                temporal_segment = q_data.get("temporal_segment", [0, duration])

                qvh_entry = {
                    "qid": qid,
                    "query": query_text,
                    "vid": video_id,
                    "duration": duration,
                    "relevant_windows": [temporal_segment],
                }
                out_f.write(json.dumps(qvh_entry) + "\n")
                qid += 1

    print(f"   ✅ Converti {qid} queries → {output_file}")
    return output_file


def convert_to_rmot_format(data_root, output_dir, dataset_name):
    """
    Convertit les annotations SVAG-Bench au format RMOT.

    Requis par TempRMOT (Section 5.3):
        Format compatible avec le setup Refer-KITTI-V2.

    Format RMOT:
        - expressions.json: {video_id: {obj_id: {"exp": "query text"}}}
        - annotations au format COCO avec tracking
    """
    ann_path = os.path.join(data_root, "annotations")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"Annotations non trouvées: {ann_path}")

    expressions = {}
    for f in sorted(os.listdir(ann_path)):
        if not f.endswith(".json"):
            continue
        with open(os.path.join(ann_path, f)) as fp:
            data = json.load(fp)

        video_id = data.get("video_id", f.replace(".json", ""))
        queries = data.get("queries", [data]) if "queries" in data else [data]

        expressions[video_id] = {}
        for i, q_data in enumerate(queries):
            track_ids = q_data.get("track_ids", [i])
            query_text = q_data.get("query", "")
            for tid in track_ids:
                expressions[video_id][str(tid)] = {"exp": query_text}

    expressions_file = os.path.join(output_dir, f"{dataset_name}_expressions.json")
    with open(expressions_file, "w") as f:
        json.dump(expressions, f, indent=2)

    print(f"   ✅ Expressions RMOT sauvegardées → {expressions_file}")
    return expressions_file
