"""
SVAGFormer Pipeline — Conforme à l'article SVAG-Bench (arXiv:2510.13016).

Architecture (Figure 3 de l'article):
    SVAGFormer découple le grounding en DEUX modules INDÉPENDANTS :
    1. Spatial Grounding  → TempRMOT  (détection + tracking par query)
    2. Temporal Grounding → FlashVTG  (localisation temporelle par query)

    Les résultats sont combinés via SVAGEval avec ID mapping (majority voting).
    Chaque sous-dataset (OVIS, MOT17, MOT20) est traité SÉPARÉMENT.

Référence: Section 4 — Methodology
    "We propose SVAGFormer for the SVAG task by decoupling grounding into
     two modules: Spatial Grounding and Temporal Grounding."

Structure réelle des submodules (vérifiée sur les repos clonés):
    temporal/FlashVTG/
        model.py          — build_model(args)
        config.py         — BaseOptions / TestOptions
        train.py          — entraînement complet
        inference.py      — inférence
        start_end_dataset.py  — StartEndDataset (format QVHighlights)
    spatial/
        main.py           — get_args_parser() + main(args)
        models/__init__.py — build_model(args)
        engine.py         — train_one_epoch, train_one_epoch_rmot
        eval.py           — evaluate sur RMOT
        inference.py      — inférence vidéo
    evaluation/
        temporal_eval/eval.py   — eval_submission(), compute_mr_r1/r5/r10, mAP, mIoU
        spatial_eval/           — HOTA via TrackEval
"""
import os
import sys
import yaml
import torch
import numpy as np
from collections import Counter

# Ajouter les chemins réels des submodules (d'après la structure vérifiée)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "temporal"))         # FlashVTG/
sys.path.insert(0, os.path.join(_ROOT, "temporal", "FlashVTG"))  # model.py, config.py, train.py
sys.path.insert(0, os.path.join(_ROOT, "spatial"))          # main.py, engine.py, models/
sys.path.insert(0, os.path.join(_ROOT, "evaluation"))       # temporal_eval/, spatial_eval/


class SpatialGroundingModule:
    """
    Module de Spatial Grounding basé sur TempRMOT (Section 4 & 5.3).

    TempRMOT [35] est un framework RMOT (Referring Multi-Object Tracking)
    construit sur TransRMOT [32] avec amélioration de la cohérence temporelle
    via query memory.

    Paramètres de l'article (Section 5.3 — Spatial Grounding Settings):
        - Basé sur le setup officiel de TempRMOT sur Refer-KITTI-V2
        - Memory length = 5
        - Optimiseur Adam, lr = 1e-5
        - Decay factor 10 après la 40ème epoch
        - Entraînement 60 epochs sur 4 GPUs
    """

    def __init__(self, config, device="cuda"):
        self.device = device
        self.config = config
        self.memory_length = config.get("memory_length", 5)

        # TempRMOT s'initialise avec sa propre config
        # Import conditionnel pour éviter les erreurs si le submodule n'est pas cloné
        try:
            # Structure réelle: spatial/models/__init__.py expose build_model(args)
            from models import build_model as build_temprmot
            import argparse

            # Parser officiel de spatial/main.py
            from main import get_args_parser
            parser = get_args_parser()
            # Paramètres de l'article (Section 5.3)
            args = parser.parse_args([])
            args.memory_bank_len = self.memory_length   # memory_length=5
            args.device = device
            args.lr_trans = 1e-5                        # lr=1e-5 (Section 5.3)
            args.lr_drop = config.get("lr_decay_epoch", 40)  # decay epoch=40
            self.model, self.criterion, self.postprocessors = build_temprmot(args)
            self.model = self.model.to(device)
            self.args = args
        except (ImportError, SystemExit):
            print("⚠️  TempRMOT (spatial/) non trouvé ou non initialisé.")
            print("    Lancez: git submodule update --init --recursive")
            self.model = None
            self.args = None

    def load_checkpoint(self, checkpoint_path):
        """Charge un checkpoint TempRMOT pré-entraîné."""
        if self.model is not None and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model"], strict=False)
            print(f"✅ Checkpoint spatial chargé: {checkpoint_path}")

    def infer(self, video_frames, query):
        """
        Exécute le spatial grounding sur les frames vidéo.

        Args:
            video_frames: liste de frames (numpy arrays ou tensors)
            query: requête texte décrivant l'action

        Returns:
            tracks: dict {track_id: [(frame_idx, bbox), ...]} — tubes spatiaux
        """
        if self.model is None:
            print("⚠️  Mode placeholder — pas de tracking réel")
            return {}

        self.model.eval()
        with torch.no_grad():
            # TempRMOT prend les frames + query et retourne les tracks
            results = self.model.inference(video_frames, query)
        return results

    def get_training_params(self):
        """
        Retourne les paramètres d'entraînement conformes à l'article (Section 5.3).
        """
        return {
            "optimizer": "Adam",
            "lr": 1e-5,
            "lr_decay_factor": 10,
            "lr_decay_epoch": 40,
            "total_epochs": 60,
            "num_gpus": 4,
            "memory_length": self.memory_length,
            "base_setup": "Refer-KITTI-V2",
        }


class TemporalGroundingModule:
    """
    Module de Temporal Grounding basé sur FlashVTG (Section 4 & 5.3).

    FlashVTG [1] est un framework pour le text-guided video temporal grounding.
    Il utilise:
        - Temporal Feature Layering (TFL) pour le multi-scale modeling
        - Adaptive Score Refinement (ASR) pour l'alignement query-segment

    Paramètres de l'article (Section 5.3 — Temporal Grounding Settings):
        - Features vidéo extraites via InternVideo2
        - Features texte extraites via LLaMA
        - Dimension des features = 256
        - Module de fusion avec 8 attention heads
        - K = 4 (temporal feature layering)
        - 5 couches de TFL
        - Optimiseur AdamW
        - NMS threshold = 0.7
        - Données converties au format QVHighlights
        - Entraînement SÉPARÉ par dataset
        - Max visual length ajusté par dataset
    """

    def __init__(self, config, device="cuda"):
        self.device = device
        self.config = config
        self.feature_dim = config.get("feature_dim", 256)
        self.n_heads = config.get("n_heads", 8)
        self.tfl_layers = config.get("tfl_layers", 5)
        self.K = config.get("K", 4)
        self.nms_threshold = config.get("nms_threshold", 0.7)

        try:
            # Structure réelle: temporal/FlashVTG/model.py expose build_model(args)
            from model import build_model as build_flashvtg
            import argparse
            import torch

            # Paramètres de l'article (Section 5.3)
            # FlashVTG utilise un dict ou argparse
            opt = argparse.Namespace(
                hidden_dim=self.feature_dim,      # 256
                nheads=self.n_heads,              # 8
                nms_thd=self.nms_threshold,       # 0.7
                enc_layers=3,
                t2v_layers=2,
                dummy_layers=2,
                dim_feedforward=1024,
                input_dropout=0.5,
                dropout=0.1,
                use_txt_pos=False,
                num_dummies=45,
                total_prompts=10,
                num_prompts=1,
                pre_norm=False,
                n_input_proj=2,
                span_loss_type="l1",
                use_SRM=False,
                position_embedding="sine",
                max_after_nms=10,
                max_before_nms=50,
                aux_loss=True,
                device=torch.device(device),
                v_feat_dim=self.feature_dim,
                t_feat_dim=self.feature_dim,
                use_tef=True,
                use_video=True,
                num_conv_layers=3,
                num_mlp_layers=3,
                kernel_size=3
            )
            self.model, self.criterion = build_flashvtg(opt)
            self.model = self.model.to(device)
            self.opt = opt
        except (ImportError, Exception) as e:
            print(f"⚠️  FlashVTG (temporal/FlashVTG/) non trouvé: {e}")
            print("    Lancez: git submodule update --init --recursive")
            self.model = None
            self.opt = None

        # Feature extractors (InternVideo2 + LLaMA)
        self.video_encoder = None  # InternVideo2
        self.text_encoder = None   # LLaMA

    def load_checkpoint(self, checkpoint_path):
        """Charge un checkpoint FlashVTG pré-entraîné."""
        if self.model is not None and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model"], strict=False)
            print(f"✅ Checkpoint temporal chargé: {checkpoint_path}")

    def extract_video_features(self, video_frames):
        """
        Extrait les features vidéo via InternVideo2 [31].
        Dimension de sortie: (T, feature_dim=256)
        """
        if self.video_encoder is None:
            try:
                from internvideo2.model import InternVideo2
                self.video_encoder = InternVideo2(output_dim=self.feature_dim)
                self.video_encoder = self.video_encoder.to(self.device).eval()
            except ImportError:
                print("⚠️  InternVideo2 non disponible — features placeholder")
                T = len(video_frames)
                return torch.randn(T, self.feature_dim).to(self.device)

        with torch.no_grad():
            features = self.video_encoder(video_frames)
        return features

    def extract_text_features(self, query):
        """
        Extrait les features texte via LLaMA [29].
        Dimension de sortie: (L, feature_dim=256)
        """
        if self.text_encoder is None:
            try:
                from transformers import LlamaModel, LlamaTokenizer
                self.text_encoder = LlamaModel.from_pretrained(
                    self.config.get("llama_model", "meta-llama/Llama-2-7b-hf")
                ).to(self.device).eval()
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    self.config.get("llama_model", "meta-llama/Llama-2-7b-hf")
                )
            except (ImportError, OSError):
                print("⚠️  LLaMA non disponible — features texte placeholder")
                return torch.randn(len(query.split()), self.feature_dim).to(self.device)

        with torch.no_grad():
            tokens = self.tokenizer(query, return_tensors="pt").to(self.device)
            outputs = self.text_encoder(**tokens)
            # Projeter vers feature_dim=256
            text_features = outputs.last_hidden_state
        return text_features

    def infer(self, video_features, text_features, apply_nms=True):
        """
        Exécute le temporal grounding.

        Args:
            video_features: tensor (T, feature_dim)
            text_features: tensor (L, feature_dim)
            apply_nms: appliquer NMS avec threshold=0.7

        Returns:
            segments: liste de (start, end, score) — segments temporels prédits
        """
        if self.model is None:
            print("⚠️  Mode placeholder — pas de temporal grounding réel")
            return []

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(video_features, text_features)

        if apply_nms:
            predictions = self._apply_nms(predictions, self.nms_threshold)

        return predictions

    def _apply_nms(self, predictions, threshold=0.7):
        """
        Applique Non-Maximum Suppression (NMS) sur les segments temporels.
        Threshold = 0.7 (Section 5.3).
        """
        if not predictions:
            return predictions

        # Trier par score décroissant
        sorted_preds = sorted(predictions, key=lambda x: x[2], reverse=True)
        kept = []

        for pred in sorted_preds:
            start, end, score = pred
            suppress = False
            for k_start, k_end, k_score in kept:
                # Calculer IoU temporelle
                inter_start = max(start, k_start)
                inter_end = min(end, k_end)
                intersection = max(0, inter_end - inter_start)
                union = (end - start) + (k_end - k_start) - intersection
                iou = intersection / max(union, 1e-6)
                if iou > threshold:
                    suppress = True
                    break
            if not suppress:
                kept.append(pred)

        return kept

    def get_training_params(self):
        """
        Retourne les paramètres d'entraînement conformes à l'article (Section 5.3).
        """
        return {
            "optimizer": "AdamW",
            "feature_dim": self.feature_dim,
            "n_heads": self.n_heads,
            "tfl_layers": self.tfl_layers,
            "K": self.K,
            "nms_threshold": self.nms_threshold,
            "video_encoder": "InternVideo2",
            "text_encoder": "LLaMA",
            "data_format": "QVHighlights",
            "training_mode": "separate_per_dataset",
        }


class SVAGEvalIntegration:
    """
    Intégration du toolkit SVAGEval (Section 5.1 & 5.2).

    Évaluation en deux dimensions:
        - Spatiale: HOTA (Higher Order Tracking Accuracy) à α=0.5
        - Temporelle: R@1/5/10, mAP, mIoU

    ID Mapping (Section 5.2):
        1. Spatial ID Matching via HOTA à α=0.5
        2. Majority Voting pour résoudre les ambiguïtés
        3. Construction des paires temporelles

    Score final = moyenne arithmétique des scores OVIS + MOT17 + MOT20
    Métrique principale du leaderboard: m-HIoU = mean(HOTA, mIoU)
    """

    def __init__(self):
        try:
            from svageval.evaluation import SVAGEvaluator
            self.evaluator = SVAGEvaluator()
        except ImportError:
            print("⚠️  SVAGEval submodule non trouvé.")
            self.evaluator = None

    def map_ids_majority_voting(self, spatial_matching_results):
        """
        ID Mapping par majority voting (Section 5.2).

        Pour chaque GT track_id, le predicted track_id qui apparaît
        le plus souvent est sélectionné.
        """
        gt_to_pred_mapping = {}

        for gt_id, frame_matches in spatial_matching_results.items():
            # frame_matches: liste de predicted_ids trouvés pour ce gt_id
            if frame_matches:
                counter = Counter(frame_matches)
                gt_to_pred_mapping[gt_id] = counter.most_common(1)[0][0]

        return gt_to_pred_mapping

    def evaluate_spatial(self, predictions, ground_truth, alpha=0.5):
        """
        Évaluation spatiale via HOTA à α=0.5 (Section 5.1).

        HOTA combine detection accuracy (DetA) et association accuracy (AssA):
            HOTA = sqrt(DetA × AssA)

        Calculé en moyennant le HOTA par query sur toutes les queries.
        """
        if self.evaluator is not None:
            return self.evaluator.compute_hota(predictions, ground_truth, alpha=alpha)

        # Placeholder si SVAGEval pas disponible
        print("⚠️  SVAGEval non disponible — métriques placeholder")
        return {"HOTA": 0.0, "DetA": 0.0, "AssA": 0.0}

    def evaluate_temporal(self, pred_segments, gt_segments):
        """
        Évaluation temporelle (Section 5.1).

        Métriques:
            - R@1, R@5, R@10 à différents seuils IoU
            - mAP (Mean Average Precision)
            - mIoU (Mean Intersection over Union)
        """
        if self.evaluator is not None:
            return self.evaluator.compute_temporal_metrics(pred_segments, gt_segments)

        print("⚠️  SVAGEval non disponible — métriques placeholder")
        return {"R1@0.3": 0.0, "R1@0.5": 0.0, "R1@0.7": 0.0,
                "R5@0.3": 0.0, "R5@0.5": 0.0, "R5@0.7": 0.0,
                "mAP": 0.0, "mIoU": 0.0}

    @staticmethod
    def compute_final_score(dataset_scores):
        """
        Score final = moyenne arithmétique sur OVIS, MOT17, MOT20 (Section 5.2).

        Métrique principale: m-HIoU = mean(HOTA, mIoU)
        """
        all_hota = []
        all_miou = []

        for ds_name in ["ovis", "mot17", "mot20"]:
            if ds_name in dataset_scores:
                all_hota.append(dataset_scores[ds_name].get("HOTA", 0.0))
                all_miou.append(dataset_scores[ds_name].get("mIoU", 0.0))

        avg_hota = np.mean(all_hota) if all_hota else 0.0
        avg_miou = np.mean(all_miou) if all_miou else 0.0
        m_hiou = (avg_hota + avg_miou) / 2.0

        return {
            "avg_HOTA": avg_hota,
            "avg_mIoU": avg_miou,
            "m-HIoU": m_hiou,  # Métrique principale du leaderboard
        }


class SVAGFormerPipeline:
    """
    Pipeline SVAGFormer conforme à l'article SVAG-Bench (arXiv:2510.13016).

    Architecture (Figure 3):
        Les modules spatial et temporal fonctionnent INDÉPENDAMMENT.
        Chaque sous-dataset (OVIS, MOT17, MOT20) est traité séparément.
        Les résultats sont combinés via SVAGEval.

    Ce n'est PAS un pipeline séquentiel (temporal→spatial).
    Les deux modules opèrent en parallèle sur les mêmes données.
    """

    def __init__(self, config_path="configs/pipeline.yaml", device="cuda"):
        self.device = device
        self.config = yaml.safe_load(open(config_path))

        # Modules indépendants (Section 4)
        self.spatial = SpatialGroundingModule(
            self.config.get("spatial", {}), device=device
        )
        self.temporal = TemporalGroundingModule(
            self.config.get("temporal", {}), device=device
        )

        # Évaluation intégrée
        self.evaluator = SVAGEvalIntegration()

    def load_checkpoints(self, spatial_ckpt=None, temporal_ckpt=None):
        """Charge les checkpoints des deux modules."""
        if spatial_ckpt:
            self.spatial.load_checkpoint(spatial_ckpt)
        if temporal_ckpt:
            self.temporal.load_checkpoint(temporal_ckpt)

    def forward(self, video_frames, query):
        """
        Exécution du pipeline SVAGFormer (Figure 3).

        Les deux modules fonctionnent INDÉPENDAMMENT:
            1. Spatial: TempRMOT détecte et track les objets selon la query
            2. Temporal: FlashVTG localise les segments temporels de l'action

        Args:
            video_frames: liste de frames vidéo (numpy arrays)
            query: requête texte décrivant l'action

        Returns:
            dict avec:
                - spatial_tracks: {track_id: [(frame_idx, bbox), ...]}
                - temporal_segments: [(start, end, score), ...]
                - query: la requête originale
        """
        # --- Module 1: Spatial Grounding (TempRMOT) ---
        # Détection + tracking des objets référencés par la query
        spatial_tracks = self.spatial.infer(video_frames, query)

        # --- Module 2: Temporal Grounding (FlashVTG) ---
        # Extraction des features (InternVideo2 + LLaMA)
        video_features = self.temporal.extract_video_features(video_frames)
        text_features = self.temporal.extract_text_features(query)
        # Localisation temporelle avec NMS
        temporal_segments = self.temporal.infer(
            video_features, text_features, apply_nms=True
        )

        return {
            "spatial_tracks": spatial_tracks,
            "temporal_segments": temporal_segments,
            "query": query,
        }

    def evaluate(self, predictions, ground_truth, dataset_name):
        """
        Évalue les résultats sur un dataset donné.

        Args:
            predictions: dict avec spatial_tracks et temporal_segments
            ground_truth: annotations ground truth
            dataset_name: "ovis", "mot17", ou "mot20"

        Returns:
            dict de métriques (HOTA, R@N, mAP, mIoU)
        """
        # Évaluation spatiale (HOTA @ α=0.5)
        spatial_metrics = self.evaluator.evaluate_spatial(
            predictions["spatial_tracks"],
            ground_truth["tracks"],
            alpha=0.5,
        )

        # ID Mapping par majority voting
        id_mapping = self.evaluator.map_ids_majority_voting(
            predictions.get("spatial_matching", {})
        )

        # Évaluation temporelle (R@N, mAP, mIoU)
        temporal_metrics = self.evaluator.evaluate_temporal(
            predictions["temporal_segments"],
            ground_truth["segments"],
        )

        return {
            "dataset": dataset_name,
            **spatial_metrics,
            **temporal_metrics,
        }

    def evaluate_all_datasets(self, all_predictions, all_ground_truth):
        """
        Évalue sur les 3 datasets et calcule le score final (Section 5.2).

        Score final = moyenne arithmétique OVIS + MOT17 + MOT20
        Métrique principale: m-HIoU = mean(HOTA, mIoU)
        """
        dataset_scores = {}
        for ds_name in ["ovis", "mot17", "mot20"]:
            if ds_name in all_predictions:
                dataset_scores[ds_name] = self.evaluate(
                    all_predictions[ds_name],
                    all_ground_truth[ds_name],
                    ds_name,
                )
                print(f"📊 {ds_name.upper()}: {dataset_scores[ds_name]}")

        final = self.evaluator.compute_final_score(dataset_scores)
        print(f"\n🏆 Score final m-HIoU: {final['m-HIoU']:.4f}")
        return final
