"""
Entraînement complet du pipeline SVAGFormer — Conforme à SVAG-Bench.

Selon l'article (Section 5.3):
    - Les modules spatial et temporal sont entraînés SÉPARÉMENT
    - Chaque dataset (OVIS, MOT17, MOT20) est entraîné SÉPARÉMENT
    - "We train our model separately on OVIS, MOT17, and MOT20,
       rather than jointly across all datasets."

Ce script orchestre l'entraînement des deux modules sur chaque dataset.
"""
import os
import sys
import yaml
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.train_temporal import train_temporal
from scripts.train_spatial import train_spatial


def train_full_pipeline(pipeline_config_path="configs/pipeline.yaml"):
    """
    Entraîne le pipeline complet SVAGFormer.

    Ordre d'entraînement pour chaque dataset:
        1. TempRMOT (spatial) — 60 epochs
        2. FlashVTG (temporal) — séparé

    L'article entraîne chaque module indépendamment sur chaque dataset.
    """
    config = yaml.safe_load(open(pipeline_config_path))
    datasets = config.get("datasets", ["ovis", "mot17", "mot20"])

    print("=" * 60)
    print("  SVAGFormer — Entraînement Complet")
    print("  Conforme à SVAG-Bench (arXiv:2510.13016)")
    print("=" * 60)
    print(f"  Datasets: {', '.join(d.upper() for d in datasets)}")
    print(f"  Mode: Entraînement SÉPARÉ par dataset (Section 5.3)")
    print("=" * 60)

    for ds_name in datasets:
        ds_config_path = f"configs/{ds_name}.yaml"

        if not os.path.exists(ds_config_path):
            print(f"\n⚠️  Config non trouvée: {ds_config_path}, skip {ds_name}")
            continue

        # --- Entraînement Spatial (TempRMOT) ---
        print(f"\n{'='*60}")
        print(f"  [{ds_name.upper()}] Phase 1: Spatial Grounding (TempRMOT)")
        print(f"{'='*60}")
        train_spatial(ds_name, ds_config_path, pipeline_config_path)

        # --- Entraînement Temporal (FlashVTG) ---
        print(f"\n{'='*60}")
        print(f"  [{ds_name.upper()}] Phase 2: Temporal Grounding (FlashVTG)")
        print(f"{'='*60}")
        train_temporal(ds_name, ds_config_path, pipeline_config_path)

    print("\n" + "=" * 60)
    print("  ✅ Entraînement complet terminé!")
    print("  Prochaine étape: évaluer avec scripts/evaluate.py")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train full SVAGFormer pipeline")
    parser.add_argument("--pipeline-config", type=str, default="configs/pipeline.yaml")
    args = parser.parse_args()

    train_full_pipeline(args.pipeline_config)
