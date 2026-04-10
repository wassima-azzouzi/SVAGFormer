"""
Entraînement du module Temporal (FlashVTG) — Conforme à SVAG-Bench Section 5.3.

Temporal Grounding Settings (de l'article):
    - Features vidéo: InternVideo2 [31]
    - Features texte: LLaMA [29]
    - Dimension features: 256
    - Module de fusion: 8 attention heads
    - K = 4
    - Temporal Feature Layering: 5 couches
    - Optimiseur: AdamW
    - NMS threshold: 0.7
    - Format données: QVHighlights
    - Entraînement SÉPARÉ sur chaque dataset
    - Max visual length ajusté par dataset
"""
import os
import sys
import yaml
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "temporal"))


def train_temporal(dataset_name, dataset_config_path, pipeline_config_path="configs/pipeline.yaml"):
    """
    Entraîne FlashVTG sur un dataset spécifique.
    L'article entraîne SÉPARÉMENT sur OVIS, MOT17 et MOT20.
    """
    # Charger les configs
    pipeline_config = yaml.safe_load(open(pipeline_config_path))
    dataset_config = yaml.safe_load(open(dataset_config_path))
    temporal_config = pipeline_config.get("temporal", {})
    ds_temporal = dataset_config.get("temporal_training", {})

    device = pipeline_config.get("device", "cuda")
    feature_dim = temporal_config.get("feature_dim", 256)
    n_heads = temporal_config.get("n_heads", 8)
    tfl_layers = temporal_config.get("tfl_layers", 5)
    K = temporal_config.get("K", 4)
    nms_threshold = temporal_config.get("nms_threshold", 0.7)
    max_visual_length = ds_temporal.get("max_visual_length", 1000)

    print("=" * 60)
    print(f"  FlashVTG — Entraînement sur {dataset_name.upper()}")
    print("=" * 60)
    print(f"  Feature dim    : {feature_dim}")
    print(f"  Attention heads : {n_heads}")
    print(f"  TFL layers     : {tfl_layers}")
    print(f"  K              : {K}")
    print(f"  NMS threshold  : {nms_threshold}")
    print(f"  Max vis length : {max_visual_length}")
    print(f"  Optimizer      : AdamW")
    print(f"  Data format    : QVHighlights")
    print(f"  Device         : {device}")
    print("=" * 60)

    # --- Step 1: Conversion des données au format QVHighlights ---
    print("\n📦 Étape 1: Conversion des données au format QVHighlights...")
    data_root = dataset_config["dataset"]["root"]
    qvh_output_dir = os.path.join(data_root, "qvhighlights_format")
    os.makedirs(qvh_output_dir, exist_ok=True)

    try:
        from utils.dataset import convert_to_qvhighlights_format
        convert_to_qvhighlights_format(data_root, qvh_output_dir, dataset_name)
        print(f"   ✅ Données converties: {qvh_output_dir}")
    except (ImportError, FileNotFoundError) as e:
        print(f"   ⚠️  Conversion non effectuée: {e}")
        print(f"   → Assurez-vous que les données existent dans {data_root}")

    # --- Step 2: Extraction des features ---
    print("\n🎬 Étape 2: Extraction des features...")
    print(f"   Vidéo encoder : InternVideo2 [31]")
    print(f"   Texte encoder : LLaMA [29]")

    # --- Step 3: Construction du modèle FlashVTG ---
    print("\n🏗️  Étape 3: Construction du modèle FlashVTG...")
    try:
        from flashvtg.model.flashvtg import build_model as build_flashvtg
        from flashvtg.config.defaults import get_cfg_defaults

        cfg = get_cfg_defaults()
        cfg.HIDDEN_DIM = feature_dim
        cfg.N_HEADS = n_heads
        cfg.TFL_LAYERS = tfl_layers
        cfg.K = K
        cfg.MAX_VISUAL_LENGTH = max_visual_length

        model = build_flashvtg(cfg)
        print(f"   ✅ Modèle FlashVTG construit")
    except ImportError:
        print("   ⚠️  FlashVTG submodule non disponible")
        print("   → Clonez avec: git clone --recurse-submodules")
        return

    # --- Step 4: Boucle d'entraînement ---
    print("\n🚀 Étape 4: Entraînement...")
    print("   TODO: Implémenter la boucle d'entraînement FlashVTG")
    print("   Référence: temporal/flashvtg/train.py")
    print("   Ou utiliser directement le script d'entraînement officiel de FlashVTG")
    print("   avec les données converties au format QVHighlights")

    # L'article recommande d'utiliser le script officiel de FlashVTG:
    # python temporal/flashvtg/train.py \
    #     --data_dir <qvh_output_dir> \
    #     --hidden_dim 256 \
    #     --n_heads 8 \
    #     --tfl_layers 5 \
    #     --K 4 \
    #     --optimizer adamw \
    #     --nms_threshold 0.7

    print(f"\n✅ Configuration terminée pour {dataset_name.upper()}")


def main():
    parser = argparse.ArgumentParser(description="Train FlashVTG temporal grounding module")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["ovis", "mot17", "mot20"],
                        help="Dataset to train on (trained separately per article)")
    parser.add_argument("--config", type=str, default=None,
                        help="Dataset config path (auto-detected if not provided)")
    parser.add_argument("--pipeline-config", type=str, default="configs/pipeline.yaml",
                        help="Pipeline config path")
    args = parser.parse_args()

    # Auto-detect dataset config
    if args.config is None:
        args.config = f"configs/{args.dataset}.yaml"

    train_temporal(args.dataset, args.config, args.pipeline_config)


if __name__ == "__main__":
    main()
