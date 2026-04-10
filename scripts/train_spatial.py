"""
Entraînement du module Spatial (TempRMOT) — Conforme à SVAG-Bench Section 5.3.

Spatial Grounding Settings (de l'article):
    - Basé sur le setup officiel de TempRMOT sur Refer-KITTI-V2
    - Memory length: 5
    - Optimiseur: Adam, lr = 1e-5
    - Decay factor 10 après la 40ème epoch
    - Entraînement: 60 epochs sur 4 GPUs
    - TempRMOT [35] est construit sur TransRMOT [32]
    - Amélioré avec query memory pour la cohérence temporelle
"""
import os
import sys
import yaml
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "spatial"))


def train_spatial(dataset_name, dataset_config_path, pipeline_config_path="configs/pipeline.yaml"):
    """
    Entraîne TempRMOT sur un dataset spécifique.
    L'article utilise le setup officiel de TempRMOT.
    """
    # Charger les configs
    pipeline_config = yaml.safe_load(open(pipeline_config_path))
    dataset_config = yaml.safe_load(open(dataset_config_path))
    spatial_config = pipeline_config.get("spatial", {})
    ds_spatial = dataset_config.get("spatial_training", {})

    device = pipeline_config.get("device", "cuda")
    memory_length = spatial_config.get("memory_length", 5)
    lr = spatial_config.get("lr", 1e-5)
    lr_decay_factor = spatial_config.get("lr_decay_factor", 10)
    lr_decay_epoch = spatial_config.get("lr_decay_epoch", 40)
    total_epochs = ds_spatial.get("epochs", 60)
    num_gpus = ds_spatial.get("num_gpus", 4)

    print("=" * 60)
    print(f"  TempRMOT — Entraînement sur {dataset_name.upper()}")
    print("=" * 60)
    print(f"  Base setup     : Refer-KITTI-V2")
    print(f"  Memory length  : {memory_length}")
    print(f"  Optimizer      : Adam")
    print(f"  Learning rate  : {lr}")
    print(f"  LR decay       : ×{lr_decay_factor} après epoch {lr_decay_epoch}")
    print(f"  Total epochs   : {total_epochs}")
    print(f"  GPUs           : {num_gpus}")
    print(f"  Device         : {device}")
    print("=" * 60)

    # --- Step 1: Préparation des données RMOT ---
    print("\n📦 Étape 1: Préparation des données au format RMOT...")
    data_root = dataset_config["dataset"]["root"]
    print(f"   Data root: {data_root}")

    # --- Step 2: Construction du modèle TempRMOT ---
    print("\n🏗️  Étape 2: Construction du modèle TempRMOT...")
    try:
        from temprmot.models.temprmot import build_model as build_temprmot
        from temprmot.config.defaults import get_cfg_defaults

        cfg = get_cfg_defaults()
        cfg.MEMORY_LENGTH = memory_length
        cfg.TRAIN.LR = lr
        cfg.TRAIN.EPOCHS = total_epochs
        cfg.TRAIN.LR_DECAY_FACTOR = lr_decay_factor
        cfg.TRAIN.LR_DECAY_EPOCH = lr_decay_epoch

        model = build_temprmot(cfg)
        print(f"   ✅ Modèle TempRMOT construit (basé sur TransRMOT + query memory)")
    except ImportError:
        print("   ⚠️  TempRMOT submodule non disponible")
        print("   → Clonez avec: git clone --recurse-submodules")
        return

    # --- Step 3: Optimiseur Adam avec lr decay ---
    print("\n⚙️  Étape 3: Configuration de l'optimiseur...")
    print(f"   Adam, lr={lr}")
    print(f"   StepLR: decay ×{lr_decay_factor} à epoch {lr_decay_epoch}")

    # import torch
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer, step_size=lr_decay_epoch, gamma=1.0/lr_decay_factor
    # )

    # --- Step 4: Boucle d'entraînement ---
    print(f"\n🚀 Étape 4: Entraînement ({total_epochs} epochs)...")
    print("   TODO: Implémenter la boucle d'entraînement TempRMOT")
    print("   Référence: spatial/temprmot/train.py")
    print("   Ou utiliser directement le script d'entraînement officiel de TempRMOT")

    # L'article recommande d'utiliser le script officiel de TempRMOT:
    # python -m torch.distributed.launch --nproc_per_node=4 \
    #     spatial/temprmot/train.py \
    #     --dataset <dataset_name> \
    #     --data_dir <data_root> \
    #     --memory_length 5 \
    #     --lr 1e-5 \
    #     --lr_decay_factor 10 \
    #     --lr_decay_epoch 40 \
    #     --epochs 60

    print(f"\n✅ Configuration terminée pour {dataset_name.upper()}")


def main():
    parser = argparse.ArgumentParser(description="Train TempRMOT spatial grounding module")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["ovis", "mot17", "mot20"],
                        help="Dataset to train on")
    parser.add_argument("--config", type=str, default=None,
                        help="Dataset config path (auto-detected if not provided)")
    parser.add_argument("--pipeline-config", type=str, default="configs/pipeline.yaml",
                        help="Pipeline config path")
    args = parser.parse_args()

    if args.config is None:
        args.config = f"configs/{args.dataset}.yaml"

    train_spatial(args.dataset, args.config, args.pipeline_config)


if __name__ == "__main__":
    main()
