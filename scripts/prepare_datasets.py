"""
Script de préparation des datasets — Conforme à SVAG-Bench.

SVAG-Bench (Section 3):
    - 688 vidéos, 19590 annotated records, 903 unique verbs
    - Vidéos issues de MOT17 [4], MOT20 [5], et OVIS [22]
    - Annotations action-centric (pas appearance-based)
    - Moyenne 28.47 queries par vidéo
    - Longueur moyenne des queries: 9.58 mots
    - Annotations augmentées via GPT-3.5 [21] + vérification humaine

Crée les dossiers nécessaires et prépare la structure pour chaque dataset.
"""
import os
import yaml


DATASETS = {
    "mot17": {
        "description": "Multi-Object Tracking — crowded urban scenes [4]",
        "url": "https://motchallenge.net/data/MOT17/",
        "root": "data/mot17",
        "reference": "Dendorfer et al., IJCV 2021",
        "characteristics": "pedestrians, long videos (hundreds of frames)",
    },
    "mot20": {
        "description": "Multi-Object Tracking — crowded scenes [5]",
        "url": "https://motchallenge.net/data/MOT20/",
        "root": "data/mot20",
        "reference": "Dendorfer et al., 2020",
        "characteristics": "dense objects, very long videos (thousands of frames)",
    },
    "ovis": {
        "description": "Occluded Video Instance Segmentation [22]",
        "url": "http://songbai.site/ovis/",
        "root": "data/ovis",
        "reference": "Qi et al., IJCV 2022",
        "characteristics": "25 categories, complex occlusions, shorter videos",
    },
}


def prepare_dataset(name, info):
    """Prépare la structure de dossiers pour un dataset SVAG-Bench."""
    print(f"📂 Préparation de {name.upper()}...")
    print(f"   Description: {info['description']}")
    print(f"   Référence: {info['reference']}")

    root = info["root"]
    # Structure standard pour chaque dataset
    subdirs = [
        os.path.join(root, "images"),           # Frames vidéo
        os.path.join(root, "annotations"),       # Annotations SVAG-Bench (JSON)
        os.path.join(root, "qvhighlights_format"),  # Données FlashVTG (Section 5.3)
        os.path.join(root, "rmot_format"),       # Données TempRMOT (Section 5.3)
        os.path.join(root, "features"),          # Features pré-extraites
        os.path.join(root, "features", "internvideo2"),  # Features vidéo InternVideo2
        os.path.join(root, "features", "llama"),  # Features texte LLaMA
    ]

    for subdir in subdirs:
        os.makedirs(subdir, exist_ok=True)

    print(f"   ✅ Dossiers créés: {root}/")
    print(f"   📁 images/                — frames vidéo")
    print(f"   📁 annotations/           — annotations SVAG-Bench")
    print(f"   📁 qvhighlights_format/   — format FlashVTG")
    print(f"   📁 rmot_format/           — format TempRMOT")
    print(f"   📁 features/internvideo2/ — features vidéo pré-extraites")
    print(f"   📁 features/llama/        — features texte pré-extraites")
    print(f"   ℹ️  Téléchargez les données depuis: {info['url']}")
    print(f"   ℹ️  Annotations SVAG-Bench depuis: https://github.com/Shuaicong97/SVAGEval")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("  SVAGFormer — Préparation des datasets")
    print("  Conforme à SVAG-Bench (arXiv:2510.13016)")
    print("=" * 60)
    print()
    print("  SVAG-Bench: 688 vidéos, 19590 records, 903 verbes")
    print("  Sources: MOT17, MOT20, OVIS")
    print()

    for name, info in DATASETS.items():
        prepare_dataset(name, info)

    print("=" * 60)
    print("  ✅ Tous les dossiers sont prêts!")
    print()
    print("  Prochaines étapes:")
    print("  1. Téléchargez les vidéos depuis les URLs ci-dessus")
    print("  2. Téléchargez les annotations SVAG-Bench")
    print("  3. Convertissez au format QVHighlights: python utils/dataset.py")
    print("  4. Extrayez les features InternVideo2 + LLaMA")
    print("  5. Lancez l'entraînement: python scripts/train_full_pipeline.py")
    print("=" * 60)
