# SVAGFormer

**Spatio-temporal Video-text Aligned Grounding Transformer**

Implémentation du pipeline SVAGFormer basée sur l'article **SVAG-Bench** (arXiv:2510.13016).

> **Référence:** Tanveer Hannan, Shuaicong Wu, Mark Weber, et al. *"SVAG-Bench: A Large-Scale Benchmark for Multi-Instance Spatio-temporal Video Action Grounding."* arXiv:2510.13016, October 2025.

## 🏗️ Architecture (Figure 3 de l'article)

SVAGFormer est un framework modulaire qui découple le grounding en **deux modules indépendants** :

1. **Spatial Grounding** — [TempRMOT](https://github.com/zyn213/TempRMOT) [35]: Détection et tracking des objets référencés par la query. Construit sur TransRMOT [32] avec query memory pour la cohérence temporelle.
2. **Temporal Grounding** — [FlashVTG](https://github.com/Zhuo-Cao/FlashVTG) [1]: Localisation temporelle des segments d'action via Temporal Feature Layering (TFL) et Adaptive Score Refinement (ASR).
3. **Évaluation** — [SVAGEval](https://github.com/Shuaicong97/SVAGEval): Toolkit d'évaluation standardisé avec ID mapping par majority voting.

> **Note importante:** Les modules spatial et temporal fonctionnent **indépendamment** (pas en cascade). Les résultats sont combinés via SVAGEval.

## 📁 Structure du Projet

```
svagformer/
├── configs/
│   ├── pipeline.yaml             # Config globale (paramètres de l'article)
│   ├── mot17.yaml                # Config MOT17
│   ├── mot20.yaml                # Config MOT20
│   └── ovis.yaml                 # Config OVIS
├── data/                         # Datasets (auto-créé par prepare_datasets.py)
│   ├── mot17/
│   │   ├── images/               # Frames vidéo
│   │   ├── annotations/          # Annotations SVAG-Bench
│   │   ├── qvhighlights_format/  # Format FlashVTG
│   │   ├── rmot_format/          # Format TempRMOT
│   │   └── features/             # Features InternVideo2 + LLaMA
│   ├── mot20/
│   └── ovis/
├── models/
│   ├── __init__.py
│   └── svagformer_pipeline.py    # Pipeline principal (modules indépendants)
├── temporal/                     # FlashVTG (git submodule)
├── spatial/                      # TempRMOT (git submodule)
├── evaluation/                   # SVAGEval (git submodule)
├── scripts/
│   ├── prepare_datasets.py       # Préparation des données
│   ├── train_temporal.py         # Entraînement FlashVTG
│   ├── train_spatial.py          # Entraînement TempRMOT
│   ├── train_full_pipeline.py    # Entraînement complet
│   ├── inference.py              # Inférence
│   └── evaluate.py               # Évaluation SVAGEval
├── utils/
│   ├── dataset.py                # Chargement + conversion (QVHighlights, RMOT)
│   └── visualization.py          # Visualisation des résultats
├── requirements.txt
├── README.md
├── .gitmodules
└── LICENSE
```

## 🚀 Installation

```bash
# Cloner avec les submodules
git clone --recurse-submodules https://github.com/wassima-azzouzi/svagformer.git
cd svagformer

# Installer les dépendances
pip install -r requirements.txt

# Préparer la structure des datasets
python scripts/prepare_datasets.py
```

## ⚙️ Paramètres Clés (Section 5.3 de l'article)

### Spatial Grounding (TempRMOT)
| Paramètre | Valeur |
|-----------|--------|
| Base setup | Refer-KITTI-V2 |
| Memory length | 5 |
| Optimiseur | Adam |
| Learning rate | 1e-5 |
| LR decay | ×10 après epoch 40 |
| Epochs | 60 |
| GPUs | 4 |

### Temporal Grounding (FlashVTG)
| Paramètre | Valeur |
|-----------|--------|
| Video features | InternVideo2 [31] |
| Text features | LLaMA [29] |
| Feature dimension | 256 |
| Attention heads | 8 |
| K (TFL) | 4 |
| TFL layers | 5 |
| Optimiseur | AdamW |
| NMS threshold | 0.7 |
| Format données | QVHighlights |
| Entraînement | Séparé par dataset |

## 💡 Mettre en place et entraîner sur un dataset (Tutoriel)

Voici la marche à suivre pour entraîner le pipeline sur un dataset officiel de SVAG-Bench (ex: MOT17).

### 1. Préparer les données
Cette étape crée l'arborescence requise :
```bash
python scripts/prepare_datasets.py
```
> Placez ensuite les vidéos dans `data/mot17/images/` et les annotations json de SVAG-Bench dans `data/mot17/annotations/`.

### 2. Formater les données (QVHighlights & RMOT)
Le script de transformation adapte les annotations pour FlashVTG et TempRMOT :
```bash
python utils/dataset.py  # (cette fonction est en cours d'intégration dans le prepare_dataset)
```

### 3. Lancer l'entraînement (Méthode 1 : Tout automatique)
Pour entraîner les deux modules séquentiellement (sur 60 epochs) pour OVIS, MOT17 et MOT20 :
```bash
python scripts/train_full_pipeline.py
```

### 4. Lancer l'entraînement (Méthode 2 : Contrôle manuel par dataset)
Si vous voulez entraîner spécifiquement le modèle **temporal (FlashVTG)** sur **OVIS** :
```bash
python scripts/train_temporal.py --dataset ovis
```

Si vous voulez entraîner le modèle **spatial (TempRMOT)** sur **MOT17** (Utilisera 4 GPUs par défaut) :
```bash
python scripts/train_spatial.py --dataset mot17
```

### Inférence (Test sur une nouvelle vidéo)
```bash
python scripts/inference.py \
    --video path/to/video.mp4 \
    --query "a person is dancing in the open area" \
    --spatial-ckpt checkpoints/temprmot_ovis.pth \
    --temporal-ckpt checkpoints/flashvtg_ovis.pth
```

### Évaluation (Score SVAGEval officiel)
```bash
python scripts/evaluate.py \
    --predictions-dir outputs/predictions \
    --gt-dir data/ground_truth \
    --datasets ovis mot17 mot20
```

## 📊 Métriques d'Évaluation (Section 5.1)

### Spatial Grounding
- **HOTA** (Higher Order Tracking Accuracy) à α=0.5 — métrique principale
- **DetA** (Detection Accuracy)
- **AssA** (Association Accuracy)

### Temporal Grounding
- **R@N** (Recall at N) : R1@0.3/0.5/0.7, R5@0.3/0.5/0.7, R10@0.3/0.5/0.7
- **mAP** (Mean Average Precision)
- **mIoU** (Mean Intersection over Union)

### Score Final (Section 5.2)
- Score = moyenne arithmétique des scores OVIS + MOT17 + MOT20
- **m-HIoU = mean(HOTA, mIoU)** — métrique principale du leaderboard

## 📊 Datasets Supportés (SVAG-Bench)

| Dataset | Type | Classes | Frames | Référence |
|---------|------|---------|--------|-----------|
| MOT17 | Multi-Object Tracking | 1 (pedestrian) | Long (centaines) | [motchallenge.net](https://motchallenge.net/data/MOT17/) |
| MOT20 | Multi-Object Tracking | 1 (pedestrian) | Très long (milliers) | [motchallenge.net](https://motchallenge.net/data/MOT20/) |
| OVIS | Occluded Video Instance Seg. | 25 | Court | [songbai.site/ovis](http://songbai.site/ovis/) |

**SVAG-Bench total:** 688 vidéos, 19 590 records annotés, 903 verbes uniques, 28.47 queries/vidéo en moyenne.

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 📖 Citation

```bibtex
@article{hannan2025svagbench,
    title={SVAG-Bench: A Large-Scale Benchmark for Multi-Instance Spatio-temporal Video Action Grounding},
    author={Hannan, Tanveer and Wu, Shuaicong and Weber, Mark and Shit, Suprosanna and Gu, Jindong and Koner, Rajat and Ošep, Aljoša and Leal-Taixé, Laura and Seidl, Thomas},
    journal={arXiv preprint arXiv:2510.13016},
    year={2025}
}

@inproceedings{cao2025flashvtg,
    title={FlashVTG: Feature Layering and Adaptive Score Handling Network for Video Temporal Grounding},
    author={Cao, Zhuo and Zhang, Bingqing and Du, Heming and Yu, Xin and Li, Xue and Wang, Sen},
    booktitle={WACV},
    year={2025}
}

@article{zhang2024temprmot,
    title={Bootstrapping Referring Multi-Object Tracking},
    author={Zhang, Yani and Wu, Dongming and Han, Wencheng and Dong, Xingping},
    journal={arXiv preprint arXiv:2406.05039},
    year={2024}
}
```
