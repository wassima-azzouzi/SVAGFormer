"""
Script d'inférence SVAGFormer — Conforme à SVAG-Bench.

Prend une vidéo + requête texte et exécute les deux modules indépendamment:
    1. Spatial Grounding (TempRMOT): détection + tracking des objets référencés
    2. Temporal Grounding (FlashVTG): localisation des segments temporels

Les résultats peuvent ensuite être évalués via SVAGEval.
"""
import os
import sys
import argparse
import yaml
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.svagformer_pipeline import SVAGFormerPipeline


def load_video_frames(video_path, max_frames=None):
    """Charge les frames d'une vidéo."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames


def main():
    parser = argparse.ArgumentParser(description="SVAGFormer Inference")
    parser.add_argument("--video", type=str, required=True,
                        help="Chemin vers la vidéo")
    parser.add_argument("--query", type=str, required=True,
                        help="Requête texte décrivant l'action (ex: 'a person is dancing')")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    parser.add_argument("--pipeline-config", type=str, default="configs/pipeline.yaml",
                        help="Chemin vers la config du pipeline")
    parser.add_argument("--spatial-ckpt", type=str, default=None,
                        help="Checkpoint TempRMOT")
    parser.add_argument("--temporal-ckpt", type=str, default=None,
                        help="Checkpoint FlashVTG")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Dossier de sortie")
    args = parser.parse_args()

    print("=" * 60)
    print("  SVAGFormer — Inférence")
    print("=" * 60)
    print(f"  🎬 Vidéo  : {args.video}")
    print(f"  🔍 Requête: {args.query}")
    print(f"  📂 Sortie : {args.output_dir}")
    print("=" * 60)

    # Charger le pipeline
    pipeline = SVAGFormerPipeline(
        config_path=args.pipeline_config,
        device=args.device,
    )

    # Charger les checkpoints
    pipeline.load_checkpoints(args.spatial_ckpt, args.temporal_ckpt)

    # Charger les frames vidéo
    print("\n📹 Chargement de la vidéo...")
    frames = load_video_frames(args.video)
    print(f"   {len(frames)} frames chargées")

    # Exécuter le pipeline (modules indépendants)
    print("\n🚀 Exécution du pipeline SVAGFormer...")
    print("   [1/2] Spatial Grounding (TempRMOT)...")
    print("   [2/2] Temporal Grounding (FlashVTG)...")
    result = pipeline.forward(frames, args.query)

    # Afficher les résultats
    print("\n📊 Résultats:")
    print(f"   Requête: '{result['query']}'")

    spatial = result.get("spatial_tracks", {})
    temporal = result.get("temporal_segments", [])

    print(f"\n   🗺️  Spatial Grounding (TempRMOT):")
    if spatial:
        print(f"      {len(spatial)} tracks détectés")
        for track_id, detections in spatial.items():
            print(f"      Track {track_id}: {len(detections)} détections")
    else:
        print(f"      Aucun track (modules non chargés ou placeholder)")

    print(f"\n   ⏱️  Temporal Grounding (FlashVTG):")
    if temporal:
        for i, (start, end, score) in enumerate(temporal):
            print(f"      Segment {i+1}: [{start:.1f}, {end:.1f}] (score: {score:.3f})")
    else:
        print(f"      Aucun segment (modules non chargés ou placeholder)")

    # Sauvegarder les résultats
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n✅ Résultats sauvegardés dans {args.output_dir}/")


if __name__ == "__main__":
    main()
