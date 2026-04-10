"""
Utilitaires de visualisation pour SVAGFormer.
Affiche les bounding boxes, segments temporels, et résultats de tracking.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def draw_bbox(frame, bbox, label="", color=(0, 255, 0), thickness=2):
    """Dessine une bounding box sur une frame."""
    x1, y1, x2, y2 = [int(c) for c in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def visualize_temporal_segment(video_length, start_frame, end_frame, query=""):
    """Visualise le segment temporel groundé."""
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.barh(0, video_length, color="lightgray", edgecolor="black")
    ax.barh(0, end_frame - start_frame, left=start_frame, color="green", alpha=0.7)
    ax.set_xlim(0, video_length)
    ax.set_yticks([])
    ax.set_xlabel("Frame")
    ax.set_title(f"Temporal Grounding: '{query}' → [{start_frame}, {end_frame}]")
    plt.tight_layout()
    plt.savefig("temporal_segment.png", dpi=150)
    plt.show()


def visualize_tracks(frames, tracks, output_path="tracking_result.mp4"):
    """Sauvegarde une vidéo avec les tracks dessinés."""
    if not frames:
        print("⚠️ Aucune frame à visualiser.")
        return
    
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, 25, (w, h))
    
    colors = {}
    for i, (frame, frame_tracks) in enumerate(zip(frames, tracks)):
        for track in frame_tracks:
            track_id = track.get("id", 0)
            bbox = track.get("bbox", [0, 0, 0, 0])
            
            if track_id not in colors:
                colors[track_id] = tuple(np.random.randint(0, 255, 3).tolist())
            
            frame = draw_bbox(frame, bbox, f"ID:{track_id}", colors[track_id])
        
        writer.write(frame)
    
    writer.release()
    print(f"✅ Vidéo sauvegardée: {output_path}")
