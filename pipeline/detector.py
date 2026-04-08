"""YOLO-based person and ball detection with built-in tracking."""

import numpy as np
from ultralytics import YOLO


# COCO class IDs
PERSON_CLASS = 0
SPORTS_BALL_CLASS = 32

# Custom ball model uses class 0 (single-class)
CUSTOM_BALL_CLASS = 0


def _parse_boxes(results, class_id: int) -> list[dict]:
    """Extract detections for a given class from YOLO results."""
    detections = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls != class_id:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            track_id = int(box.id[0]) if box.id is not None else -1
            detections.append({
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "conf": conf,
                "track_id": track_id,
            })
    return detections


def detect_and_track(
    frame: np.ndarray,
    model: YOLO,
    confidence: float = 0.3,
    max_persons: int = 20,
    ball_model: YOLO | None = None,
    ball_confidence: float = 0.25,
    imgsz: int = 1280,
) -> tuple[list[dict], list[dict]]:
    """
    Detect persons and sports balls using YOLO's built-in ByteTrack tracker.

    If ball_model is provided, it is used for ball detection instead of
    the generic COCO sports_ball class from the person model.

    Returns:
        (persons, balls) — each is a list of dicts with keys:
            bbox: (x1, y1, x2, y2)
            conf: float
            track_id: int  (-1 if tracker lost the object)
    """
    if ball_model is not None:
        # Separate models: person model for people, ball model for handball
        person_results = model.track(
            frame,
            classes=[PERSON_CLASS],
            conf=confidence,
            imgsz=imgsz,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )
        ball_results = ball_model.track(
            frame,
            conf=ball_confidence,
            imgsz=imgsz,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )
        persons = _parse_boxes(person_results, PERSON_CLASS)
        balls = _parse_boxes(ball_results, CUSTOM_BALL_CLASS)
    else:
        # Single model: detect both persons and balls from COCO
        results = model.track(
            frame,
            classes=[PERSON_CLASS, SPORTS_BALL_CLASS],
            conf=confidence,
            imgsz=imgsz,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )
        persons = _parse_boxes(results, PERSON_CLASS)
        balls = _parse_boxes(results, SPORTS_BALL_CLASS)

    # Sort persons by confidence descending and limit
    persons.sort(key=lambda d: d["conf"], reverse=True)
    persons = persons[:max_persons]

    # Keep best ball detection only
    balls.sort(key=lambda d: d["conf"], reverse=True)
    balls = balls[:1]

    return persons, balls
