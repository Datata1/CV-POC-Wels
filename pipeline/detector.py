"""YOLO-based person and ball detection."""

import numpy as np
from ultralytics import YOLO


# COCO class IDs
PERSON_CLASS = 0
SPORTS_BALL_CLASS = 32


def detect_objects(
    frame: np.ndarray,
    model: YOLO,
    confidence: float = 0.3,
    max_persons: int = 20,
) -> tuple[list[dict], list[dict]]:
    """
    Detect persons and sports balls in a frame.

    Returns:
        (persons, balls) — each is a list of dicts with keys:
            bbox: (x1, y1, x2, y2)
            conf: float
    """
    results = model(
        frame,
        classes=[PERSON_CLASS, SPORTS_BALL_CLASS],
        conf=confidence,
        verbose=False,
    )

    persons = []
    balls = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            entry = {"bbox": (int(x1), int(y1), int(x2), int(y2)), "conf": conf}

            if cls == PERSON_CLASS:
                persons.append(entry)
            elif cls == SPORTS_BALL_CLASS:
                balls.append(entry)

    # Sort persons by confidence descending and limit
    persons.sort(key=lambda d: d["conf"], reverse=True)
    persons = persons[:max_persons]

    # Keep best ball detection only
    balls.sort(key=lambda d: d["conf"], reverse=True)
    balls = balls[:1]

    return persons, balls
