"""MediaPipe-based pose estimation."""

import cv2
import mediapipe as mp
import numpy as np

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarksConnections = mp.tasks.vision.PoseLandmarksConnections
RunningMode = mp.tasks.vision.RunningMode

POSE_CONNECTIONS = PoseLandmarksConnections.POSE_LANDMARKS


def create_landmarker(model_path: str) -> PoseLandmarker:
    """Create and return a PoseLandmarker instance."""
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
    )
    return PoseLandmarker.create_from_options(options)


def estimate_pose(
    frame_rgb: np.ndarray,
    bbox: tuple[int, int, int, int],
    landmarker: PoseLandmarker,
) -> dict | None:
    """
    Run pose estimation on a bounding-box crop.

    Args:
        frame_rgb: full frame in RGB
        bbox: (x1, y1, x2, y2)
        landmarker: MediaPipe PoseLandmarker

    Returns:
        dict with 'landmarks' (list of 33 dicts) and 'offset' (x1, y1, cw, ch)
        or None if no pose detected.
    """
    x1, y1, x2, y2 = bbox
    fh, fw = frame_rgb.shape[:2]
    w, h = x2 - x1, y2 - y1

    # Padding for better detection
    pad_x, pad_y = int(w * 0.15), int(h * 0.1)
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(fw, x2 + pad_x)
    cy2 = min(fh, y2 + pad_y)

    crop = np.ascontiguousarray(frame_rgb[cy1:cy2, cx1:cx2])
    if crop.size == 0:
        return None

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop)
    results = landmarker.detect(mp_image)

    if not results.pose_landmarks:
        return None

    crop_h, crop_w = crop.shape[:2]
    raw = results.pose_landmarks[0]

    landmarks = []
    for lm in raw:
        landmarks.append({
            "x": lm.x * crop_w + cx1,
            "y": lm.y * crop_h + cy1,
            "z": lm.z,
            "visibility": lm.visibility,
        })

    return {
        "landmarks": landmarks,
        "offset": (cx1, cy1, crop_w, crop_h),
        "raw": raw,
    }
