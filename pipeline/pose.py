"""YOLO11-pose GPU-accelerated pose estimation.

Uses the ultralytics YOLO11-pose model running on CUDA for fast, accurate
17-keypoint (COCO) pose estimation.  Replaces the previous MediaPipe-based
CPU implementation.
"""

import numpy as np
import torch
from ultralytics import YOLO

# COCO 17-keypoint skeleton connections (index pairs)
POSE_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # head
    (5, 6),                                     # shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),           # arms
    (5, 11), (6, 12), (11, 12),                 # torso
    (11, 13), (13, 15), (12, 14), (14, 16),     # legs
]

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def create_pose_model(model_path: str = "yolo11m-pose.pt") -> YOLO:
    """Load a YOLO11-pose model onto GPU (falls back to CPU if unavailable)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)
    model.to(device)
    return model


def estimate_pose(
    frame_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    pose_model: YOLO,
) -> dict | None:
    """
    Run GPU-accelerated pose estimation on a bounding-box crop.

    Args:
        frame_bgr: full frame in BGR (OpenCV native format)
        bbox: (x1, y1, x2, y2)
        pose_model: YOLO11-pose model

    Returns:
        dict with 'landmarks' (list of 17 dicts) and 'offset' (x1, y1, cw, ch)
        or None if no pose detected.
    """
    x1, y1, x2, y2 = bbox
    fh, fw = frame_bgr.shape[:2]
    w, h = x2 - x1, y2 - y1

    # Padding for better detection
    pad_x, pad_y = int(w * 0.15), int(h * 0.1)
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(fw, x2 + pad_x)
    cy2 = min(fh, y2 + pad_y)

    crop = np.ascontiguousarray(frame_bgr[cy1:cy2, cx1:cx2])
    if crop.size == 0:
        return None

    results = pose_model(crop, verbose=False)

    if not results or not results[0].keypoints or len(results[0].keypoints) == 0:
        return None

    kpts = results[0].keypoints
    # Take the most confident detection
    if kpts.conf is None or kpts.conf.shape[0] == 0:
        return None

    best_idx = int(kpts.conf.mean(dim=1).argmax())
    xy = kpts.xy[best_idx].cpu().numpy()       # (17, 2)
    conf = kpts.conf[best_idx].cpu().numpy()    # (17,)

    crop_h, crop_w = crop.shape[:2]
    landmarks = []
    for i in range(len(xy)):
        landmarks.append({
            "x": float(xy[i][0]) + cx1,
            "y": float(xy[i][1]) + cy1,
            "z": 0.0,
            "visibility": float(conf[i]),
        })

    return {
        "landmarks": landmarks,
        "offset": (cx1, cy1, crop_w, crop_h),
    }


def estimate_poses_batch(
    frame_bgr: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
    pose_model: YOLO,
) -> list[dict | None]:
    """
    Run pose estimation on the full frame and match detections to bboxes.

    This is more GPU-efficient than cropping per-player: a single forward
    pass finds all poses, then each is matched to the closest bbox.

    Args:
        frame_bgr: full BGR frame
        bboxes: list of (x1, y1, x2, y2) player bounding boxes
        pose_model: YOLO11-pose model

    Returns:
        list aligned with bboxes, each entry is a pose dict or None.
    """
    if not bboxes:
        return []

    results = pose_model(frame_bgr, verbose=False)
    if not results or not results[0].keypoints or len(results[0].keypoints) == 0:
        return [None] * len(bboxes)

    kpts = results[0].keypoints
    pose_boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else None

    if pose_boxes is None or len(pose_boxes) == 0:
        return [None] * len(bboxes)

    xy_all = kpts.xy.cpu().numpy()      # (N_poses, 17, 2)
    conf_all = kpts.conf.cpu().numpy()  # (N_poses, 17)

    # Match each player bbox to the closest pose detection by center distance
    bbox_centers = np.array([((b[0]+b[2])/2, (b[1]+b[3])/2) for b in bboxes])
    pose_centers = np.array([((p[0]+p[2])/2, (p[1]+p[3])/2) for p in pose_boxes])

    results_list: list[dict | None] = [None] * len(bboxes)
    used_poses = set()

    # Compute all pairwise distances
    dists = np.linalg.norm(bbox_centers[:, None] - pose_centers[None, :], axis=2)

    for _ in range(min(len(bboxes), len(pose_centers))):
        # Mask already-used poses
        for pi in used_poses:
            dists[:, pi] = float("inf")

        bi, pi = np.unravel_index(dists.argmin(), dists.shape)
        if dists[bi, pi] == float("inf"):
            break

        # Only match if centers are reasonably close
        bw = bboxes[bi][2] - bboxes[bi][0]
        bh = bboxes[bi][3] - bboxes[bi][1]
        max_dist = max(bw, bh) * 0.75
        if dists[bi, pi] > max_dist:
            dists[bi, :] = float("inf")
            continue

        xy = xy_all[pi]
        conf = conf_all[pi]
        landmarks = []
        for k in range(len(xy)):
            landmarks.append({
                "x": float(xy[k][0]),
                "y": float(xy[k][1]),
                "z": 0.0,
                "visibility": float(conf[k]),
            })

        results_list[bi] = {
            "landmarks": landmarks,
            "offset": (0, 0, frame_bgr.shape[1], frame_bgr.shape[0]),
        }

        used_poses.add(pi)
        dists[bi, :] = float("inf")

    return results_list
