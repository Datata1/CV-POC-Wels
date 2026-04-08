"""Drawing utilities for annotated video output."""

import cv2
import numpy as np

from pipeline.pose import POSE_CONNECTIONS

# Team colors (BGR)
TEAM_COLORS = {
    "A": (255, 100, 0),    # blue-ish
    "B": (0, 100, 255),    # orange-ish
    "unknown": (180, 180, 180),
}
BALL_COLOR = (0, 255, 255)    # yellow
SKELETON_COLOR = (0, 255, 255)
JOINT_COLOR = (0, 0, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.45


def draw_player(frame: np.ndarray, det: dict):
    """Draw bounding box, label, team color, and pose for one player."""
    x1, y1, x2, y2 = det["bbox"]
    team = det.get("team", "unknown")
    color = TEAM_COLORS.get(team, TEAM_COLORS["unknown"])
    tid = det.get("track_id", "?")

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Label
    label = f"#{tid} Team {team} ({det['conf']:.0%})"
    (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4), FONT, FONT_SCALE, (255, 255, 255), 1)

    # On-court indicator
    if not det.get("on_court", True):
        cv2.putText(frame, "OFF-COURT", (x1, y2 + 15), FONT, 0.4, (0, 0, 200), 1)

    # Pose skeleton
    pose = det.get("pose")
    if pose:
        _draw_pose(frame, pose["landmarks"], pose["offset"])


def _draw_pose(frame: np.ndarray, landmarks: list[dict], offset: tuple):
    """Draw skeleton connections and joint dots."""
    h, w = frame.shape[:2]

    for start_idx, end_idx in POSE_CONNECTIONS:
        s = landmarks[start_idx]
        e = landmarks[end_idx]
        sx, sy = int(s["x"]), int(s["y"])
        ex, ey = int(e["x"]), int(e["y"])

        if (
            s["visibility"] > 0.5
            and e["visibility"] > 0.5
            and 0 <= sx < w and 0 <= sy < h
            and 0 <= ex < w and 0 <= ey < h
        ):
            cv2.line(frame, (sx, sy), (ex, ey), SKELETON_COLOR, 2)

    for lm in landmarks:
        px, py = int(lm["x"]), int(lm["y"])
        if lm["visibility"] > 0.5 and 0 <= px < w and 0 <= py < h:
            cv2.circle(frame, (px, py), 3, JOINT_COLOR, -1)


def draw_ball(frame: np.ndarray, ball: dict):
    """Draw ball marker."""
    x1, y1, x2, y2 = ball["bbox"]
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    r = max((x2 - x1) // 2, 8)
    cv2.circle(frame, (cx, cy), r, BALL_COLOR, 2)
    cv2.putText(
        frame, f"Ball ({ball['conf']:.0%})", (cx + r + 4, cy + 4),
        FONT, FONT_SCALE, BALL_COLOR, 1,
    )


def draw_hud(frame: np.ndarray, state: dict):
    """Draw a small heads-up display with frame stats."""
    lines = [
        f"Frame {state['frame_id']} | {state['timestamp_s']:.1f}s",
        f"Players: {state['on_court_count']}/{state['player_count']}"
        + (f" | Ball: yes" if state["ball"] else " | Ball: --"),
    ]
    y = 25
    for line in lines:
        cv2.putText(frame, line, (10, y), FONT, 0.5, (255, 255, 255), 1)
        y += 20


# Keypoint visualisation colors
KP_DOT_COLOR = (0, 255, 0)       # green
KP_LABEL_COLOR = (0, 255, 0)
KP_DOT_RADIUS = 5


def draw_keypoints(
    frame: np.ndarray,
    src_pts: np.ndarray,
    names: list[str],
):
    """Draw detected court keypoints as labeled dots on the frame."""
    for (px, py), name in zip(src_pts, names):
        x, y = int(px), int(py)
        cv2.circle(frame, (x, y), KP_DOT_RADIUS, KP_DOT_COLOR, -1)
        cv2.circle(frame, (x, y), KP_DOT_RADIUS, (255, 255, 255), 1)
        cv2.putText(frame, name, (x + 8, y - 4), FONT, 0.35, KP_LABEL_COLOR, 1)


def overlay_court(frame: np.ndarray, court_img: np.ndarray, scale: float = 0.55):
    """Overlay the court top-down view at the bottom-center of the frame."""
    h, w = frame.shape[:2]
    ch, cw = court_img.shape[:2]
    new_w = int(cw * scale)
    new_h = int(ch * scale)
    resized = cv2.resize(court_img, (new_w, new_h))

    # Position: bottom-center with small margin
    margin = 10
    y1 = h - new_h - margin
    x1 = (w - new_w) // 2

    # Semi-transparent background
    roi = frame[y1:y1 + new_h, x1:x1 + new_w]
    blended = cv2.addWeighted(roi, 0.3, resized, 0.7, 0)
    frame[y1:y1 + new_h, x1:x1 + new_w] = blended
