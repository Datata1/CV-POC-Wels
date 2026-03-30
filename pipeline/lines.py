"""Automatic court line detection and homography estimation.

POC: Detects white/bright court lines via HSV filtering + Hough transform,
classifies them as horizontal (sideline/midline) or vertical (baseline),
computes intersections, and derives a homography to map pixel → court metres.
"""

import cv2
import numpy as np

# Handball court dimensions (metres)
COURT_LENGTH = 40.0
COURT_WIDTH = 20.0


def detect_lines(
    frame_bgr: np.ndarray,
    hsv_lower: tuple = (0, 0, 180),
    hsv_upper: tuple = (180, 50, 255),
    min_line_length: int = 80,
    max_line_gap: int = 15,
    hough_threshold: int = 60,
) -> np.ndarray | None:
    """Detect line segments via HSV color filter + HoughLinesP.

    Returns Nx4 array of (x1, y1, x2, y2) or None.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    edges = cv2.Canny(mask, 50, 150)
    raw = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if raw is None:
        return None
    return raw.reshape(-1, 4)


def _line_angle(x1, y1, x2, y2) -> float:
    """Return angle in degrees [0, 180)."""
    return float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180)


def classify_lines(
    lines: np.ndarray,
    horizontal_tolerance: float = 25.0,
    vertical_tolerance: float = 25.0,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Split lines into horizontal and vertical groups (by angle).

    Returns:
        (horizontal_lines, vertical_lines) — each is a list of (x1,y1,x2,y2).
    """
    h_lines, v_lines = [], []
    for seg in lines:
        x1, y1, x2, y2 = seg
        angle = _line_angle(x1, y1, x2, y2)
        if angle < horizontal_tolerance or angle > (180 - horizontal_tolerance):
            h_lines.append(seg)
        elif abs(angle - 90) < vertical_tolerance:
            v_lines.append(seg)
    return h_lines, v_lines


def _fit_line_params(segments: list[np.ndarray]) -> list[tuple[float, float]]:
    """Fit each segment to (slope, intercept) in y = slope*x + intercept.

    Vertical-ish lines return (inf, x_intercept).
    """
    params = []
    for x1, y1, x2, y2 in segments:
        dx = x2 - x1
        if abs(dx) < 1:
            params.append((float("inf"), float(x1)))
        else:
            slope = (y2 - y1) / dx
            intercept = y1 - slope * x1
            params.append((slope, intercept))
    return params


def _cluster_values(values: list[float], gap: float) -> list[list[int]]:
    """Group indices whose values differ by less than `gap`."""
    if not values:
        return []
    order = sorted(range(len(values)), key=lambda i: values[i])
    clusters: list[list[int]] = [[order[0]]]
    for idx in order[1:]:
        if values[idx] - values[clusters[-1][-1]] < gap:
            clusters[-1].append(idx)
        else:
            clusters.append([idx])
    return clusters


def merge_lines(
    segments: list[np.ndarray], cluster_gap: float = 30.0,
) -> list[np.ndarray]:
    """Cluster nearby parallel segments and merge each cluster into one representative."""
    if not segments:
        return []
    # Use midpoint y for horizontal, midpoint x for vertical
    mid_vals = []
    for x1, y1, x2, y2 in segments:
        angle = _line_angle(x1, y1, x2, y2)
        if angle < 45 or angle > 135:
            mid_vals.append((y1 + y2) / 2.0)
        else:
            mid_vals.append((x1 + x2) / 2.0)

    clusters = _cluster_values(mid_vals, cluster_gap)
    merged = []
    for group in clusters:
        pts = []
        for i in group:
            x1, y1, x2, y2 = segments[i]
            pts.append((x1, y1))
            pts.append((x2, y2))
        pts_arr = np.array(pts)
        # Fit a line through all endpoints
        vx, vy, cx, cy = cv2.fitLine(pts_arr.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        # Project endpoints onto the fitted line to get the full extent
        t_vals = [(p[0] - cx) * vx + (p[1] - cy) * vy for p in pts]
        t_min, t_max = min(t_vals), max(t_vals)
        x1r = int(cx + vx * t_min)
        y1r = int(cy + vy * t_min)
        x2r = int(cx + vx * t_max)
        y2r = int(cy + vy * t_max)
        merged.append(np.array([x1r, y1r, x2r, y2r]))
    return merged


def _intersect(seg_a: np.ndarray, seg_b: np.ndarray) -> tuple[float, float] | None:
    """Compute intersection point of two infinite lines through segments."""
    x1, y1, x2, y2 = seg_a.astype(float)
    x3, y3, x4, y4 = seg_b.astype(float)
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    ix = x1 + t * (x2 - x1)
    iy = y1 + t * (y2 - y1)
    return (ix, iy)


def find_intersections(
    h_lines: list[np.ndarray],
    v_lines: list[np.ndarray],
    frame_shape: tuple,
) -> list[tuple[float, float]]:
    """Find intersections between horizontal and vertical merged lines.

    Only returns points inside the frame.
    """
    h, w = frame_shape[:2]
    pts = []
    for hl in h_lines:
        for vl in v_lines:
            pt = _intersect(hl, vl)
            if pt and 0 <= pt[0] < w and 0 <= pt[1] < h:
                pts.append(pt)
    return pts


def estimate_homography_from_lines(
    frame_bgr: np.ndarray,
    hsv_lower: tuple = (0, 0, 180),
    hsv_upper: tuple = (180, 50, 255),
) -> tuple[np.ndarray | None, dict]:
    """Full pipeline: detect lines → merge → intersect → homography.

    Returns:
        (H or None, debug_info dict with detected lines / intersections)
    """
    h, w = frame_bgr.shape[:2]

    raw = detect_lines(frame_bgr, hsv_lower, hsv_upper)
    if raw is None or len(raw) < 4:
        return None, {"raw_lines": 0, "h_merged": [], "v_merged": [], "intersections": []}

    h_raw, v_raw = classify_lines(raw)
    h_merged = merge_lines(h_raw, cluster_gap=h * 0.05)
    v_merged = merge_lines(v_raw, cluster_gap=w * 0.05)

    debug = {
        "raw_lines": len(raw),
        "h_merged": h_merged,
        "v_merged": v_merged,
        "intersections": [],
    }

    intersections = find_intersections(h_merged, v_merged, frame_bgr.shape)
    debug["intersections"] = intersections

    if len(intersections) < 4 or len(h_merged) < 2 or len(v_merged) < 2:
        return None, debug

    # --- Assign court coordinates ---
    # Sort horizontal lines by y (top → bottom), vertical by x (left → right)
    h_sorted = sorted(h_merged, key=lambda s: (s[1] + s[3]) / 2)
    v_sorted = sorted(v_merged, key=lambda s: (s[0] + s[2]) / 2)

    # Take the two outermost horizontals as sidelines, two outermost verticals as baselines
    top_line = h_sorted[0]
    bot_line = h_sorted[-1]
    left_line = v_sorted[0]
    right_line = v_sorted[-1]

    # Four "corner" intersections
    corners_px = []
    corners_court = []

    mapping = [
        (top_line, left_line, (0.0, 0.0)),        # TL
        (top_line, right_line, (COURT_LENGTH, 0.0)),   # TR
        (bot_line, right_line, (COURT_LENGTH, COURT_WIDTH)),  # BR
        (bot_line, left_line, (0.0, COURT_WIDTH)),      # BL
    ]

    for hl, vl, court_pt in mapping:
        pt = _intersect(hl, vl)
        if pt is None:
            return None, debug
        corners_px.append(pt)
        corners_court.append(court_pt)

    src = np.float32(corners_px)
    dst = np.float32(corners_court)
    H, _ = cv2.findHomography(src, dst)
    debug["corners_px"] = corners_px
    debug["corners_court"] = corners_court
    return H, debug


def draw_debug_lines(
    frame: np.ndarray, debug: dict, alpha: float = 0.6,
) -> np.ndarray:
    """Draw detected/merged lines and intersections on the frame for debugging."""
    overlay = frame.copy()

    for seg in debug.get("h_merged", []):
        x1, y1, x2, y2 = seg
        cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for seg in debug.get("v_merged", []):
        x1, y1, x2, y2 = seg
        cv2.line(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)

    for pt in debug.get("intersections", []):
        cv2.circle(overlay, (int(pt[0]), int(pt[1])), 6, (0, 0, 255), -1)

    for pt in debug.get("corners_px", []):
        cv2.circle(overlay, (int(pt[0]), int(pt[1])), 10, (0, 255, 255), 3)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame
