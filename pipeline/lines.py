"""Court line detection via blue-floor segmentation + adaptive homography.

Strategy for broadcast handball with a panning camera:
1. Segment the blue court floor via HSV to create an ROI mask.
2. Detect white lines *only* inside the court mask (eliminates ads/jerseys).
3. Identify the two sidelines (longest near-horizontal lines) — these are
   almost always partially visible in any camera angle.
4. Identify vertical lines (center line, baselines, 7m line) and label them
   using line length + position heuristics.
5. Compute homography from available correspondences (≥4 points needed).
6. Smooth the homography temporally via EMA + optical-flow compensation.
"""

import cv2
import numpy as np

# Handball court dimensions (metres)
COURT_LENGTH = 40.0
COURT_WIDTH = 20.0


# ── 1. Court floor segmentation ──────────────────────────────────────────

def segment_court(
    frame_bgr: np.ndarray,
    blue_lower: tuple = (95, 30, 80),
    blue_upper: tuple = (125, 255, 255),
    min_area_frac: float = 0.05,
) -> np.ndarray | None:
    """Segment the blue court floor and return a binary mask.

    Returns the mask covering the largest blue region, or None if
    the blue area is too small (e.g. camera on crowd / replay).
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(blue_lower), np.array(blue_upper))

    # Morphology to fill gaps (players, shadows, small logos on floor)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Keep only the largest connected component
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8,
    )
    if n_labels < 2:
        return None

    areas = stats[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(areas)) + 1
    total_px = frame_bgr.shape[0] * frame_bgr.shape[1]
    if areas[best - 1] < total_px * min_area_frac:
        return None

    court_mask = (labels == best).astype(np.uint8) * 255
    return court_mask


# ── 2. Line detection inside court mask ──────────────────────────────────

def detect_court_lines(
    frame_bgr: np.ndarray,
    court_mask: np.ndarray,
    min_line_length: int = 60,
    max_line_gap: int = 20,
    hough_threshold: int = 50,
) -> np.ndarray | None:
    """Detect white lines inside the court mask only."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, np.array([0, 0, 170]), np.array([180, 60, 255]))

    # Restrict to court area
    white_mask = cv2.bitwise_and(white_mask, court_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

    edges = cv2.Canny(white_mask, 50, 150)
    raw = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if raw is None:
        return None
    return raw.reshape(-1, 4)


# ── 3. Line classification and merging ───────────────────────────────────

def _line_angle(x1, y1, x2, y2) -> float:
    """Return angle in degrees [0, 180)."""
    return float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180)


def _line_length(seg) -> float:
    x1, y1, x2, y2 = seg
    return float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))


def classify_lines(
    lines: np.ndarray,
    horizontal_tolerance: float = 25.0,
    vertical_tolerance: float = 25.0,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Split lines into horizontal and vertical groups by angle."""
    h_lines, v_lines = [], []
    for seg in lines:
        x1, y1, x2, y2 = seg
        angle = _line_angle(x1, y1, x2, y2)
        if angle < horizontal_tolerance or angle > (180 - horizontal_tolerance):
            h_lines.append(seg)
        elif abs(angle - 90) < vertical_tolerance:
            v_lines.append(seg)
    return h_lines, v_lines


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
    """Cluster nearby parallel segments and merge each cluster."""
    if not segments:
        return []
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
        vx, vy, cx, cy = cv2.fitLine(
            pts_arr.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01,
        ).flatten()
        t_vals = [(p[0] - cx) * vx + (p[1] - cy) * vy for p in pts]
        t_min, t_max = min(t_vals), max(t_vals)
        merged.append(np.array([
            int(cx + vx * t_min), int(cy + vy * t_min),
            int(cx + vx * t_max), int(cy + vy * t_max),
        ]))
    return merged


def _intersect(seg_a: np.ndarray, seg_b: np.ndarray) -> tuple[float, float] | None:
    """Intersection of two infinite lines through segments."""
    x1, y1, x2, y2 = seg_a.astype(float)
    x3, y3, x4, y4 = seg_b.astype(float)
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    ix = x1 + t * (x2 - x1)
    iy = y1 + t * (y2 - y1)
    return (ix, iy)


# ── 4. Identify which vertical line is which ─────────────────────────────

def _assign_court_x(
    v_sorted: list[np.ndarray],
    top_line: np.ndarray,
    bot_line: np.ndarray,
) -> list[tuple[np.ndarray, float]]:
    """Label each vertical line with its court x-coordinate.

    The center line (x=20m) is the most commonly visible vertical and is
    usually the longest.  Other verticals are assigned based on typical
    handball court features.
    """
    if not v_sorted:
        return []

    # Score each vertical: longer lines that span more of the court
    # are more likely to be the center line or end lines
    lengths = [_line_length(vl) for vl in v_sorted]
    longest_idx = int(np.argmax(lengths))

    if len(v_sorted) == 1:
        return [(v_sorted[0], COURT_LENGTH / 2)]

    # Assume longest = center line
    center_vl = v_sorted[longest_idx]
    center_px = (center_vl[0] + center_vl[2]) / 2.0

    assigned: list[tuple[np.ndarray, float]] = []
    for i, vl in enumerate(v_sorted):
        mid_x = (vl[0] + vl[2]) / 2.0
        if i == longest_idx:
            assigned.append((vl, COURT_LENGTH / 2))
        elif mid_x < center_px:
            # Left of center: could be baseline (0), 6m (6), 7m (7), 9m (9)
            # Use distance ratio to center to guess
            # For now: if it's far from center it's likely baseline/9m
            dist_ratio = (center_px - mid_x) / center_px if center_px > 0 else 0
            if dist_ratio > 0.5:
                assigned.append((vl, 0.0))  # baseline
            else:
                assigned.append((vl, 9.0))  # 9m line
        else:
            # Right of center
            dist_ratio = (mid_x - center_px) / (1920 - center_px) if center_px < 1920 else 0
            if dist_ratio > 0.5:
                assigned.append((vl, COURT_LENGTH))  # baseline
            else:
                assigned.append((vl, COURT_LENGTH - 9.0))  # 9m line

    return assigned


# ── 5. Homography estimation ─────────────────────────────────────────────

def estimate_homography(
    frame_bgr: np.ndarray,
) -> tuple[np.ndarray | None, dict]:
    """Full pipeline: segment court → detect lines → identify → homography."""
    h, w = frame_bgr.shape[:2]
    debug: dict = {
        "court_mask": False,
        "raw_lines": 0,
        "h_merged": [],
        "v_merged": [],
        "intersections": [],
        "src_pts": [],
        "dst_pts": [],
    }

    # 1. Segment court
    court_mask = segment_court(frame_bgr)
    if court_mask is None:
        return None, debug
    debug["court_mask"] = True

    # 2. Detect lines inside court
    raw = detect_court_lines(frame_bgr, court_mask)
    if raw is None or len(raw) < 3:
        debug["raw_lines"] = 0 if raw is None else len(raw)
        return None, debug
    debug["raw_lines"] = len(raw)

    # 3. Classify + merge
    h_raw, v_raw = classify_lines(raw)
    h_merged = merge_lines(h_raw, cluster_gap=h * 0.04)
    v_merged = merge_lines(v_raw, cluster_gap=w * 0.04)
    debug["h_merged"] = h_merged
    debug["v_merged"] = v_merged

    if len(h_merged) < 2:
        return None, debug

    # Take the two most separated horizontal lines as sidelines
    h_sorted = sorted(h_merged, key=lambda s: (s[1] + s[3]) / 2)
    top_line = h_sorted[0]
    bot_line = h_sorted[-1]

    top_y = (top_line[1] + top_line[3]) / 2
    bot_y = (bot_line[1] + bot_line[3]) / 2
    if (bot_y - top_y) < h * 0.15:
        return None, debug

    # 4. Identify vertical lines
    v_sorted = sorted(v_merged, key=lambda s: (s[0] + s[2]) / 2)
    assigned_v = _assign_court_x(v_sorted, top_line, bot_line)
    if not assigned_v:
        return None, debug

    # 5. Build correspondences from line intersections
    src_pts: list[tuple[float, float]] = []
    dst_pts: list[tuple[float, float]] = []

    for vl, court_x in assigned_v:
        pt_top = _intersect(vl, top_line)
        pt_bot = _intersect(vl, bot_line)
        if pt_top and 0 <= pt_top[0] < w and 0 <= pt_top[1] < h:
            src_pts.append(pt_top)
            dst_pts.append((court_x, 0.0))
        if pt_bot and 0 <= pt_bot[0] < w and 0 <= pt_bot[1] < h:
            src_pts.append(pt_bot)
            dst_pts.append((court_x, COURT_WIDTH))

    debug["src_pts"] = src_pts
    debug["dst_pts"] = dst_pts
    debug["intersections"] = src_pts

    if len(src_pts) < 4:
        return None, debug

    src = np.float32(src_pts)
    dst = np.float32(dst_pts)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        return None, debug

    # Sanity check: frame center should map to somewhere on/near the court
    center_pt = cv2.perspectiveTransform(np.float32([[[w / 2, h / 2]]]), H)
    cx, cy = float(center_pt[0][0][0]), float(center_pt[0][0][1])
    if not (-10 <= cx <= 50 and -10 <= cy <= 30):
        debug["sanity_fail"] = (cx, cy)
        return None, debug

    debug["corners_px"] = src_pts[:4]
    debug["corners_court"] = dst_pts[:4]
    return H, debug


# ── 6. Temporal smoothing ────────────────────────────────────────────────

class HomographyTracker:
    """Smooths homography over time using EMA + optical flow for panning.

    Usage:
        tracker = HomographyTracker()
        # Each frame:
        H_smooth = tracker.update(frame, raw_H_or_None)
    """

    def __init__(self, ema_alpha: float = 0.3, max_stale_frames: int = 150):
        self._H: np.ndarray | None = None
        self._prev_gray: np.ndarray | None = None
        self._ema_alpha = ema_alpha
        self._stale_frames = 0
        self._max_stale = max_stale_frames

    @property
    def current_H(self) -> np.ndarray | None:
        return self._H

    def update(
        self, frame_bgr: np.ndarray, new_H: np.ndarray | None,
    ) -> np.ndarray | None:
        """Update with a new frame and optional fresh homography.

        If new_H is provided, blend it with current estimate via EMA.
        If new_H is None, compensate for camera pan using optical flow.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if new_H is not None:
            if self._H is None:
                self._H = new_H.copy()
            else:
                self._H = self._ema_alpha * new_H + (1 - self._ema_alpha) * self._H
            self._stale_frames = 0
        elif self._H is not None:
            # Compensate for camera motion via optical flow
            if self._prev_gray is not None:
                flow_H = self._estimate_camera_motion(self._prev_gray, gray)
                if flow_H is not None:
                    try:
                        self._H = self._H @ np.linalg.inv(flow_H)
                    except np.linalg.LinAlgError:
                        pass
            self._stale_frames += 1
            if self._stale_frames > self._max_stale:
                self._H = None

        self._prev_gray = gray
        return self._H

    @staticmethod
    def _estimate_camera_motion(
        prev_gray: np.ndarray, curr_gray: np.ndarray,
    ) -> np.ndarray | None:
        """Estimate global camera motion homography between two frames."""
        pts0 = cv2.goodFeaturesToTrack(
            prev_gray, maxCorners=200, qualityLevel=0.01,
            minDistance=30, blockSize=7,
        )
        if pts0 is None or len(pts0) < 10:
            return None

        pts1, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, pts0, None,
            winSize=(21, 21), maxLevel=3,
        )
        if pts1 is None:
            return None

        good = status.flatten() == 1
        if good.sum() < 8:
            return None

        src = pts0[good].reshape(-1, 2)
        dst = pts1[good].reshape(-1, 2)
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        return H


# ── Legacy wrapper (for backward compat with diagnose.py) ────────────────

def estimate_homography_from_lines(
    frame_bgr: np.ndarray,
    hsv_lower: tuple = (0, 0, 140),
    hsv_upper: tuple = (180, 80, 255),
) -> tuple[np.ndarray | None, dict]:
    """Legacy wrapper — redirects to the new court-segmentation pipeline."""
    return estimate_homography(frame_bgr)


def detect_lines(frame_bgr, **kw):
    """Legacy: detect lines using the new court-masked approach."""
    court_mask = segment_court(frame_bgr)
    if court_mask is None:
        return None
    return detect_court_lines(frame_bgr, court_mask, **kw)


def find_intersections(h_lines, v_lines, frame_shape):
    h, w = frame_shape[:2]
    pts = []
    for hl in h_lines:
        for vl in v_lines:
            pt = _intersect(np.asarray(hl), np.asarray(vl))
            if pt and 0 <= pt[0] < w and 0 <= pt[1] < h:
                pts.append(pt)
    return pts


# ── Debug visualization ──────────────────────────────────────────────────

def draw_debug_lines(
    frame: np.ndarray, debug: dict, alpha: float = 0.6,
) -> np.ndarray:
    """Draw detected/merged lines and intersections for debugging."""
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
