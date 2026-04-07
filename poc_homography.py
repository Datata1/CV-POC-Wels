#!/usr/bin/env python3
"""
POC: Manual Court Keypoint → Homography Validation

Proves that placing visible court keypoints manually enables accurate
homography mapping of detected players/ball to real court coordinates,
even when only part of the court is visible.

Steps:
    1. Read pre-selected frame images from input/poc-homography/
    2. Interactively place known court keypoints (same as Roboflow labeling)
    3. Run object detection (persons + ball) on the frame
    4. Compute homography and visualize mapped positions on 2D court

Output per frame goes to output/poc-homography/<image_stem>/

Usage:
    # Place images in input/poc-homography/ then:
    uv run python poc_homography.py

    # Or point to a different image directory:
    uv run python poc_homography.py --input path/to/images/

    # Legacy: extract from video
    uv run python poc_homography.py --input video.mp4 --frame 500 1200
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from pipeline.court_viz import (
    render_court,
    _draw_court_base,
    _m2px,
    CANVAS_W,
    CANVAS_H,
    MARGIN,
    COURT_LENGTH,
    COURT_WIDTH,
    BG_COLOR,
    LINE_COLOR,
)

INPUT_DIR = Path(__file__).parent / "input"
POC_INPUT_DIR = Path(__file__).parent / "input" / "poc-homography"
OUTPUT_DIR = Path(__file__).parent / "output" / "poc-homography"

FONT = cv2.FONT_HERSHEY_SIMPLEX

# ── Court keypoints with real-world coordinates (metres) ─────────────
# Based on IHF standard handball court (40m × 20m)
# x: 0..40 along length, y: 0..20 along width
# y=0 = far sideline, y=20 = near sideline (camera side)
#
# Names match the Roboflow annotation classes (roboflow_classes.csv).

COURT_KEYPOINTS = {
    # Court corners
    "court_TL":       (0.0, 0.0),
    "court_TR":       (40.0, 0.0),
    "court_BL":       (0.0, 20.0),
    "court_BR":       (40.0, 20.0),
    # Center line
    "center_T":       (20.0, 0.0),
    "center_B":       (20.0, 20.0),
    "center_spot":    (20.0, 10.0),
    # Left goal area
    "goalpost_L_T":   (0.0, 8.5),
    "goalpost_L_B":   (0.0, 11.5),
    "6m_base_L_T":    (0.0, 7.0),
    "6m_base_L_B":    (0.0, 13.0),
    "6m_vertex_L_T":  (6.0, 8.5),
    "6m_vertex_L_B":  (6.0, 11.5),
    "9m_base_L_T":    (3.0, 0.0),
    "9m_base_L_B":    (3.0, 20.0),
    "9m_vertex_L_T":  (9.0, 8.5),
    "9m_vertex_L_B":  (9.0, 11.5),
    "7m_L":           (7.0, 10.0),
    "4m_L":           (4.0, 10.0),
    # Right goal area
    "goalpost_R_T":   (40.0, 8.5),
    "goalpost_R_B":   (40.0, 11.5),
    "6m_base_R_T":    (40.0, 7.0),
    "6m_base_R_B":    (40.0, 13.0),
    "6m_vertex_R_T":  (34.0, 8.5),
    "6m_vertex_R_B":  (34.0, 11.5),
    "9m_base_R_T":    (37.0, 0.0),
    "9m_base_R_B":    (37.0, 20.0),
    "9m_vertex_R_T":  (31.0, 8.5),
    "9m_vertex_R_B":  (31.0, 11.5),
    "7m_R":           (33.0, 10.0),
    "4m_R":           (36.0, 10.0),
}

KEYPOINT_NAMES = list(COURT_KEYPOINTS.keys())

# Colors
KP_PLACED = (0, 255, 0)
KP_CURRENT = (0, 0, 255)
PERSON_COLOR = (255, 100, 0)
BALL_COLOR = (0, 255, 255)


# ── Helpers ──────────────────────────────────────────────────────────


def find_video(input_dir: Path) -> Path:
    for f in sorted(input_dir.iterdir()):
        if f.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
            return f
    raise FileNotFoundError(f"No video in {input_dir}")


def extract_frame(video_path: Path, frame_num: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_num >= total:
        print(f"Frame {frame_num} out of range (video has {total} frames)")
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


# ── Interactive keypoint placement ───────────────────────────────────


# Side-panel dimensions for GUI
PANEL_W = 420
PANEL_BG = (25, 25, 25)


class KeypointPlacer:
    """GUI for manually placing court keypoints on a video frame.

    Shows the frame on the left and a large info panel on the right with:
    - Current keypoint name + court position in big text
    - A clear court diagram with T/B/L/R labels
    - Placed keypoints list + controls
    """

    def __init__(self, frame: np.ndarray):
        self.original = frame.copy()
        self.fh, self.fw = frame.shape[:2]
        self.current_idx = 0
        self.placed: dict[str, tuple[int, int]] = {}
        self.order: list[str] = []
        self.done = False
        self.canvas: np.ndarray | None = None

    # ── panel drawing helpers ──

    def _build_court_diagram(self, panel: np.ndarray, y_start: int) -> int:
        """Draw a clear court diagram on the side panel. Returns y after."""
        pw = PANEL_W
        cw, ch = pw - 40, int((pw - 40) * 0.5)  # court aspect ~2:1
        x0, y0 = 20, y_start

        # Background
        cv2.rectangle(panel, (x0 - 2, y0 - 2), (x0 + cw + 2, y0 + ch + 2),
                      (60, 60, 60), -1)
        cv2.rectangle(panel, (x0, y0), (x0 + cw, y0 + ch), (40, 80, 40), -1)

        sx = cw / 40.0
        sy = ch / 20.0

        def m2p(mx, my):
            return int(x0 + mx * sx), int(y0 + my * sy)

        # Court outline + center
        cv2.rectangle(panel, m2p(0, 0), m2p(40, 20), (120, 180, 90), 1)
        cv2.line(panel, m2p(20, 0), m2p(20, 20), (100, 140, 70), 1)

        # 6m arcs (approximate as rectangles for clarity)
        cv2.rectangle(panel, m2p(0, 7), m2p(6, 13), (100, 140, 70), 1)
        cv2.rectangle(panel, m2p(34, 7), m2p(40, 13), (100, 140, 70), 1)

        # Corner labels: T=top(far), B=bottom(near)
        cv2.putText(panel, "T (far sideline, y=0)",
                    (x0, y0 - 8), FONT, 0.42, (180, 180, 180), 1)
        cv2.putText(panel, "B (near/camera, y=20)",
                    (x0, y0 + ch + 16), FONT, 0.42, (180, 180, 180), 1)
        cv2.putText(panel, "L", (x0 - 15, y0 + ch // 2 + 4),
                    FONT, 0.45, (180, 180, 180), 1)
        cv2.putText(panel, "R", (x0 + cw + 4, y0 + ch // 2 + 4),
                    FONT, 0.45, (180, 180, 180), 1)

        # Goal labels
        cv2.putText(panel, "GOAL", (m2p(0, 9)[0] + 2, m2p(0, 10)[1] + 4),
                    FONT, 0.3, (200, 200, 200), 1)
        cv2.putText(panel, "GOAL", (m2p(37, 9)[0], m2p(37, 10)[1] + 4),
                    FONT, 0.3, (200, 200, 200), 1)

        # Draw all keypoint positions
        for name, (cx, cy) in COURT_KEYPOINTS.items():
            px, py = m2p(cx, cy)
            if name in self.placed:
                cv2.circle(panel, (px, py), 4, KP_PLACED, -1)
                cv2.circle(panel, (px, py), 4, (255, 255, 255), 1)
            elif (self.current_idx < len(KEYPOINT_NAMES)
                  and name == KEYPOINT_NAMES[self.current_idx]):
                # Highlight current with large pulsing marker
                cv2.circle(panel, (px, py), 8, KP_CURRENT, 2)
                cv2.circle(panel, (px, py), 3, KP_CURRENT, -1)
                # Label near it
                lx = px + 10 if cx < 30 else px - 80
                cv2.putText(panel, name, (lx, py - 10),
                            FONT, 0.35, KP_CURRENT, 1)
            else:
                cv2.circle(panel, (px, py), 2, (80, 80, 80), -1)

        return y0 + ch + 24

    def _build_panel(self) -> np.ndarray:
        """Build the right-side info panel."""
        panel = np.full((self.fh, PANEL_W, 3), PANEL_BG, dtype=np.uint8)
        y = 20

        # ── Title ──
        cv2.putText(panel, "COURT KEYPOINT PLACER",
                    (10, y + 16), FONT, 0.55, (200, 200, 200), 1)
        y += 35
        cv2.line(panel, (10, y), (PANEL_W - 10, y), (80, 80, 80), 1)
        y += 15

        # ── Current keypoint ──
        if self.current_idx < len(KEYPOINT_NAMES):
            name = KEYPOINT_NAMES[self.current_idx]
            cx, cy = COURT_KEYPOINTS[name]

            cv2.putText(panel, f"[{self.current_idx+1}/{len(KEYPOINT_NAMES)}] Place:",
                        (10, y + 14), FONT, 0.45, (150, 150, 150), 1)
            y += 28

            # Big keypoint name
            cv2.putText(panel, name,
                        (10, y + 24), FONT, 0.85, KP_CURRENT, 2)
            y += 38

            # Court coords
            cv2.putText(panel, f"Court: ({cx}, {cy}) metres",
                        (10, y + 14), FONT, 0.5, (180, 180, 255), 1)
            y += 22

            # Explain what T/B means for this keypoint
            hint = ""
            if "_T" in name:
                hint = "T = TOP = far sideline (away from camera)"
            elif "_B" in name:
                hint = "B = BOTTOM = near sideline (camera side)"
            if "_L" in name:
                hint += ("  |  " if hint else "") + "L = LEFT goal side (x=0)"
            elif "_R" in name:
                hint += ("  |  " if hint else "") + "R = RIGHT goal side (x=40)"
            if hint:
                # Wrap long hints
                words = hint.split("  |  ")
                for line in words:
                    cv2.putText(panel, line.strip(),
                                (10, y + 14), FONT, 0.38, (100, 200, 255), 1)
                    y += 18
            y += 8
        else:
            cv2.putText(panel, "All keypoints reviewed.",
                        (10, y + 14), FONT, 0.5, (200, 200, 200), 1)
            y += 28
            cv2.putText(panel, "Press 'd' to finish.",
                        (10, y + 14), FONT, 0.5, (200, 200, 200), 1)
            y += 30

        # ── Court diagram ──
        cv2.line(panel, (10, y), (PANEL_W - 10, y), (80, 80, 80), 1)
        y += 10
        y = self._build_court_diagram(panel, y)
        y += 5

        # ── Controls ──
        cv2.line(panel, (10, y), (PANEL_W - 10, y), (80, 80, 80), 1)
        y += 18
        controls = [
            ("click", "place keypoint"),
            ("n", "skip (not visible)"),
            ("u", "undo last"),
            ("d / Enter", "done (need 4+)"),
            ("q / Esc", "quit / cancel"),
        ]
        for key, desc in controls:
            cv2.putText(panel, key, (15, y), FONT, 0.42, (100, 220, 255), 1)
            cv2.putText(panel, f"- {desc}", (100, y), FONT, 0.40, (170, 170, 170), 1)
            y += 20
        y += 8

        # ── Placed counter ──
        cv2.line(panel, (10, y), (PANEL_W - 10, y), (80, 80, 80), 1)
        y += 20
        color = KP_PLACED if len(self.placed) >= 4 else (0, 120, 255)
        cv2.putText(panel, f"Placed: {len(self.placed)}  (min 4)",
                    (10, y), FONT, 0.55, color, 1)
        y += 22

        # List last ~15 placed keypoints (most recent first)
        shown = list(self.placed.keys())[-15:]
        for name in reversed(shown):
            cx, cy = COURT_KEYPOINTS[name]
            cv2.putText(panel, f"  {name} ({cx},{cy})",
                        (10, y), FONT, 0.33, KP_PLACED, 1)
            y += 15

        return panel

    # ── compositing ──

    def _redraw(self):
        frame_vis = self.original.copy()

        # Draw placed keypoints on frame
        for name, (px, py) in self.placed.items():
            cv2.circle(frame_vis, (px, py), 8, KP_PLACED, -1)
            cv2.circle(frame_vis, (px, py), 8, (255, 255, 255), 2)
            cx, cy = COURT_KEYPOINTS[name]
            cv2.putText(frame_vis, f"{name}",
                        (px + 12, py - 8), FONT, 0.5, KP_PLACED, 1)
            cv2.putText(frame_vis, f"({cx},{cy})",
                        (px + 12, py + 12), FONT, 0.4, (200, 255, 200), 1)

        # Crosshair hint at current keypoint label on frame top bar
        bar_h = 36
        cv2.rectangle(frame_vis, (0, 0), (self.fw, bar_h), (0, 0, 0), -1)
        if self.current_idx < len(KEYPOINT_NAMES):
            name = KEYPOINT_NAMES[self.current_idx]
            cv2.putText(frame_vis,
                        f"Click to place: {name}",
                        (10, 26), FONT, 0.65, KP_CURRENT, 2)
        else:
            cv2.putText(frame_vis,
                        "All done - press 'd' to finish",
                        (10, 26), FONT, 0.65, (200, 200, 200), 2)

        # Build side panel
        panel = self._build_panel()

        # Composite: frame | panel
        self.canvas = np.hstack([frame_vis, panel])

    # ── event handling ──

    def _on_mouse(self, event, x, y, flags, param):
        # Only accept clicks on the frame region (not the panel)
        if (event == cv2.EVENT_LBUTTONDOWN
                and x < self.fw
                and self.current_idx < len(KEYPOINT_NAMES)):
            name = KEYPOINT_NAMES[self.current_idx]
            self.placed[name] = (x, y)
            self.order.append(name)
            self.current_idx += 1
            self._redraw()
            cv2.imshow("Place Court Keypoints", self.canvas)

    def run(self) -> dict[str, tuple[int, int]] | None:
        """Open interactive window. Returns placed keypoints or None if cancelled."""
        win = "Place Court Keypoints"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        # Size: frame width scaled to ~1100px + panel
        disp_w = 1100 + PANEL_W
        scale = 1100 / self.fw
        disp_h = int(self.fh * scale)
        cv2.resizeWindow(win, disp_w, disp_h)
        cv2.setMouseCallback(win, self._on_mouse)

        self._redraw()
        cv2.imshow(win, self.canvas)

        while not self.done:
            key = cv2.waitKey(30) & 0xFF

            if key == ord("n"):  # skip keypoint
                if self.current_idx < len(KEYPOINT_NAMES):
                    self.current_idx += 1
                    self._redraw()
                    cv2.imshow(win, self.canvas)

            elif key == ord("u"):  # undo last
                if self.order:
                    last = self.order.pop()
                    del self.placed[last]
                    self.current_idx = KEYPOINT_NAMES.index(last)
                    self._redraw()
                    cv2.imshow(win, self.canvas)

            elif key in (ord("d"), 13):  # done
                if len(self.placed) >= 4:
                    self.done = True
                else:
                    msg = self.canvas.copy()
                    h, w = msg.shape[:2]
                    cv2.putText(msg, "Need at least 4 keypoints!",
                                (w // 4, h // 2), FONT, 1.2, (0, 0, 255), 3)
                    cv2.imshow(win, msg)
                    cv2.waitKey(1500)
                    self._redraw()
                    cv2.imshow(win, self.canvas)

            elif key in (ord("q"), 27):  # quit
                cv2.destroyAllWindows()
                return None

        cv2.destroyAllWindows()
        return self.placed


# ── Step 1: keypoints visualization ──────────────────────────────────


def draw_step1_keypoints(frame: np.ndarray,
                         placed: dict[str, tuple[int, int]]) -> np.ndarray:
    vis = frame.copy()
    h, w = vis.shape[:2]

    for name, (px, py) in placed.items():
        cx, cy = COURT_KEYPOINTS[name]
        cv2.circle(vis, (px, py), 8, KP_PLACED, -1)
        cv2.circle(vis, (px, py), 8, (255, 255, 255), 2)
        label = f"{name} ({cx},{cy}m)"
        cv2.putText(vis, label, (px + 10, py - 6), FONT, 0.45, KP_PLACED, 1)

    # Connect keypoints that share an edge on the court boundary
    _draw_connections(vis, placed)

    cv2.rectangle(vis, (0, 0), (w, 35), (0, 0, 0), -1)
    cv2.putText(vis,
                f"Step 1: Manual Court Keypoints ({len(placed)} placed)",
                (10, 25), FONT, 0.6, KP_PLACED, 1)
    return vis


def _draw_connections(vis, placed):
    """Draw lines between placed keypoints that form court edges."""
    edges = [
        ("court_TL", "court_TR"), ("court_TR", "court_BR"),
        ("court_BR", "court_BL"), ("court_BL", "court_TL"),
        ("center_T", "center_B"),
        ("court_TL", "center_T"), ("center_T", "court_TR"),
        ("court_BL", "center_B"), ("center_B", "court_BR"),
        ("6m_base_L_T", "6m_vertex_L_T"), ("6m_vertex_L_T", "6m_vertex_L_B"),
        ("6m_vertex_L_B", "6m_base_L_B"),
        ("6m_base_R_T", "6m_vertex_R_T"), ("6m_vertex_R_T", "6m_vertex_R_B"),
        ("6m_vertex_R_B", "6m_base_R_B"),
        ("goalpost_L_T", "goalpost_L_B"), ("goalpost_R_T", "goalpost_R_B"),
    ]
    for a, b in edges:
        if a in placed and b in placed:
            cv2.line(vis, placed[a], placed[b], (0, 180, 0), 1, cv2.LINE_AA)


# ── Step 2: object detection ─────────────────────────────────────────


def detect_objects(frame: np.ndarray,
                   person_model: YOLO,
                   ball_model: YOLO | None) -> tuple[list[dict], list[dict]]:
    """Single-frame detection (no tracking) for persons and ball."""
    import time
    t0 = time.perf_counter()
    person_res = person_model.predict(frame, classes=[0], conf=0.3, verbose=False)
    t_person = time.perf_counter() - t0
    persons = []
    for r in person_res:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            persons.append({
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "conf": float(box.conf[0]),
            })
    persons.sort(key=lambda d: d["conf"], reverse=True)
    persons = persons[:20]
    print(f"    person model: {len(persons)} detections in {t_person:.2f}s")

    balls = []
    if ball_model:
        t0 = time.perf_counter()
        ball_res = ball_model.predict(frame, conf=0.25, verbose=False)
        for r in ball_res:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                balls.append({
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "conf": float(box.conf[0]),
                })
        t_ball = time.perf_counter() - t0
        balls.sort(key=lambda d: d["conf"], reverse=True)
        balls = balls[:1]
        print(f"    ball model:   {len(balls)} detections in {t_ball:.2f}s")

    return persons, balls


def draw_step2_detections(frame: np.ndarray,
                          persons: list[dict],
                          balls: list[dict]) -> np.ndarray:
    vis = frame.copy()
    h, w = vis.shape[:2]

    for i, p in enumerate(persons):
        x1, y1, x2, y2 = p["bbox"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), PERSON_COLOR, 2)
        lbl = f"Person {i+1} ({p['conf']:.0%})"
        cv2.putText(vis, lbl, (x1, y1 - 8), FONT, 0.4, PERSON_COLOR, 1)
        # Foot point
        fx, fy = (x1 + x2) // 2, y2
        cv2.circle(vis, (fx, fy), 4, (255, 255, 255), -1)

    for b in balls:
        x1, y1, x2, y2 = b["bbox"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), BALL_COLOR, 2)
        cv2.putText(vis, f"Ball ({b['conf']:.0%})", (x1, y1 - 8),
                    FONT, 0.4, BALL_COLOR, 1)

    cv2.rectangle(vis, (0, 0), (w, 35), (0, 0, 0), -1)
    cv2.putText(vis,
                f"Step 2: Object Detection ({len(persons)} persons, {len(balls)} balls)",
                (10, 25), FONT, 0.6, PERSON_COLOR, 1)
    return vis


# ── Step 3: homography ───────────────────────────────────────────────


def compute_homography(placed: dict[str, tuple[int, int]]):
    """Compute homography from pixel↔court correspondences."""
    src, dst = [], []
    for name, (px, py) in placed.items():
        cx, cy = COURT_KEYPOINTS[name]
        src.append([px, py])
        dst.append([cx, cy])

    src = np.float32(src)
    dst = np.float32(dst)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    inliers = int(mask.sum()) if mask is not None else 0
    return H, inliers, len(src)


def draw_step3_court(H: np.ndarray,
                     persons: list[dict],
                     balls: list[dict],
                     placed: dict[str, tuple[int, int]],
                     frame_num: int) -> np.ndarray:
    """Render top-down court with mapped detections."""
    # Map persons
    court_players = []
    for p in persons:
        x1, y1, x2, y2 = p["bbox"]
        foot = np.float32([[[(x1 + x2) / 2.0, float(y2)]]])
        mapped = cv2.perspectiveTransform(foot, H)
        cx, cy = float(mapped[0][0][0]), float(mapped[0][0][1])
        court_players.append({
            "court_pos": [cx, cy],
            "team": "unknown",
            "track_id": "",
        })

    # Map ball
    court_ball = None
    if balls:
        b = balls[0]
        x1, y1, x2, y2 = b["bbox"]
        center = np.float32([[[(x1 + x2) / 2.0, (y1 + y2) / 2.0]]])
        mapped = cv2.perspectiveTransform(center, H)
        court_ball = {"court_pos": [float(mapped[0][0][0]), float(mapped[0][0][1])]}

    court_img = render_court(court_players, court_ball, frame_id=frame_num)

    # Draw placed keypoints on the court (green diamonds) to verify mapping
    for name in placed:
        cx, cy = COURT_KEYPOINTS[name]
        px, py = _m2px(cx, cy)
        pts = np.array([(px, py-5), (px+5, py), (px, py+5), (px-5, py)], np.int32)
        cv2.fillPoly(court_img, [pts], KP_PLACED)
        cv2.putText(court_img, name, (px + 7, py - 2), FONT, 0.25, KP_PLACED, 1)

    # Title
    cv2.rectangle(court_img, (0, 0), (CANVAS_W, 35), (0, 0, 0), -1)
    cv2.putText(court_img,
                f"Step 3: Homography — {len(court_players)} players mapped to court",
                (10, 25), FONT, 0.5, (0, 200, 255), 1)

    return court_img, court_players, court_ball


def draw_step4_warp(frame: np.ndarray,
                    H: np.ndarray,
                    placed: dict[str, tuple[int, int]]) -> np.ndarray:
    """Warp the video frame onto the 2D court canvas (bird's-eye overlay)."""
    # H maps pixel → court_metres.  We need pixel → canvas_pixels.
    _sx = (CANVAS_W - 2 * MARGIN) / COURT_LENGTH
    _sy = (CANVAS_H - 2 * MARGIN) / COURT_WIDTH
    S = np.float64([[_sx, 0, MARGIN],
                     [0, _sy, MARGIN],
                     [0,  0,  1]])
    H_canvas = S @ H

    warped = cv2.warpPerspective(frame, H_canvas, (CANVAS_W, CANVAS_H))

    # Court base
    canvas = np.full((CANVAS_H, CANVAS_W, 3), BG_COLOR, dtype=np.uint8)
    _draw_court_base(canvas)

    # Blend warped frame with court lines
    mask = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    mask = (mask > 10).astype(np.uint8) * 255
    blended = canvas.copy()
    alpha = 0.5
    np.copyto(blended, cv2.addWeighted(canvas, 1 - alpha, warped, alpha, 0),
              where=(mask[..., None] > 0))

    # Redraw court lines on top for visibility
    _draw_court_base(blended)

    cv2.rectangle(blended, (0, 0), (CANVAS_W, 35), (0, 0, 0), -1)
    cv2.putText(blended,
                "Step 4: Warped Frame (bird's-eye overlay)",
                (10, 25), FONT, 0.5, (200, 200, 0), 1)
    return blended


# ── Combined summary image ───────────────────────────────────────────


def create_combined(kp_img, det_img, court_img, warp_img):
    """Stacked: detection image on top, court mapping on bottom."""
    target_w = 1280

    def fit(img, tw, th):
        """Resize img to fit inside tw×th, letterbox with black."""
        h, w = img.shape[:2]
        scale = min(tw / w, th / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (nw, nh))
        canvas = np.zeros((th, tw, 3), dtype=np.uint8)
        y0 = (th - nh) // 2
        x0 = (tw - nw) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = resized
        return canvas

    top = fit(det_img, target_w, 540)
    bot = fit(court_img, target_w, 400)
    return np.vstack([top, bot])


# ── Main ─────────────────────────────────────────────────────────────


def collect_frames(args) -> list[tuple[str, np.ndarray]]:
    """Return list of (name, image) pairs from either images dir or video."""
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    # Mode 1: images in input/poc-homography/
    images_dir = Path(args.input) if args.input else POC_INPUT_DIR
    if images_dir.is_dir():
        files = sorted(
            f for f in images_dir.iterdir()
            if f.suffix.lower() in image_exts
        )
        if not files:
            print(f"No images found in {images_dir}")
            sys.exit(1)
        print(f"Found {len(files)} images in {images_dir}")
        frames = []
        for f in files:
            img = cv2.imread(str(f))
            if img is not None:
                frames.append((f.stem, img))
            else:
                print(f"  WARNING: could not read {f.name}")
        return frames

    # Mode 2: extract from video (legacy --frame support)
    if args.input and Path(args.input).is_file():
        video_path = Path(args.input)
    else:
        video_path = find_video(INPUT_DIR)
    print(f"Video: {video_path}")
    frame_nums = args.frame or [0]
    frames = []
    for n in frame_nums:
        img = extract_frame(video_path, n)
        if img is not None:
            frames.append((f"frame_{n:05d}", img))
    return frames


def main():
    parser = argparse.ArgumentParser(
        description="POC: Manual court keypoints → homography validation")
    parser.add_argument("--frame", type=int, nargs="+", default=None,
                        help="Frame number(s) to extract from video")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to image directory or video file")
    parser.add_argument("--person-model", type=str, default="yolo11n.pt",
                        help="YOLO model for person detection")
    parser.add_argument("--ball-model", type=str, default="models/handball_ball.pt",
                        help="YOLO model for ball detection")
    args = parser.parse_args()

    frames = collect_frames(args)
    if not frames:
        print("No frames to process.")
        sys.exit(1)

    print("Loading models...")
    print(f"  Person model: {args.person_model}")
    person_model = YOLO(args.person_model)
    ball_path = Path(args.ball_model)
    if ball_path.exists():
        print(f"  Ball model:   {ball_path}")
        ball_model = YOLO(str(ball_path))
    else:
        print(f"  Ball model not found ({ball_path}), using COCO sports-ball class")
        ball_model = None

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for frame_name, frame in frames:
        print(f"\n{'='*60}")
        print(f"{frame_name}")
        print(f"{'='*60}")

        frame_dir = OUTPUT_DIR / frame_name
        frame_dir.mkdir(parents=True, exist_ok=True)

        # Save raw
        cv2.imwrite(str(frame_dir / "00_raw.jpg"), frame)

        # ── Step 1: interactive keypoint placement ──
        print("\nKeypoint placement GUI")
        print("  click = place | n = skip | u = undo | d = done | q = quit")
        placer = KeypointPlacer(frame)
        placed = placer.run()
        if placed is None:
            print("Cancelled.")
            continue

        print(f"\n  Placed {len(placed)} keypoints:")
        for name, (px, py) in placed.items():
            cx, cy = COURT_KEYPOINTS[name]
            print(f"    {name}: pixel({px},{py}) → court({cx},{cy})m")

        kp_img = draw_step1_keypoints(frame, placed)
        cv2.imwrite(str(frame_dir / "01_keypoints.jpg"), kp_img)
        print(f"  → {frame_name}/01_keypoints.jpg")

        # ── Step 2: detection ──
        print("\nRunning detection...")
        persons, balls = detect_objects(frame, person_model, ball_model)
        det_img = draw_step2_detections(frame, persons, balls)
        cv2.imwrite(str(frame_dir / "02_detections.jpg"), det_img)
        print(f"  {len(persons)} persons, {len(balls)} balls")
        print(f"  → {frame_name}/02_detections.jpg")

        # ── Step 3: homography + court mapping ──
        print("\nComputing homography...")
        H, inliers, total = compute_homography(placed)
        if H is None:
            print("  ERROR: homography failed!")
            continue
        print(f"  {inliers}/{total} inliers")

        court_img, court_players, court_ball = draw_step3_court(
            H, persons, balls, placed, 0)
        cv2.imwrite(str(frame_dir / "03_court.jpg"), court_img)
        print(f"  → {frame_name}/03_court.jpg")

        # Mapped positions
        for i, cp in enumerate(court_players):
            cx, cy = cp["court_pos"]
            tag = "  ON" if 0 <= cx <= 40 and 0 <= cy <= 20 else " OFF"
            print(f"    Player {i+1}: ({cx:5.1f}, {cy:5.1f})m{tag}")
        if court_ball:
            bx, by = court_ball["court_pos"]
            print(f"    Ball:     ({bx:5.1f}, {by:5.1f})m")

        # ── Step 4: warped overlay ──
        warp_img = draw_step4_warp(frame, H, placed)
        cv2.imwrite(str(frame_dir / "04_warp.jpg"), warp_img)
        print(f"  → {frame_name}/04_warp.jpg")

        # ── Combined ──
        combined = create_combined(kp_img, det_img, court_img, warp_img)
        cv2.imwrite(str(frame_dir / "05_combined.jpg"), combined)
        print(f"  → {frame_name}/05_combined.jpg")

    print(f"\nAll outputs in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
