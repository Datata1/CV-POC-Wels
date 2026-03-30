"""2D court visualizer — draws a top-down handball court with player positions.

Renders a bird's-eye-view image of the handball court (40m × 20m) with
player dots color-coded by team and an optional ball marker.
"""

import cv2
import numpy as np

# Court dimensions (metres)
COURT_LENGTH = 40.0
COURT_WIDTH = 20.0

# Visualisation canvas settings
CANVAS_W = 800   # pixels
CANVAS_H = 400   # pixels
MARGIN = 40      # pixels around the court

# Derived scale: metres → pixels
_COURT_PX_W = CANVAS_W - 2 * MARGIN
_COURT_PX_H = CANVAS_H - 2 * MARGIN
_SCALE_X = _COURT_PX_W / COURT_LENGTH
_SCALE_Y = _COURT_PX_H / COURT_WIDTH

# Colors (BGR)
BG_COLOR = (40, 40, 40)
COURT_COLOR = (80, 120, 60)
LINE_COLOR = (255, 255, 255)
TEAM_COLORS = {
    "A": (255, 100, 0),
    "B": (0, 100, 255),
    "unknown": (180, 180, 180),
}
BALL_COLOR = (0, 255, 255)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _m2px(mx: float, my: float) -> tuple[int, int]:
    """Convert court metres to canvas pixel coordinates."""
    px = int(MARGIN + mx * _SCALE_X)
    py = int(MARGIN + my * _SCALE_Y)
    return px, py


def _draw_court_base(canvas: np.ndarray):
    """Draw the handball court outline and markings."""
    tl = _m2px(0, 0)
    br = _m2px(COURT_LENGTH, COURT_WIDTH)
    cv2.rectangle(canvas, tl, br, COURT_COLOR, -1)
    cv2.rectangle(canvas, tl, br, LINE_COLOR, 2)

    # Center line
    ct = _m2px(COURT_LENGTH / 2, 0)
    cb = _m2px(COURT_LENGTH / 2, COURT_WIDTH)
    cv2.line(canvas, ct, cb, LINE_COLOR, 1)

    # Center circle (not official in handball, but useful marker)
    cc = _m2px(COURT_LENGTH / 2, COURT_WIDTH / 2)
    r = int(1.0 * _SCALE_X)  # ~1m radius visual marker
    cv2.circle(canvas, cc, r, LINE_COLOR, 1)

    # 6m goal-area arcs (left and right)
    for goal_x in [0.0, COURT_LENGTH]:
        cx, cy = _m2px(goal_x, COURT_WIDTH / 2)
        axes = (int(6.0 * _SCALE_X), int(6.0 * _SCALE_Y))
        if goal_x == 0:
            cv2.ellipse(canvas, (cx, cy), axes, 0, -90, 90, LINE_COLOR, 1)
        else:
            cv2.ellipse(canvas, (cx, cy), axes, 0, 90, 270, LINE_COLOR, 1)

    # 9m dashed arcs
    for goal_x in [0.0, COURT_LENGTH]:
        cx, cy = _m2px(goal_x, COURT_WIDTH / 2)
        axes = (int(9.0 * _SCALE_X), int(9.0 * _SCALE_Y))
        if goal_x == 0:
            for a in range(-90, 90, 10):
                cv2.ellipse(canvas, (cx, cy), axes, 0, a, a + 5, LINE_COLOR, 1)
        else:
            for a in range(90, 270, 10):
                cv2.ellipse(canvas, (cx, cy), axes, 0, a, a + 5, LINE_COLOR, 1)

    # Goal rectangles (3m wide)
    for goal_x in [0.0, COURT_LENGTH]:
        top = _m2px(goal_x, COURT_WIDTH / 2 - 1.5)
        bot = _m2px(goal_x, COURT_WIDTH / 2 + 1.5)
        if goal_x == 0:
            g_tl = (top[0] - 12, top[1])
            g_br = (bot[0], bot[1])
        else:
            g_tl = (top[0], top[1])
            g_br = (bot[0] + 12, bot[1])
        cv2.rectangle(canvas, g_tl, g_br, LINE_COLOR, 2)

    # 7m marks
    for x7 in [7.0, COURT_LENGTH - 7.0]:
        pt = _m2px(x7, COURT_WIDTH / 2)
        cv2.line(canvas, (pt[0] - 4, pt[1]), (pt[0] + 4, pt[1]), LINE_COLOR, 2)

    # Dimension labels
    cv2.putText(canvas, "40m", _m2px(COURT_LENGTH / 2 - 1.5, -1.5), FONT, 0.35, LINE_COLOR, 1)
    cv2.putText(canvas, "20m", (5, MARGIN + _COURT_PX_H // 2), FONT, 0.35, LINE_COLOR, 1)


def render_court(
    players: list[dict],
    ball: dict | None = None,
    frame_id: int = 0,
    timestamp_s: float = 0.0,
) -> np.ndarray:
    """Render a single frame of the 2D court with player positions.

    Args:
        players: list of dicts with 'court_pos' [x, y], 'team', 'track_id'
        ball: dict with 'court_pos' [x, y] or None
        frame_id: for HUD display
        timestamp_s: for HUD display

    Returns:
        BGR image (CANVAS_H x CANVAS_W)
    """
    canvas = np.full((CANVAS_H, CANVAS_W, 3), BG_COLOR, dtype=np.uint8)
    _draw_court_base(canvas)

    # Draw players
    for p in players:
        cp = p.get("court_pos")
        if cp is None:
            continue
        mx, my = cp
        if not (-5 <= mx <= COURT_LENGTH + 5 and -5 <= my <= COURT_WIDTH + 5):
            continue
        px, py = _m2px(mx, my)
        team = p.get("team", "unknown")
        color = TEAM_COLORS.get(team, TEAM_COLORS["unknown"])
        cv2.circle(canvas, (px, py), 7, color, -1)
        cv2.circle(canvas, (px, py), 7, (255, 255, 255), 1)
        tid = p.get("track_id", "?")
        cv2.putText(canvas, str(tid), (px + 9, py + 4), FONT, 0.32, color, 1)

    # Draw ball
    if ball and ball.get("court_pos"):
        bx, by = ball["court_pos"]
        bpx, bpy = _m2px(bx, by)
        cv2.circle(canvas, (bpx, bpy), 5, BALL_COLOR, -1)
        cv2.circle(canvas, (bpx, bpy), 5, (255, 255, 255), 1)

    # HUD
    cv2.putText(
        canvas,
        f"Frame {frame_id} | {timestamp_s:.1f}s | Players on court: {len([p for p in players if p.get('court_pos')])}",
        (MARGIN, CANVAS_H - 10), FONT, 0.35, (200, 200, 200), 1,
    )

    return canvas
