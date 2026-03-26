"""Court detection and homography for mapping player positions to 2D court coordinates.

This POC uses a configurable 4-point manual calibration.  You mark 4 known court
points once (e.g. the four corners of the court) and the module computes a
perspective transform to map pixel positions → real-world court metres.

For automatic court detection in a future iteration you would replace
`load_calibration` with a line/keypoint detector.
"""

import json
from pathlib import Path

import cv2
import numpy as np

# Official handball court dimensions (metres)
COURT_LENGTH = 40.0
COURT_WIDTH = 20.0

# Default 4 destination points on the 2D court (metres) corresponding to
# the four calibration source points.  Order: TL, TR, BR, BL of the court.
DEFAULT_DST = np.float32([
    [0.0, 0.0],
    [COURT_LENGTH, 0.0],
    [COURT_LENGTH, COURT_WIDTH],
    [0.0, COURT_WIDTH],
])


class CourtMapper:
    """Maps pixel coordinates to 2D court coordinates via homography."""

    def __init__(self, calibration_path: Path | None = None):
        self._H: np.ndarray | None = None
        self._src_pts: np.ndarray | None = None
        if calibration_path and calibration_path.exists():
            self.load_calibration(calibration_path)

    @property
    def is_calibrated(self) -> bool:
        return self._H is not None

    def load_calibration(self, path: Path):
        """
        Load a JSON calibration file mapping 4 pixel points to court corners.

        Expected format:
        {
            "src": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
            "dst": [[0,0], [40,0], [40,20], [0,20]]   // optional, defaults used
        }
        """
        data = json.loads(path.read_text())
        src = np.float32(data["src"])
        dst = np.float32(data.get("dst", DEFAULT_DST.tolist()))
        self._src_pts = src
        self._H, _ = cv2.findHomography(src, dst)

    def save_calibration(self, src_points: list[list[float]], path: Path):
        """Save calibration points to JSON."""
        data = {
            "src": [list(map(float, p)) for p in src_points],
            "dst": DEFAULT_DST.tolist(),
        }
        path.write_text(json.dumps(data, indent=2))
        self.load_calibration(path)

    def pixel_to_court(self, px: float, py: float) -> tuple[float, float] | None:
        """Map a pixel position to court coordinates (metres). Returns None if uncalibrated."""
        if self._H is None:
            return None
        pt = np.float32([[[px, py]]])
        mapped = cv2.perspectiveTransform(pt, self._H)
        cx, cy = float(mapped[0][0][0]), float(mapped[0][0][1])
        return (cx, cy)

    def is_on_court(self, px: float, py: float, margin: float = 3.0) -> bool:
        """Return True if a pixel position maps to within the court bounds (with margin)."""
        coords = self.pixel_to_court(px, py)
        if coords is None:
            return True  # If uncalibrated, assume on-court
        cx, cy = coords
        return (
            -margin <= cx <= COURT_LENGTH + margin
            and -margin <= cy <= COURT_WIDTH + margin
        )

    def foot_position(self, bbox: tuple[int, int, int, int]) -> tuple[float, float]:
        """Estimate foot position as bottom-center of bounding box (pixel coords)."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, float(y2))
