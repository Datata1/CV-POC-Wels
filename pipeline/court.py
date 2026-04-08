"""Court detection and homography for mapping player positions to 2D court coordinates.

Supports two modes:
1. Manual 4-point calibration (JSON file with known pixel↔court pairs).
2. Automatic keypoint detection: a YOLO object-detection model where each
   of the 30 court-landmark classes is a tiny bounding box.  The bbox
   center gives a pixel location; a CSV lookup gives the court coordinate.
   With ≥4 non-collinear detections → cv2.findHomography(RANSAC).
"""

import csv
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


# ── Keypoint-based automatic homography ──────────────────────────────────

# Default path to the court-keypoint coordinate mapping CSV.
_DEFAULT_KP_CSV = Path(__file__).resolve().parent.parent / "keypoints" / "court_keypoints.csv"


def _load_keypoint_map(csv_path: Path) -> dict[int, tuple[str, float, float]]:
    """Load keypoint CSV → {class_id: (name, court_x, court_y)}."""
    mapping: dict[int, tuple[str, float, float]] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            cid = int(row["id"])
            mapping[cid] = (row["name"], float(row["court_x_m"]), float(row["court_y_m"]))
    return mapping


class KeypointDetector:
    """Detects court landmarks via a YOLO object-detection model (30 classes).

    Each class corresponds to a specific court landmark.  The bbox center of
    each detection provides the pixel location; the CSV lookup provides the
    known real-world court coordinate.

    Usage::

        kp_det = KeypointDetector("models/handball_court_kp.pt")
        src, dst, names = kp_det.detect(frame)
        if len(src) >= 4:
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    """

    def __init__(
        self,
        model_path: str | Path,
        csv_path: Path = _DEFAULT_KP_CSV,
        conf_threshold: float = 0.2,
    ):
        from ultralytics import YOLO

        self._model = YOLO(str(model_path))
        self._kp_map = _load_keypoint_map(csv_path)
        self._conf = conf_threshold

    def detect(
        self,
        frame_bgr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Run detection on a single frame.

        Returns
        -------
        src_pts : np.ndarray, shape (N, 2)  — pixel coordinates (bbox centres)
        dst_pts : np.ndarray, shape (N, 2)  — court coordinates (metres)
        names   : list[str]                 — keypoint names for debugging
        """
        results = self._model.track(
            frame_bgr,
            conf=self._conf,
            imgsz=1920,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )
        src: list[tuple[float, float]] = []
        dst: list[tuple[float, float]] = []
        names: list[str] = []

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in self._kp_map:
                    continue
                name, cx_m, cy_m = self._kp_map[cls_id]
                # Pixel position = bbox centre
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                px = (x1 + x2) / 2.0
                py = (y1 + y2) / 2.0
                src.append((px, py))
                dst.append((cx_m, cy_m))
                names.append(name)

        if not src:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), []
        return np.float32(src), np.float32(dst), names

    def estimate_homography(
        self,
        frame_bgr: np.ndarray,
        ransac_thresh: float = 5.0,
    ) -> tuple[np.ndarray | None, dict]:
        """Detect keypoints and compute homography in one call.

        Returns (H, debug_info).  H is None when < 4 valid points detected.
        """
        src, dst, names = self.detect(frame_bgr)
        debug: dict = {
            "n_keypoints": len(names),
            "names": names,
            "src_pts": src.tolist() if len(src) else [],
            "dst_pts": dst.tolist() if len(dst) else [],
        }

        if len(src) < 4:
            return None, debug

        # Check for collinearity — need points not all on one line
        xs = dst[:, 0]
        ys = dst[:, 1]
        if np.ptp(xs) < 1.0 or np.ptp(ys) < 1.0:
            debug["rejected"] = "collinear"
            return None, debug

        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
        if H is None:
            return None, debug

        # Sanity: frame center should map roughly onto the court
        h, w = frame_bgr.shape[:2]
        center = cv2.perspectiveTransform(np.float32([[[w / 2, h / 2]]]), H)
        cx, cy = float(center[0][0][0]), float(center[0][0][1])
        if not (-10 <= cx <= 50 and -10 <= cy <= 30):
            debug["sanity_fail"] = (cx, cy)
            return None, debug

        inliers = int(mask.sum()) if mask is not None else len(src)
        debug["inliers"] = inliers
        return H, debug
