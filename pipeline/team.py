"""Team classification via jersey color clustering (HSV histogram + K-Means)."""

import cv2
import numpy as np
from sklearn.cluster import KMeans


class TeamClassifier:
    """
    Assigns each player to a team based on jersey color.

    Works by extracting a color histogram from the torso region of each
    player crop, then clustering all histograms in the frame into N teams.
    Referees / goalkeepers may form a third cluster.

    The classifier re-fits every `refit_interval` frames using a sliding
    window of recent histograms to maintain stable cluster assignments.
    """

    def __init__(self, n_teams: int = 2, refit_interval: int = 30):
        self._n_teams = n_teams
        self._refit_interval = refit_interval
        self._kmeans: KMeans | None = None
        self._frame_count = 0
        self._hist_buffer: list[np.ndarray] = []
        # Cluster-to-team label mapping (stable across refits)
        self._label_map: dict[int, str] | None = None

    @staticmethod
    def _extract_torso_hist(frame_bgr: np.ndarray, bbox: tuple) -> np.ndarray | None:
        """Extract an HSV color histogram from the torso region of a bounding box."""
        x1, y1, x2, y2 = bbox
        h = y2 - y1
        w = x2 - x1
        if h < 10 or w < 10:
            return None

        # Torso: middle 40% vertically, center 60% horizontally
        ty1 = y1 + int(h * 0.2)
        ty2 = y1 + int(h * 0.6)
        tx1 = x1 + int(w * 0.2)
        tx2 = x2 - int(w * 0.2)

        torso = frame_bgr[ty1:ty2, tx1:tx2]
        if torso.size == 0:
            return None

        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        # 2D histogram: Hue (16 bins) x Saturation (8 bins), ignore very dark pixels
        hist = cv2.calcHist(
            [hsv], [0, 1], None, [16, 8], [0, 180, 30, 256]
        )
        cv2.normalize(hist, hist)
        return hist.flatten()

    def classify(
        self, frame_bgr: np.ndarray, detections: list[dict],
    ) -> list[dict]:
        """
        Add a 'team' key ("A", "B", or "unknown") to each detection dict.

        Args:
            frame_bgr: full BGR frame
            detections: list of dicts with 'bbox' (x1, y1, x2, y2)
        """
        hists = []
        valid_indices = []
        for i, det in enumerate(detections):
            h = self._extract_torso_hist(frame_bgr, det["bbox"])
            if h is not None:
                hists.append(h)
                valid_indices.append(i)
            else:
                det["team"] = "unknown"

        if len(hists) < 2:
            for i in valid_indices:
                detections[i]["team"] = "unknown"
            return detections

        hists_arr = np.array(hists)
        self._hist_buffer.extend(hists)
        # Keep buffer bounded
        max_buf = 200
        if len(self._hist_buffer) > max_buf:
            self._hist_buffer = self._hist_buffer[-max_buf:]

        self._frame_count += 1

        # Fit or refit KMeans periodically
        if self._kmeans is None or self._frame_count % self._refit_interval == 0:
            buf = np.array(self._hist_buffer)
            self._kmeans = KMeans(
                n_clusters=self._n_teams, n_init=10, random_state=42,
            )
            self._kmeans.fit(buf)
            # Build stable label map from cluster centers sorted by hue peak
            centers = self._kmeans.cluster_centers_
            # Use the index of the max hue bin as a sort key for deterministic labels
            order = sorted(range(self._n_teams), key=lambda c: np.argmax(centers[c][:16]))
            labels = ["A", "B", "C", "D"][:self._n_teams]
            self._label_map = {order[j]: labels[j] for j in range(self._n_teams)}

        preds = self._kmeans.predict(hists_arr)
        for idx, vi in enumerate(valid_indices):
            cluster = int(preds[idx])
            detections[vi]["team"] = self._label_map.get(cluster, "unknown")

        return detections
