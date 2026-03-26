"""Simple IoU-based multi-object tracker for consistent player IDs across frames."""

import numpy as np


def _iou(box_a: tuple, box_b: tuple) -> float:
    """Compute IoU between two (x1, y1, x2, y2) boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter)


class SimpleTracker:
    """
    Greedy IoU tracker that assigns persistent IDs to detections across frames.

    For a POC this avoids heavy dependencies (ByteTrack, etc.) while still
    providing stable track IDs suitable for team assignment and state export.
    """

    def __init__(self, iou_threshold: float = 0.3, max_lost: int = 15):
        self._next_id = 1
        self._tracks: dict[int, dict] = {}  # track_id -> {bbox, lost}
        self._iou_threshold = iou_threshold
        self._max_lost = max_lost

    def update(self, detections: list[dict]) -> list[dict]:
        """
        Match new detections to existing tracks via IoU.

        Args:
            detections: list of dicts with 'bbox' key (x1, y1, x2, y2).

        Returns:
            Same detections list, each augmented with a 'track_id' key.
        """
        if not self._tracks:
            for det in detections:
                det["track_id"] = self._next_id
                self._tracks[self._next_id] = {"bbox": det["bbox"], "lost": 0}
                self._next_id += 1
            return detections

        # Build cost matrix
        track_ids = list(self._tracks.keys())
        matched_tracks = set()
        matched_dets = set()

        # Greedy matching: for each detection find best track
        pairs = []
        for di, det in enumerate(detections):
            best_iou = 0.0
            best_tid = None
            for tid in track_ids:
                if tid in matched_tracks:
                    continue
                score = _iou(det["bbox"], self._tracks[tid]["bbox"])
                if score > best_iou:
                    best_iou = score
                    best_tid = tid
            if best_tid is not None and best_iou >= self._iou_threshold:
                pairs.append((di, best_tid, best_iou))

        # Sort by IoU descending to resolve conflicts
        pairs.sort(key=lambda p: p[2], reverse=True)
        for di, tid, _ in pairs:
            if di in matched_dets or tid in matched_tracks:
                continue
            detections[di]["track_id"] = tid
            self._tracks[tid] = {"bbox": detections[di]["bbox"], "lost": 0}
            matched_tracks.add(tid)
            matched_dets.add(di)

        # New tracks for unmatched detections
        for di, det in enumerate(detections):
            if di not in matched_dets:
                det["track_id"] = self._next_id
                self._tracks[self._next_id] = {"bbox": det["bbox"], "lost": 0}
                self._next_id += 1

        # Increment lost counter for unmatched tracks; prune old ones
        for tid in track_ids:
            if tid not in matched_tracks:
                self._tracks[tid]["lost"] += 1
        self._tracks = {
            tid: t for tid, t in self._tracks.items() if t["lost"] <= self._max_lost
        }

        return detections
