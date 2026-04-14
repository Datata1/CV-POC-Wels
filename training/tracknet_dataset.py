"""Dataset for training TrackNet on handball footage.

Pairs YOLO-format bbox labels (from Roboflow export) with their sibling
(t-2, t-1, t) image triplets on disk, and produces (input_tensor, target_heatmap)
pairs suitable for per-pixel 256-way cross-entropy loss.

Directory layout expected:

    triplets_dir/
        <stem>_<fidx>_t-2.jpg
        <stem>_<fidx>_t-1.jpg
        <stem>_<fidx>_t.jpg
    labels_dir/
        <stem>_<fidx>_t.txt        # YOLO: "<cls> cx cy w h" (normalised)
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


INPUT_W = 640
INPUT_H = 360


def gaussian_heatmap(w: int, h: int, cx: float, cy: float, sigma: float) -> np.ndarray:
    """Return a (h, w) uint8 image with a Gaussian blob of peak 255 at (cx, cy)."""
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2))
    return (g * 255.0).astype(np.uint8)


def _read_yolo_center(label_file: Path) -> Tuple[float, float] | None:
    """Return (cx, cy) normalised in [0, 1] of the first box in a YOLO label file."""
    for line in label_file.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            return float(parts[1]), float(parts[2])
    return None


@dataclass
class Sample:
    t2: Path
    t1: Path
    t0: Path
    cx_norm: float
    cy_norm: float


class HandballTrackNetDataset(Dataset):
    def __init__(
        self,
        triplets_dir: Path | str,
        labels_dir: Path | str,
        sigma: float = 5.0,
        hflip: bool = True,
    ) -> None:
        self.triplets_dir = Path(triplets_dir)
        self.labels_dir = Path(labels_dir)
        self.sigma = sigma
        self.hflip = hflip

        self.samples: List[Sample] = []
        for label_file in sorted(self.labels_dir.glob("*_t.txt")):
            center = _read_yolo_center(label_file)
            if center is None:
                continue
            stem = label_file.stem  # "<stem>_<fidx>_t"
            base = stem[:-2]  # strip "_t"
            t0 = self.triplets_dir / f"{base}_t.jpg"
            t1 = self.triplets_dir / f"{base}_t-1.jpg"
            t2 = self.triplets_dir / f"{base}_t-2.jpg"
            if not (t0.exists() and t1.exists() and t2.exists()):
                continue
            self.samples.append(Sample(t2, t1, t0, center[0], center[1]))

        if not self.samples:
            raise RuntimeError(
                f"No samples found. Looked for *_t.txt in {self.labels_dir} "
                f"with matching triplets in {self.triplets_dir}."
            )

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _load_resized(path: Path) -> np.ndarray:
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"Could not read {path}")
        img = cv2.resize(img, (INPUT_W, INPUT_H))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        f2 = self._load_resized(s.t2)
        f1 = self._load_resized(s.t1)
        f0 = self._load_resized(s.t0)
        cx = s.cx_norm * INPUT_W
        cy = s.cy_norm * INPUT_H

        if self.hflip and random.random() < 0.5:
            f2 = f2[:, ::-1, :].copy()
            f1 = f1[:, ::-1, :].copy()
            f0 = f0[:, ::-1, :].copy()
            cx = INPUT_W - 1 - cx

        stacked = np.concatenate([f2, f1, f0], axis=2)  # (H, W, 9)
        inp = torch.from_numpy(stacked).permute(2, 0, 1).contiguous()  # (9, H, W)

        heatmap = gaussian_heatmap(INPUT_W, INPUT_H, cx, cy, self.sigma)
        target = torch.from_numpy(heatmap.astype(np.int64)).flatten()  # (H*W,)

        gt_xy = torch.tensor([cx, cy], dtype=torch.float32)
        return inp, target, gt_xy


def split_dataset(
    ds: HandballTrackNetDataset, val_frac: float = 0.1, seed: int = 0
) -> Tuple[List[int], List[int]]:
    """Return (train_indices, val_indices). Split by sample index."""
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    k = max(1, int(len(idx) * val_frac))
    return idx[k:], idx[:k]
