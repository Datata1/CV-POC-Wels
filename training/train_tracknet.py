"""Fine-tune TrackNet on handball triplets.

Warm-starts from the pretrained tennis weights at `models/tracknet.pt` and
saves the best checkpoint (by val detection rate @ 5 px) to
`models/tracknet_handball.pt`.

Usage:
    uv run python training/train_tracknet.py \
        --triplets annotation/ball/triplets \
        --labels   annotation/ball/labels \
        --pretrained models/tracknet.pt \
        --output     models/tracknet_handball.pt \
        --epochs 30 --batch-size 4
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from pipeline.tracknet import INPUT_H, INPUT_W, BallTrackerNet
from training.tracknet_dataset import HandballTrackNetDataset, split_dataset


def heatmap_to_point(
    heatmap: np.ndarray, threshold: int = 128, min_area: int = 2
) -> Optional[Tuple[float, float]]:
    _, binary = cv2.threshold(heatmap, threshold, 255, cv2.THRESH_BINARY)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = int(np.argmax(areas)) + 1
    if stats[idx, cv2.CC_STAT_AREA] < min_area:
        return None
    cx, cy = centroids[idx]
    return float(cx), float(cy)


@torch.no_grad()
def detection_rate(model: BallTrackerNet, loader: DataLoader, device: torch.device, radius: float = 5.0) -> float:
    model.eval()
    hits = total = 0
    for inp, _target, gt_xy in loader:
        inp = inp.to(device)
        probs = model(inp)  # softmax (B, 256, H*W)
        pred = probs.argmax(dim=1).reshape(-1, INPUT_H, INPUT_W).cpu().numpy().astype(np.uint8)
        for h, g in zip(pred, gt_xy.numpy()):
            pt = heatmap_to_point(h)
            if pt is not None and np.hypot(pt[0] - g[0], pt[1] - g[1]) <= radius:
                hits += 1
            total += 1
    return hits / max(total, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--triplets", type=Path, default=Path("annotation/ball/triplets"))
    ap.add_argument("--labels", type=Path, default=Path("annotation/ball/labels"))
    ap.add_argument("--pretrained", type=Path, default=Path("models/tracknet.pt"))
    ap.add_argument("--output", type=Path, default=Path("models/tracknet_handball.pt"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1.0, help="Adadelta lr; try 1e-3 for Adam")
    ap.add_argument("--optimizer", choices=["adadelta", "adam"], default="adadelta")
    ap.add_argument("--sigma", type=float, default=5.0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--radius", type=float, default=5.0, help="Detection-rate tolerance (px @ 640x360)")
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[train] device: {device}")

    ds = HandballTrackNetDataset(args.triplets, args.labels, sigma=args.sigma, hflip=True)
    print(f"[train] samples: {len(ds)}")
    train_idx, val_idx = split_dataset(ds, val_frac=args.val_frac)
    train_loader = DataLoader(
        Subset(ds, train_idx), batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_ds = HandballTrackNetDataset(args.triplets, args.labels, sigma=args.sigma, hflip=False)
    val_loader = DataLoader(
        Subset(val_ds, val_idx), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    print(f"[train] train={len(train_idx)}  val={len(val_idx)}")

    model = BallTrackerNet(out_channels=256).to(device)
    if args.pretrained.exists():
        state = torch.load(args.pretrained, map_location=device, weights_only=False)
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        model.load_state_dict(state)
        print(f"[train] warm-started from {args.pretrained}")
    else:
        print(f"[train] no pretrained weights at {args.pretrained}, training from scratch")

    if args.optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        opt = torch.optim.Adadelta(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_dr = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for inp, target, _gt in train_loader:
            inp = inp.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            logits = model.forward_logits(inp)  # (B, 256, H*W)
            loss = loss_fn(logits, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * inp.size(0)
            n += inp.size(0)
        train_loss = running / max(n, 1)

        dr = detection_rate(model, val_loader, device, radius=args.radius)
        print(f"[train] epoch {epoch:03d}  loss={train_loss:.4f}  val_dr@{args.radius:g}px={dr * 100:.1f}%")

        if dr > best_dr:
            best_dr = dr
            torch.save(model.state_dict(), args.output)
            print(f"[train]   ↳ new best, saved → {args.output}")

    print(f"[train] done. best val detection rate: {best_dr * 100:.1f}%")


if __name__ == "__main__":
    main()
