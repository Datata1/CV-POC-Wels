"""Extract (t-2, t-1, t) triplets for TrackNet training/labelling.

For every Nth frame of a source video, writes three JPEGs:
    <stem>_<fidx>_t-2.jpg
    <stem>_<fidx>_t-1.jpg
    <stem>_<fidx>_t.jpg

Only the `_t.jpg` files should be uploaded to Roboflow for labelling. The
dataset loader finds the sibling `_t-1` / `_t-2` files by filename.

Usage:
    uv run python training/extract_tracknet_triplets.py \
        --input input/game.mp4 --stride 30 \
        --out annotation/ball/triplets
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("annotation/ball/triplets"))
    ap.add_argument("--stride", type=int, default=30, help="Keep every Nth frame")
    ap.add_argument("--start", type=int, default=2, help="Earliest frame index to keep (>=2)")
    ap.add_argument("--max-frames", type=int, default=0, help="Stop after N kept (0 = all)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.input}")

    prev2 = prev1 = None
    fidx = 0
    kept = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if prev2 is not None and fidx >= args.start and fidx % args.stride == 0:
            base = args.out / f"{stem}_{fidx:06d}"
            cv2.imwrite(str(base) + "_t-2.jpg", prev2)
            cv2.imwrite(str(base) + "_t-1.jpg", prev1)
            cv2.imwrite(str(base) + "_t.jpg", frame)
            kept += 1
            if args.max_frames and kept >= args.max_frames:
                break
        prev2 = prev1
        prev1 = frame
        fidx += 1

    cap.release()
    print(f"[triplets] wrote {kept} triplets to {args.out}")
    print(f"[triplets] upload only *_t.jpg to Roboflow for labelling")


if __name__ == "__main__":
    main()
