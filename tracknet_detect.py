"""Standalone TrackNet ball-detection pipeline.

Processes a handball video, runs TrackNet on every sliding triplet of frames,
writes an annotated video (with a fading trajectory trail) and a JSONL file of
per-frame ball positions.

Usage:
    uv run python tracknet_detect.py --input input/game.mp4 \
        --model models/tracknet.pt --trail-length 20
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import deque
from pathlib import Path
from typing import Deque, Optional, Tuple

import cv2
import numpy as np

from pipeline.tracknet import TrackNetDetector


def find_first_video(folder: Path) -> Optional[Path]:
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
        matches = sorted(folder.glob(ext))
        if matches:
            return matches[0]
    return None


def draw_trail(
    frame: np.ndarray,
    trail: Deque[Tuple[int, int]],
    color: Tuple[int, int, int] = (0, 255, 255),
) -> None:
    """Draw a fading trail: older points are smaller and more transparent."""
    n = len(trail)
    for i, (x, y) in enumerate(trail):
        age = (n - i) / max(n, 1)
        radius = max(1, int(6 * (1.0 - age)))
        alpha = 1.0 - age
        overlay = frame.copy()
        cv2.circle(overlay, (x, y), radius, color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_ball(
    frame: np.ndarray, xy: Tuple[float, float], conf: float
) -> None:
    x, y = int(xy[0]), int(xy[1])
    cv2.circle(frame, (x, y), 10, (0, 255, 255), 2)
    cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
    cv2.putText(
        frame,
        f"{conf:.2f}",
        (x + 12, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_hud(
    frame: np.ndarray,
    frame_idx: int,
    total: int,
    fps: float,
    detect_rate: float,
) -> None:
    text = (
        f"TrackNet | frame {frame_idx}/{total} | "
        f"{fps:.1f} FPS | detect {detect_rate * 100:.1f}%"
    )
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )


def open_writer(
    path: Path, fps: float, size: Tuple[int, int]
) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, size)


def maybe_transcode_to_h264(src: Path) -> None:
    """Best-effort re-encode mp4v → H.264 via ffmpeg so browsers play the file."""
    if not src.exists():
        return
    dst = src.with_suffix(".h264.mp4")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(src), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "23", "-preset", "veryfast", str(dst),
            ],
            check=True,
        )
        dst.replace(src)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass  # leave mp4v file as-is


def main() -> None:
    ap = argparse.ArgumentParser(description="TrackNet ball detection on a handball video")
    ap.add_argument("--input", type=Path, default=None, help="Input video (default: first in input/)")
    ap.add_argument("--output-dir", type=Path, default=Path("output"))
    ap.add_argument("--model", type=Path, default=Path("models/tracknet.pt"))
    ap.add_argument("--threshold", type=int, default=128, help="Heatmap binarisation threshold (0-255)")
    ap.add_argument("--trail-length", type=int, default=20, help="Number of past positions in trajectory trail")
    ap.add_argument("--chunk-seconds", type=int, default=60, help="Split output into chunks of N seconds")
    ap.add_argument("--preview", action="store_true", help="Show live preview window")
    ap.add_argument("--device", type=str, default=None, help="'cuda' or 'cpu' (auto if omitted)")
    ap.add_argument("--save-heatmaps", action="store_true", help="Also write a side-by-side heatmap video")
    args = ap.parse_args()

    input_path = args.input or find_first_video(Path("input"))
    if input_path is None or not input_path.exists():
        raise SystemExit(f"No input video found (looked for {input_path or 'input/'})")
    if not args.model.exists():
        raise SystemExit(
            f"TrackNet weights not found at {args.model}. "
            "See guides/TRACKNET_GUIDE.md for download instructions."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    chunk_frames = max(1, int(args.chunk_seconds * fps))

    print(f"[tracknet] input:  {input_path}  ({W}x{H} @ {fps:.2f} fps, {total} frames)")
    print(f"[tracknet] model:  {args.model}")
    print(f"[tracknet] device: {args.device or 'auto'}")

    detector = TrackNetDetector(
        weights_path=str(args.model),
        device=args.device,
        heatmap_threshold=args.threshold,
    )

    jsonl_path = args.output_dir / f"{stem}_tracknet.jsonl"
    jsonl_file = jsonl_path.open("w")

    trail: Deque[Tuple[int, int]] = deque(maxlen=args.trail_length)
    writer: Optional[cv2.VideoWriter] = None
    heatmap_writer: Optional[cv2.VideoWriter] = None
    chunk_idx = 0
    chunk_paths: list[Path] = []
    detections = 0
    frame_idx = 0
    t_start = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Start a new chunk when needed
            if frame_idx % chunk_frames == 0:
                if writer is not None:
                    writer.release()
                if heatmap_writer is not None:
                    heatmap_writer.release()
                chunk_idx += 1
                sec_start = int(frame_idx / fps)
                sec_end = int(min(total, frame_idx + chunk_frames) / fps)
                chunk_name = f"{stem}_tracknet_chunk{chunk_idx:03d}_{sec_start}s-{sec_end}s.mp4"
                chunk_path = args.output_dir / chunk_name
                chunk_paths.append(chunk_path)
                writer = open_writer(chunk_path, fps, (W, H))
                if args.save_heatmaps:
                    hm_path = args.output_dir / chunk_name.replace(".mp4", "_heatmap.mp4")
                    heatmap_writer = open_writer(hm_path, fps, (W, H))

            xy, conf, heatmap = detector.detect(frame)

            record = {
                "frame_id": frame_idx,
                "timestamp_s": frame_idx / fps,
                "ball": None,
            }
            if xy is not None:
                detections += 1
                trail.append((int(xy[0]), int(xy[1])))
                record["ball"] = {
                    "x": round(xy[0], 2),
                    "y": round(xy[1], 2),
                    "conf": round(conf, 4),
                }
                draw_ball(frame, xy, conf)
            draw_trail(frame, trail)

            elapsed = time.time() - t_start
            proc_fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0.0
            detect_rate = detections / (frame_idx + 1)
            draw_hud(frame, frame_idx, total, proc_fps, detect_rate)

            writer.write(frame)
            if heatmap_writer is not None and heatmap is not None:
                hm_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
                hm_color = cv2.resize(hm_color, (W, H))
                heatmap_writer.write(hm_color)

            jsonl_file.write(json.dumps(record) + "\n")

            if args.preview:
                cv2.imshow("tracknet", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(
                    f"[tracknet] {frame_idx}/{total}  "
                    f"detect={detect_rate * 100:.1f}%  proc={proc_fps:.1f} fps"
                )
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if heatmap_writer is not None:
            heatmap_writer.release()
        jsonl_file.close()
        if args.preview:
            cv2.destroyAllWindows()

    print(f"[tracknet] done. frames={frame_idx}  detections={detections} "
          f"({100 * detections / max(frame_idx, 1):.1f}%)")
    print(f"[tracknet] states → {jsonl_path}")
    for p in chunk_paths:
        maybe_transcode_to_h264(p)
        print(f"[tracknet] video  → {p}")


if __name__ == "__main__":
    main()
