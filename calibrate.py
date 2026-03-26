"""
Court Calibration Tool

Opens the first frame of a video and lets you click 4 court corner points.
Saves a calibration JSON for use with analyze.py --calibration.

Usage:
    uv run python calibrate.py
    uv run python calibrate.py --input input/match.mp4 --output court_cal.json
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

INPUT_DIR = Path(__file__).parent / "input"

INSTRUCTIONS = """
=== Court Calibration ===

Click 4 points on the court IN THIS ORDER:
  1. Top-left corner of the court
  2. Top-right corner of the court
  3. Bottom-right corner of the court
  4. Bottom-left corner of the court

These map to the handball court corners:
  (0,0) — (40,0) — (40,20) — (0,20) metres

Press 'r' to reset, 'q' to quit, 's' to save.
"""


def find_video_file(input_dir: Path) -> Path:
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
    for f in sorted(input_dir.iterdir()):
        if f.suffix.lower() in video_extensions:
            return f
    raise FileNotFoundError(f"No video file found in '{input_dir}'.")


points: list[list[int]] = []
frame_display = None


def mouse_callback(event, x, y, flags, param):
    global frame_display
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append([x, y])
        frame_display = param.copy()
        _draw_points(frame_display)


def _draw_points(img):
    labels = ["TL", "TR", "BR", "BL"]
    for i, (px, py) in enumerate(points):
        color = (0, 255, 0) if i < len(points) - 1 else (0, 0, 255)
        cv2.circle(img, (px, py), 6, color, -1)
        cv2.putText(
            img, f"{labels[i]} ({px},{py})", (px + 10, py - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
        )
    if len(points) >= 2:
        for i in range(len(points)):
            p1 = tuple(points[i])
            p2 = tuple(points[(i + 1) % len(points)]) if i + 1 < len(points) else None
            if p2:
                cv2.line(img, p1, p2, (0, 200, 0), 1)
        if len(points) == 4:
            cv2.line(img, tuple(points[3]), tuple(points[0]), (0, 200, 0), 1)


def main():
    global points, frame_display

    parser = argparse.ArgumentParser(description="Court calibration tool.")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default="court_cal.json")
    args = parser.parse_args()

    if args.input:
        video_path = Path(args.input)
    else:
        INPUT_DIR.mkdir(exist_ok=True)
        video_path = find_video_file(INPUT_DIR)

    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Error: Could not read first frame.")
        sys.exit(1)

    print(INSTRUCTIONS)
    frame_orig = frame.copy()
    frame_display = frame.copy()

    cv2.namedWindow("Calibrate Court", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibrate Court", 1280, 720)
    cv2.setMouseCallback("Calibrate Court", mouse_callback, frame_orig)

    while True:
        status = f"Points: {len(points)}/4"
        if len(points) == 4:
            status += "  |  Press 's' to save, 'r' to reset"
        disp = frame_display.copy()
        cv2.putText(disp, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Calibrate Court", disp)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            points = []
            frame_display = frame_orig.copy()
        elif key == ord("s") and len(points) == 4:
            out_path = Path(args.output)
            data = {
                "src": points,
                "dst": [[0, 0], [40, 0], [40, 20], [0, 20]],
                "video": video_path.name,
            }
            out_path.write_text(json.dumps(data, indent=2))
            print(f"\nCalibration saved to: {out_path}")
            print(f"Use with:  uv run python analyze.py --calibration {out_path}")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
