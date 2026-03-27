"""
Handball Match - Person Detection & Pose Estimation POC

Detects people in a handball match video using YOLO11, draws bounding boxes
with confidence scores, and overlays joint/pose positions using YOLO11-pose
running on GPU for fast inference.

Usage:
    1. Place your video file in the `input/` directory
    2. Run: uv run python detect.py
    3. Output is saved to `output/`
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from pipeline.pose import create_pose_model, estimate_poses_batch, POSE_CONNECTIONS

# Colors (BGR)
BBOX_COLOR = (0, 255, 0)
BBOX_THICKNESS = 2
TEXT_COLOR = (255, 255, 255)
TEXT_BG_COLOR = (0, 180, 0)
FONT_SCALE = 0.5
JOINT_COLOR = (0, 0, 255)
SKELETON_COLOR = (0, 255, 255)

INPUT_DIR = Path(__file__).parent / "input"
OUTPUT_DIR = Path(__file__).parent / "output"
DEFAULT_POSE_MODEL = "yolo11m-pose.pt"
HAS_FFMPEG = shutil.which("ffmpeg") is not None


def find_video_file(input_dir: Path) -> Path:
    """Find the first video file in the input directory."""
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
    for f in sorted(input_dir.iterdir()):
        if f.suffix.lower() in video_extensions:
            return f
    raise FileNotFoundError(
        f"No video file found in '{input_dir}'. "
        f"Supported formats: {', '.join(sorted(video_extensions))}"
    )


def remux_to_h264(raw_path: Path, final_path: Path):
    """Remux raw AVI to H.264 MP4 using ffmpeg, then remove the raw file."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(raw_path),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(final_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    raw_path.unlink()


def detect_persons_yolo(
    frame: np.ndarray, model: YOLO, confidence: float = 0.3,
) -> list[tuple[int, int, int, int, float]]:
    """
    Detect persons using YOLO11.
    Returns list of (x, y, w, h, conf) bounding boxes.
    """
    results = model(frame, classes=[0], conf=confidence, verbose=False)
    boxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            boxes.append((x1, y1, x2 - x1, y2 - y1, conf))
    return boxes


def draw_pose_on_frame(
    frame: np.ndarray,
    landmarks: list[dict],
):
    """Draw pose landmarks on the full frame."""
    h, w = frame.shape[:2]

    # Draw skeleton connections
    for start_idx, end_idx in POSE_CONNECTIONS:
        s = landmarks[start_idx]
        e = landmarks[end_idx]

        start_x = int(s["x"])
        start_y = int(s["y"])
        end_x = int(e["x"])
        end_y = int(e["y"])

        if (
            s["visibility"] > 0.5
            and e["visibility"] > 0.5
            and 0 <= start_x < w
            and 0 <= start_y < h
            and 0 <= end_x < w
            and 0 <= end_y < h
        ):
            cv2.line(frame, (start_x, start_y), (end_x, end_y), SKELETON_COLOR, 2)

    # Draw joint points
    for lm in landmarks:
        px = int(lm["x"])
        py = int(lm["y"])
        if lm["visibility"] > 0.5 and 0 <= px < w and 0 <= py < h:
            cv2.circle(frame, (px, py), 4, JOINT_COLOR, -1)


def process_frame(
    frame: np.ndarray,
    yolo_model: YOLO,
    pose_model: YOLO,
    max_persons: int,
    yolo_confidence: float,
) -> np.ndarray:
    """Detect persons and estimate pose for a single frame. Returns annotated frame."""
    boxes = detect_persons_yolo(frame, yolo_model, yolo_confidence)

    bboxes_xyxy = []
    for x, y, w, h, conf in boxes[:max_persons]:
        bboxes_xyxy.append((x, y, x + w, y + h))

    poses = estimate_poses_batch(frame, bboxes_xyxy, pose_model)

    for i, (x, y, w, h, conf) in enumerate(boxes[:max_persons]):
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), BBOX_COLOR, BBOX_THICKNESS)

        # Label with confidence
        label = f"Person {i + 1} ({conf:.0%})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 1)
        cv2.rectangle(frame, (x, y - th - 8), (x + tw + 4, y), TEXT_BG_COLOR, -1)
        cv2.putText(
            frame, label, (x + 2, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR, 1,
        )

        # Pose estimation per detected person
        if poses[i] is not None:
            draw_pose_on_frame(frame, poses[i]["landmarks"])

    return frame


def process_chunk(
    video_path: Path,
    output_path: Path,
    start_frame: int,
    end_frame: int,
    chunk_idx: int,
    total_chunks: int,
    fps: float,
    width: int,
    height: int,
    yolo_model: YOLO,
    pose_model: YOLO,
    show_preview: bool = False,
    max_persons: int = 20,
    yolo_confidence: float = 0.3,
) -> int:
    """Process a range of frames [start_frame, end_frame) and write to output_path."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Write raw AVI first (always works), then remux to H.264 MP4 via ffmpeg
    if HAS_FFMPEG:
        raw_path = output_path.with_suffix(".raw.avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    else:
        raw_path = output_path
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(str(raw_path), fourcc, fps, (width, height))

    chunk_total = end_frame - start_frame
    frames_written = 0

    try:
        for i in range(chunk_total):
            ret, frame = cap.read()
            if not ret:
                break

            frame = process_frame(frame, yolo_model, pose_model, max_persons, yolo_confidence)
            out.write(frame)
            frames_written += 1

            if i % 30 == 0 or i == 0:
                progress = (i / chunk_total * 100) if chunk_total > 0 else 0
                print(
                    f"  Chunk {chunk_idx + 1}/{total_chunks} | "
                    f"Frame {i + 1}/{chunk_total} ({progress:.1f}%)",
                    end="\r",
                )

            if show_preview:
                preview = cv2.resize(frame, (960, 540)) if width > 960 else frame
                cv2.imshow("Handball Pose Detection", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\nStopped by user.")
                    break
    finally:
        cap.release()
        out.release()

    # Remux to proper H.264 MP4
    if HAS_FFMPEG and raw_path != output_path:
        print(f"\n  Remuxing to H.264...", end="")
        remux_to_h264(raw_path, output_path)

    return frames_written


def process_video(
    video_path: Path,
    output_dir: Path,
    output_stem: str,
    show_preview: bool = False,
    max_persons: int = 20,
    chunk_seconds: int = 60,
    yolo_confidence: float = 0.3,
    yolo_model_size: str = "yolo11n.pt",
    pose_model_size: str = "yolo11m-pose.pt",
):
    """Split video into chunks by duration and process each sequentially."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Cannot open video '{video_path}'")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    frames_per_chunk = int(fps * chunk_seconds)
    total_chunks = (total_frames + frames_per_chunk - 1) // frames_per_chunk
    total_duration = total_frames / fps

    print(f"Processing: {video_path.name}")
    print(f"Resolution: {width}x{height} | FPS: {fps:.1f} | Duration: {total_duration:.0f}s")
    print(f"Splitting into {total_chunks} chunk(s) of ~{chunk_seconds}s each")
    print(f"YOLO model: {yolo_model_size} | Confidence: {yolo_confidence}")
    print(f"ffmpeg remux: {'yes' if HAS_FFMPEG else 'no (install ffmpeg for H.264 output)'}")
    print(f"Output dir: {output_dir}")
    print()

    # Initialize YOLO11 person detector
    print("Loading YOLO detection model...")
    yolo_model = YOLO(yolo_model_size)

    # Initialize YOLO11-pose model (GPU-accelerated)
    print("Loading YOLO11-pose model (GPU)...")
    pose_model = create_pose_model(pose_model_size)

    chunk_paths = []
    total_processed = 0

    try:
        for chunk_idx in range(total_chunks):
            start_frame = chunk_idx * frames_per_chunk
            end_frame = min(start_frame + frames_per_chunk, total_frames)

            start_sec = start_frame / fps
            end_sec = end_frame / fps
            chunk_path = output_dir / f"{output_stem}_chunk{chunk_idx + 1:03d}_{start_sec:.0f}s-{end_sec:.0f}s.mp4"

            print(f"\n--- Chunk {chunk_idx + 1}/{total_chunks} ({start_sec:.0f}s - {end_sec:.0f}s) ---")

            written = process_chunk(
                video_path=video_path,
                output_path=chunk_path,
                start_frame=start_frame,
                end_frame=end_frame,
                chunk_idx=chunk_idx,
                total_chunks=total_chunks,
                fps=fps,
                width=width,
                height=height,
                yolo_model=yolo_model,
                pose_model=pose_model,
                show_preview=show_preview,
                max_persons=max_persons,
                yolo_confidence=yolo_confidence,
            )

            total_processed += written
            chunk_paths.append(chunk_path)
            print(f"\n  Done! Saved: {chunk_path.name}")
            print(f"  >>> You can watch this chunk now while the rest processes.")

    finally:
        if show_preview:
            cv2.destroyAllWindows()

    print(f"\n{'=' * 60}")
    print(f"All done! Processed {total_processed} frames in {total_chunks} chunk(s).")
    print(f"Output files:")
    for p in chunk_paths:
        print(f"  {p}")


def main():
    parser = argparse.ArgumentParser(
        description="Detect players and estimate pose in handball match videos."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input video file. If omitted, picks the first video in input/",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for output chunks. Defaults to output/",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show a live preview window while processing (press 'q' to quit)",
    )
    parser.add_argument(
        "--max-persons",
        type=int,
        default=20,
        help="Maximum number of persons to process per frame (default: 20)",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=int,
        default=60,
        help="Duration of each output chunk in seconds (default: 60)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="YOLO detection confidence threshold (default: 0.3)",
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="yolo11n.pt",
        choices=["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"],
        help="YOLO model size: n=nano(fast), s=small, m=medium, l=large, x=xlarge(accurate)",
    )
    parser.add_argument(
        "--pose-model",
        type=str,
        default="yolo11m-pose.pt",
        choices=["yolo11n-pose.pt", "yolo11s-pose.pt", "yolo11m-pose.pt", "yolo11l-pose.pt", "yolo11x-pose.pt"],
        help="YOLO11-pose model size (default: yolo11m-pose.pt, runs on GPU)",
    )
    args = parser.parse_args()

    # Resolve input
    if args.input:
        video_path = Path(args.input)
    else:
        INPUT_DIR.mkdir(exist_ok=True)
        video_path = find_video_file(INPUT_DIR)

    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    # Resolve output directory
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)

    process_video(
        video_path=video_path,
        output_dir=output_dir,
        output_stem=video_path.stem,
        show_preview=args.preview,
        max_persons=args.max_persons,
        chunk_seconds=args.chunk_seconds,
        yolo_confidence=args.confidence,
        yolo_model_size=args.yolo_model,
        pose_model_size=args.pose_model,
    )


if __name__ == "__main__":
    main()
