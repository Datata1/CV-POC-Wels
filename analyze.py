"""
Handball Match Analysis Pipeline (POC)

Full pipeline: person detection, ball detection, tracking, team classification,
pose estimation, court mapping, and per-frame state export.

Usage:
    uv run python analyze.py                    # basic run
    uv run python analyze.py --no-pose          # skip pose (faster)
    uv run python analyze.py --calibration court_cal.json  # with court mapping

Output:
    output/<name>_chunk*.mp4       — annotated video chunks
    output/<name>_states.jsonl     — per-frame state data (JSON-lines)
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from pipeline.detector import detect_objects
from pipeline.tracker import SimpleTracker
from pipeline.team import TeamClassifier
from pipeline.pose import create_landmarker, estimate_pose
from pipeline.court import CourtMapper
from pipeline.state import build_frame_state, StateExporter
from pipeline.draw import draw_player, draw_ball, draw_hud

INPUT_DIR = Path(__file__).parent / "input"
OUTPUT_DIR = Path(__file__).parent / "output"
MODEL_PATH = Path(__file__).parent / "models" / "pose_landmarker_heavy.task"
HAS_FFMPEG = shutil.which("ffmpeg") is not None


def find_video_file(input_dir: Path) -> Path:
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
    for f in sorted(input_dir.iterdir()):
        if f.suffix.lower() in video_extensions:
            return f
    raise FileNotFoundError(
        f"No video file found in '{input_dir}'. "
        f"Supported: {', '.join(sorted(video_extensions))}"
    )


def remux_to_h264(raw_path: Path, final_path: Path):
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(raw_path),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(final_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    raw_path.unlink()


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
    tracker: SimpleTracker,
    team_clf: TeamClassifier,
    landmarker,
    court_mapper: CourtMapper,
    state_exporter: StateExporter,
    show_preview: bool = False,
    max_persons: int = 20,
    yolo_confidence: float = 0.3,
    skip_pose: bool = False,
) -> int:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

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

            abs_frame = start_frame + i
            timestamp = abs_frame / fps
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 1. Detect persons + balls
            persons, balls = detect_objects(
                frame, yolo_model, yolo_confidence, max_persons,
            )

            # 2. Track
            persons = tracker.update(persons)

            # 3. Team classification
            persons = team_clf.classify(frame, persons)

            # 4. Court filtering
            for p in persons:
                foot = court_mapper.foot_position(p["bbox"])
                p["on_court"] = court_mapper.is_on_court(foot[0], foot[1])

            # 5. Pose estimation (optional)
            if not skip_pose and landmarker is not None:
                for p in persons:
                    pose = estimate_pose(frame_rgb, p["bbox"], landmarker)
                    p["pose"] = pose
            else:
                for p in persons:
                    p["pose"] = None

            # 6. Build & export state
            state = build_frame_state(
                frame_id=abs_frame,
                timestamp_s=timestamp,
                players=persons,
                balls=balls,
                court_mapper=court_mapper,
            )
            state_exporter.write(state)

            # 7. Draw annotations
            for p in persons:
                draw_player(frame, p)
            for b in balls:
                draw_ball(frame, b)
            draw_hud(frame, state)

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
                cv2.imshow("Handball Analysis", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\nStopped by user.")
                    break
    finally:
        cap.release()
        out.release()

    if HAS_FFMPEG and raw_path != output_path:
        print(f"\n  Remuxing to H.264...", end="")
        remux_to_h264(raw_path, output_path)

    return frames_written


def run(
    video_path: Path,
    output_dir: Path,
    output_stem: str,
    show_preview: bool = False,
    max_persons: int = 20,
    chunk_seconds: int = 60,
    yolo_confidence: float = 0.3,
    yolo_model_size: str = "yolov8n.pt",
    skip_pose: bool = False,
    calibration_path: Path | None = None,
    n_teams: int = 2,
):
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

    print(f"{'=' * 60}")
    print(f"Handball Match Analysis Pipeline (POC)")
    print(f"{'=' * 60}")
    print(f"Input:       {video_path.name}")
    print(f"Resolution:  {width}x{height} | FPS: {fps:.1f} | Duration: {total_duration:.0f}s")
    print(f"Chunks:      {total_chunks} x ~{chunk_seconds}s")
    print(f"YOLO model:  {yolo_model_size} | Confidence: {yolo_confidence}")
    print(f"Pose:        {'OFF' if skip_pose else 'ON'}")
    print(f"Court cal:   {calibration_path or 'none (set with --calibration)'}")
    print(f"Teams:       {n_teams}")
    print(f"ffmpeg:      {'yes' if HAS_FFMPEG else 'no'}")
    print(f"Output:      {output_dir}")
    print()

    # --- Initialize components ---
    print("Loading YOLO model...")
    yolo_model = YOLO(yolo_model_size)

    landmarker = None
    if not skip_pose:
        if not MODEL_PATH.exists():
            print(f"Warning: Pose model not found at '{MODEL_PATH}', skipping pose.")
            print("Download with:  make download-model")
            skip_pose = True
        else:
            print("Loading pose model...")
            landmarker = create_landmarker(str(MODEL_PATH))

    tracker = SimpleTracker(iou_threshold=0.3, max_lost=15)
    team_clf = TeamClassifier(n_teams=n_teams)
    court_mapper = CourtMapper(calibration_path)

    if court_mapper.is_calibrated:
        print("Court calibration loaded.")
    else:
        print("No court calibration — court positions will be unavailable.")

    state_path = output_dir / f"{output_stem}_states.jsonl"
    chunk_paths = []
    total_processed = 0

    with StateExporter(state_path) as exporter:
        try:
            for chunk_idx in range(total_chunks):
                start_frame = chunk_idx * frames_per_chunk
                end_frame = min(start_frame + frames_per_chunk, total_frames)

                start_sec = start_frame / fps
                end_sec = end_frame / fps
                chunk_path = output_dir / (
                    f"{output_stem}_chunk{chunk_idx + 1:03d}"
                    f"_{start_sec:.0f}s-{end_sec:.0f}s.mp4"
                )

                print(
                    f"\n--- Chunk {chunk_idx + 1}/{total_chunks}"
                    f" ({start_sec:.0f}s - {end_sec:.0f}s) ---"
                )

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
                    tracker=tracker,
                    team_clf=team_clf,
                    landmarker=landmarker,
                    court_mapper=court_mapper,
                    state_exporter=exporter,
                    show_preview=show_preview,
                    max_persons=max_persons,
                    yolo_confidence=yolo_confidence,
                    skip_pose=skip_pose,
                )

                total_processed += written
                chunk_paths.append(chunk_path)
                print(f"\n  Saved: {chunk_path.name}")

        finally:
            if landmarker is not None:
                landmarker.close()
            if show_preview:
                cv2.destroyAllWindows()

    print(f"\n{'=' * 60}")
    print(f"Done! {total_processed} frames in {total_chunks} chunk(s).")
    print(f"State data: {state_path}")
    print(f"Video chunks:")
    for p in chunk_paths:
        print(f"  {p}")


def main():
    parser = argparse.ArgumentParser(
        description="Handball match analysis: detection, tracking, team ID, pose, state export."
    )
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--max-persons", type=int, default=20)
    parser.add_argument("--chunk-seconds", type=int, default=60)
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument(
        "--yolo-model", type=str, default="yolov8n.pt",
        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
    )
    parser.add_argument(
        "--no-pose", action="store_true",
        help="Skip pose estimation (much faster)",
    )
    parser.add_argument(
        "--calibration", type=str, default=None,
        help="Path to court calibration JSON (see calibrate.py)",
    )
    parser.add_argument(
        "--n-teams", type=int, default=2,
        help="Number of teams to cluster (default: 2)",
    )
    args = parser.parse_args()

    if args.input:
        video_path = Path(args.input)
    else:
        INPUT_DIR.mkdir(exist_ok=True)
        video_path = find_video_file(INPUT_DIR)

    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)

    cal_path = Path(args.calibration) if args.calibration else None

    run(
        video_path=video_path,
        output_dir=output_dir,
        output_stem=video_path.stem,
        show_preview=args.preview,
        max_persons=args.max_persons,
        chunk_seconds=args.chunk_seconds,
        yolo_confidence=args.confidence,
        yolo_model_size=args.yolo_model,
        skip_pose=args.no_pose,
        calibration_path=cal_path,
        n_teams=args.n_teams,
    )


if __name__ == "__main__":
    main()
