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

from pipeline.detector import detect_and_track
from pipeline.team import TeamClassifier
from pipeline.pose import create_pose_model, estimate_poses_batch
from pipeline.court import CourtMapper
from pipeline.state import build_frame_state, StateExporter
from pipeline.draw import draw_player, draw_ball, draw_hud
from pipeline.lines import estimate_homography, HomographyTracker
from pipeline.court_viz import render_court, CANVAS_W, CANVAS_H

INPUT_DIR = Path(__file__).parent / "input"
OUTPUT_DIR = Path(__file__).parent / "output"
DEFAULT_POSE_MODEL = "yolo11m-pose.pt"
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
    team_clf: TeamClassifier,
    pose_model,
    court_mapper: CourtMapper,
    state_exporter: StateExporter,
    show_preview: bool = False,
    max_persons: int = 20,
    yolo_confidence: float = 0.3,
    skip_pose: bool = False,
    ball_model: YOLO | None = None,
    ball_confidence: float = 0.25,
    enable_lines: bool = False,
    court_video_path: Path | None = None,
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

    # Court top-down video writer
    court_out = None
    court_raw_path = None
    if court_video_path:
        if HAS_FFMPEG:
            court_raw_path = court_video_path.with_suffix(".raw.avi")
            court_out = cv2.VideoWriter(
                str(court_raw_path),
                cv2.VideoWriter_fourcc(*"MJPG"), fps, (CANVAS_W, CANVAS_H),
            )
        else:
            court_raw_path = court_video_path
            court_out = cv2.VideoWriter(
                str(court_video_path),
                cv2.VideoWriter_fourcc(*"mp4v"), fps, (CANVAS_W, CANVAS_H),
            )

    h_tracker = HomographyTracker(max_stale_frames=int(fps * 5))
    chunk_total = end_frame - start_frame
    frames_written = 0

    try:
        for i in range(chunk_total):
            ret, frame = cap.read()
            if not ret:
                break

            abs_frame = start_frame + i
            timestamp = abs_frame / fps

            # 1. Detect persons + balls (with built-in ByteTrack tracking)
            persons, balls = detect_and_track(
                frame, yolo_model, yolo_confidence, max_persons,
                ball_model=ball_model, ball_confidence=ball_confidence,
            )

            # 2. Team classification
            persons = team_clf.classify(frame, persons)

            # 2b. Court homography (if enabled and no manual calibration)
            if enable_lines and not court_mapper.is_calibrated:
                raw_H, _ = estimate_homography(frame)
                h_tracker.update(frame, raw_H)

            # 3. Court filtering (use manual calibration or line-based H)
            active_H = None
            if court_mapper.is_calibrated:
                active_H = court_mapper._H
            elif h_tracker.current_H is not None:
                active_H = h_tracker.current_H

            for p in persons:
                foot = court_mapper.foot_position(p["bbox"])
                if active_H is not None and not court_mapper.is_calibrated:
                    pt = cv2.perspectiveTransform(
                        np.float32([[[foot[0], foot[1]]]]), active_H,
                    )
                    cx, cy = float(pt[0][0][0]), float(pt[0][0][1])
                    p["court_pos_line"] = (cx, cy)
                    p["on_court"] = (
                        -3 <= cx <= 43 and -3 <= cy <= 23
                    )
                else:
                    p["on_court"] = court_mapper.is_on_court(foot[0], foot[1])

            # 4. Pose estimation (optional, GPU-accelerated)
            if not skip_pose and pose_model is not None:
                bboxes = [p["bbox"] for p in persons]
                poses = estimate_poses_batch(frame, bboxes, pose_model)
                for p, pose in zip(persons, poses):
                    p["pose"] = pose
            else:
                for p in persons:
                    p["pose"] = None

            # 5. Build & export state
            state = build_frame_state(
                frame_id=abs_frame,
                timestamp_s=timestamp,
                players=persons,
                balls=balls,
                court_mapper=court_mapper,
            )

            # Inject line-based court positions into state if available
            if active_H is not None and not court_mapper.is_calibrated:
                for ps, p in zip(state["players"], persons):
                    if "court_pos_line" in p:
                        cx, cy = p["court_pos_line"]
                        ps["court_pos"] = [round(cx, 2), round(cy, 2)]
                # Ball position via line homography
                if balls and state["ball"]:
                    bx, by = state["ball"]["center_px"]
                    bpt = cv2.perspectiveTransform(
                        np.float32([[[bx, by]]]), active_H,
                    )
                    state["ball"]["court_pos"] = [
                        round(float(bpt[0][0][0]), 2),
                        round(float(bpt[0][0][1]), 2),
                    ]

            state_exporter.write(state)

            # 5b. Render court top-down view
            if court_out is not None:
                court_frame = render_court(
                    players=state["players"],
                    ball=state.get("ball"),
                    frame_id=abs_frame,
                    timestamp_s=timestamp,
                )
                court_out.write(court_frame)

            # 6. Draw annotations
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
        if court_out is not None:
            court_out.release()

    if HAS_FFMPEG and raw_path != output_path:
        print(f"\n  Remuxing to H.264...", end="")
        remux_to_h264(raw_path, output_path)

    if court_out is not None and HAS_FFMPEG and court_raw_path != court_video_path:
        print(f"\n  Remuxing court video...", end="")
        remux_to_h264(court_raw_path, court_video_path)

    return frames_written


def run(
    video_path: Path,
    output_dir: Path,
    output_stem: str,
    show_preview: bool = False,
    max_persons: int = 20,
    chunk_seconds: int = 60,
    yolo_confidence: float = 0.3,
    yolo_model_size: str = "yolo11n.pt",
    skip_pose: bool = False,
    pose_model_size: str = "yolo11m-pose.pt",
    calibration_path: Path | None = None,
    n_teams: int = 2,
    ball_model_path: Path | None = None,
    ball_confidence: float = 0.25,
    enable_lines: bool = False,
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
    print(f"Pose:        {'OFF' if skip_pose else pose_model_size + ' (GPU)'}")
    print(f"Ball model:  {ball_model_path or 'COCO generic (use --ball-model for fine-tuned)'}")
    print(f"Court cal:   {calibration_path or 'none (set with --calibration)'}")
    print(f"Teams:       {n_teams}")
    print(f"Lines POC:   {'ON' if enable_lines else 'OFF (use --lines to enable)'}")
    print(f"ffmpeg:      {'yes' if HAS_FFMPEG else 'no'}")
    print(f"Output:      {output_dir}")
    print()

    # --- Initialize components ---
    print("Loading YOLO detection model...")
    yolo_model = YOLO(yolo_model_size)

    ball_model = None
    if ball_model_path and ball_model_path.exists():
        print(f"Loading custom ball model: {ball_model_path.name}")
        ball_model = YOLO(str(ball_model_path))
    elif ball_model_path:
        print(f"Warning: Ball model not found at '{ball_model_path}', using COCO generic.")

    pose_model = None
    if not skip_pose:
        print("Loading YOLO11-pose model (GPU)...")
        pose_model = create_pose_model(pose_model_size)

    team_clf = TeamClassifier(n_teams=n_teams)
    court_mapper = CourtMapper(calibration_path)

    if court_mapper.is_calibrated:
        print("Court calibration loaded.")
    else:
        print("No court calibration — court positions will be unavailable.")

    state_path = output_dir / f"{output_stem}_states.jsonl"
    court_video_path = output_dir / f"{output_stem}_court.mp4" if enable_lines or court_mapper.is_calibrated else None
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

                chunk_court_path = (
                    output_dir / f"{output_stem}_court_chunk{chunk_idx + 1:03d}.mp4"
                    if court_video_path else None
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
                    team_clf=team_clf,
                    pose_model=pose_model,
                    court_mapper=court_mapper,
                    state_exporter=exporter,
                    show_preview=show_preview,
                    max_persons=max_persons,
                    yolo_confidence=yolo_confidence,
                    skip_pose=skip_pose,
                    ball_model=ball_model,
                    ball_confidence=ball_confidence,
                    enable_lines=enable_lines,
                    court_video_path=chunk_court_path,
                )

                total_processed += written
                chunk_paths.append(chunk_path)
                print(f"\n  Saved: {chunk_path.name}")

        finally:
            if show_preview:
                cv2.destroyAllWindows()

    print(f"\n{'=' * 60}")
    print(f"Done! {total_processed} frames in {total_chunks} chunk(s).")
    print(f"State data: {state_path}")
    if court_video_path:
        print(f"Court view: {output_dir}/{output_stem}_court_chunk*.mp4")
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
        "--yolo-model", type=str, default="yolo11n.pt",
        choices=["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"],
    )
    parser.add_argument(
        "--no-pose", action="store_true",
        help="Skip pose estimation (much faster)",
    )
    parser.add_argument(
        "--pose-model", type=str, default="yolo11m-pose.pt",
        choices=["yolo11n-pose.pt", "yolo11s-pose.pt", "yolo11m-pose.pt", "yolo11l-pose.pt", "yolo11x-pose.pt"],
        help="YOLO11-pose model size (default: yolo11m-pose.pt, runs on GPU)",
    )
    parser.add_argument(
        "--calibration", type=str, default=None,
        help="Path to court calibration JSON (see calibrate.py)",
    )
    parser.add_argument(
        "--n-teams", type=int, default=2,
        help="Number of teams to cluster (default: 2)",
    )
    parser.add_argument(
        "--ball-model", type=str, default=None,
        help="Path to fine-tuned ball detection model (e.g. models/handball_ball.pt)",
    )
    parser.add_argument(
        "--ball-confidence", type=float, default=0.25,
        help="Ball detection confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--lines", action="store_true",
        help="Enable automatic line detection + court homography (POC)",
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
    ball_path = Path(args.ball_model) if args.ball_model else None

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
        pose_model_size=args.pose_model,
        calibration_path=cal_path,
        n_teams=args.n_teams,
        ball_model_path=ball_path,
        ball_confidence=args.ball_confidence,
        enable_lines=args.lines,
    )


if __name__ == "__main__":
    main()
