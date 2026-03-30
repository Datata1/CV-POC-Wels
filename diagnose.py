"""Diagnostic tool for pose estimation and homography issues.

Extracts a single frame (or a few), runs each stage separately,
and saves intermediate debug images so you can see exactly what
each pipeline step produces.

Usage:
    uv run python diagnose.py                         # frame 100 from input/
    uv run python diagnose.py --frame 500             # specific frame
    uv run python diagnose.py --input path/to/video.mp4
    uv run python diagnose.py --frame 100 --frame 200 --frame 300  # multiple
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from pipeline.detector import detect_and_track, PERSON_CLASS
from pipeline.pose import create_pose_model, estimate_poses_batch, POSE_CONNECTIONS
from pipeline.lines import (
    segment_court, detect_court_lines, classify_lines, merge_lines,
    find_intersections, estimate_homography, _line_angle, draw_debug_lines,
)

INPUT_DIR = Path(__file__).parent / "input"
DIAG_DIR = Path(__file__).parent / "output" / "diagnostics"


def find_video(input_dir: Path) -> Path:
    for f in sorted(input_dir.iterdir()):
        if f.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
            return f
    raise FileNotFoundError(f"No video in {input_dir}")


def extract_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read frame {frame_idx} from {video_path}")
    return frame


# ── Pose diagnostics ────────────────────────────────────────────────────

def diagnose_pose(frame: np.ndarray, tag: str, yolo_model: str, pose_model_path: str, ball_model_path: str | None):
    print(f"\n{'='*60}")
    print(f"  POSE DIAGNOSTICS — {tag}")
    print(f"{'='*60}")

    # 1. Detection
    model = YOLO(yolo_model)
    ball_model = YOLO(ball_model_path) if ball_model_path else None
    persons, balls = detect_and_track(frame, model, 0.3, 20, ball_model=ball_model)
    print(f"  Detected {len(persons)} persons, {len(balls)} balls")

    # Draw detection boxes on a copy
    det_img = frame.copy()
    for p in persons:
        x1, y1, x2, y2 = p["bbox"]
        cv2.rectangle(det_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(det_img, f"#{p['track_id']} {p['conf']:.2f}",
                    (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(str(DIAG_DIR / f"{tag}_1_detections.jpg"), det_img)
    print(f"  Saved: {tag}_1_detections.jpg")

    # 2. Pose (batch — full frame)
    pose_model = create_pose_model(pose_model_path)

    # 2a. Raw pose model output (no bbox matching)
    raw_results = pose_model(frame, verbose=False)
    raw_img = frame.copy()
    n_raw_poses = 0
    if raw_results and raw_results[0].keypoints and raw_results[0].keypoints.xy is not None:
        kpts = raw_results[0].keypoints
        xy_all = kpts.xy.cpu().numpy()
        conf_all = kpts.conf.cpu().numpy() if kpts.conf is not None else None
        n_raw_poses = len(xy_all)
        print(f"  Raw pose model detected {n_raw_poses} poses")

        if raw_results[0].boxes is not None:
            pose_boxes = raw_results[0].boxes.xyxy.cpu().numpy()
            pose_confs = raw_results[0].boxes.conf.cpu().numpy()
            print(f"  Pose model boxes: {len(pose_boxes)}")
            for i, (box, conf) in enumerate(zip(pose_boxes, pose_confs)):
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(raw_img, (x1, y1), (x2, y2), (255, 0, 255), 1)
                cv2.putText(raw_img, f"pose#{i} {conf:.2f}",
                            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

        for pi in range(n_raw_poses):
            color = [(0, 255, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 128, 255)][pi % 5]
            xy = xy_all[pi]
            conf = conf_all[pi] if conf_all is not None else np.ones(17)
            for k in range(17):
                if conf[k] > 0.3:
                    cv2.circle(raw_img, (int(xy[k][0]), int(xy[k][1])), 4, color, -1)
            for s, e in POSE_CONNECTIONS:
                if conf[s] > 0.3 and conf[e] > 0.3:
                    cv2.line(raw_img, (int(xy[s][0]), int(xy[s][1])),
                             (int(xy[e][0]), int(xy[e][1])), color, 2)
    else:
        print(f"  Raw pose model: NO poses detected!")

    cv2.imwrite(str(DIAG_DIR / f"{tag}_2a_raw_poses.jpg"), raw_img)
    print(f"  Saved: {tag}_2a_raw_poses.jpg")

    # 2b. Matched poses (batch matching as in the pipeline)
    bboxes = [p["bbox"] for p in persons]
    matched_poses = estimate_poses_batch(frame, bboxes, pose_model)

    matched_img = frame.copy()
    n_matched = sum(1 for p in matched_poses if p is not None)
    print(f"  Matched poses: {n_matched}/{len(bboxes)}")

    for i, (p, pose) in enumerate(zip(persons, matched_poses)):
        x1, y1, x2, y2 = p["bbox"]
        if pose is None:
            cv2.rectangle(matched_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(matched_img, f"#{p['track_id']} NO POSE",
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            cv2.rectangle(matched_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            lms = pose["landmarks"]
            for k in range(17):
                if lms[k]["visibility"] > 0.3:
                    cv2.circle(matched_img, (int(lms[k]["x"]), int(lms[k]["y"])), 4, (0, 0, 255), -1)
            for s, e in POSE_CONNECTIONS:
                if lms[s]["visibility"] > 0.3 and lms[e]["visibility"] > 0.3:
                    cv2.line(matched_img,
                             (int(lms[s]["x"]), int(lms[s]["y"])),
                             (int(lms[e]["x"]), int(lms[e]["y"])),
                             (0, 255, 255), 2)

    cv2.imwrite(str(DIAG_DIR / f"{tag}_2b_matched_poses.jpg"), matched_img)
    print(f"  Saved: {tag}_2b_matched_poses.jpg")

    # 2c. Print matching distances for debugging jitter
    if raw_results and raw_results[0].boxes is not None:
        pose_boxes = raw_results[0].boxes.xyxy.cpu().numpy()
        bbox_centers = np.array([((b[0]+b[2])/2, (b[1]+b[3])/2) for b in bboxes])
        pose_centers = np.array([((p[0]+p[2])/2, (p[1]+p[3])/2) for p in pose_boxes])
        if len(bbox_centers) > 0 and len(pose_centers) > 0:
            dists = np.linalg.norm(bbox_centers[:, None] - pose_centers[None, :], axis=2)
            print(f"\n  Bbox-to-pose distance matrix (min per bbox):")
            for bi in range(min(len(bboxes), 5)):
                min_d = dists[bi].min()
                bw = bboxes[bi][2] - bboxes[bi][0]
                bh = bboxes[bi][3] - bboxes[bi][1]
                threshold = max(bw, bh) * 0.75
                print(f"    bbox#{bi} (track={persons[bi]['track_id']}): "
                      f"min_dist={min_d:.1f}  threshold={threshold:.1f}  "
                      f"{'MATCH' if min_d <= threshold else 'REJECTED'}")


# ── Line / homography diagnostics ───────────────────────────────────────

def diagnose_lines(frame: np.ndarray, tag: str):
    print(f"\n{'='*60}")
    print(f"  LINE / HOMOGRAPHY DIAGNOSTICS — {tag}")
    print(f"{'='*60}")

    h, w = frame.shape[:2]
    print(f"  Frame size: {w}x{h}")

    # 1. Court segmentation (blue floor)
    court_mask = segment_court(frame)
    if court_mask is not None:
        court_pct = np.count_nonzero(court_mask) / (h * w) * 100
        print(f"  Court mask: {court_pct:.1f}% of frame")
        # Save court mask + overlay
        mask_color = cv2.cvtColor(court_mask, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(frame, 0.5, mask_color, 0.5, 0)
        cv2.imwrite(str(DIAG_DIR / f"{tag}_3a_court_mask.jpg"), court_mask)
        cv2.imwrite(str(DIAG_DIR / f"{tag}_3a_court_overlay.jpg"), overlay)
        print(f"  Saved: {tag}_3a_court_mask.jpg, {tag}_3a_court_overlay.jpg")
    else:
        print(f"  Court mask: FAILED (no blue area found)")
        cv2.imwrite(str(DIAG_DIR / f"{tag}_0_original.jpg"), frame)
        return

    # 2. White lines inside court
    raw_lines = detect_court_lines(frame, court_mask)
    n_raw = len(raw_lines) if raw_lines is not None else 0
    print(f"  Raw Hough lines (inside court): {n_raw}")

    lines_img = frame.copy()
    if raw_lines is not None:
        for x1, y1, x2, y2 in raw_lines:
            cv2.line(lines_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
    cv2.imwrite(str(DIAG_DIR / f"{tag}_3b_raw_lines.jpg"), lines_img)
    print(f"  Saved: {tag}_3b_raw_lines.jpg")

    # 3. Classify + merge
    if raw_lines is not None and len(raw_lines) > 0:
        h_raw, v_raw = classify_lines(raw_lines)
        print(f"  Classified: {len(h_raw)} horizontal, {len(v_raw)} vertical")

        h_merged = merge_lines(h_raw, cluster_gap=h * 0.04)
        v_merged = merge_lines(v_raw, cluster_gap=w * 0.04)
        print(f"  After merge: {len(h_merged)} horizontal, {len(v_merged)} vertical")

        merged_img = frame.copy()
        for seg in h_merged:
            x1, y1, x2, y2 = seg
            cv2.line(merged_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(merged_img, "H", ((x1+x2)//2, (y1+y2)//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        for seg in v_merged:
            x1, y1, x2, y2 = seg
            cv2.line(merged_img, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.putText(merged_img, "V", ((x1+x2)//2, (y1+y2)//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        intersections = find_intersections(h_merged, v_merged, frame.shape)
        print(f"  Intersections: {len(intersections)}")
        for pt in intersections:
            cv2.circle(merged_img, (int(pt[0]), int(pt[1])), 8, (0, 0, 255), -1)

        cv2.imwrite(str(DIAG_DIR / f"{tag}_3c_merged_lines.jpg"), merged_img)
        print(f"  Saved: {tag}_3c_merged_lines.jpg")

        print(f"\n  Line details:")
        print(f"  {'Type':<6} {'Angle':>7} {'MidY/X':>8} {'Length':>8}")
        for seg in h_merged:
            x1, y1, x2, y2 = seg
            angle = _line_angle(x1, y1, x2, y2)
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            mid = (y1+y2)/2
            print(f"  {'H':<6} {angle:>7.1f}° {mid:>8.1f} {length:>8.1f}px")
        for seg in v_merged:
            x1, y1, x2, y2 = seg
            angle = _line_angle(x1, y1, x2, y2)
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            mid = (x1+x2)/2
            print(f"  {'V':<6} {angle:>7.1f}° {mid:>8.1f} {length:>8.1f}px")

    # 4. Full homography pipeline
    H, debug = estimate_homography(frame)
    if H is not None:
        print(f"\n  Homography FOUND")
        print(f"  Src pts: {debug.get('src_pts', [])}")
        print(f"  Dst pts: {debug.get('dst_pts', [])}")

        test_pts = np.float32([
            [[w/2, h/2]],
            [[0, 0]],
            [[w, h]],
        ])
        mapped = cv2.perspectiveTransform(test_pts, H)
        print(f"  Frame center → court: ({mapped[0][0][0]:.1f}, {mapped[0][0][1]:.1f}) m")
        print(f"  Top-left     → court: ({mapped[1][0][0]:.1f}, {mapped[1][0][1]:.1f}) m")
        print(f"  Bot-right    → court: ({mapped[2][0][0]:.1f}, {mapped[2][0][1]:.1f}) m")

        cx, cy = mapped[0][0]
        if not (-10 <= cx <= 50 and -10 <= cy <= 30):
            print(f"  WARNING: Frame center maps outside court!")

        homo_img = frame.copy()
        draw_debug_lines(homo_img, debug)
        cv2.imwrite(str(DIAG_DIR / f"{tag}_4_homography.jpg"), homo_img)
        print(f"  Saved: {tag}_4_homography.jpg")
    else:
        print(f"\n  Homography FAILED")
        print(f"  court_mask={debug.get('court_mask')}, "
              f"raw_lines={debug.get('raw_lines')}, "
              f"h_merged={len(debug.get('h_merged', []))}, "
              f"v_merged={len(debug.get('v_merged', []))}, "
              f"src_pts={len(debug.get('src_pts', []))}")
        if debug.get("sanity_fail"):
            print(f"  Sanity fail: center mapped to {debug['sanity_fail']}")

    cv2.imwrite(str(DIAG_DIR / f"{tag}_0_original.jpg"), frame)
    print(f"  Saved: {tag}_0_original.jpg")


def main():
    parser = argparse.ArgumentParser(description="Diagnose pose / homography issues")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--frame", type=int, action="append", default=None,
                        help="Frame index to analyze (can repeat). Default: [100]")
    parser.add_argument("--yolo-model", type=str, default="yolo11n.pt")
    parser.add_argument("--pose-model", type=str, default="yolo11m-pose.pt")
    parser.add_argument("--ball-model", type=str, default=None)
    args = parser.parse_args()

    frames = args.frame or [100]

    if args.input:
        video_path = Path(args.input)
    else:
        video_path = find_video(INPUT_DIR)

    print(f"Video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"Total frames: {total}, FPS: {fps:.1f}")

    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    for fidx in frames:
        if fidx >= total:
            print(f"Skipping frame {fidx} (only {total} frames)")
            continue

        print(f"\n{'#'*60}")
        print(f"  FRAME {fidx}  ({fidx/fps:.1f}s)")
        print(f"{'#'*60}")

        frame = extract_frame(video_path, fidx)
        tag = f"f{fidx:05d}"

        diagnose_pose(frame, tag, args.yolo_model, args.pose_model, args.ball_model)
        diagnose_lines(frame, tag)

    print(f"\n{'='*60}")
    print(f"All diagnostics saved to: {DIAG_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
