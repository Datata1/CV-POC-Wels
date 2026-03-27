# Handball Match Analysis

Detects players in handball match videos, draws **bounding boxes** around each person, and overlays **joint/pose positions** (skeleton) using YOLO11 + YOLO11-pose running on **GPU (CUDA)**.

## How it works

1. **Person & ball detection** — YOLO11 detects people and sports balls with built-in ByteTrack tracking for stable IDs across frames.
2. **Pose estimation** — YOLO11-pose runs a single GPU forward pass on the full frame, returning 17 COCO keypoints per person. Poses are matched to player bounding boxes by center distance.
3. **Team classification** — HSV torso histograms are clustered via K-Means to assign team labels.
4. **Court mapping** — Optional 4-point calibration maps pixel positions to real-world court coordinates (metres).
5. **Visualization** — Bounding boxes, skeleton connections, joint dots, team colors, and a HUD are drawn onto each frame.
6. **State export** — Per-frame JSON-lines data with player positions, poses, ball location, and court coordinates.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- **NVIDIA GPU** with CUDA support (e.g. RTX 3060) for GPU-accelerated pose estimation

## Quick Start

```bash
# 1. Install dependencies (includes PyTorch CUDA)
uv sync

# 2. Place your video in the input directory
cp /path/to/handball_match.mp4 input/

# 3. Run the full pipeline (detection + tracking + teams + pose + export)
uv run python analyze.py

# 4. Fast mode (skip pose estimation)
uv run python analyze.py --no-pose

# 5. Live preview while processing
uv run python analyze.py --preview
```

Output: `output/<name>_chunk*.mp4` (annotated video) and `output/<name>_states.jsonl` (per-frame data).

## CLI Options (analyze.py)

| Flag | Description |
|------|-------------|
| `--input PATH` | Path to a specific video file (default: first video in `input/`) |
| `--output-dir PATH` | Output directory (default: `output/`) |
| `--preview` | Show a live preview window (press `q` to quit) |
| `--max-persons N` | Max persons per frame (default: 20) |
| `--chunk-seconds N` | Duration per output chunk (default: 60) |
| `--confidence F` | YOLO detection threshold (default: 0.3) |
| `--yolo-model` | Detection model: yolo11n/s/m/l/x.pt (default: yolo11n.pt) |
| `--pose-model` | Pose model: yolo11n/s/m/l/x-pose.pt (default: yolo11m-pose.pt) |
| `--no-pose` | Skip pose estimation (faster) |
| `--ball-model PATH` | Fine-tuned ball detection model (e.g. `models/handball_ball.pt`) |
| `--ball-confidence F` | Ball detection threshold (default: 0.25) |
| `--calibration PATH` | Court calibration JSON (see `calibrate.py`) |
| `--n-teams N` | Number of teams to cluster (default: 2) |

## Project Structure

```
opencv/
├── input/              ← Drop your video file here
├── output/             ← Annotated videos + state JSONL
├── models/             ← Pose models (auto-downloaded by ultralytics)
├── pipeline/
│   ├── pose.py         ← YOLO11-pose GPU pose estimation
│   ├── detector.py     ← YOLO11 person/ball detection + ByteTrack
│   ├── draw.py         ← Annotation drawing utilities
│   ├── team.py         ← Jersey color team classification
│   ├── court.py        ← Court homography mapping
│   ├── state.py        ← Per-frame state export (JSONL)
│   └── tracker.py      ← Fallback IoU tracker
├── detect.py           ← Legacy detection script
├── analyze.py          ← Full analysis pipeline
├── calibrate.py        ← Court calibration tool
├── pyproject.toml      ← Project config & dependencies
└── README.md
```

## GPU Acceleration

All pose estimation runs on GPU via PyTorch CUDA. The pipeline automatically detects CUDA availability and falls back to CPU if needed. On an NVIDIA RTX 3060:

- **YOLO11-pose (medium)**: ~35+ FPS for pose estimation (~20% faster than YOLOv8)
- **YOLO11 detection**: runs on GPU via ultralytics
- **Batch pose estimation**: single forward pass for all players per frame

### Pose Model Sizes

| Model | Speed | Accuracy | Recommended for |
|-------|-------|----------|----------------|
| `yolo11n-pose.pt` | Fastest | Good | Real-time / low-res |
| `yolo11s-pose.pt` | Fast | Better | Balanced |
| `yolo11m-pose.pt` | Medium | High | **Default — best accuracy/speed trade-off** |
| `yolo11l-pose.pt` | Slower | Higher | High accuracy needs |
| `yolo11x-pose.pt` | Slowest | Highest | Maximum precision |
