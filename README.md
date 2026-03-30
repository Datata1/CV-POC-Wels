# Handball Match Analysis

A computer vision and machine learning tool for handball trainers to **analyze match
recordings** — automatically detecting players, tracking the ball, identifying teams,
mapping positions to a 2D court, and predicting player actions.

Built as a university project for an external customer (handball club).

## Overview

The system consists of two processing stages:

1. **Stage 1 — Computer Vision Pipeline** (existing)
   Processes match video frame-by-frame to extract structured data.
2. **Stage 2 — Action Prediction** (planned)
   Uses graph neural networks on the structured data to predict player actions.

```
📹 Match Video
  → Stage 1: Detection, Tracking, Team ID, Court Mapping
    → 🗄️ DuckDB (structured match data)
      → Stage 2: GCN + LSTM → Action Prediction (pass / shot / dribble / …)
      → 📊 Analytics (heatmaps, possession, formations)
      → 🎬 Annotated Video
```

## How it works (Stage 1)

1. **Person & ball detection** — YOLO11 detects people and sports balls with built-in ByteTrack tracking for stable IDs across frames.
2. **Pose estimation** — YOLO11-pose runs a single GPU forward pass on the full frame, returning 17 COCO keypoints per person.
3. **Team classification** — HSV torso histograms are clustered via K-Means to assign team labels.
4. **Court mapping** — Calibration maps pixel positions to real-world court coordinates (metres).
5. **Visualization** — Bounding boxes, skeleton connections, joint dots, team colors, and a HUD are drawn onto each frame.
6. **Data export** — Per-frame structured data with player positions, poses, ball location, and court coordinates.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- **NVIDIA GPU** with CUDA support (e.g. RTX 3060 12GB) for GPU-accelerated inference

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

# 6. With court calibration
uv run python analyze.py --calibration court_cal.json

# 7. With custom ball model
uv run python analyze.py --ball-model models/handball_ball.pt
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
├── analyze.py              ← Full analysis pipeline (Stage 1)
├── detect.py               ← Legacy detection script
├── calibrate.py            ← Court calibration tool
├── pyproject.toml          ← Project config & dependencies
├── Makefile                ← Common commands
│
├── pipeline/               ← Stage 1 core modules
│   ├── detector.py         ← YOLO11 person/ball detection + ByteTrack
│   ├── pose.py             ← YOLO11-pose GPU pose estimation
│   ├── team.py             ← Jersey color team classification
│   ├── court.py            ← Court homography mapping
│   ├── state.py            ← Per-frame state export
│   ├── draw.py             ← Annotation drawing utilities
│   └── tracker.py          ← Fallback IoU tracker
│
├── input/                  ← Drop your video file here
├── output/                 ← Annotated videos + state data
├── models/                 ← Trained model weights
├── annotation/             ← Training data (ball labels, team crops)
│   └── ball/               ← Ball detection dataset (Roboflow export)
├── runs/                   ← YOLO training runs & results
│
├── guides/                 ← Project documentation
│   ├── PROPOSAL.md         ← Customer project proposal
│   ├── ANNOTATION_GUIDE.md ← How to label training data
│   ├── ACTION_PREDICTION_GUIDE.md  ← Stage 2: GCN + LSTM design
│   └── DUCKDB_STORAGE_GUIDE.md     ← DuckDB data storage design
│
└── assets/                 ← Diagrams & visual documentation
    └── context.md          ← C4 Context diagram
```

## Documentation

| Document | Description |
|----------|-------------|
| [guides/PROPOSAL.md](guides/PROPOSAL.md) | Project proposal for the customer — problem, solution, deliverables, timeline |
| [guides/ANNOTATION_GUIDE.md](guides/ANNOTATION_GUIDE.md) | How to label training data (ball detection, team crops, court calibration) |
| [guides/ACTION_PREDICTION_GUIDE.md](guides/ACTION_PREDICTION_GUIDE.md) | Stage 2 design — graph neural networks for action prediction |
| [guides/DUCKDB_STORAGE_GUIDE.md](guides/DUCKDB_STORAGE_GUIDE.md) | DuckDB as structured data store (replacing JSONL) |
| [assets/context.md](assets/context.md) | C4 Context diagram — high-level system overview |

## GPU Acceleration

All inference runs on GPU via PyTorch CUDA. The pipeline automatically detects CUDA availability and falls back to CPU if needed. On an NVIDIA RTX 3060:

- **YOLO11-pose (medium)**: ~35+ FPS for pose estimation
- **YOLO11 detection**: runs on GPU via ultralytics
- **Batch pose estimation**: single forward pass for all players per frame
- **Stage 2 (GCN + LSTM)**: < 1 GB VRAM — trivial for a 12 GB GPU

### Pose Model Sizes

| Model | Speed | Accuracy | Recommended for |
|-------|-------|----------|----------------|
| `yolo11n-pose.pt` | Fastest | Good | Real-time / low-res |
| `yolo11s-pose.pt` | Fast | Better | Balanced |
| `yolo11m-pose.pt` | Medium | High | **Default — best accuracy/speed trade-off** |
| `yolo11l-pose.pt` | Slower | Higher | High accuracy needs |
| `yolo11x-pose.pt` | Slowest | Highest | Maximum precision |
