# Handball Pose Detection POC

Detects players in handball match videos, draws **bounding boxes** around each person, and overlays **joint/pose positions** (skeleton) using OpenCV + MediaPipe.

## How it works

1. **Person detection** — OpenCV's built-in HOG + SVM pedestrian detector finds people in each frame and applies Non-Maximum Suppression to remove overlapping boxes.
2. **Pose estimation** — For each detected person, the bounding box region is cropped and fed into [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose.html), which returns 33 body landmarks (joints).
3. **Visualization** — Bounding boxes, skeleton connections, and joint dots are drawn onto each frame. The annotated video is written to the `output/` directory.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Place your video in the input directory
cp /path/to/handball_match.mp4 input/

# 3. Run detection
uv run python detect.py

# 4. (Optional) Live preview while processing
uv run python detect.py --preview
```

The output video will be saved as `output/<video_name>_detected.mp4`.

## CLI Options

| Flag | Description |
|------|-------------|
| `--input PATH` | Path to a specific video file (default: first video in `input/`) |
| `--output PATH` | Custom output path (default: `output/<name>_detected.mp4`) |
| `--preview` | Show a live preview window (press `q` to quit) |
| `--max-persons N` | Max persons to process per frame (default: 20) |

## Project Structure

```
opencv/
├── input/              ← Drop your video file here
├── output/             ← Annotated videos appear here
├── detect.py           ← Main detection & pose estimation script
├── pyproject.toml      ← Project config & dependencies
└── README.md
```

## Limitations & Next Steps

This is a **proof-of-concept**. Known limitations:

- **HOG detector** works for upright people but may miss fast-moving or partially occluded players. For production, consider switching to a YOLO-based detector (e.g. YOLOv8 via `ultralytics`) for much better accuracy.
- **MediaPipe Pose** processes one person at a time (cropped). For multi-person pose, consider using [MMPose](https://github.com/open-mmlab/mmpose) or MediaPipe's newer Holistic model.
- Processing speed depends on resolution and number of detections. Downscaling input or skipping frames can help.

### Upgrading to YOLO (recommended for production)

Add `ultralytics` to dependencies and replace the HOG detector:

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")  # nano model, fast
results = model(frame, classes=[0])  # class 0 = person
```

This gives significantly better detection in sports scenarios.
