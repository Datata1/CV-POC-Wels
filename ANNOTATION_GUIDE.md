# Annotation Guide — Handball Match Analysis

This guide explains how to collect and annotate training data for the analysis pipeline.

---

## 1. Court Calibration (one-time per camera angle)

```bash
make calibrate
# or: uv run python calibrate.py --input input/match.mp4
```

A window opens showing the first frame. Click 4 court corners **in order**:

| Click | Court point          | Coordinates (metres) |
|-------|---------------------|----------------------|
| 1     | Top-left corner     | (0, 0)               |
| 2     | Top-right corner    | (40, 0)              |
| 3     | Bottom-right corner | (40, 20)             |
| 4     | Bottom-left corner  | (0, 20)              |

> Use the outer sideline intersections at the baseline. If corners are off-screen,
> use the nearest visible court marking and adjust `dst` in the JSON manually.

Press `s` to save → produces `court_cal.json`.

Then run the pipeline with calibration:
```bash
uv run python analyze.py --calibration court_cal.json
```

---

## 2. Ball Detection — Labeling Data for Fine-Tuning YOLO

The pipeline currently uses YOLO's built-in `sports ball` class (COCO class 32),
which works for some scenarios but is not trained specifically on handballs.
For better accuracy, fine-tune YOLOv8 on your own data.

### Step 1: Extract frames

```bash
# Extract every 10th frame as a JPEG
mkdir -p annotation/ball/images
ffmpeg -i input/match.mp4 -vf "select=not(mod(n\,10))" -vsync vfr \
    annotation/ball/images/frame_%05d.jpg
```

### Step 2: Label with CVAT or Roboflow

**Option A — Roboflow (recommended, free tier available):**
1. Go to [app.roboflow.com](https://app.roboflow.com) and create a project
2. Upload the extracted frames
3. Label each ball with a bounding box (class: `handball`)
4. Export in **YOLOv8 format** (you'll get a `data.yaml` + `labels/` directory)

**Option B — CVAT (self-hosted):**
1. Go to [app.cvat.ai](https://app.cvat.ai) or run CVAT locally
2. Create a task, upload frames
3. Draw bounding boxes around every visible ball, label as `handball`
4. Export as **YOLO 1.1**

### Step 3: Labeling tips
- Label even partially occluded balls (held by a player)
- Skip frames where the ball is fully invisible
- Aim for **500–1000+ labeled frames** for decent results
- Use data augmentation (Roboflow does this automatically)

### Step 4: Train

```bash
# After exporting data to annotation/ball/
yolo detect train data=annotation/ball/data.yaml model=yolov8n.pt epochs=50 imgsz=640
```

The trained model will be at `runs/detect/train/weights/best.pt`.
Place it in `models/ball_yolov8.pt` and update the detector to use it.

---

## 3. Team Classification — When Color Clustering Isn't Enough

The pipeline's default team classifier uses unsupervised HSV color clustering,
which works well when jersey colors are distinct. If it struggles (e.g., similar
colors, bad lighting), you can train a small classifier:

### Step 1: Extract player crops

Run the pipeline once to get tracked detections:
```bash
uv run python analyze.py --no-pose --chunk-seconds 10
```

Then extract crops from the state file:
```python
import json, cv2
from pathlib import Path

video = cv2.VideoCapture("input/match.mp4")
states = Path("output/match_states.jsonl").read_text().strip().split("\n")

out = Path("annotation/teams")
for team in ["A", "B", "referee", "other"]:
    (out / team).mkdir(parents=True, exist_ok=True)

for line in states[::30]:  # every 30th frame
    s = json.loads(line)
    video.set(cv2.CAP_PROP_POS_FRAMES, s["frame_id"])
    ret, frame = video.read()
    if not ret:
        continue
    for p in s["players"]:
        x1, y1, x2, y2 = p["bbox"]
        crop = frame[y1:y2, x1:x2]
        # Save to "unsorted" first, then manually sort into team folders
        fname = f"f{s['frame_id']}_t{p['track_id']}.jpg"
        cv2.imwrite(str(out / "A" / fname), crop)  # sort manually after

video.release()
```

### Step 2: Sort crops into folders
Move each crop into the correct folder:
```
annotation/teams/
  A/           ← Team A player crops
  B/           ← Team B player crops
  referee/     ← Referees
  other/       ← Coaches, spectators, etc.
```

Aim for **100+ crops per class**.

### Step 3: Train (if needed in future)
A small MobileNet or ResNet-18 classifier can be trained on these crops.
This is a future enhancement — the color clustering works for the POC.

---

## 4. Player vs. Non-Player Filtering

With court calibration active, the pipeline automatically filters players
by checking if their foot position maps inside the court boundaries.

If you don't have calibration, you can manually annotate which persons
are players vs. non-players (coaches, spectators):

1. Run the pipeline to get the JSONL state file
2. Review detections and note which `track_id`s are non-players
3. Create a filter file:

```json
{
    "exclude_track_ids": [15, 23, 31],
    "reason": "coaches and spectators"
}
```

This can be used as a post-processing step on the state data.

---

## 5. State Data Format

The pipeline exports a `.jsonl` file (one JSON object per line, one line per frame):

```json
{
    "frame_id": 150,
    "timestamp_s": 5.0,
    "ball": {
        "bbox": [520, 340, 545, 365],
        "conf": 0.72,
        "center_px": [532.5, 352.5],
        "court_pos": [22.5, 11.3]
    },
    "players": [
        {
            "track_id": 3,
            "team": "A",
            "bbox": [200, 150, 260, 350],
            "conf": 0.91,
            "foot_px": [230.0, 350.0],
            "court_pos": [15.2, 8.7],
            "on_court": true,
            "pose": [{"x": 230.1, "y": 165.3, "z": -0.12, "vis": 0.95}, ...]
        }
    ],
    "player_count": 14,
    "on_court_count": 12
}
```

### Loading for ML training

```python
import json
import pandas as pd

states = []
with open("output/match_states.jsonl") as f:
    for line in f:
        states.append(json.loads(line))

# Convert to DataFrame for analysis
df = pd.json_normalize(states)
print(f"Loaded {len(df)} frames")
```

---

## Quick Reference

| Task                    | Command                                              |
|------------------------|------------------------------------------------------|
| Install dependencies   | `make install`                                       |
| Download pose model    | `make download-model`                                |
| Calibrate court        | `make calibrate`                                     |
| Run full pipeline      | `make analyze`                                       |
| Run without pose       | `make analyze-fast`                                  |
| Run with preview       | `make analyze-preview`                               |
| Run with calibration   | `uv run python analyze.py --calibration court_cal.json` |
| Extract frames for labeling | `ffmpeg -i input/match.mp4 -vf "select=not(mod(n\,10))" -vsync vfr annotation/images/frame_%05d.jpg` |
| Clean output           | `make clean`                                         |
