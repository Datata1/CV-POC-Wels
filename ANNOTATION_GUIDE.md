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

## 2. Ball Detection — Labeling & Fine-Tuning YOLO11

The pipeline currently uses YOLO's generic `sports ball` class (COCO class 32),
which often misses handballs — they're small, fast, skin-colored, and frequently
occluded by hands. Fine-tuning on your own data makes a **huge** difference.

### Overview

```
Extract frames → Label in Roboflow → Export YOLO format → Train → Plug into pipeline
```

### Step 1: Extract frames from your match videos

```bash
# Every 10th frame (adjust for more/less data)
mkdir -p annotation/ball/raw_images
ffmpeg -i input/match.mp4 -vf "select=not(mod(n\,10))" -vsync vfr \
    annotation/ball/raw_images/frame_%05d.jpg
```

> **Tip:** Extract from multiple matches/camera angles for a more robust model.
> Aim for **1000–2000 frames** total, of which ~60-70% will contain a visible ball.

### Step 2: Label with Roboflow (recommended)

1. Go to [app.roboflow.com](https://app.roboflow.com) → Create Project
   - Project type: **Object Detection**
   - Name: e.g. `handball-ball`
2. Upload the extracted frames
3. Label each visible ball with a **tight bounding box**
   - Class name: `handball`
   - Use the **single class** — don't create multiple ball classes
4. Labeling guidelines:
   - **Label every visible ball**, even if partially occluded by a hand
   - **Skip frames** where the ball is completely invisible
   - **Tight boxes** — the box should be as close to the ball edge as possible
   - Ball in flight, ball in hand, ball on ground — label all of them
   - When in doubt whether something is the ball → label it (false positives are easier to fix than missing data)
5. Use Roboflow's **Smart Polygon / AI-assist** to speed up labeling
6. Apply a **70/20/10 train/valid/test split** in Roboflow
7. Enable augmentations in Roboflow:
   - Flip horizontal ✅
   - Rotation ±15° ✅
   - Brightness ±25% ✅
   - Blur up to 2.5px ✅
   - Mosaic ✅ (very effective for small objects)
8. **Export** → Format: **YOLOv8** (compatible with YOLO11) → Download zip

**Alternative: CVAT (self-hosted)**
1. Go to [app.cvat.ai](https://app.cvat.ai) or run CVAT locally
2. Create a task, upload frames, label as `handball`
3. Export as **YOLO 1.1** format
4. Create `data.yaml` manually (see below)

### Step 3: Prepare the dataset structure

After export, your folder should look like this:

```
annotation/ball/
├── data.yaml          ← class names + paths
├── train/
│   ├── images/        ← training JPEGs
│   └── labels/        ← one .txt per image (YOLO format)
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

The `data.yaml` should contain:
```yaml
train: annotation/ball/train/images
val: annotation/ball/valid/images
test: annotation/ball/test/images

nc: 1
names: ['handball']
```

> Roboflow generates this automatically. If using CVAT, create it manually.

### Step 4: Train on GPU

```bash
# Fine-tune YOLO11n on your labeled handball data (fast, good for small objects)
yolo detect train \
    data=annotation/ball/data.yaml \
    model=yolo11n.pt \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    device=0 \
    name=handball_ball

# For higher accuracy (slower training):
yolo detect train \
    data=annotation/ball/data.yaml \
    model=yolo11s.pt \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    device=0 \
    name=handball_ball_s
```

**Training tips:**
- Start with `yolo11n.pt` — nano is fast to iterate and good for single-class detection
- Use `imgsz=640` (default) or `imgsz=1280` if the ball is very small in your footage
- With ~1000 labeled frames, 100 epochs is a good starting point
- Watch `val/mAP50` — it should plateau around epoch 60–80
- The best model is saved automatically: `runs/detect/handball_ball/weights/best.pt`

### Step 5: Validate the trained model

```bash
# Run validation on the test set
yolo detect val \
    data=annotation/ball/data.yaml \
    model=runs/detect/handball_ball/weights/best.pt \
    device=0

# Visual check: run inference on a few frames
yolo detect predict \
    model=runs/detect/handball_ball/weights/best.pt \
    source=annotation/ball/test/images \
    device=0 \
    save=True
```

Check `runs/detect/predict/` for visual results. You want:
- **mAP50 > 0.7** for a usable model
- **mAP50 > 0.85** for a solid model
- Few false positives (hands/feet detected as ball)

### Step 6: Plug into the pipeline

```bash
# Copy trained model
cp runs/detect/handball_ball/weights/best.pt models/handball_ball.pt

# Run with custom ball model
uv run python analyze.py --ball-model models/handball_ball.pt
```

> **Note:** The `--ball-model` flag requires a pipeline update (see below).
> Until then, you can replace the detection model manually in `pipeline/detector.py`.

### How many labels do I need?

| Labels | Expected quality |
|--------|------------------|
| 200    | Barely usable — lots of missed detections |
| 500    | Decent — catches most balls in clear view |
| 1000   | Good — works in most game situations |
| 2000+  | Excellent — robust to occlusion, blur, lighting |

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
| Download pose model    | `make download-pose-model`                           |
| Calibrate court        | `make calibrate`                                     |
| Run full pipeline      | `make analyze`                                       |
| Run without pose       | `make analyze-fast`                                  |
| Run with preview       | `make analyze-preview`                               |
| Run with calibration   | `uv run python analyze.py --calibration court_cal.json` |
| Extract frames for labeling | `ffmpeg -i input/match.mp4 -vf "select=not(mod(n\,10))" -vsync vfr annotation/images/frame_%05d.jpg` |
| Clean output           | `make clean`                                         |
