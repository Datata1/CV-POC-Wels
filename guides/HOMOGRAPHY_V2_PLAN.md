# Homography v2 — Goal + Court Segmentation (Action Plan)

Step-by-step plan for replacing the broken line-detection approach with
goal detection + trained court segmentation for homography. Follow in order.

**Why the current approach fails:**
- Line detection: HSV picks up ads, jerseys, goal net — not just court lines.
- Court segmentation (HSV blue mask): Captures the blue **advertising boards**
  behind the far sideline ("Official Fresh Food Partner", "pixum", etc.)
  as part of the "court". Contour corners land on the ad edge, not the
  actual sideline. This is unfixable with color filters — the ads are the
  same blue as the floor.

**Two things need to work:**
1. **Goal detection** → 2 stable anchor points (post foot-points on baseline)
2. **Court floor segmentation** → clean boundary polygon → 2–4 corner points

Both require annotation + training. You can **share the same extracted frames**
for both — extract once, annotate twice (bbox for goals, polygon for court).

**End goal:** Per-frame homography that maps player pixel positions to real
court coordinates (40m × 20m), surviving camera pans.

---

## Phase 0 — Extract Shared Frames

Both goal and court annotation need labeled frames from your video.
Extract once, use for both.

### Step 1: Extract frames

```bash
make extract-goal-frames
```

This produces ~970 frames in `annotation/goal/raw_images/` (every 300th
frame from the video). You'll upload the same images to both Roboflow
projects. Pick frames with good variety:

- Various camera pan positions (left attack, midfield, right attack)
- Both goals visible at times
- Different zoom levels
- Some tight shots, some wide shots

---

## Phase 1 — Goal Detection Model

### Step 2: Label goals in Roboflow

1. Go to [app.roboflow.com](https://app.roboflow.com) → workspace `hsc-wels`
2. Create new project:
   - Type: **Object Detection**
   - Name: `handball-goal`
3. Upload the extracted frames from step 1
4. Label each visible goal with a **bounding box**:
   - Class name: **`goal`** (single class)
   - The box should cover the **full goal frame** — both posts and crossbar
   - Include the net area if it's inside the structural frame
   - If both goals are visible, label both
   - If a goal is partially cut off at the edge, still label the visible part
5. Labeling targets:
   - **Minimum: 150 labeled frames** (at least 200 goal instances)
   - Label both goals when visible → ~250–350 total annotations
6. Apply **70/20/10** train/valid/test split
7. Augmentations:
   - Horizontal flip ✅
   - Brightness ±20% ✅
   - Blur up to 2px ✅
   - No rotation (goals are always upright)
   - No mosaic (goals are large objects)
8. Export → Format: **YOLOv8** → Download zip

### Step 3: Train goal model

```bash
unzip ~/Downloads/handball-goal-*.zip -d annotation/goal/
make train-goal
```

Watch for `val/mAP50 > 0.85` (goals are large, distinctive).

### Step 4: Validate and install

```bash
make validate-goal
make predict-goal    # check runs/detect/predict/ visually
make install-goal    # → models/handball_goal.pt
```

---

## Phase 2 — Court Floor Segmentation Model

**Why HSV fails:** The current `segment_court()` uses HSV blue filtering.
It picks up blue advertising boards behind the far sideline, blue elements
in the crowd, and blue clothing — making the mask boundary useless for
extracting actual court corners.

**The fix:** Train a YOLO11-seg instance segmentation model. The model
**learns** what "court floor" looks like vs. "blue advertising" — something
a color filter fundamentally cannot do.

### Step 5: Label court floor polygons in Roboflow

1. In Roboflow workspace `hsc-wels`, create a **new project**:
   - Type: **Instance Segmentation** (NOT object detection)
   - Name: `handball-court`
2. Upload the **same frames** from step 1
3. For each frame, draw a **polygon** around the visible court floor:
   - Class name: **`court`** (single class)
   - Trace the **actual sidelines and baselines** — NOT the advertising
     boards behind them
   - The polygon should follow the court edges precisely:
     - Far sideline = the white line separating court from the ad band
     - Near sideline = the white line separating court from bench area
     - Baselines = vertical lines at each end (or frame edge if cut off)
   - If the court extends off-frame, trace along the frame edge
   - **Include the goal area** (6m zone) as part of the court polygon
   - **Exclude** players, advertising overlays on the floor, and areas
     beyond the sidelines
4. Labeling guidelines:
   - You don't need pixel-perfect edges — roughly following the sidelines
     is fine. YOLO-seg will generalize.
   - **100–150 frames is enough** for a single-class segmentation task
   - Include frames from all camera positions (panned left/center/right)
   - Include frames where court is partially obscured (replays, zooms)
     → label what's visible
5. Augmentations:
   - Horizontal flip ✅
   - Brightness ±20% ✅
   - Blur up to 2px ✅
   - Scale ±15% ✅
6. Export → Format: **YOLOv8-seg** → Download zip

> **Tip:** Roboflow's Smart Polygon tool can auto-trace the blue floor.
> You'll just need to manually fix the boundary where it bleeds into the
> ad boards — much faster than tracing from scratch.

### Step 6: Train court segmentation model

```bash
unzip ~/Downloads/handball-court-*.zip -d annotation/court/
make train-court
```

This uses `yolo segment train` (not `detect train`). Expect:
- `val/mAP50-mask > 0.85` — court floor is a large, consistent region
- Training is slightly slower than bbox detection due to mask computation

### Step 7: Validate and install

```bash
make validate-court
make predict-court    # check runs/segment/predict/ visually
make install-court    # → models/handball_court.pt
```

**What to check in predictions:**
- Court polygon follows the **actual sidelines**, not the ad boards
- Near sideline boundary stops at the bench area
- Goal area included inside the polygon
- No major gaps or holes from players/floor ads

### What this replaces

The trained segmentation model replaces `segment_court()` (the HSV
blue filter). The difference:

| | HSV `segment_court()` | YOLO-seg model |
|---|---|---|
| Blue ad boards behind sideline | ❌ Included in mask | ✅ Excluded |
| Blue clothing / crowd | ❌ May leak in | ✅ Excluded |
| Floor advertising overlays | ❌ Creates holes | ✅ Handled |
| Different halls / colours | ❌ Needs HSV tuning | ✅ Generalizes |
| Boundary quality | ❌ Jagged, noisy | ✅ Clean polygon |

---

## Phase 3 — Goal Corner Extraction

Once you have a working goal detector, extract the structural corners
of the goal within the detected bounding box.

### Step 8: Implement goal corner refinement

In `pipeline/lines.py` (or a new `pipeline/goal.py`), add:

1. **Run YOLO goal detection** on the frame → get goal bounding boxes
2. **Crop the ROI** (the bbox region + small padding)
3. **Find the goal posts** inside the ROI:
   - Convert to grayscale
   - Use `cv2.goodFeaturesToTrack()` or edge detection to find the
     vertical post edges and horizontal crossbar
   - The posts are silver/white metal — they contrast with the blue floor
     and the dark net
4. **Extract the 2 foot-points** (where posts meet the ground):
   - These are the bottom-left and bottom-right corners of the goal frame
   - They lie on the baseline → known court coordinates

**Court coordinates of goal foot-points:**

| Point | Court (x, y) metres |
|-------|---------------------|
| Left goal, left post foot   | (0.0, 8.5)  |
| Left goal, right post foot  | (0.0, 11.5) |
| Right goal, left post foot  | (40.0, 11.5) |
| Right goal, right post foot | (40.0, 8.5)  |

> Note: "left/right post" is from the camera's perspective. You'll need to
> determine which post is which by comparing x-coordinates in the image.

**Practical approach for the POC:**
Start with the YOLO bbox corners directly — the bottom-left and bottom-right
of the bbox are rough estimates of the post foot-points. You can refine later
with `goodFeaturesToTrack` or edge analysis. The bbox approach is good enough
to test the full pipeline end-to-end.

---

## Phase 4 — Court Boundary Corners from Segmentation

You need ≥ 4 non-collinear points for a homography. The goal gives 2
(both on the baseline → collinear). The court segmentation polygon gives
the remaining points.

### Step 9: Extract corner points from court polygon

The YOLO-seg model outputs a polygon mask per frame. Extract the court
boundary corners from it:

1. **Run court segmentation model** on the frame → get polygon vertices
   (YOLO-seg returns a list of (x, y) polygon points)
2. **Approximate the polygon** to a quadrilateral:
   ```
   cv2.approxPolyDP(polygon, epsilon, closed=True)
   ```
   Increase epsilon until you get 4–6 vertices. The court in perspective
   is a trapezoid, so 4 vertices is the target.
3. **Sort the 4 corners** into: top-left, top-right, bottom-right, bottom-left
   (by angle from centroid, or by position)
4. **Assign court coordinates** to each corner

**Corner assignment logic:**

The camera is on the sideline (elevated), looking across the court. In
perspective:
- **Top edge** of trapezoid = far sideline (y = 0m in court coords)
- **Bottom edge** of trapezoid = near sideline (y = 20m in court coords)
- **Left/right edges** = baselines or where the court exits the frame

Use the goal position as anchor:
- If goal is on the left → left-side corners are near x ≈ 0m
- If goal is on the right → right-side corners are near x ≈ 40m
- If no goal visible → estimate x from the court polygon's extent
  relative to known court proportions (aspect ratio 2:1)

**What this gives you:**

| Visible scenario | Points from goal | Points from court polygon | Total |
|------------------|-----------------|-------------------------:|------:|
| Left attack      | 2 (left goal feet) | 2–4 (trapezoid corners) | 4–6 ✅ |
| Midfield         | 0–2 (goal at edge) | 3–4 (both sides visible) | 3–6 ✅ |
| Right attack     | 2 (right goal feet) | 2–4 (trapezoid corners) | 4–6 ✅ |

---

## Phase 5 — Homography Computation + Pipeline Integration

### Step 10: Combine points and compute homography

For each frame:

```
1. Run goal detection → 0 or 2 foot-points (with known court coords)
2. Run court segmentation model → polygon → approxPolyDP → 2-4 corners
3. Assign court coordinates to polygon corners (use goal as anchor)
4. Collect all pixel→court point pairs
5. If ≥ 4 non-collinear pairs → cv2.findHomography(RANSAC)
6. Feed into HomographyTracker (already implemented) for temporal smoothing
```

### Step 11: Add CLI flags to analyze.py

Like `--ball-model`, add:

```
--goal-model models/handball_goal.pt
--court-model models/handball_court.pt
```

When provided:
- Load goal YOLO model + court YOLO-seg model alongside person/ball models
- Run both per frame
- Goal → foot-points, Court → polygon corners
- Combine → homography → HomographyTracker
- Replace the current line-detection logic in the `--lines` code path

### Step 12: Update the drawing pipeline

Use the computed homography to transform player foot positions (bottom
center of bbox) to court coordinates, then pass to `render_court()` in
`court_viz.py` — this part already works, it just needs real homography
data instead of the broken line-detection output.

---

## Phase 6 — Test and Iterate

### Step 13: Run the diagnostic script

Update `diagnose.py` to test the new goal + court segmentation pipeline:

```bash
uv run python diagnose.py --frame 100 --frame 300 --frame 600
```

Check:
- [ ] Goal detected in all frames where it's visible?
- [ ] Foot-points roughly at the base of the posts?
- [ ] Court polygon follows the actual sidelines (not the ad boards)?
- [ ] Polygon approximated to 4–6 vertices?
- [ ] ≥ 4 points collected per frame?
- [ ] Player dots on the 2D court map are in plausible positions?

### Step 14: Run the full pipeline

```bash
make analyze-goal
```

Watch the output video. Player dots on the court minimap should:
- Stay roughly in position (not jitter wildly)
- Move when the actual player moves
- Be on the correct half of the court

---

## Quick Reference — Makefile Targets

| Target | What it does |
|--------|-------------|
| `extract-goal-frames` | Extract frames from video for annotation (shared) |
| `train-goal` | Fine-tune YOLO11m on goal bbox annotations |
| `validate-goal` | Run goal validation metrics on test set |
| `predict-goal` | Visual goal predictions on test images |
| `install-goal` | Copy best.pt → models/handball_goal.pt |
| `train-court` | Train YOLO11m-seg on court polygon annotations |
| `validate-court` | Run court segmentation validation |
| `predict-court` | Visual court segmentation predictions |
| `install-court` | Copy best.pt → models/handball_court.pt |
| `analyze-goal` | Full pipeline with goal + court models |

---

## Annotation Summary

You need **two Roboflow projects** (different task types):

| Project | Type | Class | Annotations | Effort |
|---------|------|-------|-------------|--------|
| `handball-goal` | Object Detection | `goal` (bbox) | 150–200 frames | ~1–2 hours |
| `handball-court` | Instance Segmentation | `court` (polygon) | 100–150 frames | ~2–3 hours |

Use the **same extracted frames** for both. Total annotation effort: ~3–5 hours.

---

## What's Already Done (reuse these)

- `HomographyTracker` in `pipeline/lines.py` — EMA + optical flow smoothing ✅
- `render_court()` in `pipeline/court_viz.py` — 2D top-down court drawing ✅
- Ball annotation/training workflow in Makefile — pattern to follow ✅
- Roboflow workspace `hsc-wels` set up ✅
- Goal Makefile targets already added ✅

---

## Known Risks

| Risk | Mitigation |
|------|-----------|
| Goal partially off-screen | HomographyTracker persists last good H for up to 150 frames; court polygon alone may give enough corners |
| Goal posts occluded by goalkeeper | Bbox still detected (YOLO robust to partial occlusion); foot-points from bbox bottom edge |
| Court polygon imprecise at edges | RANSAC in findHomography tolerates outlier points; polygon approximation smooths noise |
| Midfield views with no goal visible | Rely on court polygon corners only (need ≥ 4) + temporal smoothing from last known H |
| Corner assignment wrong | Use goal position as anchor for left/right; validate visually with diagnostic script |
| Different halls / floor colours | Court seg model generalizes better than HSV; for new halls, add 20–30 frames to training set |
| YOLO-seg polygon too detailed | `approxPolyDP` with increasing epsilon until ≤ 6 vertices; or use convex hull first |
