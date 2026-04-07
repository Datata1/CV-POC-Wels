# Homography v2 — Goal Detection + Court Keypoint Pose (Action Plan)

Step-by-step plan for replacing the broken line-detection approach with
goal detection + court keypoint detection for homography. Follow in order.

**Why the current approach fails:**
- Line detection: HSV picks up ads, jerseys, goal net — not just court lines.
- Court segmentation (HSV blue mask): Captures the blue **advertising boards**
  behind the far sideline ("Official Fresh Food Partner", "pixum", etc.)
  as part of the "court". Contour corners land on the ad edge, not the
  actual sideline. This is unfixable with color filters — the ads are the
  same blue as the floor.

**Two things need to work:**
1. **Goal detection** → locate the goal, know which side of the court is
   visible (left/right anchor). The bbox alone is NOT precise enough for
   exact post foot-points — see Phase 3 assessment.
2. **Court keypoint pose model** → directly detect court landmarks (corners,
   line intersections, penalty marks) as keypoints with known court
   coordinates → direct pixel↔world correspondences for homography.

Both require annotation + training. You can **share the same extracted frames**
for both — extract once, annotate twice (bbox for goals, keypoints for court).

**Why keypoints instead of court segmentation:**
- **Direct correspondences.** Homography needs pixel↔world point pairs.
  Keypoints give these directly — no polygon extraction, `approxPolyDP`,
  or error-prone corner assignment needed.
- **More points, better distributed.** A handball court has 12–15+
  landmarks spread across the field → more stable homography.
- **Handles partial views naturally.** When the camera pans, just use
  whichever keypoints are detected. No "assign corners using goal anchor".
- **SOTA approach.** SoccerNet camera calibration and similar sports
  analytics systems use exactly this pattern.

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

## Phase 2 — Court Keypoint Pose Model

**Why keypoints instead of segmentation:**
Homography needs pixel↔world point pairs. A segmentation polygon
requires `approxPolyDP` + fragile corner assignment. Keypoints give
correspondences directly — the model outputs the pixel location of
each court landmark, and you already know its world coordinate.
This is the SOTA approach (SoccerNet, etc.).

### Step 5: Define keypoint skeleton

A handball court has these annotatable landmarks (12 keypoints):

| # | Keypoint name | Court (x, y) m | Notes |
|---|---------------|-----------------|-------|
| 0 | court_top_left | (0, 0) | Far sideline × left baseline |
| 1 | court_top_right | (40, 0) | Far sideline × right baseline |
| 2 | court_bottom_left | (0, 20) | Near sideline × left baseline |
| 3 | court_bottom_right | (40, 20) | Near sideline × right baseline |
| 4 | center_top | (20, 0) | Center line × far sideline |
| 5 | center_bottom | (20, 20) | Center line × near sideline |
| 6 | goal_area_top_left | (0, 7) | Left 6m arc meets baseline (far) |
| 7 | goal_area_bottom_left | (0, 13) | Left 6m arc meets baseline (near) |
| 8 | goal_area_top_right | (40, 7) | Right 6m arc meets baseline (far) |
| 9 | goal_area_bottom_right | (40, 13) | Right 6m arc meets baseline (near) |
| 10 | seven_m_left | (7, 10) | Left 7-metre mark |
| 11 | seven_m_right | (33, 10) | Right 7-metre mark |

> **Visibility:** Mark keypoints as **not-visible** when occluded or
> off-frame. YOLO-pose handles visibility flags natively (0 = not labeled,
> 1 = labeled but occluded, 2 = labeled and visible).

### Step 6: Label court keypoints in Roboflow

1. In Roboflow workspace `hsc-wels`, create a **new project**:
   - Type: **Keypoint Detection**
   - Name: `handball-court-keypoints`
   - Define the skeleton with the 12 keypoints above
2. Upload the **same frames** from Phase 0
3. For each frame, place a **bounding box** around the visible court area,
   then mark each visible keypoint:
   - The bbox should cover the entire visible court floor
   - Place keypoints precisely on the court landmark locations
   - Only mark keypoints that are **actually visible** in the frame
   - For a typical panned-left view: you'll see keypoints 0–7, 10
     but NOT 1, 3, 8, 9, 11 (right side off-frame)
   - For midfield views: you may see 4, 5 and some of both sides
4. Labeling guidelines:
   - **150–200 frames** for good coverage across all camera positions
   - Include wide shots (more keypoints visible) and tight shots (fewer)
   - Be precise on keypoint placement — these directly become your
     homography correspondences
   - Include frames from all camera angles (panned left/center/right)
5. Augmentations:
   - Horizontal flip ✅ (flip keypoint indices accordingly)
   - Brightness ±20% ✅
   - Blur up to 2px ✅
   - No rotation (court is always roughly level)
6. Export → Format: **YOLOv8-pose** → Download zip

### Step 7: Train court keypoint model

```bash
unzip ~/Downloads/handball-court-keypoints-*.zip -d annotation/court/
make train-court-keypoints
```

This uses `yolo pose train` with `yolo11m-pose.pt` as the base model.
The base model already understands human pose — fine-tuning it to
detect court keypoints instead is straightforward.

Expect:
- `val/mAP50 > 0.80` — court is always present and large
- Keypoint OKS (Object Keypoint Similarity) should be high since the
  landmarks are well-separated

### Step 8: Validate and install

```bash
make validate-court-keypoints
make predict-court-keypoints    # check runs/pose/predict/ visually
make install-court-keypoints    # → models/handball_court_kp.pt
```

**What to check in predictions:**
- Keypoint dots land on the correct court landmarks
- Visibility flags are correct (no phantom keypoints off-screen)
- Works across all camera positions (panned left/center/right)
- Keypoints are stable across consecutive frames (not jumping)

---

## Phase 3 — Goal Bbox Assessment for Corner Extraction

> **Assessment (based on predict2 results):** The goal bbox is **NOT precise
> enough** for extracting exact post foot-points. Looking at the predictions:
>
> - The bbox wraps the **goal mouth + goalkeeper + net area**, not the
>   structural frame edge-to-edge.
> - The bottom-left/bottom-right of the bbox do NOT align with where the
>   posts actually meet the ground — they're off by ~20–30px.
> - The goalkeeper standing in the goal shifts the bbox boundaries.
> - At an angle (most frames), the goal is a trapezoid but the bbox is a
>   rectangle — corners don't correspond to post positions.
>
> **Verdict:** The goal model is useful as a **side-of-court anchor**
> (left goal visible → we're looking at the left half), but should NOT
> be used for precise homography point correspondences. The court keypoint
> model (Phase 2) is the primary source of accurate pixel↔world pairs.
>
> If you later want goal-post foot-points, train a separate **goal keypoint
> model** (pose model on just the goal, with 4 keypoints: 2 post tops +
> 2 post feet). But for the POC, the court keypoints alone should give
> enough well-distributed points.

---

## Phase 5 — Homography Computation + Pipeline Integration

### Step 10: Combine points and compute homography

For each frame:

```
1. Run court keypoint model → N visible keypoints (each has known court coords)
2. Run goal detection → side-of-court anchor (left/right)
3. Filter keypoints with visibility > threshold (e.g. conf > 0.5)
4. Build pixel→court point pairs from detected keypoints
5. If ≥ 4 non-collinear pairs → cv2.findHomography(RANSAC)
6. Feed into HomographyTracker (already implemented) for temporal smoothing
```

The keypoint model is the **primary source** of correspondences. The goal
detector provides context (which half of the court is visible) but is not
needed for the homography math itself.

### Step 11: Add CLI flags to analyze.py

Like `--ball-model`, add:

```
--goal-model models/handball_goal.pt
--court-kp-model models/handball_court_kp.pt
```

When provided:
- Load goal YOLO model + court keypoint YOLO-pose model alongside person/ball
- Run both per frame
- Court keypoints → direct pixel↔world pairs for homography
- Goal detection → side anchor (optional sanity check)
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

Update `diagnose.py` to test the new goal + court keypoint pipeline:

```bash
uv run python diagnose.py --frame 100 --frame 300 --frame 600
```

Check:
- [ ] Goal detected in all frames where it's visible?
- [ ] Court keypoints land on the correct landmarks?
- [ ] Visibility flags correct (no phantom keypoints off-screen)?
- [ ] ≥ 4 non-collinear keypoints detected per frame?
- [ ] Homography produces plausible court coordinates?
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
| `train-court-keypoints` | Train YOLO11m-pose on court keypoint annotations |
| `validate-court-keypoints` | Run court keypoint validation |
| `predict-court-keypoints` | Visual court keypoint predictions |
| `install-court-keypoints` | Copy best.pt → models/handball_court_kp.pt |
| `analyze-goal` | Full pipeline with goal + court keypoint models |

---

## Annotation Summary

You need **two Roboflow projects** (different task types):

| Project | Type | Class | Annotations | Effort |
|---------|------|-------|-------------|--------|
| `handball-goal` | Object Detection | `goal` (bbox) | 150–200 frames | ~1–2 hours |
| `handball-court-keypoints` | Keypoint Detection | `court` (12 keypoints) | 150–200 frames | ~3–4 hours |

Use the **same extracted frames** for both. Total annotation effort: ~4–6 hours.

> **Note:** Keypoint annotation takes longer per frame than polygon annotation
> because you need to precisely place each landmark. However, the result is
> much more useful — direct pixel↔world correspondences with no post-processing.

---

## What's Already Done (reuse these)

- `HomographyTracker` in `pipeline/lines.py` — EMA + optical flow smoothing ✅
- `render_court()` in `pipeline/court_viz.py` — 2D top-down court drawing ✅
- Ball annotation/training workflow in Makefile — pattern to follow ✅
- Roboflow workspace `hsc-wels` set up ✅
- Goal detection model trained + installed → `models/handball_goal.pt` ✅
- Goal Makefile targets already added ✅

---

## Known Risks

| Risk | Mitigation |
|------|-----------|
| Goal partially off-screen | HomographyTracker persists last good H for up to 150 frames; court keypoints alone give enough points |
| Not enough keypoints visible | Camera always shows ≥ one half of the court → ≥ 5–6 keypoints; temporal smoothing covers brief gaps |
| Keypoint placement imprecise | RANSAC in findHomography tolerates outlier points; only need 4 good ones |
| Midfield views with no goal visible | Court keypoints from both halves visible → plenty of points; goal detection is just a sanity check |
| Keypoints confused by floor ads | Landmarks are at court lines/intersections, not on the floor surface; model learns to distinguish |
| Different halls / floor colours | Keypoint model generalizes to line intersections regardless of floor colour; for new halls, add 20–30 frames |
| Horizontal flip swaps left↔right keypoints | Roboflow handles keypoint index remapping on flip augmentation |
