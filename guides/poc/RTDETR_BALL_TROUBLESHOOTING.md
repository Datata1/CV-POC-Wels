# Troubleshooting: RT-DETRv4 Ball Detection

## Problem
Inference produced bounding boxes in **wrong locations** — random points on the field instead of the actual ball.
Training metrics (mAP=0.495, mAP50=0.771) looked healthy, but visual results on video were garbage.

## Root Cause
**`orig_target_sizes` dimension ordering was swapped in the inference script.**

RT-DETRv4's `PostProcessor` expects `orig_target_sizes` as `[width, height]`, not `[height, width]`.

```python
# WRONG — causes x/y coordinate swapping
orig_size = torch.tensor([[orig_h, orig_w]])

# CORRECT — matches RT-DETRv4 dataloader convention
orig_size = torch.tensor([[orig_w, orig_h]])
```

### Why it looked like "random points"
The PostProcessor multiplies normalized box coordinates (0–1) by `orig_target_sizes`:
- With `[1080, 1920]` (wrong): x-coords scaled by 1080, y-coords by 1920 → boxes shifted and stretched
- With `[1920, 1080]` (correct): coordinates map to actual pixel positions

### Why training metrics were fine
The training/validation pipeline uses its own `DataLoader` which sets `orig_size` correctly from `torchvision.transforms.v2`. The bug was **only in the custom inference script** (`scripts/rtdetr_predict.py`).

## Verification
```python
# Ground truth ball position: x=770, y=605

# WRONG [h,w]: top detection at [432.7, 1075.6] — bottom edge of frame
# RIGHT [w,h]: top detection at [769.2, 605.0] — matches ground truth perfectly
```

## Fix Applied
**File**: `scripts/rtdetr_predict.py`, line in `preprocess()`:
```python
orig_size = torch.tensor([[orig_w, orig_h]])  # RT-DETRv4 expects [width, height]
```

## Key Takeaway
When writing custom inference for RT-DETRv4 (or any DETR variant):
1. Check the `orig_size` convention by inspecting the **val_dataloader** target dict
2. RT-DETRv4 uses `[width, height]` — opposite of PyTorch's common `[height, width]` (e.g. in `torch.Size`)
3. If detections are spatially "off" but metrics look good → suspect coordinate ordering mismatch

## Validation Results (confirmed correct)
| Metric | Value |
|--------|-------|
| mAP@50:95 | 0.495 |
| mAP@50 | 0.771 |
| mAP@75 | 0.567 |
| AR@50 | 0.918 |

## Files
- Inference script: `scripts/rtdetr_predict.py`
- Checkpoint: `local/rtdetrv4/outputs/rtv4_handball_ball/best_stg1.pth`
- Preview video: `output/rtdetr_ball_preview.mp4`

---

## Next Steps: Improving Stability & Precision

### Current Weaknesses
- mAP50-95 = 0.495 → box localization is imprecise (good at finding the ball, bad at tight boxes)
- Ball is ~1% of image area (~18×18px in 1920×1080) → tiny target problem
- 760 training images → limited data diversity
- Mosaic augmentation shrinks images to 320px → ball becomes ~3px, essentially invisible during many training steps
- Stagnated at epoch 16/100 → model converged too early

### Action 1: Disable/Fix Mosaic for Small Objects (HIGH IMPACT)
The default config applies `Mosaic(output_size=320)` which composites 4 images into 320×320. For a ~18px ball, this shrinks it to ~3px — below detectable. Either:

**Option A**: Disable Mosaic entirely in ball config:
```yaml
# In rtv4_ball.yml — override augmentation pipeline
train_dataloader:
  dataset:
    transforms:
      ops:
        - {type: RandomPhotometricDistort, p: 0.5}
        - {type: RandomZoomOut, fill: 0}
        - {type: RandomIoUCrop, p: 0.8}
        - {type: SanitizeBoundingBoxes, min_size: 1}
        - {type: RandomHorizontalFlip}
        - {type: Resize, size: [640, 640]}
        - {type: SanitizeBoundingBoxes, min_size: 1}
        - {type: ConvertPILImage, dtype: 'float32', scale: True}
        - {type: ConvertBoxes, fmt: 'cxcywh', normalize: True}
      policy:
        name: stop_epoch
        epoch: [90]
        ops: ['RandomPhotometricDistort', 'RandomZoomOut', 'RandomIoUCrop']
  collate_fn:
    type: BatchImageCollateFunction
    base_size: 640
    mixup_prob: 0.0
```

**Option B**: Increase Mosaic output_size to 640 (keeps augmentation but preserves ball size).

### Action 2: Train at Higher Resolution (HIGH IMPACT)
The ball at 640px is ~6px after resize. At 1280px it doubles to ~12px. Court-KP saw +56% mAP going 640→1920.

```yaml
# In rtv4_ball.yml
train_dataloader:
  dataset:
    transforms:
      ops:
        - {type: Resize, size: [1280, 1280]}
        # ...
eval_spatial_size: [1280, 1280]
```

**Trade-off**: batch_size must drop to 1 at 1280px on RTX 3060 12GB. Consider gradient accumulation or mixed precision (`use_amp: True`).

### Action 3: Enable AMP (QUICK WIN)
Currently `use_amp: False` in ball config. Enabling it halves VRAM for activations, allowing larger batch or resolution.

```yaml
use_amp: True
```

### Action 4: Tune Learning Rate Schedule (MEDIUM IMPACT)
Model peaked at epoch 16 and stagnated → LR may be too high after warmup or flat phase too long.

```yaml
# Shorter flat phase, earlier cosine decay
flat_epoch: 30       # was 50
no_aug_epoch: 15     # was 10 — stop augmentation earlier to refine
epoches: 150         # train longer with lower LR tail
warmup_iter: 1000    # was 2000 — dataset is small, don't need long warmup
```

### Action 5: Add More Training Data (MEDIUM IMPACT)
760 images is limited. Options:
1. Extract more frames from matches (different camera angles, lighting)
2. Use Roboflow to add more augmented exports
3. Generate synthetic hard examples (ball near similar-colored objects)

### Action 6: Test Set Evaluation (ESSENTIAL)
Currently only validating on val split. Run on test split for unbiased metrics:
```bash
# Create a test config or modify val_dataloader temporarily
# Point ann_file to annotation/ball/coco/test.json
```

### Action 7: Video-Level Post-Processing (STABILITY)
For video inference, add temporal smoothing to reduce flickering:
1. **Confidence smoothing**: Exponential moving average of scores across frames
2. **Tracking**: Use simple IoU-based tracker (e.g. SORT) to maintain ball identity
3. **Interpolation**: Fill gaps when ball is occluded for 1–2 frames

### Action 8: Evaluate Without Distillation (DIAGNOSTIC)
DINOv3 distillation adds complexity. Try a run without teacher model to check if distillation helps or hurts on this small dataset:
```yaml
# Comment out teacher_model section in rtv4_ball.yml
RTv4Criterion:
  weight_dict:
    loss_distill: 0.0  # disable distillation loss
```

### Recommended Priority Order
| # | Action | Expected Impact | Effort |
|---|--------|----------------|--------|
| 1 | Disable Mosaic | +++ | 5 min (config change) |
| 2 | Enable AMP | ++ | 1 min |
| 3 | Increase resolution to 1280 | +++ | Config + retrain |
| 4 | Tune LR schedule | ++ | Config + retrain |
| 5 | Test without distillation | diagnostic | Config + retrain |
| 6 | More training data | ++ | Hours (annotation) |
| 7 | Video post-processing | +++ (stability) | ~1h coding |
| 8 | Test set evaluation | essential | 5 min |

### Suggested v2 Training Run
Combine actions 1–4 into a single retrain:
```bash
# Create rtv4_ball_v2.yml with:
# - No Mosaic, no Mixup
# - AMP enabled
# - 1280px input (if VRAM allows) or 640px + larger batch
# - flat_epoch=30, epoches=150
# Then:
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True make train-rtdetr-ball-v2
```
