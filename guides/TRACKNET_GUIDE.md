# TrackNet Ball Detection — POC Guide

This guide explains the TrackNet ball-detection pipeline added to the project:
what it is, why it exists, how to run it, and what happens under the hood.

---

## 1. Why TrackNet?

Our existing ball detection uses **YOLO** (fine-tuned on Roboflow labels). YOLO
is a single-frame object detector: it looks at one image at a time and asks
*"where are the objects?"*.

That works poorly for a **small, fast-moving handball** because:

- When the ball moves fast it becomes a **blurred streak**, not a clean disk
- On many frames the ball occupies only ~10–20 pixels and YOLO's anchor grid
  misses it entirely
- The ball gets **occluded** by players and reappears a frame later — YOLO
  treats every frame independently, so it can't use context

**TrackNet** (Huang et al., 2019) tackles exactly this problem. It takes
**three consecutive frames** stacked together and predicts a heatmap of where
the ball is in the *current* frame. Because the model sees motion, it can
essentially **detect the ball by its trajectory** rather than just its
appearance — even when the ball is a blur, partially occluded, or nearly
invisible in a single frame.

This implementation is a PyTorch port based on
<https://github.com/yastrebksv/TrackNet>.

---

## 2. How it works — under the hood

### 2.1 Input: three frames, nine channels

At frame `t`, we feed the network the stack of frames `[t-2, t-1, t]`,
resized to **640 × 360** and concatenated along the channel axis:

```
frame t-2  (3 RGB channels)  ┐
frame t-1  (3 RGB channels)  ├── 9-channel input, 640x360
frame t    (3 RGB channels)  ┘
```

The first two frames of the video therefore produce **no output** — we need
three frames before the sliding window is full.

### 2.2 Architecture: VGG encoder → upsampling decoder

```
Input  640×360×9
   │
   ▼
┌─────────────── Encoder (VGG-like) ───────────────┐
│ Conv(64)  → Conv(64)   → MaxPool ↓2              │
│ Conv(128) → Conv(128)  → MaxPool ↓2              │
│ Conv(256) ×3           → MaxPool ↓2              │
│ Conv(512) ×3                                     │
└──────────────────────────────────────────────────┘
   │    (feature map 80×45×512)
   ▼
┌─────────────── Decoder (bilinear up) ────────────┐
│ Upsample ↑2 → Conv(256) ×3                       │
│ Upsample ↑2 → Conv(128) ×2                       │
│ Upsample ↑2 → Conv(64)  ×2                       │
│ Conv → 256 channels                              │
└──────────────────────────────────────────────────┘
   │
   ▼
Softmax over channel dim → Heatmap 640×360
```

Every `Conv` is `Conv2d(3×3) + ReLU + BatchNorm`. The decoder uses **bilinear
upsampling** (no transposed-convolution artefacts) and restores the output to
the same spatial size as the input.

### 2.3 Output: a 256-class per-pixel classifier

Instead of regressing a single continuous value per pixel, TrackNet treats
**each pixel as a 256-class classification problem** — one class per possible
grayscale intensity (0–255). During training the ground truth is a Gaussian
blob centred on the ball (peak = 255, decaying outward); the loss is per-pixel
categorical cross-entropy.

At inference we just take `argmax` over the 256 channels to recover a
`uint8` heatmap of shape `(360, 640)`.

### 2.4 Heatmap → (x, y)

```
heatmap (uint8)                → threshold at 128
       │
       ▼
binary mask                    → connectedComponentsWithStats
       │
       ▼
largest blob                   → centroid (cx, cy)
       │
       ▼
scale back to original frame   → (x * W/640, y * H/360)
```

Confidence is reported as the **mean heatmap value inside the blob**,
normalised to `[0, 1]`. A confident detection is a bright, sharply localised
blob; a low-confidence one is a diffuse smear.

### 2.5 Trajectory trail (visualisation only)

The detected `(x, y)` of the last N frames is stored in a ring buffer and
drawn as a fading trail on top of the output video. This is purely cosmetic —
the model does not use the trail for prediction.

---

## 3. Getting the weights

The original repo ships pretrained tennis weights via a **Google Drive link
in its README** (there is no GitHub Release / direct URL).

1. Open <https://github.com/yastrebksv/TrackNet>
2. In the *Pretrained model* section, click the `model_best.pt` Google Drive link
3. Save the downloaded file to:

    ```
    models/tracknet.pt
    ```

`make download-tracknet-model` just prints these instructions — Google Drive
blocks automated downloads for files over ~100 MB, so we don't try to script
it. If you prefer the CLI, install `gdown` and pass the file ID from the
share URL:

```bash
uv run gdown <FILE_ID> -O models/tracknet.pt
```

> **Note:** these weights are trained on **tennis** footage. They often work
> out-of-the-box on handball because the task (detect a small ball using
> motion cues) generalises reasonably well, but detection rate will improve
> significantly after fine-tuning on handball data (see §5).

---

## 4. Running it

### Quick start

```bash
# 1. Put a match video in input/
# 2. Put the weights at models/tracknet.pt
make tracknet-detect
```

### Full CLI

```bash
uv run python tracknet_detect.py \
    --input input/game.mp4 \
    --model models/tracknet.pt \
    --output-dir output \
    --threshold 128 \
    --trail-length 20 \
    --chunk-seconds 60 \
    --save-heatmaps        # optional: emit heatmap side-video
    --preview              # optional: live window
    --device cuda          # or cpu; auto-detected if omitted
```

### Outputs

| File | Meaning |
|---|---|
| `output/<name>_tracknet_chunk001_0s-60s.mp4` | Annotated video (ball circle + trail + HUD) |
| `output/<name>_tracknet_chunk001_0s-60s_heatmap.mp4` | *(opt.)* Heatmap side-video, hot colormap |
| `output/<name>_tracknet.jsonl` | Per-frame ball position (JSON-lines) |

Each line in the JSONL:

```json
{"frame_id": 42, "timestamp_s": 1.40, "ball": {"x": 812.4, "y": 334.1, "conf": 0.83}}
{"frame_id": 43, "timestamp_s": 1.43, "ball": null}
```

`ball` is `null` when the heatmap peak was below the threshold — i.e. the
model is not confident enough.

---

## 5. Fine-tuning on handball (future work)

The pretrained tennis weights are a starting point. To get reliable handball
detection you will want to fine-tune:

1. **Label data.** Use the Roboflow ball-labelling workflow we already have
   (`make extract-frames` → label → export YOLO format). Convert the
   centre-of-bbox to a Gaussian heatmap on the fly at training time.
2. **Training loop.** Use the loss and optimiser from yastrebksv/TrackNet
   (Adadelta, per-pixel categorical CE). A few hundred labelled frames are
   usually enough to see big gains because the pretrained backbone already
   knows what "moving round thing" looks like.
3. **Validation metric.** Report *detection rate* (% frames where the
   predicted point lands within `r` pixels of GT, usually `r = 5`).

A training script is **not** part of this POC — the goal here is to measure
how far the out-of-the-box model gets us before investing in labels.

---

## 6. Comparison with our YOLO ball detector

| Aspect | YOLO (fine-tuned) | TrackNet |
|---|---|---|
| Frames considered | 1 | 3 consecutive |
| Input size | 640×640 | 640×360 |
| Output | Bounding boxes | Heatmap → single point |
| Motion-aware | No | **Yes** |
| Handles motion blur | Poorly | **Well** |
| Works when ball is occluded 1 frame | No | **Often yes** (uses neighbour frames) |
| Gives a bbox size | Yes | No (point only) |
| Training data needed | ~500 labelled frames | ~500 labelled frames + heatmap generator |
| Inference speed (RTX 3060) | ~60 fps | ~40 fps (POC estimate) |

You can run **both** in parallel and fuse the results — TrackNet for
hard-to-see frames, YOLO when you need a bbox — but that fusion is out of
scope for this POC.

---

## 7. Files added by this POC

```
pipeline/tracknet.py        # Model definition + TrackNetDetector wrapper
tracknet_detect.py          # Standalone CLI script
guides/TRACKNET_GUIDE.md    # This file
Makefile                    # + tracknet-detect, download-tracknet-model
```

Nothing in the existing `analyze.py` pipeline is modified.
