# Training a Custom TrackNet on Handball Footage

The shipped weights (`models/tracknet.pt`) come from yastrebksv/TrackNet and were
trained on **tennis**. They transfer to handball only partially — the ball is
bigger, the background is busier, and the camera motion is different. Fine-
tuning on your own footage is how you fix the low detection rate you're seeing.

This guide walks through the full pipeline: sample frames → label → build a
heatmap dataset → fine-tune `BallTrackerNet` → drop the new `.pt` into the
existing inference script.

---

## 1. What you're training

TrackNet is a **per-pixel 256-way classifier**: for each pixel it predicts an
intensity 0–255. Ground truth is a Gaussian blob centred on the ball (peak 255,
σ ≈ 5 px). Loss is categorical cross-entropy over the 256 classes.

Concretely:

- **Input:** 3 consecutive RGB frames, resized to 640×360, concatenated along
  the channel axis → tensor of shape `(9, 360, 640)`.
- **Target:** one 360×640 heatmap (`uint8`) with a Gaussian blob at the ball
  in the **third** (current) frame.
- **Label unit:** a single labelled frame = one `(x, y)` ball centre. You also
  need the two frames immediately before it (unlabelled — they're just input).

So you don't label triplets; you label single frames, but you must keep the
video context so the dataloader can grab `t-2, t-1, t`.

---

## 2. How much data do you need?

Rough numbers from the TrackNet paper and community reports:

| Dataset size | What to expect |
|---|---|
| 200–500 frames | Noticeable improvement over tennis weights on your own match |
| 1 000–2 000 | Solid, generalises across camera angles of the same venue |
| 5 000+ | Production-quality, robust across venues |

Start with **~500 labelled frames from 2–3 different matches** (or different
halves). Label every ~15–30th frame so you get variety without redundancy.

---

## 3. Sample frames

You already have `make extract-frames` — reuse it. It grabs every 10th frame
from `input/`. For TrackNet you want variety, so either:

```bash
make extract-frames
```

…or pull from multiple videos by dropping them in `input/` one at a time and
rerunning. Output lands in `annotation/ball/images/` (same folder YOLO uses).

> **Important:** TrackNet needs the two preceding frames at inference time,
> but at training time you can extract them lazily from the source video.
> Keep a mapping `labelled_frame → (source_video, frame_index)` so the loader
> can fetch `t-2` and `t-1`. The easiest way: name extracted frames
> `<video_stem>_<frame_index>.jpg` (which `make extract-frames` already does).

---

## 4. Label the ball

Two reasonable options:

**Option A — reuse the Roboflow bbox workflow you already use for YOLO.**
Label a tight bbox around the ball; at training time take the bbox centre as
the `(x, y)`. This is the path of least resistance since the tooling exists.

**Option B — point labels in CVAT / Label Studio.** Slightly more accurate
(bbox-centre drifts if the ball is a blurred streak), but more setup.

Go with **A** unless you find bbox-centre drift is hurting accuracy.

Export as **YOLO format** into `annotation/ball/` exactly as you do today.
You'll get `labels/<name>.txt` files with lines like:

```
0 0.412 0.588 0.012 0.021   # class cx cy w h  (normalised)
```

Only `cx`, `cy` matter — `w`, `h` are discarded.

---

## 5. Build the training dataset

Create `training/tracknet_dataset.py`. Pseudocode for the loader:

```python
class HandballTrackNetDataset(Dataset):
    def __init__(self, labels_dir, videos_dir, sigma=5):
        self.samples = []  # list of (video_path, frame_idx, x_norm, y_norm)
        for label_file in Path(labels_dir).glob("*.txt"):
            stem = label_file.stem                 # "2025_11_16_hard_wels_000450"
            video_stem, _, fidx = stem.rpartition("_")
            video = Path(videos_dir) / f"{video_stem}.mp4"
            x, y = read_first_bbox_center(label_file)
            self.samples.append((video, int(fidx), x, y))
        self.sigma = sigma

    def __getitem__(self, i):
        video, fidx, xn, yn = self.samples[i]
        frames = read_frames(video, [fidx - 2, fidx - 1, fidx])   # BGR → RGB
        frames = [cv2.resize(f, (640, 360)) for f in frames]
        inp = np.concatenate(frames, axis=2).astype(np.float32) / 255.0
        inp = torch.from_numpy(inp).permute(2, 0, 1)              # (9, 360, 640)

        x_px, y_px = xn * 640, yn * 360
        heatmap = gaussian_heatmap(640, 360, x_px, y_px, self.sigma)  # uint8 (360,640)
        target = torch.from_numpy(heatmap).long().flatten()            # (360*640,)
        return inp, target
```

Key helpers:

```python
def gaussian_heatmap(W, H, cx, cy, sigma):
    xs = np.arange(W); ys = np.arange(H)
    xx, yy = np.meshgrid(xs, ys)
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return (g * 255).astype(np.uint8)

def read_frames(video, indices):
    cap = cv2.VideoCapture(str(video))
    out = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, i))
        ok, f = cap.read()
        out.append(f if ok else np.zeros((1080, 1920, 3), np.uint8))
    cap.release()
    return out
```

Caching tip: `cap.set(POS_FRAMES)` is slow. Pre-extract the needed triplets to
PNG once (`annotation/ball/triplets/<stem>_{t-2,t-1,t}.jpg`) before training.

Split 80/10/10 train/val/test by **match**, not by frame — otherwise nearby
frames leak across splits and your val score will look better than reality.

---

## 6. Training loop

Create `training/train_tracknet.py`:

```python
model = BallTrackerNet(out_channels=256).to(device)
# Warm-start from tennis weights:
state = torch.load("models/tracknet.pt", map_location=device, weights_only=False)
model.load_state_dict(state)

opt = torch.optim.Adadelta(model.parameters(), lr=1.0)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    for inp, target in train_loader:
        inp, target = inp.to(device), target.to(device)
        logits = model_forward_logits(model, inp)   # (B, 256, H*W) before softmax
        loss = loss_fn(logits, target)
        opt.zero_grad(); loss.backward(); opt.step()
```

Two important notes on the existing model:

1. `BallTrackerNet.forward` returns **softmax probabilities**, not logits.
   For CE loss you want raw logits. Either add a `return_logits=True` kwarg
   to the model, or duplicate the forward and skip the final `self.softmax`.
2. `target` must be `long` with values in `[0, 255]` — it's the quantised
   Gaussian heatmap flattened.

Training hyperparameters that work well on the original repo:

| Hparam | Value |
|---|---|
| Optimiser | Adadelta, lr=1.0 |
| Batch size | 2–8 (GPU-dependent; input is 9×360×640 so memory-heavy) |
| Epochs | 20–50 (fine-tune); 100+ from scratch |
| σ (Gaussian) | 5 px at 640×360 |
| Augmentation | Horizontal flip only (careful: flip all 3 frames + flip x) |

Save the best checkpoint by **validation detection rate** (see §7), not by
loss — loss is dominated by the zero-background pixels.

---

## 7. Validation metric

Detection rate @ radius r = the % of val frames where the predicted
`(x, y)` is within `r` pixels of GT (at model resolution). Standard choice:
`r = 5`.

```python
def detection_rate(model, loader, device, r=5):
    hits = total = 0
    for inp, target in loader:
        pred = model(inp.to(device))              # use softmax output
        heat = pred.argmax(dim=1).reshape(-1, 360, 640).cpu().numpy().astype(np.uint8)
        for h, t in zip(heat, target):
            gt = decode_center(t.view(360, 640).numpy())
            pt = heatmap_to_point(h)              # reuse TrackNetDetector logic
            if pt is not None and np.hypot(pt[0] - gt[0], pt[1] - gt[1]) <= r:
                hits += 1
            total += 1
    return hits / total
```

---

## 8. Drop the new weights in

Once training finishes, save:

```python
torch.save(model.state_dict(), "models/tracknet_handball.pt")
```

…then just run the existing inference:

```bash
uv run python tracknet_detect.py --model models/tracknet_handball.pt
```

No code changes needed — `TrackNetDetector` already accepts any weights file
that matches the architecture.

---

## 9. Minimal file layout this adds

```
annotation/ball/                        # existing — Roboflow export
  images/
  labels/
training/
  tracknet_dataset.py                   # new: Dataset + helpers
  train_tracknet.py                     # new: training loop
models/
  tracknet.pt                           # pretrained (tennis) — unchanged
  tracknet_handball.pt                  # new: your fine-tuned weights
```

Add a Makefile target when you're ready:

```makefile
train-tracknet:
	uv run python training/train_tracknet.py \
	    --labels annotation/ball/labels \
	    --videos input \
	    --pretrained models/tracknet.pt \
	    --output models/tracknet_handball.pt
```

---

## 10. Debugging checklist

If detection rate is still bad after training:

- **Visualise a batch.** Dump input triplets + target heatmap as images. The
  blob must be on the ball in the **third** frame, not the first.
- **Check frame indexing.** Off-by-one between `t-2, t-1, t` and the labelled
  index is the single most common bug. Your label is for frame `t`.
- **Class balance.** 99.99% of target pixels are class 0. That's fine —
  don't try to "fix" it with class weights; the Gaussian already does enough.
- **Overfit on 10 frames first.** If the model can't memorise 10 samples to
  near-zero loss, something is wrong in the loader, not the training.
- **Learning rate.** If loss plateaus early, try `Adam(lr=1e-3)` instead of
  Adadelta.
- **σ too small / large.** Too small → gradient is too sparse; too large →
  localisation is fuzzy. 5 px at 640×360 is the safe default.

---

## 11. References

- Huang et al., *TrackNet: A Deep Learning Network for Tracking High-speed
  and Tiny Objects*, 2019.
- yastrebksv/TrackNet — PyTorch port this project is based on.
- See `guides/TRACKNET_GUIDE.md` for the inference-side architecture details.
