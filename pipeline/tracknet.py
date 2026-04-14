"""TrackNet: small-ball detection via 3-frame heatmap regression.

Architecture mirrors yastrebksv/TrackNet (a PyTorch re-implementation of the
original TrackNet paper, Huang et al. 2019). The network takes three
consecutive RGB frames stacked as a 9-channel tensor and outputs a per-pixel
heatmap whose peak marks the ball position.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn


# TrackNet was trained on 640x360 input (WxH). Keep the exact size so the
# pretrained weights load cleanly.
INPUT_W = 640
INPUT_H = 360


class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(out_c),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BallTrackerNet(nn.Module):
    """TrackNet — VGG-style encoder + bilinear-upsampling decoder.

    Output is a per-pixel distribution over 256 intensity classes (0-255).
    At inference we take argmax along the class dim to get a grayscale heatmap.
    """

    def __init__(self, out_channels: int = 256) -> None:
        super().__init__()
        self.out_channels = out_channels

        self.conv1 = ConvBlock(9, 64)
        self.conv2 = ConvBlock(64, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = ConvBlock(128, 256)
        self.conv6 = ConvBlock(256, 256)
        self.conv7 = ConvBlock(256, 256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv8 = ConvBlock(256, 512)
        self.conv9 = ConvBlock(512, 512)
        self.conv10 = ConvBlock(512, 512)
        self.ups1 = nn.Upsample(scale_factor=2)
        self.conv11 = ConvBlock(512, 256)
        self.conv12 = ConvBlock(256, 256)
        self.conv13 = ConvBlock(256, 256)
        self.ups2 = nn.Upsample(scale_factor=2)
        self.conv14 = ConvBlock(256, 128)
        self.conv15 = ConvBlock(128, 128)
        self.ups3 = nn.Upsample(scale_factor=2)
        self.conv16 = ConvBlock(128, 64)
        self.conv17 = ConvBlock(64, 64)
        self.conv18 = ConvBlock(64, out_channels)
        self.softmax = nn.Softmax(dim=1)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw per-pixel class logits, shape (B, 256, H*W). Used for training."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.ups1(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.ups2(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.ups3(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        return x.reshape(x.shape[0], self.out_channels, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax(self.forward_logits(x))


class TrackNetDetector:
    """Sliding-window wrapper around BallTrackerNet.

    Feed frames one at a time; detector buffers the last 3 and returns a ball
    position (or None) for the current frame. For the first two frames the
    buffer is not yet full, so None is returned.
    """

    def __init__(
        self,
        weights_path: str,
        device: Optional[str] = None,
        heatmap_threshold: int = 128,
        min_blob_area: int = 2,
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.threshold = heatmap_threshold
        self.min_blob_area = min_blob_area

        self.model = BallTrackerNet(out_channels=256).to(self.device)
        state = torch.load(weights_path, map_location=self.device, weights_only=False)
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        self.model.load_state_dict(state)
        self.model.eval()

        self.frame_buffer: deque[np.ndarray] = deque(maxlen=3)
        self._orig_size: Optional[Tuple[int, int]] = None  # (W, H)

    @staticmethod
    def _preprocess_frame(frame_bgr: np.ndarray) -> np.ndarray:
        """Resize to model input and convert BGR→RGB float32 in [0, 1]."""
        resized = cv2.resize(frame_bgr, (INPUT_W, INPUT_H))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return rgb  # (H, W, 3)

    def _heatmap_to_point(
        self, heatmap: np.ndarray
    ) -> Tuple[Optional[Tuple[float, float]], float]:
        """Convert a (H, W) uint8 heatmap to (x, y) in model coords + confidence."""
        _, binary = cv2.threshold(heatmap, self.threshold, 255, cv2.THRESH_BINARY)
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        if num <= 1:
            return None, 0.0
        # Pick largest non-background blob
        areas = stats[1:, cv2.CC_STAT_AREA]
        idx = int(np.argmax(areas)) + 1
        if stats[idx, cv2.CC_STAT_AREA] < self.min_blob_area:
            return None, 0.0
        cx, cy = centroids[idx]
        # Confidence = mean heatmap value inside the blob, normalised
        blob_mask = labels == idx
        conf = float(heatmap[blob_mask].mean()) / 255.0
        return (float(cx), float(cy)), conf

    @torch.no_grad()
    def detect(
        self, frame_bgr: np.ndarray
    ) -> Tuple[Optional[Tuple[float, float]], float, Optional[np.ndarray]]:
        """Add `frame_bgr` to the buffer and run inference if buffer is full.

        Returns `(xy_in_original_pixels, confidence, heatmap_uint8)`.
        xy is None until the sliding window has 3 frames, or if no ball found.
        The heatmap is at model resolution (360x640); None when buffer not full.
        """
        H, W = frame_bgr.shape[:2]
        self._orig_size = (W, H)
        self.frame_buffer.append(self._preprocess_frame(frame_bgr))
        if len(self.frame_buffer) < 3:
            return None, 0.0, None

        # Stack 3 frames along channel dim → (9, H, W)
        stacked = np.concatenate(list(self.frame_buffer), axis=2)  # (H, W, 9)
        tensor = torch.from_numpy(stacked).permute(2, 0, 1).unsqueeze(0).to(self.device)

        out = self.model(tensor)  # (1, 256, H*W)
        pred = out.argmax(dim=1).reshape(INPUT_H, INPUT_W).cpu().numpy().astype(np.uint8)

        pt, conf = self._heatmap_to_point(pred)
        if pt is None:
            return None, 0.0, pred
        # Scale back to original frame coords
        x = pt[0] * (W / INPUT_W)
        y = pt[1] * (H / INPUT_H)
        return (x, y), conf, pred
