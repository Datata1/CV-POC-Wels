"""Run RT-DETRv4 inference on a video and draw detections."""
import argparse
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_script_dir)
sys.path.insert(0, os.path.join(_project_dir, "local", "rtdetrv4"))

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from engine.core import YAMLConfig


def load_model(config_path, checkpoint_path, device="cuda"):
    cfg = YAMLConfig(config_path, resume=checkpoint_path)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Build model
    model = cfg.model
    if "ema" in checkpoint and checkpoint["ema"] is not None:
        ema_state = checkpoint["ema"]["module"]
        model.load_state_dict(ema_state)
    else:
        model.load_state_dict(checkpoint["model"])

    # Build postprocessor
    postprocessor = cfg.postprocessor

    model.to(device).eval()
    postprocessor.to(device).eval()

    return model, postprocessor


def preprocess(frame_bgr, size=(640, 640)):
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    orig_w, orig_h = img.size

    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
    ])
    tensor = transform(img).unsqueeze(0)
    orig_size = torch.tensor([[orig_w, orig_h]])  # RT-DETRv4 expects [width, height]
    return tensor, orig_size


def draw_detections(frame, labels, boxes, scores, class_names, threshold=0.5):
    for label, box, score in zip(labels, boxes, scores):
        if score < threshold:
            continue
        x1, y1, x2, y2 = box.int().tolist()
        name = class_names[label] if label < len(class_names) else str(label)
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{name} {score:.2f}"
        cv2.putText(frame, text, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def main():
    parser = argparse.ArgumentParser(description="RT-DETRv4 video inference")
    parser.add_argument("-c", "--config", required=True, help="YAML config path")
    parser.add_argument("-r", "--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("-v", "--video", required=True, help="Input video path")
    parser.add_argument("-o", "--output", default="output/rtdetr_inference.mp4", help="Output video")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames (0=all)")
    parser.add_argument("--classes", nargs="+", default=["Ball"], help="Class names")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    model, postprocessor = load_model(args.config, args.checkpoint, device)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    max_frames = args.max_frames if args.max_frames > 0 else total
    frame_idx = 0

    print(f"Processing {min(max_frames, total)} frames from {args.video}...")
    with torch.no_grad():
        while cap.isOpened() and frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            tensor, orig_size = preprocess(frame)
            tensor = tensor.to(device)
            orig_size = orig_size.to(device)

            outputs = model(tensor)
            results = postprocessor(outputs, orig_size)

            labels = results[0]["labels"].cpu()
            boxes = results[0]["boxes"].cpu()
            scores = results[0]["scores"].cpu()

            frame = draw_detections(frame, labels, boxes, scores, args.classes, args.threshold)

            writer.write(frame)
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  {frame_idx}/{min(max_frames, total)}")

    cap.release()
    writer.release()
    print(f"Done. Output: {args.output} ({frame_idx} frames)")


if __name__ == "__main__":
    main()
