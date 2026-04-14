#!/usr/bin/env python3
"""Convert YOLO-format annotations to COCO-JSON for RT-DETRv4 training.

Handles both:
- YOLO detection (class_id cx cy w h)
- YOLO segmentation/polygon (class_id x1 y1 x2 y2 ...) → converts to bounding box

Usage:
    python scripts/yolo_to_coco.py --data annotation/ball/data.yaml --output annotation/ball/coco
    python scripts/yolo_to_coco.py --data annotation/court/data.yaml --output annotation/court/coco
    python scripts/yolo_to_coco.py --data annotation/goal/data.yaml --output annotation/goal/coco
"""

import argparse
import json
from pathlib import Path

import yaml
from PIL import Image


def parse_yolo_label(label_path: Path, img_w: int, img_h: int) -> list[dict]:
    """Parse a YOLO label file, returning list of annotations with absolute coords.

    Detects format automatically:
    - 5 tokens → detection: class_id cx cy w h (normalized)
    - >5 tokens → polygon: class_id x1 y1 x2 y2 ... → derive bbox
    """
    annotations = []
    text = label_path.read_text().strip()
    if not text:
        return annotations

    for line in text.splitlines():
        tokens = line.strip().split()
        if len(tokens) < 5:
            continue

        class_id = int(tokens[0])
        coords = [float(t) for t in tokens[1:]]

        if len(coords) == 4:
            # Standard detection: cx cy w h (normalized)
            cx, cy, w, h = coords
            x = (cx - w / 2) * img_w
            y = (cy - h / 2) * img_h
            bw = w * img_w
            bh = h * img_h
        else:
            # Polygon: x1 y1 x2 y2 ... (normalized) → derive bounding box
            xs = [coords[i] * img_w for i in range(0, len(coords), 2)]
            ys = [coords[i] * img_h for i in range(1, len(coords), 2)]
            x = min(xs)
            y = min(ys)
            bw = max(xs) - x
            bh = max(ys) - y

        if bw > 0 and bh > 0:
            annotations.append({
                "category_id": class_id,
                "bbox": [round(x, 2), round(y, 2), round(bw, 2), round(bh, 2)],
                "area": round(bw * bh, 2),
            })

    return annotations


def convert_split(
    images_dir: Path,
    labels_dir: Path,
    categories: list[dict],
    split_name: str,
) -> dict:
    """Convert one split (train/val/test) to COCO format."""
    coco = {
        "images": [],
        "annotations": [],
        "categories": categories,
    }

    ann_id = 1
    image_files = sorted(
        f for f in images_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    for img_id, img_path in enumerate(image_files, start=1):
        img = Image.open(img_path)
        w, h = img.size

        coco["images"].append({
            "id": img_id,
            "file_name": img_path.name,
            "width": w,
            "height": h,
        })

        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        for ann in parse_yolo_label(label_path, w, h):
            ann["id"] = ann_id
            ann["image_id"] = img_id
            ann["iscrowd"] = 0
            coco["annotations"].append(ann)
            ann_id += 1

    print(f"  {split_name}: {len(coco['images'])} images, {len(coco['annotations'])} annotations")
    return coco


def main():
    parser = argparse.ArgumentParser(description="Convert YOLO dataset to COCO-JSON")
    parser.add_argument("--data", required=True, help="Path to YOLO data.yaml")
    parser.add_argument("--output", required=True, help="Output directory for COCO JSONs")
    args = parser.parse_args()

    data_yaml = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)

    nc = cfg["nc"]
    names = cfg["names"]
    categories = [{"id": i, "name": name} for i, name in enumerate(names)]

    base_dir = data_yaml.parent

    # Map split names to their directory structures
    splits = {
        "train": ("train/images", "train/labels"),
        "val": ("valid/images", "valid/labels"),
        "test": ("test/images", "test/labels"),
    }

    print(f"Converting {data_yaml} ({nc} classes: {names})")

    for split_name, (img_rel, lbl_rel) in splits.items():
        images_dir = base_dir / img_rel
        labels_dir = base_dir / lbl_rel

        if not images_dir.exists():
            print(f"  {split_name}: skipped (no {images_dir})")
            continue

        coco = convert_split(images_dir, labels_dir, categories, split_name)

        out_path = output_dir / f"{split_name}.json"
        with open(out_path, "w") as f:
            json.dump(coco, f)
        print(f"  → {out_path}")


if __name__ == "__main__":
    main()
