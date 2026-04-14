# POC: YOLO Ball Detection

## Ziel
Erkennung des Handballs im Videobild. Zwei Varianten getestet: YOLO11n (nano) und YOLO11m (medium).

## Architektur
- **Modell v1**: YOLO11n (nano, ~2.6M params)
- **Modell v2**: YOLO11m (medium, ~20M params)
- **Ansatz**: Standard Object Detection, 1 Klasse

## Dataset
| Split | Bilder |
|-------|--------|
| Train | 760    |
| Val   | 102    |
| Test  | 76     |

- 1 Klasse: `Ball`
- Format: YOLO (txt-Labels mit Polygonen)
- Quelle: Roboflow `ball-detection-pq2gf` v3
- Pfad: `annotation/ball/`

## Training-Parameter
| Parameter | v1 (nano) | v2 (medium) |
|-----------|-----------|-------------|
| Basismodell | yolo11n.pt | yolo11m.pt |
| Epochs | 100 | 100 |
| Batch Size | 16 | 16 |
| Input Size | 640×640 | 640×640 |
| GPU | RTX 3060 12GB | RTX 3060 12GB |
| Dauer | ~8.5 min | ~32 min |

## Ergebnisse

### v1 — YOLO11n
| Metrik | Best | Final (Ep. 100) |
|--------|------|-----------------|
| Precision | — | 0.780 |
| Recall | — | 0.526 |
| **mAP50** | **0.691** (Ep. 74) | 0.631 |
| **mAP50-95** | — | 0.334 |

### v2 — YOLO11m
| Metrik | Best | Final (Ep. 100) |
|--------|------|-----------------|
| Precision | — | 0.878 |
| Recall | — | 0.485 |
| **mAP50** | **0.699** (Ep. 36) | 0.638 |
| **mAP50-95** | — | 0.357 |

### Fazit
- Medium-Modell bringt minimal bessere mAP50 (+0.008) bei 4× längerer Trainingszeit.
- Ball ist klein und schnell → mAP50-95 bleibt unter 0.36.
- `val/cls_loss` zeigte Inf/NaN in v2 (Epochs 2–4), stabilisierte sich danach.

## Commands
```bash
# Training (Makefile-Defaults)
make train-ball

# Oder manuell:
uv run yolo detect train model=yolo11m.pt data=annotation/ball/data.yaml epochs=100 imgsz=640 batch=16
```

## Output
- v1: `runs/detect/handball_ball/`
- v2: `runs/detect/handball_ball2/`
- Bestes Modell: `weights/best.pt` → kopiert nach `models/handball_ball.pt`
