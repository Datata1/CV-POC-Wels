# POC: RT-DETRv4 Ball Detection

## Ziel
Fine-tuning von RT-DETRv4-L auf das Handball-Ball-Dataset als Alternative zu YOLO11m.
Vergleich der mAP-Werte beider Architekturen.

## Architektur
- **Modell**: RT-DETRv4-L (HGNetv2-B4 Backbone + Transformer Decoder)
- **Teacher**: DINOv3 ViT-B/16 (Knowledge Distillation während Training)
- **Basis**: COCO-pretrained → fine-tuned auf Handball-Ball

## Dataset
| Split | Bilder | Annotationen |
|-------|--------|-------------|
| Train | 760    | 750         |
| Val   | 102    | 97          |
| Test  | 76     | 78          |

- 1 Klasse: `Ball`
- Format: COCO-JSON (konvertiert aus YOLO via `scripts/yolo_to_coco.py`)
- Pfad: `annotation/ball/coco/{train,val,test}.json`

## Training-Parameter
| Parameter | Wert |
|-----------|------|
| Epochs | 100 |
| Batch Size | 2 |
| Input Size | 640×640 |
| Optimizer | AdamW (lr=5e-4, backbone lr=2.5e-5) |
| AMP | Ja |
| GPU | RTX 3060 12GB |
| Geschätzte Dauer | ~4.5h |

## Commands
```bash
# Setup (einmalig)
make setup-rtdetr
make convert-datasets

# Training
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True make train-rtdetr-ball

# Validierung
make validate-rtdetr-ball
```

## Output
- Checkpoints: `local/rtdetrv4/outputs/rtv4_handball_ball/`
- Bestes Modell: `best_stg1.pth`
- Metriken pro Epoch: `log.txt` (JSON mit mAP, Loss)
- TensorBoard: `summary/`

## Fortschritt prüfen
```bash
grep '"epoch"' local/rtdetrv4/outputs/rtv4_handball_ball/log.txt | wc -l
```

## Baseline (YOLO11m)
Vergleichsergebnisse: `runs/detect/handball_ball/results.csv`

## Config
`local/rtdetrv4/configs/handball/rtv4_ball.yml`
