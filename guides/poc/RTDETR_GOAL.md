# POC: RT-DETRv4 Goal Detection

## Ziel
Fine-tuning von RT-DETRv4-L auf das Handball-Goal-Dataset als Alternative zu YOLO11m.
Vergleich der mAP-Werte beider Architekturen.

## Architektur
- **Modell**: RT-DETRv4-L (HGNetv2-B4 Backbone + Transformer Decoder)
- **Teacher**: DINOv3 ViT-B/16 (Knowledge Distillation)
- **Basis**: COCO-pretrained → fine-tuned auf Goal-Erkennung

## Dataset
| Split | Bilder | Annotationen |
|-------|--------|-------------|
| Train | 638    | 717         |
| Val   | 37     | 40          |
| Test  | 18     | 20          |

- 1 Klasse: `goal`
- Format: COCO-JSON (konvertiert via `scripts/yolo_to_coco.py`)
- Pfad: `annotation/goal/coco/{train,val,test}.json`

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
# Training
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True make train-rtdetr-goal

# Validierung
make validate-rtdetr-goal
```

## Output
- Checkpoints: `local/rtdetrv4/outputs/rtv4_handball_goal/`
- Bestes Modell: `best_stg1.pth`
- Metriken: `log.txt`

## Baseline (YOLO11m)
Vergleichsergebnisse: `runs/detect/handball_goal/results.csv`

## Config
`local/rtdetrv4/configs/handball/rtv4_goal.yml`
