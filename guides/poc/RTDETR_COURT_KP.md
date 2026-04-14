# POC: RT-DETRv4 Court Keypoint Detection

## Ziel
Fine-tuning von RT-DETRv4-L auf 30 Court-Landmark-Klassen für die automatische Homographie-Berechnung.
Vergleich mit YOLO11m Court-KP-Modell.

## Architektur
- **Modell**: RT-DETRv4-L (HGNetv2-B4 + Transformer Decoder)
- **Teacher**: DINOv3 ViT-B/16 (Knowledge Distillation)
- **Ansatz**: Jeder Landmark = eigene Objektklasse, BBox-Zentren → Pixel-Koordinaten → Homographie via RANSAC

## Dataset
| Split | Bilder | Annotationen |
|-------|--------|-------------|
| Train | 297    | 5235        |
| Val   | 9      | 144         |
| Test  | 4      | 88          |

- 30 Klassen: `4m_L`, `4m_R`, `6m_base_*`, `6m_vertex_*`, `7m_*`, `9m_*`, `center_*`, `court_*`, `goalpost_*`
- Format: COCO-JSON (konvertiert via `scripts/yolo_to_coco.py`)
- Mapping: `keypoints/court_keypoints.csv` (Klasse → Meter-Koordinaten auf dem 40×20m Feld)

## Training-Parameter
| Parameter | Wert |
|-----------|------|
| Epochs | 200 |
| Batch Size | 4 |
| Input Size | 640×640 |
| Optimizer | AdamW (lr=5e-4, backbone lr=2.5e-5) |
| AMP | Ja |
| GPU | RTX 3060 12GB |
| Geschätzte Dauer | ~9–10h |

## Risiko
Court-Keypoints sind sehr kleine BBoxen (Punktlandmarks). RT-DETR ist für natürliche Objekte optimiert — könnte schlechter als YOLO abschneiden.

## Commands
```bash
# Training
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True make train-rtdetr-court-kp

# Validierung
make validate-rtdetr-court-kp
```

## Output
- Checkpoints: `local/rtdetrv4/outputs/rtv4_handball_court_kp/`
- Bestes Modell: `best_stg1.pth`
- Metriken: `log.txt`

## Baseline (YOLO11m)
Vergleichsergebnisse: `runs/detect/handball_court_kp/results.csv`

## Config
`local/rtdetrv4/configs/handball/rtv4_court_kp.yml`
