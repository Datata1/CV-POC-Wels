# POC: YOLO Goal Detection

## Ziel
Erkennung des Handball-Tors im Videobild. Das Tor dient als Seitenindikator (links/rechts auf dem Feld)
und potenziell als Anker für die Homographie-Berechnung.

## Architektur
- **Modell**: YOLO11m (medium)
- **Ansatz**: Standard Object Detection, 1 Klasse

## Dataset
| Split | Bilder | Annotationen |
|-------|--------|-------------|
| Train | 638    | 717         |
| Val   | 37     | 40          |
| Test  | 18     | 20          |

- 1 Klasse: `goal`
- Quelle: Roboflow `goal-detection-pt6sz` v2
- Pfad: `annotation/goal/`

## Training-Parameter
| Parameter | Wert |
|-----------|------|
| Basismodell | yolo11m.pt |
| Epochs | 100 |
| Batch Size | 16 |
| Input Size | 640×640 |
| GPU | RTX 3060 12GB |
| Dauer | ~28.7 min |

## Ergebnisse
| Metrik | Best | Final (Ep. 100) |
|--------|------|-----------------|
| Precision | — | 0.999 |
| Recall | — | 0.975 |
| **mAP50** | **0.985** | 0.985 |
| **mAP50-95** | **0.955** (Ep. 66) | 0.945 |

### Fazit
- **Quasi perfekte Erkennung.** Tore sind groß und visuell eindeutig.
- Modell ist produktionsreif für die Pipeline.
- Erkenntnis aus Homography-v2: Goal-Bbox ist **nicht** präzise genug für Torpfosten-Extraktion.
  Empfehlung: Tor nur als Seiten-Anker (links/rechts) nutzen, nicht als Homographie-Punkt.

## Commands
```bash
make train-goal

# Oder manuell:
uv run yolo detect train model=yolo11m.pt data=annotation/goal/data.yaml \
  epochs=100 imgsz=640 batch=16 name=handball_goal
```

## Output
- Run: `runs/detect/handball_goal/`
- Bestes Modell: `weights/best.pt` → kopiert nach `models/handball_goal.pt`
