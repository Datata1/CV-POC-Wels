# POC: YOLO Court Keypoint Detection

## Ziel
Erkennung von 30 Spielfeld-Landmarks als individuelle Objekt-Klassen.
Die erkannten Keypoints liefern Pixel↔Welt-Korrespondenzen für die Homographie-Berechnung.

## Architektur
- **Modell**: YOLO11m (Object Detection, NICHT Pose-Estimation)
- **Ansatz**: Jeder Landmark = eigene Klasse mit kleiner Bounding-Box → Bbox-Zentrum = Pixel-Koordinate
- **30 Klassen**: `4m_L`, `4m_R`, `6m_base_L_B`, `6m_base_L_T`, … `goalpost_R_B`, `goalpost_R_T`

## Dataset
| Split | Bilder | Annotationen |
|-------|--------|-------------|
| Train | 297    | 5235        |
| Val   | 9      | 144         |
| Test  | 4      | 88          |

- Quelle: Roboflow `field-keypoints-uu1fy` v1
- Pfad: `annotation/court/`
- Welt-Koordinaten: `keypoints/court_keypoints.csv` (40m × 20m IHF-Feld)
- **Achtung**: `court_BR` fehlt in den Trainingsdaten (nur 3 von 4 Ecken annotiert)

## Training-Parameter
| Parameter | v1 | v2 |
|-----------|----|----|
| Epochs | 100 (stop @ 62) | 100 (stop @ 48) |
| Batch Size | 16 | 2 |
| Input Size | **640×640** | **1920×1920** |
| GPU | RTX 3060 12GB | RTX 3060 12GB |
| Dauer | ~8 min | ~58 min |

## Ergebnisse

### v1 — 640px
| Metrik | Wert |
|--------|------|
| Precision | 0.397 |
| Recall | 0.345 |
| **mAP50** | **0.320** |
| **mAP50-95** | **0.081** |

### v2 — 1920px
| Metrik | Best | Final (Ep. 48) |
|--------|------|----------------|
| Precision | — | 0.571 |
| Recall | — | 0.494 |
| **mAP50** | **0.583** (Ep. ~34) | 0.500 |
| **mAP50-95** | — | 0.181 |

### Fazit
- 1920px bringt +56% relative Verbesserung bei mAP50 gegenüber 640px.
- mAP50-95 bleibt sehr niedrig → Bbox-Lokalisierung ungenau (kleine Punkte).
- 30-Klassen-Detection auf wenig Daten (297 Train) ist grundsätzlich schwierig.
- **Hauptblocker** für die Homographie-Pipeline: Keypoint-Qualität reicht noch nicht.

## Mögliche Verbesserungen
- Mehr annotierte Bilder (aktuell nur 297 Train)
- `court_BR` annotieren (4. Ecke fehlt)
- Pose-Estimation statt Detection (Regression statt Bbox-Zentrum)
- Höhere Auflösung (z.B. 2560px) oder Tile-basierter Ansatz

## Commands
```bash
# v2 (1920px, empfohlen)
make train-court-kp

# Oder manuell:
uv run yolo detect train model=yolo11m.pt data=annotation/court/data.yaml \
  epochs=200 imgsz=1920 batch=2 name=handball_court_kp2
```

## Output
- v1: `runs/detect/handball_court_kp/`
- v2: `runs/detect/handball_court_kp2/`
- Bestes Modell: `weights/best.pt` → kopiert nach `models/handball_court_kp.pt`
