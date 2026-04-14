# POC: Court Homography

## Ziel
Abbildung von Pixel-Koordinaten (Kamerabild) auf reale Spielfeld-Koordinaten (40m × 20m)
mittels Homographie-Transformation. Ermöglicht 2D-Taktikansicht von oben.

## Strategien (evaluiert)

| Strategie | Ansatz | Bewertung |
|-----------|--------|-----------|
| A | Klassische Linienerkennung (Hough Lines) | ❌ Unzuverlässig in Mehrzweckhallen |
| B | ML Keypoint Detection (SoccerNet-Style) | ✅ SOTA, gewählter Ansatz |
| C | Tor als Anker + Geometrie | ⚠️ Nur 2 Punkte, nicht ausreichend |
| D | Kombination C+A | ❌ Verworfen zugunsten von B |

**Gewählter Ansatz**: Strategie B — ML-basierte Keypoint Detection (30 Landmarks) + RANSAC-Homographie.

## Pipeline-Architektur
```
Videobild
  │
  ├─ YOLO Court-KP Modell → 30 Keypoints (Pixel-Koord.)
  ├─ YOLO Goal Modell → Tor-Bbox (Seiten-Anker: links/rechts)
  │
  ▼
court_keypoints.csv → 30 Welt-Koordinaten (Meter)
  │
  ▼
RANSAC Homographie (cv2.findHomography)
  │
  ▼
Pixel → Welt-Transformation (3×3 Matrix)
  │
  ▼
2D Court View (Taktikansicht)
```

## Keypoints-Referenz
- 30 Klassen definiert in `keypoints/court_keypoints.csv`
- Koordinatensystem: Ursprung = Ecke unten-links, X = Längsachse (0–40m), Y = Querachse (0–20m)
- Enthält: 4 Ecken (3 annotiert), Torfosten, 6m/9m-Kreise, 7m/4m-Marken, Mittellinie
- **Fehlend**: `court_BR` (40, 20) — nicht im Trainingsdatensatz

## POC-Tool
`poc_homography.py` — Interaktives Werkzeug für manuelle Keypoint-Platzierung:
1. Bild laden aus `input/poc-homography/`
2. GUI: Keypoints manuell auf Spielfeld-Landmarks setzen (Seitenpanel mit Felddiagramm)
3. YOLO-Detection (Personen + Ball) auf dem Frame
4. Homographie aus platzierten Keypoints berechnen
5. 6 Ausgabebilder: raw, keypoints, detections, court, warp, combined

**Output**: `output/poc-homography/<frame_name>/`

## Automatische Pipeline
`calibrate.py` — Manuell 4 Ecken klicken (nur bei statischer Kamera mit sichtbaren 4 Ecken).
Pipeline-Module: `pipeline/court.py`, `pipeline/court_viz.py`

## Aktueller Status

| Komponente | Status | Qualität |
|------------|--------|----------|
| Goal Detection | ✅ Fertig | 0.985 mAP50 — produktionsreif |
| Court KP Detection | ⚠️ Trainiert | 0.50 mAP50 @ 1920px — noch unzureichend |
| Homographie-Berechnung | ✅ Implementiert | Funktioniert mit manuellen Keypoints |
| Automatische Pipeline | 🔄 Blockiert | Wartet auf bessere KP-Qualität |

## Blocker & nächste Schritte
1. **Court-KP Qualität verbessern** — Hauptblocker
   - Mehr Trainingsdaten (aktuell nur 297 Bilder)
   - `court_BR` annotieren
   - Alternativen testen: RT-DETRv4, höhere Auflösung, Pose-Estimation
2. **Robustheit gegen fehlende Punkte** — nicht alle 30 Landmarks in jedem Frame sichtbar
3. **Dynamische Kamera** — aktuelle Calibration geht nur mit statischer Kamera

## Referenz-Dokumente
- Ausführlicher Guide: `guides/COURT_HOMOGRAPHY_GUIDE.md`
- v2-Plan: `guides/HOMOGRAPHY_V2_PLAN.md`
