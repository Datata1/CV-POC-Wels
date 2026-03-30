# Court Homography Guide — Felderkennung & Koordinaten-Mapping

Ziel: Spielerpositionen aus Pixel-Koordinaten des Videos in reale
Handball-Feldkoordinaten (40m × 20m) umrechnen — auch bei schwenkender
Kamera und wechselnden Hallen.

---

## 1. Das Problem

| Herausforderung | Warum es schwer ist |
|---|---|
| Kamera schwenkt mit dem Spiel | Homographie muss **pro Frame** neu berechnet werden |
| Nicht das ganze Feld sichtbar | 4-Punkt-Kalibrierung auf Feld-Ecken funktioniert nur, wenn alle 4 Ecken im Bild sind |
| Verschiedene Hallen | Boden-Farbe, Linien-Kontrast, Beleuchtung variieren stark |
| Amateur-Aufnahmen | Schlechte Auflösung, verwackelt, ungünstige Winkel |

Die aktuelle Lösung in `calibrate.py` (4 Ecken manuell klicken) funktioniert
nur für eine statische Kamera, bei der alle 4 Ecken sichtbar sind. Für
schwenkende Kameras brauchen wir **automatische, frame-weise** Ansätze.

---

## 2. Handball-Feld als Referenz — die bekannte Geometrie

Das Handball-Feld ist **standardisiert** (IHF-Regeln). Das ist euer größter Vorteil:

```
              ┌─────────────────── 40m ────────────────────┐
              │                    │                        │
         ┌────┤                    │ Mittellinie            ├────┐
         │    │   6m-Kreis         │             6m-Kreis   │    │
  20m    │ TOR│      ╭───╮        │              ╭───╮     │TOR │
         │    │   9m ╭┤   ├╮      │          9m ╭┤   ├╮    │    │
         │    │      ╰┤   ├╯      │             ╰┤   ├╯    │    │
         └────┤       ╰───╯       │              ╰───╯     ├────┘
              │                    │                        │
              └────────────────────┴────────────────────────┘
```

**Alle Maße sind bekannt:**

| Element | Maß |
|---|---|
| Spielfeld | 40m × 20m |
| Torraum (6m-Kreis) | Halbkreis, Radius 6m ab Tormitte |
| Freiwurflinie (9m) | Gestrichelt, Radius 9m ab Tormitte |
| 7m-Punkt | 7m vor Tormitte |
| Mittellinie | Bei 20m, quer über das Feld |
| Tor (innen) | 3m breit × 2m hoch |
| Torraum-Gerade | 6m-Linie, 3m parallel zur Torlinie, verbindet Bogen mit Seitenlinie |

Diese Geometrie kennt ihr — ihr müsst sie nur im Bild **wiederfinden**.

---

## 3. Verfügbare Strategien — Übersicht

```
Strategie A:  Linien-Erkennung + Geometrie-Matching (klassischer CV)
Strategie B:  Keypoint-Detektion (ML, trainiertes Modell)
Strategie C:  Tor als Anker + relative Geometrie (eure Idee!)
Strategie D:  Kombination aus A + C (empfohlen)
```

### 3.1 Strategie A — Linien-Erkennung (klassischer CV)

**Prinzip:** Feldlinien mit Computer Vision erkennen, dann bekannten
Feld-Maßen zuordnen.

```
Frame → Farb-Filter (nur Handball-Linien) → Canny Edges → Hough Lines
      → Linien-Cluster → Bekannter Geometrie zuordnen → Homographie
```

#### Das Hauptproblem: Mehrsport-Hallen-Böden

In Amateur-Sporthallen liegen **5-8 verschiedene Linienfarben**
übereinander — Handball, Basketball, Volleyball, Badminton, etc.:

```
Typisches Farbschema (variiert je nach Halle):
  Handball:     weiß oder gelb
  Basketball:   orange oder rot
  Volleyball:   blau oder grün
  Badminton:    grün oder weiß
  Boden:        hellbraun / grau / blau
```

Ein naiver "finde alle weißen Linien"-Filter produziert **massives
Rauschen** aus den anderen Sportarten, Werbebanden, Hallenmarkierungen
und Lichtreflexionen.

#### Zusätzliche Probleme bei flachem Kamerawinkel

Bei Seitenlinienkameras in Sporthallen:
- Linien werden zu **sehr dünnen Streifen** (1-2 Pixel breit)
- **Starke perspektivische Verkürzung** — nahe Linien breit, ferne kaum sichtbar
- **Hallenbeleuchtung** erzeugt Reflexionen auf dem Boden, die wie Linien aussehen
- **Spieler verdecken** große Teile der Linien

#### Verfahren (wenn ihr es trotzdem versuchen wollt):

1. **Farbfilter — aber gezielt für EINE Farbe**

   Ihr müsst die **Handball-Linienfarbe der jeweiligen Halle** kennen
   oder einmal bestimmen. Ein generischer Weißfilter reicht nicht.

   ```python
   hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

   # Option A: Weiße Linien (hoher Value, niedrige Saturation)
   # Problem: fängt auch Reflexionen, helle Kleidung, andere weiße Linien
   mask_white = cv2.inRange(hsv, (0, 0, 180), (180, 40, 255))

   # Option B: Gelbe Linien (spezifischer Hue-Bereich)
   # Besser, weil weniger andere Sport-Linien gelb sind
   mask_yellow = cv2.inRange(hsv, (18, 80, 150), (35, 255, 255))
   ```

   **Realität:** Selbst mit dem richtigen Farbfilter bekommt ihr
   immer noch Linien anderer Sportarten in ähnlichen Farben.

2. **Kanten-Erkennung + Hough-Transformation:**
   ```python
   # Rauschen entfernen
   kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
   mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # kleine Flecken weg
   mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Lücken schließen

   edges = cv2.Canny(mask, 50, 150)
   lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                            minLineLength=60, maxLineGap=15)
   ```

3. **Linien filtern — das entscheidende Problem**

   Aus den Hough-Lines müsst ihr die Handball-Linien von allem
   anderen trennen. Mögliche Filter:

   ```python
   def filter_handball_lines(lines, frame_shape):
       """Versuche Handball-Linien von anderen zu unterscheiden."""
       if lines is None:
           return []

       h, w = frame_shape[:2]
       filtered = []

       for line in lines:
           x1, y1, x2, y2 = line[0]
           length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
           angle = np.degrees(np.arctan2(y2-y1, x2-x1)) % 180

           # Filter 1: Zu kurze Linien ignorieren
           if length < 50:
               continue

           # Filter 2: Neigung — Seitenlinien sind annähernd horizontal,
           # Grundlinien annähernd vertikal (perspektivisch verzerrt!)
           # Bei flachem Winkel: "vertikal" kann 60°-80° sein
           is_roughly_horizontal = angle < 20 or angle > 160
           is_roughly_vertical = 60 < angle < 120

           if not (is_roughly_horizontal or is_roughly_vertical):
               continue  # Diagonale Linien = wahrscheinlich nicht Handball

           filtered.append(line[0])

       return filtered
   ```

   ⚠️ **Ehrliche Einschätzung:** Diese Filter sind fragil. In einer
   Halle mit 4 Sportarten und flachem Winkel werden sie viele
   falsche Linien durchlassen und echte Linien verpassen.

4. **Linien klassifizieren** — Anhand von Winkel und Position bestimmen,
   welche echte Feld-Markierung das ist:
   - Fast horizontale, lange Linien → Seitenlinien oder Mittellinie
   - Fast vertikale Linien am Bildrand → Grundlinien
   - Gebogene Segmente → 6m/9m-Kreise (sehr schwer mit HoughLinesP!)

5. **Schnittpunkte berechnen** → Homographie:
   ```python
   H, _ = cv2.findHomography(pixel_pts, court_pts, cv2.RANSAC, 5.0)
   ```

#### Ehrliche Bewertung für euren Use-Case

| Aspekt | Bewertung |
|---|---|
| Aufwand | Hoch (viel Tuning pro Halle nötig) |
| Zuverlässigkeit | **Gering** bei Mehrsport-Hallen |
| Generalisierung | **Schlecht** — jede Halle braucht andere Schwellwerte |
| Als alleinige Strategie | **Nicht empfohlen** |
| Als Ergänzung zum Tor | **Brauchbar** — wenn der Farbfilter grob stimmt, liefert es ein paar Extra-Punkte |

**Fazit:** Linien-Erkennung allein ist für Amateur-Sporthallen unzuverlässig.
Als **zusätzliche Punktquelle neben dem Tor** kann es aber helfen — wenn
eine Linie erkannt wird, ist das ein Bonus-Referenzpunkt. Wenn nicht,
hat man immer noch das Tor.

#### Was stattdessen besser funktioniert

Für das Mehrsport-Linien-Problem gibt es bessere Alternativen:

**a) Linien-Segmentierung mit ML (statt Hough Lines)**

Statt Linien mit klassischem CV zu finden, trainiert ihr ein kleines
Segmentierungsmodell, das nur Handball-Linien markiert:

```
Input:  Frame
Output: Binäre Maske "Handball-Linie ja/nein" pro Pixel
```

Vorteil: Das Modell **lernt** den Unterschied zwischen Handball- und
Basketball-Linien anhand des Kontexts (Linienverlauf, Nachbarschaft,
typische Muster). Das kann ein Farbfilter nicht.

Aufwand: ~200-300 mit Polygon-Masken annotierte Frames.

**b) Farb-Kalibrierung pro Halle (pragmatisch)**

Bevor das Spiel analysiert wird, zeigt ein Tool den ersten Frame
und lässt den User **auf eine Handball-Linie klicken**. Daraus wird
der Farbbereich automatisch bestimmt:

```python
def calibrate_line_color(frame, click_x, click_y, radius=10):
    """Bestimme den HSV-Bereich der Handball-Linien aus einem User-Klick."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    patch = hsv[
        max(0, click_y-radius):click_y+radius,
        max(0, click_x-radius):click_x+radius,
    ]
    mean_hsv = patch.mean(axis=(0, 1))
    std_hsv = patch.std(axis=(0, 1))

    # Bereich: mean ± 2*std
    lower = np.clip(mean_hsv - 2*std_hsv, [0,0,0], [180,255,255]).astype(int)
    upper = np.clip(mean_hsv + 2*std_hsv, [0,0,0], [180,255,255]).astype(int)
    return tuple(lower), tuple(upper)
```

Das ist schnell, braucht kein ML, und passt sich an jede Halle an.

**c) Nur das Tor nutzen + geometrische Ableitung (Strategie C/D)**

Da die Linien-Erkennung so problematisch ist, wird die Tor-basierte
Strategie umso wertvoller. Details dazu in Abschnitt 3.3 und 3.4.

---

### 3.2 Strategie B — Keypoint-Detektion (ML)

**Prinzip:** Ein neuronales Netz trainieren, das direkt bestimmte
Feld-Punkte im Bild erkennt (z.B. Linien-Schnittpunkte, Tor-Ecken).

**Vorhandene Projekte:**
- **SportsField** / **narya** — Fußball-spezifisch, Konzept übertragbar
- **SportsSloMo** — Keypoint-basierte Feld-Registrierung
- Eigenes YOLO11-Keypoint-Modell trainieren

**Vorgehen:**
1. Frames annotieren: Markiere 10-20 definierte Punkte pro Frame
   (Tor-Ecken, Linien-Schnittpunkte, 7m-Punkt, Mittelkreis-Mitte usw.)
2. Train YOLO11-pose (custom Keypoints) oder ein Regression-Netz
3. Pro Frame: Modell sagt Keypoint-Positionen vorher → Homographie berechnen

**Vorteile:**
- Robust gegen Verdeckungen (Netz lernt zu interpolieren)
- Einmal trainiert, funktioniert automatisch

**Nachteile:**
- **Hoher Annotationsaufwand** (hunderte Frames von verschiedenen Hallen)
- Braucht diverse Trainings-Daten aus vielen Hallen
- Für euer Team aktuell wahrscheinlich zu aufwändig

---

### 3.3 Strategie C — Tor als Anker (eure Idee)

**Prinzip:** Das Tor ist fast immer sichtbar, hat bekannte Maße (3m × 2m),
und seine Position auf dem Feld ist fix. Von den Tor-Eckpunkten lassen
sich andere Feld-Koordinaten ableiten.

```
Tor erkannt (4 Eckpunkte) → Bekannte Maße (3m × 2m)
  → Position auf dem Feld (Tormitte = Grundlinie bei 10m)
  → Weitere Punkte geometrisch ableiten
  → Homographie berechnen
```

**Warum das funktioniert:**

Das Tor ist ein **Rechteck mit bekannten Maßen** (3m breit, 2m hoch). Wenn
ihr die 4 Eckpunkte des Tors im Bild erkennt, habt ihr:

- Eine lokale Homographie für die Tor-Ebene
- Die Orientierung der Kamera relativ zum Feld
- Einen festen Ankerpunkt im Feld-Koordinatensystem

**Tor-Eckpunkte → Feld-Koordinaten:**

```
Linkes Tor (x=0):
  Pfosten unten-links:  (0.0,  8.5)   ← Tormitte bei y=10m, Tor ist 3m breit
  Pfosten unten-rechts: (0.0, 11.5)
  Latte links:          (0.0,  8.5)   + 2m Höhe (nur in 3D relevant)
  Latte rechts:         (0.0, 11.5)   + 2m Höhe

Rechtes Tor (x=40):
  Pfosten unten-links:  (40.0,  8.5)
  Pfosten unten-rechts: (40.0, 11.5)
```

**Wichtig:** Die Tor-Pfosten stehen auf dem Boden und liegen auf der
Grundlinie. Die Fußpunkte der Pfosten geben euch 2 Punkte im
Feld-Koordinatensystem. Aber **2 Punkte reichen nicht** für eine vollständige
Homographie (braucht mindestens 4). Hier braucht ihr zusätzliche Punkte.

**Zusätzliche Punkte vom Tor ableiten:**

Da das Tor auf der Grundlinie steht wirkt die Unterkante des Tors
(Bodenlinie zwischen den Pfosten) als **Referenzlinie** auf der Grundlinie.
Weitere Punkte könnt ihr gewinnen:

1. **Ecke Feld/Grundlinie** — Dort, wo Seitenlinie auf Grundlinie trifft.
   Oft am Rand des Bildes erkennbar, und der Abstand zum Tor ist bekannt
   (8.5m zum näheren Pfosten).

2. **6m-Linie trifft Grundlinie** — Der 6m-Kreisbogen berührt die
   Grundlinie 3m neben der Seitenlinie. Oft als Markierung erkennbar.

3. **Andere sichtbare Linien** — Mittellinie, 9m-Linie, 7m-Punkt.

---

### 3.4 Strategie D — Kombination (empfohlen für euch)

Die praktischste Lösung kombiniert **Tor-Erkennung** als stabilen Anker
mit **Linien-Erkennung** für zusätzliche Referenzpunkte:

```
┌─────────────────────────────────────────────────────┐
│                  Pro Frame:                         │
│                                                     │
│  1. Tor erkennen (YOLO bbox → Ecken extrahieren)    │
│        ↓                                            │
│  2. Linien erkennen (Hough Lines, gefiltert)        │
│        ↓                                            │
│  3. Punkte sammeln: Tor-Ecken + Linien-Schnitte     │
│        ↓                                            │
│  4. Jedem Punkt seine Feld-Koordinate zuweisen      │
│        ↓                                            │
│  5. Homographie berechnen (RANSAC)                  │
│        ↓                                            │
│  6. Spieler-Positionen transformieren               │
└─────────────────────────────────────────────────────┘
```

---

## 4. Machbarkeit — Tor als Anker (Detail-Analyse)

### 4.1 Tor erkennen — wie?

**Option A: YOLO Bounding Box**

Ihr habt bereits ein YOLO-Modell. Fügt "goal" als Klasse hinzu und
trainiert es auf euren Daten (→ unified model mit Person + Ball + Goal).

Das gibt euch eine **Bounding Box** um das Tor. Von da die Eckpunkte
extrahieren:

```python
def tor_ecken_aus_bbox(bbox, seite="links"):
    """
    Schätze die 4 Tor-Eckpunkte aus der YOLO Bounding Box.
    Annahme: Kamera schaut ungefähr frontal auf das Tor.
    """
    x1, y1, x2, y2 = bbox
    # Untere Kante = Boden-Linie des Tors (auf der Grundlinie)
    # Obere Kante = Latte
    return {
        "pfosten_links_unten":  (x1, y2),  # Boden links
        "pfosten_rechts_unten": (x2, y2),  # Boden rechts
        "latte_links":          (x1, y1),  # Oben links
        "latte_rechts":         (x2, y1),  # Oben rechts
    }
```

⚠️ **Achtung:** Die YOLO-BBox ist achsen-parallel, das Tor aber
perspektivisch verzerrt. Für bessere Genauigkeit solltet ihr die
tatsächlichen Ecken innerhalb der BBox finden (→ Option B).

**Option B: Corner Refinement in der BBox**

Nachdem YOLO die grobe Tor-Position liefert, könnt ihr innerhalb
des Ausschnitts die exakten Ecken finden:

```python
def refine_tor_ecken(frame, bbox):
    """Finde exakte Tor-Ecken innerhalb der YOLO BBox."""
    x1, y1, x2, y2 = bbox
    roi = frame[y1:y2, x1:x2]

    # Tor-Pfosten und Latte sind meist weiß/silber
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(
        gray, maxCorners=4, qualityLevel=0.3, minDistance=20
    )

    if corners is not None:
        # Zurück in Frame-Koordinaten
        corners = corners.reshape(-1, 2)
        corners[:, 0] += x1
        corners[:, 1] += y1
        # Sortieren: TL, TR, BR, BL
        return sort_corners(corners)
    return None
```

**Option C: Tor als Keypoint-Objekt annotieren**

Die genaueste Variante: Im Annotations-Tool (Roboflow) die 4
Tor-Eckpunkte direkt als Keypoints annotieren und ein Keypoint-Modell
trainieren. Erfordert mehr Annotationsaufwand, gibt aber die besten
Ergebnisse.

### 4.2 Von Tor-Ecken zur Homographie

Angenommen, ihr erkennt das **linke Tor** und extrahiert 2 Boden-Punkte:

```python
# Pixel-Koordinaten (aus Tor-Erkennung)
pfosten_links_px  = (120, 680)   # Beispiel
pfosten_rechts_px = (210, 680)

# Bekannte Feld-Koordinaten (Meter)
pfosten_links_m  = (0.0, 8.5)    # Grundlinie, 8.5m von Seitenlinie
pfosten_rechts_m = (0.0, 11.5)   # Grundlinie, 11.5m von Seitenlinie
```

Das sind **2 Punkte** — für eine Homographie brauchen wir **mindestens 4**
(besser 6+). Die fehlenden Punkte kommen aus:

1. **Sichtbare Linien** (Hough Lines → Schnittpunkte)
2. **Zweites Tor** (wenn sichtbar → 2 weitere Punkte)
3. **Bekannte geometrische Abstände** (6m-Punkt, 7m-Punkt, Mittellinie)

**Beispiel mit Tor + 2 Linien-Schnittpunkten:**

```python
import cv2
import numpy as np

# Gesammelte Punkt-Paare: (pixel_x, pixel_y) → (feld_x, feld_y)
pixel_pts = np.float32([
    [120, 680],    # Linker Pfosten (Boden)
    [210, 680],    # Rechter Pfosten (Boden)
    [50,  680],    # Ecke Grundlinie / Seitenlinie (links)
    [120, 420],    # 6m-Linie trifft Seitenlinie
    [680, 350],    # Mittellinie trifft Seitenlinie
])

court_pts = np.float32([
    [0.0,  8.5],   # Linker Pfosten
    [0.0, 11.5],   # Rechter Pfosten
    [0.0,  0.0],   # Ecke Grundlinie/Seitenlinie
    [6.0,  0.0],   # 6m-Linie an Seitenlinie
    [20.0, 0.0],   # Mittellinie an Seitenlinie
])

H, mask = cv2.findHomography(pixel_pts, court_pts, cv2.RANSAC, 5.0)
```

### 4.3 Was wenn nur ein Tor sichtbar ist?

Das ist der **Normalfall** bei Handball-Übertragungen! Die Kamera zeigt
meist nur eine Hälfte. Das ist kein Problem, solange ihr genug Punkte
findet:

| Szenario | Sichtbar | Punkt-Quellen |
|---|---|---|
| Angriff links | Linkes Tor + halbe Linien | Tor (2) + Linien (2-4) = ausreichend |
| Mittelfeld | Mittellinie + evtl. Tore am Rand | Mittellinie-Punkte + Seitenlinien |
| Angriff rechts | Rechtes Tor + halbe Linien | Tor (2) + Linien (2-4) = ausreichend |

**Minimum: 4 nicht-kollineare Punkte** (nicht alle auf einer Linie!).

---

## 5. Frame-weise Homographie bei schwenkender Kamera

Bei **jeder Kamera-Bewegung** ändert sich die Homographie. Ansätze:

### 5.1 Re-Compute pro Frame

Berechne die Homographie in jedem Frame neu aus den erkannten Punkten.

**Problem:** Wenn in einem Frame zu wenig Punkte erkannt werden (< 4),
gibt es keine Homographie.

**Lösung: Temporal Smoothing**

```python
class TemporalHomography:
    """Glättet die Homographie über mehrere Frames."""

    def __init__(self, buffer_size=5):
        self._buffer = []
        self._buffer_size = buffer_size

    def update(self, H_new):
        if H_new is not None:
            self._buffer.append(H_new)
            if len(self._buffer) > self._buffer_size:
                self._buffer.pop(0)

    @property
    def H(self):
        if not self._buffer:
            return None
        # Gewichteter Durchschnitt (neuere Frames stärker)
        weights = np.linspace(0.5, 1.0, len(self._buffer))
        weights /= weights.sum()
        return sum(w * h for w, h in zip(weights, self._buffer))
```

### 5.2 Optischer Fluss als Brücke

Wenn in Frame N eine gute Homographie existiert, aber Frame N+1 zu wenig
Punkte hat: **Optischen Fluss** nutzen, um die Kamera-Bewegung zu schätzen
und die Homographie anzupassen.

```python
def update_homography_with_flow(prev_gray, curr_gray, H_prev):
    """Schätze Kamera-Bewegung via optischem Fluss und update H."""
    # Sparse optical flow (Lucas-Kanade)
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, 200, 0.01, 10)
    if prev_pts is None:
        return H_prev

    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_pts, None
    )

    # Nur gute Matches
    good_prev = prev_pts[status.flatten() == 1]
    good_curr = curr_pts[status.flatten() == 1]

    # Kamera-Bewegung als Homographie
    H_delta, _ = cv2.findHomography(good_prev, good_curr, cv2.RANSAC, 3.0)

    if H_delta is not None:
        # H_new transformiert von neuen Pixeln → alten Pixeln → Court
        return H_prev @ np.linalg.inv(H_delta)
    return H_prev
```

---

## 6. Implementierungsplan (Schritt für Schritt)

### Phase 1 — Tor-Erkennung (Woche 1-2)

```
[ ] Tor in euren Videos annotieren (Bounding Box in Roboflow)
    → 100-200 Frames mit "goal" Label reichen für den Anfang
[ ] Ins unified YOLO-Modell aufnehmen (Person + Ball + Goal)
    → oder separates Modell trainieren, wenn ihr den Ball-Workflow beibehalten wollt
[ ] Tor-Erkennung testen: wird das Tor zuverlässig erkannt?
[ ] Corner Refinement implementieren (goodFeaturesToTrack in der BBox)
```

### Phase 2 — Statische Homographie mit Tor (Woche 2-3)

```
[ ] Tor-Ecken → Feld-Koordinaten Mapping implementieren
[ ] Zusätzliche Punkte: Linien-Erkennung (Hough Lines) für Grundlinie,
    Seitenlinie, Mittellinie
[ ] Homographie mit cv2.findHomography(RANSAC) berechnen
[ ] Spieler-Fuß-Positionen transformieren und auf 2D-Court zeichnen
[ ] Testen mit statischer Kamera-Aufnahme
```

### Phase 3 — Dynamische Kamera (Woche 3-4)

```
[ ] Frame-weise Homographie: in jedem Frame Punkte erkennen → H berechnen
[ ] Temporal Smoothing einbauen (Puffer über 5-10 Frames)
[ ] Optical Flow als Fallback, wenn zu wenig Punkte erkannt werden
[ ] Testen mit schwenkender Kamera
```

### Phase 4 — Qualitätssicherung (Woche 4+)

```
[ ] Homographie-Qualitäts-Score: wie viele Punkte? Wie gut passt der Fit?
[ ] Frames mit schlechter Homographie markieren (confidence < threshold)
[ ] Verschiedene Hallen testen
[ ] Farb-Filter anpassen pro Halle (oder automatisch lernen)
```

---

## 7. Bekannte Punkte am Handball-Feld (Referenz-Tabelle)

Nutzt diese Tabelle als Nachschlagewerk beim Zuordnen erkannter Punkte:

| Punkt | Feld-Koordinaten (x, y) in Metern | Wie erkennen? |
|---|---|---|
| Ecke oben-links | (0, 0) | Grundlinie × Seitenlinie |
| Ecke oben-rechts | (40, 0) | Grundlinie × Seitenlinie |
| Ecke unten-links | (0, 20) | Grundlinie × Seitenlinie |
| Ecke unten-rechts | (40, 20) | Grundlinie × Seitenlinie |
| Mittellinie × Seitenlinie oben | (20, 0) | Mittellinie × Seitenlinie |
| Mittellinie × Seitenlinie unten | (20, 20) | Mittellinie × Seitenlinie |
| Tor links, Pfosten oben | (0, 8.5) | Torpfosten-Fuß |
| Tor links, Pfosten unten | (0, 11.5) | Torpfosten-Fuß |
| Tor rechts, Pfosten oben | (40, 8.5) | Torpfosten-Fuß |
| Tor rechts, Pfosten unten | (40, 11.5) | Torpfosten-Fuß |
| 7m-Punkt links | (7, 10) | Markierung auf dem Boden |
| 7m-Punkt rechts | (33, 10) | Markierung auf dem Boden |
| 6m-Linie × Seitenlinie oben (links) | (6, 0*) | 6m-Bogen trifft Grundlinie/Gerade |
| 6m-Linie × Seitenlinie unten (links) | (6, 20*) | 6m-Bogen trifft Grundlinie/Gerade |
| Mittelpunkt | (20, 10) | Anstoßpunkt |

_* Die 6m-Linie trifft nicht direkt die Seitenlinie, sondern die Grundlinie 3m neben der Seitenlinie. Die genauen Koordinaten hängen vom Kreisbogen ab._

---

## 8. Minimal-Implementierung (Proof of Concept)

Ein einfacher erster Ansatz, der in `pipeline/court.py` integriert
werden kann:

```python
"""
Ansatz: Tor-BBox + Linien-Erkennung → Homographie pro Frame.
"""

import cv2
import numpy as np


# Handball-Feld Referenzpunkte (Meter)
FIELD_LANDMARKS = {
    "tor_links_pfosten_oben":   (0.0,  8.5),
    "tor_links_pfosten_unten":  (0.0, 11.5),
    "tor_rechts_pfosten_oben":  (40.0,  8.5),
    "tor_rechts_pfosten_unten": (40.0, 11.5),
    "ecke_oben_links":          (0.0,  0.0),
    "ecke_unten_links":         (0.0, 20.0),
    "ecke_oben_rechts":         (40.0,  0.0),
    "ecke_unten_rechts":        (40.0, 20.0),
    "mitte_oben":               (20.0,  0.0),
    "mitte_unten":              (20.0, 20.0),
}


def detect_field_lines(frame_bgr, saturation_max=50, value_min=180):
    """Erkenne weiße Linien auf dem Spielfeld."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, value_min), (180, saturation_max, 255))

    # Rauschen entfernen
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Linien finden
    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi/180,
        threshold=80, minLineLength=60, maxLineGap=15
    )
    return lines


def extract_goal_floor_points(goal_bbox, side="left"):
    """
    Extrahiere die Boden-Punkte des Tors aus einer YOLO BBox.
    Gibt 2 Pixel-Punkte zurück + ihre Feld-Koordinaten.
    """
    x1, y1, x2, y2 = goal_bbox

    pixel_pts = [(x1, y2), (x2, y2)]  # Boden links, Boden rechts

    if side == "left":
        court_pts = [(0.0, 8.5), (0.0, 11.5)]
    else:
        court_pts = [(40.0, 8.5), (40.0, 11.5)]

    return pixel_pts, court_pts


def compute_homography(pixel_points, court_points):
    """
    Berechne Homographie aus Punkt-Paaren.
    Braucht mindestens 4 Paare.
    """
    if len(pixel_points) < 4:
        return None

    src = np.float32(pixel_points)
    dst = np.float32(court_points)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return H
```

---

## 9. Empfehlung für euer Team

Angesichts eurer Erfahrung und des Projekt-Umfangs:

| Priorität | Aktion | Aufwand |
|---|---|---|
| 🟢 Sofort | Tor annotieren & in YOLO-Modell aufnehmen | Gering (100-200 Annotations) |
| 🟢 Sofort | Tor-Pfosten-Fußpunkte als Referenz nutzen | Wenig Code |
| � Sofort | Farb-Kalibrierung: User klickt auf Handball-Linie im ersten Frame → HSV-Bereich wird automatisch bestimmt | Wenig Code |
| 🟡 Dann | Gefilterte Linien-Erkennung als **Bonus** (nicht als Hauptquelle) | Mittlerer Aufwand |
| 🟡 Dann | Statische Homographie pro Halbzeit testen | Wenig Code |
| 🔴 Später | Frame-weise Homographie mit Temporal Smoothing | Mehr Aufwand |
| 🔴 Später | Optical Flow für Kamera-Schwenks | Komplex |

**Kernbotschaft:** Verlasst euch **primär auf das Tor** als stabilen
Anker — das ist euer zuverlässigstes Signal. Linien-Erkennung ist
in Mehrsport-Hallen zu unzuverlässig als Hauptstrategie, kann aber
als optionale Zusatzquelle für Bonus-Referenzpunkte dienen. Eine
kurze Farb-Kalibrierung pro Halle (User klickt auf eine Handball-Linie)
macht die Linien-Erkennung deutlich brauchbarer als ein generischer
Weißfilter.

---

## 10. Nützliche OpenCV-Funktionen (Referenz)

| Funktion | Zweck |
|---|---|
| `cv2.findHomography(src, dst, RANSAC)` | Homographie aus Punkt-Paaren (robust) |
| `cv2.perspectiveTransform(pts, H)` | Punkte mit Homographie transformieren |
| `cv2.warpPerspective(img, H, size)` | Ganzes Bild transformieren (für Visualisierung) |
| `cv2.HoughLinesP(edges, ...)` | Linien-Segmente in Kantenbild finden |
| `cv2.goodFeaturesToTrack(gray, ...)` | Markante Ecken/Punkte finden |
| `cv2.calcOpticalFlowPyrLK(...)` | Sparse Optical Flow (Punkt-Tracking) |
| `cv2.Canny(img, t1, t2)` | Kanten-Erkennung |
| `cv2.inRange(hsv, lo, hi)` | Farb-Segmentierung |
| `cv2.morphologyEx(mask, op, kernel)` | Morphologische Operationen (Rauschen entfernen) |
