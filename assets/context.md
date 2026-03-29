# C4 Context Diagram — Handball Match Analysis Tool

The **Context diagram** (C4 Level 1) shows the system as a single box, the people
who use it, and the external systems it interacts with.

---

## Context Diagram

```mermaid
flowchart TD
    subgraph PEOPLE["👤 People"]
        T["🏐 Handball Trainer\n\nReviews matches to adjust\nteam strategy and analyze\nplayer performance"]
        A["🏷️ Data Annotator\n\nLabels training data to\nimprove model accuracy"]
    end

    subgraph SYSTEM["Handball Match Analysis Tool"]
        direction TB
        S["Processes match video recordings\nand produces structured match data,\nannotated video, and tactical predictions"]
    end

    subgraph EXTERNAL["📦 External Systems"]
        V["📹 Match Video\nRecordings\n\nRaw footage from broadcast\nor sideline cameras"]
        R["🏷️ Roboflow / CVAT\n\nAnnotation platforms for\nlabeling training data"]
        Y["🤖 Pre-trained\nYOLO Models\n\nBase detection & pose\nestimation weights"]
    end

    T -->|"uploads match video"| S
    S -->|"annotated video\n2D court map\nstructured data\naction predictions"| T

    A -->|"labels ball, actions"| R
    R -->|"labeled training data\n(YOLO format)"| S

    V -->|"raw match footage\n(MP4)"| S
    Y -->|"model weights"| S

    style SYSTEM fill:#2563eb,stroke:#1d4ed8,color:#fff
    style S fill:#2563eb,stroke:#1d4ed8,color:#fff
    style T fill:#08427b,stroke:#073b6f,color:#fff
    style A fill:#08427b,stroke:#073b6f,color:#fff
    style V fill:#999,stroke:#777,color:#fff
    style R fill:#999,stroke:#777,color:#fff
    style Y fill:#999,stroke:#777,color:#fff
```

---

## Elements

| Element | Type | Role |
|---------|------|------|
| **Handball Trainer** | Person (primary user) | Uploads video, receives analysis output |
| **Data Annotator** | Person (supporting) | Labels training data to improve detection quality |
| **Analysis Tool** | System (ours) | The single blue box — everything we build |
| **Match Video** | External | Raw input from the customer |
| **Roboflow/CVAT** | External system | Where annotation happens (not built by us) |
| **YOLO Models** | External | Pre-trained weights we fine-tune (not built by us) |

---

## Reading the diagram

- **Blue box** = our system (what we build and deliver)
- **Dark blue boxes** = people who interact with the system
- **Grey boxes** = external systems we depend on but don't build
- **Arrows** = data flow (labels describe what crosses each boundary)

This is intentionally non-technical — the customer sees **what goes in, what comes out,
and who interacts with it**, without any pipeline internals.

The next C4 level (Containers) would zoom into the blue box and show the internal
structure: Stage 1 (CV pipeline), Stage 2 (prediction model), DuckDB storage, etc.
