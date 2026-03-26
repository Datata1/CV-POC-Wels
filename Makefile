.PHONY: help install download-model run preview clean analyze analyze-fast analyze-preview calibrate

CHUNK_SECONDS ?= 60

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies via uv
	uv sync

download-model: ## Download MediaPipe pose landmarker model (heavy)
	mkdir -p models
	curl -L -o models/pose_landmarker_heavy.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task

run: ## [legacy] Process video with detect.py
	uv run python detect.py --chunk-seconds $(CHUNK_SECONDS)

preview: ## [legacy] Process with live preview (press q to quit)
	uv run python detect.py --preview --chunk-seconds $(CHUNK_SECONDS)

run-accurate: ## [legacy] Process with larger YOLO model for best accuracy (slower)
	uv run python detect.py --chunk-seconds $(CHUNK_SECONDS) --yolo-model yolov8m.pt --confidence 0.25

# ── New pipeline ──────────────────────────────────────────

analyze: ## Full pipeline: detect, track, team ID, pose, state export
	uv run python analyze.py --chunk-seconds $(CHUNK_SECONDS)

analyze-fast: ## Full pipeline without pose estimation (much faster)
	uv run python analyze.py --chunk-seconds $(CHUNK_SECONDS) --no-pose

analyze-accurate: ## Full pipeline with large YOLO model
	uv run python analyze.py --chunk-seconds $(CHUNK_SECONDS) --yolo-model yolov8m.pt --confidence 0.25

analyze-preview: ## Full pipeline with live preview
	uv run python analyze.py --chunk-seconds $(CHUNK_SECONDS) --preview

calibrate: ## Open calibration tool to mark court corners
	uv run python calibrate.py

clean: ## Remove all generated output videos and state files
	rm -f output/*.mp4 output/*.jsonl
