.PHONY: help install download-model run preview clean

CHUNK_SECONDS ?= 60

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies via uv
	uv sync

download-model: ## Download MediaPipe pose landmarker model (heavy)
	mkdir -p models
	curl -L -o models/pose_landmarker_heavy.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task

run: ## Process video in 1-min chunks (override: CHUNK_SECONDS=30)
	uv run python detect.py --chunk-seconds $(CHUNK_SECONDS)

preview: ## Process with live preview (press q to quit)
	uv run python detect.py --preview --chunk-seconds $(CHUNK_SECONDS)

run-accurate: ## Process with larger YOLO model for best accuracy (slower)
	uv run python detect.py --chunk-seconds $(CHUNK_SECONDS) --yolo-model yolov8m.pt --confidence 0.25

clean: ## Remove all generated output videos
	rm -f output/*.mp4
