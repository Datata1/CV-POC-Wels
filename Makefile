.PHONY: help install download-pose-model run preview clean analyze analyze-fast analyze-preview calibrate extract-frames train-ball validate-ball analyze-ball analyze-lines analyze-ball-lines extract-goal-frames train-goal validate-goal predict-goal install-goal train-court validate-court predict-court install-court analyze-goal

CHUNK_SECONDS ?= 60
BALL_EPOCHS ?= 100
BALL_IMGSZ ?= 640
BALL_BATCH ?= 16
BALL_BASE_MODEL ?= yolo11m.pt
BALL_DATA ?= annotation/ball/data.yaml
BALL_NAME ?= handball_ball
BALL_MODEL ?= runs/detect/$(BALL_NAME)/weights/best.pt

GOAL_EPOCHS ?= 100
GOAL_IMGSZ ?= 640
GOAL_BATCH ?= 16
GOAL_BASE_MODEL ?= yolo11m.pt
GOAL_DATA ?= annotation/goal/data.yaml
GOAL_NAME ?= handball_goal
GOAL_MODEL ?= runs/detect/$(GOAL_NAME)/weights/best.pt

COURT_EPOCHS ?= 100
COURT_IMGSZ ?= 640
COURT_BATCH ?= 16
COURT_BASE_MODEL ?= yolo11m-seg.pt
COURT_DATA ?= annotation/court/data.yaml
COURT_NAME ?= handball_court
COURT_MODEL ?= runs/segment/$(COURT_NAME)/weights/best.pt

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies via uv
	uv sync

download-pose-model: ## Download YOLO11-pose model (medium, auto-downloaded on first run)
	uv run python -c "from ultralytics import YOLO; YOLO('yolo11m-pose.pt')"

run: ## [legacy] Process video with detect.py
	uv run python detect.py --chunk-seconds $(CHUNK_SECONDS)

preview: ## [legacy] Process with live preview (press q to quit)
	uv run python detect.py --preview --chunk-seconds $(CHUNK_SECONDS)

run-accurate: ## [legacy] Process with larger YOLO model for best accuracy (slower)
	uv run python detect.py --chunk-seconds $(CHUNK_SECONDS) --yolo-model yolo11m.pt --confidence 0.25

# ── New pipeline ──────────────────────────────────────────

analyze: ## Full pipeline: detect, track, team ID, pose, state export
	uv run python analyze.py --chunk-seconds $(CHUNK_SECONDS)

analyze-fast: ## Full pipeline without pose estimation (much faster)
	uv run python analyze.py --chunk-seconds $(CHUNK_SECONDS) --no-pose

analyze-accurate: ## Full pipeline with large YOLO model
	uv run python analyze.py --chunk-seconds $(CHUNK_SECONDS) --yolo-model yolo11m.pt --confidence 0.25

analyze-preview: ## Full pipeline with live preview
	uv run python analyze.py --chunk-seconds $(CHUNK_SECONDS) --preview

calibrate: ## Open calibration tool to mark court corners
	uv run python calibrate.py

# ── Ball training ─────────────────────────────────────────

extract-frames: ## Extract every 10th frame from input video for labeling
	mkdir -p annotation/ball/raw_images
	ffmpeg -i $$(ls input/*.mp4 input/*.avi input/*.mov 2>/dev/null | head -1) \
		-vf "select=not(mod(n\,150))" -vsync vfr annotation/ball/raw_images/frame_%05d.jpg
	@echo "Frames extracted to annotation/ball/raw_images/"
	@echo "Upload to Roboflow, label, export as YOLOv8 format into annotation/ball/"

train-ball: ## Train ball detection model (fine-tune YOLO11 on labeled data)
	uv run yolo detect train \
		data=$(BALL_DATA) \
		model=$(BALL_BASE_MODEL) \
		epochs=$(BALL_EPOCHS) \
		imgsz=$(BALL_IMGSZ) \
		batch=$(BALL_BATCH) \
		device=0 \
		name=$(BALL_NAME)
	@echo "Training complete. Best model: $(BALL_MODEL)"

validate-ball: ## Validate trained ball model on test set
	uv run yolo detect val \
		data=$(BALL_DATA) \
		model=$(BALL_MODEL) \
		device=0

predict-ball: ## Run visual predictions with trained ball model on test images
	uv run yolo detect predict \
		model=$(BALL_MODEL) \
		source=annotation/ball/test/images \
		device=0 \
		save=True
	@echo "Check runs/detect/predict/ for visual results"

install-ball: ## Copy trained ball model to models/ for use in pipeline
	cp $(BALL_MODEL) models/handball_ball.pt
	@echo "Installed to models/handball_ball.pt"
	@echo "Run: make analyze-ball"

analyze-ball: ## Full pipeline with fine-tuned ball model
	uv run python analyze.py --chunk-seconds $(CHUNK_SECONDS) --ball-model models/handball_ball.pt --confidence 0.10

# ── Court line detection (POC) ────────────────────────────

analyze-lines: ## Pipeline with automatic line detection + court top-down view
	uv run python analyze.py --chunk-seconds $(CHUNK_SECONDS) --no-pose --lines

analyze-ball-lines: ## Pipeline with ball model + line detection + court view
	uv run python analyze.py --chunk-seconds $(CHUNK_SECONDS) --ball-model models/handball_ball.pt --confidence 0.10 --no-pose --lines

# ── Goal training ─────────────────────────────────────────

extract-goal-frames: ## Extract frames from input video for goal labeling
	mkdir -p annotation/goal/raw_images
	ffmpeg -i $$(ls input/*.mp4 input/*.avi input/*.mov 2>/dev/null | head -1) \
		-vf "select=not(mod(n\,300))" -vsync vfr annotation/goal/raw_images/frame_%05d.jpg
	@echo "Frames extracted to annotation/goal/raw_images/"
	@echo "Upload to Roboflow, label as 'goal', export as YOLOv8 format into annotation/goal/"

train-goal: ## Train goal detection model (fine-tune YOLO11 on labeled data)
	uv run yolo detect train \
		data=$(GOAL_DATA) \
		model=$(GOAL_BASE_MODEL) \
		epochs=$(GOAL_EPOCHS) \
		imgsz=$(GOAL_IMGSZ) \
		batch=$(GOAL_BATCH) \
		device=0 \
		name=$(GOAL_NAME)
	@echo "Training complete. Best model: $(GOAL_MODEL)"

validate-goal: ## Validate trained goal model on test set
	uv run yolo detect val \
		data=$(GOAL_DATA) \
		model=$(GOAL_MODEL) \
		device=0

predict-goal: ## Run visual predictions with trained goal model on test images
	uv run yolo detect predict \
		model=$(GOAL_MODEL) \
		source=annotation/goal/test/images \
		device=0 \
		save=True
	@echo "Check runs/detect/predict/ for visual results"

install-goal: ## Copy trained goal model to models/ for use in pipeline
	cp $(GOAL_MODEL) models/handball_goal.pt
	@echo "Installed to models/handball_goal.pt"
	@echo "Run: make analyze-goal"

analyze-goal: ## Full pipeline with goal model + court mapping
	uv run python analyze.py --chunk-seconds $(CHUNK_SECONDS) --goal-model models/handball_goal.pt --court-model models/handball_court.pt --lines

# ── Court segmentation training ───────────────────────────

train-court: ## Train court segmentation model (YOLO11-seg on polygon annotations)
	uv run yolo segment train \
		data=$(COURT_DATA) \
		model=$(COURT_BASE_MODEL) \
		epochs=$(COURT_EPOCHS) \
		imgsz=$(COURT_IMGSZ) \
		batch=$(COURT_BATCH) \
		device=0 \
		name=$(COURT_NAME)
	@echo "Training complete. Best model: $(COURT_MODEL)"

validate-court: ## Validate trained court segmentation model
	uv run yolo segment val \
		data=$(COURT_DATA) \
		model=$(COURT_MODEL) \
		device=0

predict-court: ## Run visual predictions with court segmentation model
	uv run yolo segment predict \
		model=$(COURT_MODEL) \
		source=annotation/court/test/images \
		device=0 \
		save=True
	@echo "Check runs/segment/predict/ for visual results"

install-court: ## Copy trained court model to models/ for use in pipeline
	cp $(COURT_MODEL) models/handball_court.pt
	@echo "Installed to models/handball_court.pt"

# ── Video download ────────────────────────────────────────

download-video: ## Download sample handball video for testing
	mkdir -p input
	yt-dlp -S "vcodec:h264" --merge-output-format mp4 --cookies-from-browser chrome -o "input/video2.%(ext)s" "https://www.youtube.com/watch?v=5PDVclN7lY0"

clean: ## Remove all generated output videos and state files
	rm -f output/*.mp4 output/*.jsonl
