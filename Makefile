.PHONY: help install download-pose-model run preview clean analyze analyze-fast analyze-preview calibrate extract-frames train-ball validate-ball analyze-ball analyze-lines analyze-ball-lines extract-goal-frames train-goal validate-goal predict-goal install-goal train-court-keypoints validate-court-keypoints predict-court-keypoints install-court-keypoints analyze-goal setup-rtdetr convert-datasets train-rtdetr-ball train-rtdetr-court-kp train-rtdetr-goal validate-rtdetr-ball validate-rtdetr-court-kp validate-rtdetr-goal

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

COURT_KP_EPOCHS ?= 200
COURT_KP_IMGSZ ?= 1920
COURT_KP_BATCH ?= 2
COURT_KP_BASE_MODEL ?= yolo11m.pt
COURT_KP_DATA ?= annotation/court/data.yaml
COURT_KP_NAME ?= handball_court_kp
COURT_KP_MODEL ?= runs/detect/$(COURT_KP_NAME)/weights/best.pt

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
		-vf "select=not(mod(n\,30))" -vsync vfr -q:v 1 annotation/ball/raw_images/frame_%05d.jpg
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

# ── Homography POC (manual keypoints) ─────────────────────

poc-homography: ## POC: Manual keypoint placement → homography (reads images from input/poc-homography/)
	@mkdir -p input/poc-homography
	@echo "Place frame images in input/poc-homography/ then run this target."
	@[ "$$(ls input/poc-homography/*.jpg input/poc-homography/*.png 2>/dev/null | wc -l)" -gt 0 ] || \
		{ echo "No images found in input/poc-homography/"; exit 1; }
	uv run python poc_homography.py

# ── Goal training ─────────────────────────────────────────

extract-goal-frames: ## Extract frames from input video for goal labeling
	mkdir -p annotation/goal/raw_images
	ffmpeg -i $$(ls input/*.mp4 input/*.avi input/*.mov 2>/dev/null | head -1) \
		-vf "select=not(mod(n\,100))" -vsync vfr -q:v 1 annotation/goal/raw_images/frame_%05d.jpg
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

analyze-goal: ## Full pipeline with court keypoint mapping
	uv run python analyze.py --chunk-seconds $(CHUNK_SECONDS) --court-kp-model models/handball_court_kp.pt --no-pose --yolo-model yolo11m.pt

# ── Court keypoint pose training ──────────────────────────

train-court-keypoints: ## Train court keypoint detection model (each landmark = its own class)
	uv run yolo detect train \
		data=$(COURT_KP_DATA) \
		model=$(COURT_KP_BASE_MODEL) \
		epochs=$(COURT_KP_EPOCHS) \
		imgsz=$(COURT_KP_IMGSZ) \
		batch=$(COURT_KP_BATCH) \
		device=0 \
		name=$(COURT_KP_NAME)
	@echo "Training complete. Best model: $(COURT_KP_MODEL)"

validate-court-keypoints: ## Validate trained court keypoint model
	uv run yolo detect val \
		data=$(COURT_KP_DATA) \
		model=$(COURT_KP_MODEL) \
		device=0

predict-court-keypoints: ## Run visual predictions with court keypoint model
	uv run yolo detect predict \
		model=$(COURT_KP_MODEL) \
		source=annotation/court/test/images \
		device=0 \
		save=True
	@echo "Check runs/detect/predict/ for visual results"

install-court-keypoints: ## Copy trained court keypoint model to models/
	cp $(COURT_KP_MODEL) models/handball_court_kp.pt
	@echo "Installed to models/handball_court_kp.pt"
	@echo "Run: make analyze-goal"

# ── TrackNet ball detection (POC) ─────────────────────────

TRACKNET_MODEL ?= models/tracknet.pt

download-tracknet-model: ## Show instructions for downloading pretrained TrackNet weights
	@echo "The yastrebksv/TrackNet repo distributes weights via Google Drive, not a direct URL."
	@echo ""
	@echo "1. Open the README at https://github.com/yastrebksv/TrackNet"
	@echo "2. Click the 'model_best.pt' Google Drive link in the 'Pretrained model' section"
	@echo "3. Save the file to: $(TRACKNET_MODEL)"
	@echo ""
	@echo "Or, if you have gdown installed and know the file ID:"
	@echo "    mkdir -p models && uv run gdown <FILE_ID> -O $(TRACKNET_MODEL)"

tracknet-detect: ## Run TrackNet ball detection on first video in input/
	uv run python tracknet_detect.py --chunk-seconds $(CHUNK_SECONDS) --model $(TRACKNET_MODEL)

tracknet-detect-heatmaps: ## TrackNet detection + emit side-by-side heatmap video
	uv run python tracknet_detect.py --chunk-seconds $(CHUNK_SECONDS) --model $(TRACKNET_MODEL) --save-heatmaps

tracknet-detect-preview: ## TrackNet detection with live preview window
	uv run python tracknet_detect.py --chunk-seconds $(CHUNK_SECONDS) --model $(TRACKNET_MODEL) --preview

extract-tracknet-triplets: ## Extract (t-2, t-1, t) triplets from first input/ video for TrackNet labelling
	uv run python training/extract_tracknet_triplets.py \
		--input $$(ls input/*.mp4 input/*.avi input/*.mov 2>/dev/null | head -1) \
		--out annotation/ball/triplets --stride 30
	@echo "Upload only *_t.jpg files to Roboflow, label the ball, export YOLOv8 labels to annotation/ball/labels/"

train-tracknet: ## Fine-tune TrackNet on handball triplets (warm-start from models/tracknet.pt)
	uv run python training/train_tracknet.py \
		--triplets annotation/ball/triplets \
		--labels   annotation/ball/labels \
		--pretrained models/tracknet.pt \
		--output     models/tracknet_handball.pt \
		--epochs 30 --batch-size 4

# ── Video download ────────────────────────────────────────

download-video: ## Download sample handball video for testing
	mkdir -p input
	yt-dlp -S "vcodec:h264" --merge-output-format mp4 --cookies-from-browser chrome -o "input/video2.%(ext)s" "https://www.youtube.com/watch?v=5PDVclN7lY0"

# ── RT-DETRv4 POC ─────────────────────────────────────────

RTDETR_DIR = local/rtdetrv4
RTDETR_PRETRAIN = $(RTDETR_DIR)/pretrain/rtv4_hgnetv2_l_coco.pth
DINOV3_WEIGHTS = $(RTDETR_DIR)/pretrain/dinov3_vitb16_pretrain_lvd1689m.pth

setup-rtdetr: ## Download RT-DETRv4 pretrained weights + DINOv3 teacher
	mkdir -p $(RTDETR_DIR)/pretrain
	@echo "=== Step 1: RT-DETRv4-L COCO weights (Google Drive) ==="
	@[ -f $(RTDETR_PRETRAIN) ] || \
		(uv pip install gdown && \
		 uv run gdown --id 1shO9EzZvXZyKedE2urLsN4dwEv8Jqa_8 -O $(RTDETR_PRETRAIN))
	@echo "=== Step 2: Cloning DINOv3 repo ==="
	@[ -d $(RTDETR_DIR)/dinov3 ] || \
		git clone https://github.com/facebookresearch/dinov3.git $(RTDETR_DIR)/dinov3
	@echo "=== Step 3: DINOv3 ViT-B/16 weights ==="
	@[ -f $(DINOV3_WEIGHTS) ] || \
		echo "MANUAL STEP: Request DINOv3 weights at https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/" && \
		echo "Download ViT-B/16 (LVD-1689M) and save to: $(DINOV3_WEIGHTS)"
	@echo "=== RT-DETRv4 setup complete ==="

convert-datasets: ## Convert YOLO datasets to COCO-JSON for RT-DETRv4
	uv run python scripts/yolo_to_coco.py --data annotation/ball/data.yaml --output annotation/ball/coco
	uv run python scripts/yolo_to_coco.py --data annotation/court/data.yaml --output annotation/court/coco
	uv run python scripts/yolo_to_coco.py --data annotation/goal/data.yaml --output annotation/goal/coco

train-rtdetr-ball: ## Train RT-DETRv4 ball detection (fine-tune from COCO)
	cd $(RTDETR_DIR) && \
	uv run torchrun --nproc_per_node=1 train.py \
		-c configs/handball/rtv4_ball.yml \
		-t pretrain/rtv4_hgnetv2_l_coco.pth \
		--use-amp --seed=42

train-rtdetr-court-kp: ## Train RT-DETRv4 court keypoint detection
	cd $(RTDETR_DIR) && \
	uv run torchrun --nproc_per_node=1 train.py \
		-c configs/handball/rtv4_court_kp.yml \
		-t pretrain/rtv4_hgnetv2_l_coco.pth \
		--use-amp --seed=42

train-rtdetr-goal: ## Train RT-DETRv4 goal detection
	cd $(RTDETR_DIR) && \
	uv run torchrun --nproc_per_node=1 train.py \
		-c configs/handball/rtv4_goal.yml \
		-t pretrain/rtv4_hgnetv2_l_coco.pth \
		--use-amp --seed=42

validate-rtdetr-ball: ## Validate RT-DETRv4 ball model
	cd $(RTDETR_DIR) && \
	uv run torchrun --nproc_per_node=1 train.py \
		-c configs/handball/rtv4_ball.yml \
		-r outputs/rtv4_handball_ball/best_stg1.pth \
		--test-only

validate-rtdetr-court-kp: ## Validate RT-DETRv4 court keypoint model
	cd $(RTDETR_DIR) && \
	uv run torchrun --nproc_per_node=1 train.py \
		-c configs/handball/rtv4_court_kp.yml \
		-r outputs/rtv4_handball_court_kp/best_stg1.pth \
		--test-only

validate-rtdetr-goal: ## Validate RT-DETRv4 goal model
	cd $(RTDETR_DIR) && \
	uv run torchrun --nproc_per_node=1 train.py \
		-c configs/handball/rtv4_goal.yml \
		-r outputs/rtv4_handball_goal/best_stg1.pth \
		--test-only

clean: ## Remove all generated output videos and state files
	rm -f output/*.mp4 output/*.jsonl
