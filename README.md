# Decoupled Parallel Hybrid Vision & Deep Reinforcement Learning Deceleration Control for ADAS

This repository contains the complete implementation of a real-time Advanced Driver Assistance System (ADAS) pipeline combining multi-class road hazard detection, drivable area semantic segmentation, and an intelligent Deep Reinforcement Learning (DRL) deceleration agent.

---

## 🚀 Key Features

1. **Dual-Stream Parallel Architecture:**
   * **Road Segmentation Stream:** Custom ResNet-34/EfficientNet-B0 U-Net extracts lane bounds, alt lanes, and unpaved shoulders, incorporating a trapezoidal geometric fallback for weather/night segmentation failures.
   * **Hazard Detection Stream:** Fine-tuned YOLOv8s baseline (`models/best.pt`) runs in parallel with a secondary YOLOv8l ensembled model (`models/best_large.pt`) to detect hazards (Potholes, Cracks, Barriers, Manholes, Waterlogging, Speedbumps).
2. **Lower-Half Slicing-Aided Hyper Inference (LH-SAHI):**
   * Solves spatial downsampling scale issues for small hazards (sub-20-pixel details like Cracks and Cones) by running local inference windows on the lower portion of the frame and merging coordinates via class-filtered Non-Maximum Suppression (NMS).
3. **Deep Reinforcement Learning (DRL) Controller:**
   * A PyTorch Deep Q-Network (DQN) agent controls vehicle braking force. It dynamically adapts speed on the visual HUD based on state: $[D_{\text{hazard}}, v_{\text{speed}}, S_{\text{severity}}, L_{\text{lane}}]$ trained in a custom simulator (reducing collisions to 0.0%).
4. **Adaptive Preprocessing:**
   * Context-aware Contrast Limited Adaptive Histogram Equalization (CLAHE) in LAB space and channel-stretching contrast defogging stabilize visibility under fog, dust, rain, and night splits.
5. **Explainability & Epistemic Uncertainty Mapping:**
   * **HiResCAM:** Visualizes target-filtered class activation maps strictly localized to hazard bodies (guaranteeing zero background leakage).
   * **Monte Carlo Dropout:** Injects 2D dropout layers after C2f backbone blocks to calculate prediction variance (HIGH/MEDIUM/LOW certainty) via 10 stochastic inference passes.
6. **Complementary Speed Fusion:**
   * Employs complementary filtering to fuse high-frequency, responsive IMU integration with low-drift, stable GPS coordinates, reducing speed tracking error to **0.86 km/h RMSE** against CARLA simulator ground truth.

---

## 📂 Repository Structure

* `src/pipeline.py` - Core ADAS loop coordinating detection, segmentation, and control.
* `src/preprocess.py` - Visual enhancement algorithms (CLAHE & contrast defogging).
* `src/segmentation.py` - SMP U-Net wrapper with local import fallback.
* `src/segmentation_model.py` - PyTorch U-Net model definition.
* `src/yolo_detector.py` - YOLOv8 hazard bounding box detector with LH-SAHI coordinate mapping and Priority-based ensembling.
* `src/distance_risk.py` - Bounding box distance projection, lane alignment, and Time-to-Collision (TTC) calculations.
* `src/rl_deceleration_control.py` - Custom simulator and DQN training loop.
* `src/explainability.py` - Explainability heatmaps (HiResCAM, Eigen-CAM, FAM, Grad-CAM).
* `src/uncertainty.py` - Monte Carlo Dropout Hook Injection and uncertainty estimator.
* `src/test_carla_speed.py` - GPS/IMU speed estimation complementary filter test.
* `src/alerts_logging.py` - Haptic warning text, voice alerts (TTS), visual HUD rendering, and telemetry logging.
* `src/verify_pipeline.py` - Simulation test harness for verification checks.
* `run_pipeline.py` - Command-line interface to execute the pipeline on images or videos.

---

## 🛠️ Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/GOURAB-BANDHU/road-hazards-detection.git
   cd road-hazards-detection
   ```
2. **Install the package and dependencies:**
   You can install the dependencies directly:
   ```bash
   pip install -r requirements.txt
   ```
   Or install the repository as an editable Python package:
   ```bash
   pip install -e .
   ```
3. **Place the weights files:**
   Ensure the following model weight files are inside the `models/` directory:
   * `best.pt` (Primary YOLOv8s Detection Weights)
   * `best_large.pt` (Secondary YOLOv8l Ensemble Weights)
   * `best_model_augmented.pth` (Segmentation Weights)
   * `rl_deceleration_agent.pth` (RL Controller Weights)

---

## 💻 Running the Pipeline

### Run E2E Verification Check:
To test the pipeline dynamically using synthetic video inputs:
```bash
python -m src.verify_pipeline
```

### Run on Your Own Video:
To process an input video and render the RL-guided braking HUD overlay:
```bash
python run_pipeline.py --input path/to/video.mp4 --output logs/output_hud.mp4 --speed 60.0
```

### Train the DRL Agent:
To train the deceleration agent from scratch and plot new learning curves:
```bash
python src/rl_deceleration_control.py
```

