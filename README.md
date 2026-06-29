# Decoupled Parallel Hybrid Vision & Deep Reinforcement Learning Deceleration Control for ADAS

This repository contains the complete implementation of a real-time Advanced Driver Assistance System (ADAS) pipeline combining multi-class road hazard detection, drivable area semantic segmentation, and an intelligent Deep Reinforcement Learning (DRL) deceleration agent.

---

## 🚀 Key Features

1. **Dual-Stream Parallel Architecture:**
   * **Road Segmentation Stream:** Custom ResNet-34 U-Net (loaded from `models/best_model_augmented.pth`) extracts lane bounds, shoulders, and drivable paths.
   * **Hazard Detection Stream:** YOLOv8s Sequential Baseline model (`models/best.pt`) runs in parallel to detect obstacles (Potholes, Cracks, Barriers, Manholes, Waterlogging, Speedbumps).
2. **Lower-Half Slicing-Aided Hyper Inference (LH-SAHI):**
   * Resolves downsampling scale issues for small hazards (sub-20-pixel details like Cracks and Traffic Cones) by running local inference windows on the lower portion of the frame and merging coordinates via class-filtered Non-Maximum Suppression (NMS).
3. **Deep Reinforcement Learning (DRL) Controller:**
   * A PyTorch Deep Q-Network (DQN) agent (weights saved at `models/rl_deceleration_agent.pth`) controls vehicle braking force. It dynamically adapts speed on the visual HUD based on state: $[D_{\text{hazard}}, v_{\text{speed}}, S_{\text{severity}}, L_{\text{lane}}]$.
4. **Adaptive Preprocessing:**
   * Context-aware Contrast Limited Adaptive Histogram Equalization (CLAHE) and Dark Channel Prior (DCP) defogging stabilize visibility under fog, dust, rain, and night splits.

---

## 📂 Repository Structure

* `src/pipeline.py` - Core ADAS loop coordinating detection, segmentation, and control.
* `src/preprocess.py` - Visual enhancement algorithms (CLAHE & DCP defogging).
* `src/segmentation.py` - SMP U-Net wrapper with local import fallback.
* `src/segmentation_model.py` - PyTorch U-Net model definition.
* `src/yolo_detector.py` - YOLOv8 hazard bounding box detector with LH-SAHI coordinate mapping.
* `src/distance_risk.py` - Bounding box distance projection and Time-to-Collision (TTC) calculations.
* `src/rl_deceleration_control.py` - Custom simulator and DQN training loop.
* `src/alerts_logging.py` - Haptic warning text, voice alerts (TTS), visual HUD rendering, and telemetry logging.
* `src/verify_pipeline.py` - Simulation test harness for verification checks.
* `run_pipeline.py` - Command-line interface to execute the pipeline on images or videos.

---

## 🛠️ Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/road-hazards-detection.git
   cd road-hazards-detection
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Place the weights files:**
   Ensure the following model weight files are inside the `models/` directory:
   * `best.pt` (Detection Weights)
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
python run_pipeline.py --input path/to/video.mp4 --output logs/output_hud.avi --speed 60.0
```

### Train the DRL Agent:
To train the deceleration agent from scratch and plot new learning curves:
```bash
python src/rl_deceleration_control.py
```
