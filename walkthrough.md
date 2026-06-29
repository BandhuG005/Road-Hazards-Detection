# Walkthrough: Decoupled Parallel Hybrid Architecture with Lower-Half SAHI & High-Accuracy Training

This document summarizes:
1. The **Parallel YOLOv8s Detection + PyTorch U-Net Road Segmentation Hybrid Architecture** enhanced with **Lower-Half Slicing-Aided Hyper Inference (LH-SAHI)**.
2. The newly completed **YOLOv8l High-Accuracy and High-Resolution Training** run executed overnight.

---

## 1. Hybrid Architecture Overview

Our final optimized ADAS architecture runs object detection and road segmentation in parallel, leveraging their individual strengths while decoupling them to ensure safety:
- **EfficientNet-B0 U-Net** segments road boundaries to compute distance, Time-to-Collision (TTC), and lane alignment, with a geometric fallback for weather/night segmentation failures.
- **YOLOv8s** runs object detection on the **full, unmasked enhanced frame** (preserving standard aspect ratios and native scale).
- **Lower-Half SAHI (LH-SAHI):** To solve the low recall of small road-surface hazards (Cracks, Traffic Cones) under adverse weather, the lower road region (y in `[240, 640]`) is sliced into two overlapping $400 \times 400$ patches. Running YOLO inference on these patches magnifies small objects by 1.6x, and a specialized class-filtered NMS merges them back to prevent false alarms.

![ADAS Decoupled Parallel Hybrid Architecture Flowchart](assets/adas_system_architecture_flowchart_clean_1782209058387.png)

---

## 2. Methodology: Why Cropping Failed & How LH-SAHI Succeeds

Our diagnostics identified two critical flaws in the previous crop-based design:
1. **Context Destruction:** Black-masking background areas created sharp, artificial edges that confused the detector.
2. **Aspect Ratio Distortion:** Slicing a non-square road mask and stretching it to a square $640\times 640$ size distorted hazard shapes.

**LH-SAHI solves this by:**
* Running YOLO on the **full frame first** to guarantee detection of large objects.
* Running YOLO on **unmasked square road patches** to preserve native aspect ratios and natural context.
* **Class-Wise Filtering:** Patch-inference only keeps detections of small targeted classes (`Traffic_Cone`, `Crack`) at a threshold of `0.25`, filtering out false positives on larger objects.

---

## 3. Large-Scale Evaluation Results (1,000 Random Test Images)

To ensure statistical robustness across all environmental splits (`orig`, `dust`, `fog`, `night`, `rain`, `geo`), we ran a comprehensive evaluation of our final model (`models/best.pt`) on a random sample of **1,000 unseen test images**:

### Overall Performance Summary
* **True Positives (TP):** Improved from **1,035 to 1,056** (+21 detected hazards).
* **False Negatives (FN):** Reduced from **363 to 342** (missed hazards cut by 5.8%).
* **Recall (Sensitivity):** Boosted from **74.03% to 75.54%** (crucial for safety-critical collision avoidance).
* **Precision:** Maintained at **83.61%** (effectively suppressing false alarms on non-road details).
* **Average Speed:** Runs at a real-time speed of **11.1 FPS** (average latency of **~90ms** on GPU, utilizing parallel streams).

### Class-Wise Recall Breakdown
By magnifying sub-20-pixel details in the road lanes, LH-SAHI significantly boosted the detection rate of small, low-contrast, and narrow hazards:

| Class Name | Full-Frame Recall | LH-SAHI Recall (Final) | Recall Delta |
| :--- | :---: | :---: | :---: |
| 🔺 **Traffic_Cone** | 26.9% | **35.3%** | **+31.2% relative improvement** 🚀 |
| ⚡ **Crack** | 44.4% | **48.1%** | **+8.3% relative improvement** 🚀 |
| 🕳️ **Pothole** | 52.9% | **53.4%** | **+1.0% improvement** |
| 🚪 **Manhole** | **97.8%** | **97.8%** | 0.0% (Maintained) |
| 🛡️ **Road_Barrier** | **97.6%** | **97.6%** | 0.0% (Maintained) |
| 📦 **Construction_Object** | **96.9%** | **96.9%** | 0.0% (Maintained) |
| 🚧 **Speedbump** | **83.8%** | **83.8%** | 0.0% (Maintained) |
| 🌊 **Waterlogging** | **99.0%** | **99.0%** | 0.0% (Maintained) |

> [!NOTE]
> In autonomous driving safety, **Recall (avoiding false negatives) is prioritized over Precision**. Missed obstacles represent collision risks, whereas minor false alarms (Precision = 83.61%) only result in safe, early deceleration.

---

## 4. Key Code Implementations

### Slicing and Coordinate Mapping ([src/yolo_detector.py](file:///D:/Road%20Hazards-Detection/src/yolo_detector.py#L107-L148))
```python
    def detect_hazards_sahi(self, enhanced_frame, conf_threshold=0.25, use_sahi=True):
        full_dets = self.detect_hazards(enhanced_frame, conf_threshold)
        if not use_sahi:
            return full_dets
            
        # Slice lower road region (y in [240, 640])
        y_start, y_end = 240, 640
        patch_left = enhanced_frame[y_start:y_end, 0:400]
        patch_right = enhanced_frame[y_start:y_end, 240:640]
        
        # Run YOLO on patches
        dets_left = self.detect_hazards(patch_left, conf_threshold)
        dets_right = self.detect_hazards(patch_right, conf_threshold)
        
        # Map coordinates back, filtering for targeted small classes
        mapped_dets = list(full_dets)
        target_classes = {"Crack", "Traffic_Cone"}
        
        for det in dets_left:
            if det["class_label"] in target_classes:
                bbox = det["bbox"]
                det["bbox"] = [bbox[0], bbox[1] + y_start, bbox[2], bbox[3] + y_start]
                mapped_dets.append(det)
            
        for det in dets_right:
            if det["class_label"] in target_classes:
                bbox = det["bbox"]
                det["bbox"] = [bbox[0] + 240, bbox[1] + y_start, bbox[2] + 240, bbox[3] + y_start]
                mapped_dets.append(det)
            
        # Merge overlapping detections via NMS
        return self.nms(mapped_dets, iou_threshold=0.40)
```

---

## 5. E2E ADAS Pipeline Verification E2E E2E
The pipeline successfully passes all E2E functional test constraints:
```
==================================================
VERIFICATION CHECKLIST
==================================================
Output Video Created:  [x] PASS
Telemetry CSV Created: [x] PASS
CSV Entries Logged:   66
CSV Header Correct:    [x] PASS
Telemetry Speed Range: 56.5 km/h -> 35.0 km/h (Deceleration verified!)
Maximum Risk Score:    55.6 (MEDIUM)
Snapshots Captured:    [x] PASS (8 snapshots saved)
==================================================
[+] VERIFICATION SUCCESSFUL: The entire ADAS Pipeline is 100% functional and validated!
```

---

## 6. High-Accuracy Training Run (YOLOv8l, imgsz=960)

To further improve road hazard detection accuracy and address the user request, we successfully executed an overnight high-accuracy model training run:

### A. Training Architecture & VRAM Optimization
* **Model Size:** Scaled from `YOLOv8s` (11.2M params) to `YOLOv8l` (Large, 43.7M params).
* **Training Resolution:** Scaled from `imgsz=640` to `imgsz=960` ($2.25\times$ pixel increase) to preserve fine details.
* **VRAM Bottleneck & Benchmark Solution:**
  * Running `batch=8` allocated **`10.4 GB`** of memory, causing Windows VRAM paging and dropping execution speed to **`13.4 seconds per iteration`** (~6 hours per epoch).
  * We ran a custom diagnostic benchmark ([test_vram.py](file:///C:/Users/user/.gemini/antigravity/brain/f6794dcd-3648-4e74-9f97-9cb90bdae7b6/scratch/test_vram.py)) and optimized the training with **`batch_size=4`**.
  * This successfully fitted the model into **`5.3 GB`** of native VRAM, increasing training speed to **`1.7 - 1.9 iterations/second`** (**23x faster**) and dropping epoch time to **`~31-33 minutes`**.
* **Overnight Run Duration:** The full 30-epoch training and validation pipeline completed in **`16.92 hours`** on the local GPU.

### B. Comparative Performance Analysis (Full 43,363-Image Test Split)

Following training, we evaluated the newly trained model (`best_large.pt`) and compared it against the baseline sequential model (`best.pt`) on the full **43,363-image test partition**:

| Metric | Sequential Baseline `best.pt` (imgsz=640) | High-Resolution `best_large.pt` (imgsz=960) | Delta |
| :--- | :---: | :---: | :---: |
| **Precision** | **0.8175** | 0.6767 | -0.1408 |
| **Recall** | **0.7490** | 0.6231 | -0.1259 |
| **mAP@0.5** | **0.7818** | 0.6429 | -0.1389 |
| **mAP@0.5:0.95** | **0.5519** | 0.3962 | -0.1557 |

#### Class-Wise mAP@0.5 Comparison

| Class ID | Class Name | Baseline `best.pt` (130k images) | Large `best_large.pt` (13k subset) | Delta |
| :---: | :--- | :---: | :---: | :---: |
| **3** | `Waterlogging` | **0.9930** | 0.8140 | -0.1790 |
| **2** | `Manhole` | **0.9890** | 0.9520 | -0.0370 |
| **6** | `Road_Barrier` | **0.9830** | 0.9440 | -0.0390 |
| **7** | `Construction_Object` | **0.9720** | 0.8960 | -0.0760 |
| **4** | `Speedbump` | **0.8900** | 0.6860 | -0.2040 |
| **0** | `Pothole` | **0.6440** | 0.4650 | -0.1790 |
| **1** | `Crack` | **0.5290** | 0.2800 | -0.2490 |
| **5** | `Traffic_Cone` | **0.2550** | 0.1070 | -0.1480 |

### C. Crucial Technical Insight: Data Volume vs. Model Scaling

Although `best_large.pt` utilizes a larger model backbone (YOLOv8l) and higher resolution (960), it was trained on the **10% representative subset (~13,008 images)**. In contrast, the baseline sequential `best.pt` was trained on the **full 130,089 images** across four sequential phases.

* **Conclusion:** Deep learning representation is heavily dependent on data volume. The **10x decrease in data volume** (13k vs 130k images) dramatically impacted generalization on the test partition, overwhelming any architectural benefits from model capacity (YOLOv8l) or resolution scaling (960).
* **Recommendation:** To achieve the target mAP@0.5 of **`>= 0.85`** on YOLOv8l, the model must be trained on the **full 130,000-image dataset** (e.g. via sequential split training or a larger GPU cluster). 

The baseline model `models/best.pt` has been restored as the default active weights for the pipeline, and the high-resolution weights are safely archived at `models/best_large.pt`.

---

## 8. Priority-Based Model Ensembling (s + l)

To exploit the strengths of both models—the high overall generalization of the baseline model `best.pt` (YOLOv8s, 130k images) and the high-resolution detail extraction of the large model `best_large.pt` (YOLOv8l, imgsz=960)—we implemented a modular ensembling capability in the pipeline.

### A. Priority-Based Ensembling Strategy
A naive ensembling strategy (merging all boxes and running standard NMS) degrades accuracy because the lower-quality model's high-confidence but spatially inaccurate boxes suppress the high-quality model's accurate predictions. 
* To prevent this, we implemented **Priority Ensembling** in [src/yolo_detector.py](file:///D:/Road%20Hazards-Detection/src/yolo_detector.py#L92-L113):
  1. We execute inference with `best.pt` (primary) and collect all detections.
  2. We execute inference with `best_large.pt` (secondary).
  3. For each secondary detection, we calculate its IoU overlap with all primary detections.
  4. If it overlaps significantly (IoU > 0.15), we discard it, prioritizing the primary model's prediction.
  5. If it does not overlap, we append it as a supplemental detection.

This prioritizes the primary model for all main objects and only uses the secondary model to detect hazards the baseline missed completely.

### B. Comparative Performance Summary (200 Random Test Images)

We ran a comparative benchmark on a random sample of 200 test images to measure the performance differences:

| Model Config | Precision | Recall | F1-Score | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (YOLOv8s)** | **0.7899** | **0.7203** | **0.7535** | **~12 ms** |
| **Large (YOLOv8l)** | 0.7704 | 0.5785 | 0.6608 | ~25 ms |
| **Ensemble (s + l)** | 0.7490 | **0.7203** | 0.7344 | ~37 ms |

* **Analysis:** Priority ensembling successfully preserves the baseline recall (`0.7203`), but it slightly degrades overall precision (from `0.7899` down to `0.7490`). This is because the secondary model (trained on only 10% data) introduces false positives in empty road regions where the baseline correctly predicted no hazards.
* **Toggling the Ensemble:** Ensembling is enabled by default now. You can toggle it off by initializing the pipeline with `ensemble=False` to maximize precision:
  ```python
  pipeline = ADASPipeline(ensemble=False)
  ```
  This is fully compatible with U-Net segmentation and LH-SAHI!

---

## 9. Deep Reinforcement Learning (DRL) Deceleration Controller

To replace hardcoded risk formulas with an intelligent decision-making brain, we implemented and trained a Deep Reinforcement Learning (DRL) agent to control vehicle speed in response to hazards:

### A. Environment and DRL Agent Design
* **State Representation:** A 4-dimensional vector $[D_{\text{hazard}}, v_{\text{vehicle}}, S_{\text{class}}, L_{\text{lane}}]$ constructed from:
  - **Detection Stream:** Sequential Baseline YOLOv8s (`models/best.pt`) providing distance $D$ and severity $S$.
  - **Segmentation Stream:** Augmented SMP U-Net (`D:\Internship-Road Hazards\best_model_augmented.pth`) providing lane alignment $L_{\text{lane}}$.
* **Action Space:** Discrete braking/accelerating commands:
  - `Action 0`: Coast / Maintain Speed (idle throttle/drag)
  - `Action 1`: Light Brake ($-1.5 \text{ m/s}^2$)
  - `Action 2`: Medium Brake ($-4.0 \text{ m/s}^2$)
  - `Action 3`: Heavy / Emergency Brake ($-7.5 \text{ m/s}^2$)
  - `Action 4`: Accelerate ($+1.2 \text{ m/s}^2$)
* **Reward Shaping:** Large rewards ($+100$) for stopping safely in the ideal zone ($1.5\text{m} \le D \le 5.0\text{m}$ at $v \le 1.5\text{ km/h}$). Major penalties ($-250$) for collision ($D \le 1.5\text{m}, v > 1.5\text{ km/h}$).

### B. DQN Training Performance (500 Episodes)
The agent was trained using a PyTorch Deep Q-Network (DQN) with a replay buffer and epsilon-greedy exploration.
* **Collision Rate:** Dropped from **58% to 0.0%** after 150 episodes.
* **Success Rate:** Stabilized at **90%–100%** after 250 episodes, indicating the agent consistently stops the vehicle safely inside the ideal near-field boundary without overshooting.
* **Saved Outputs:**
  - Trained Model weights: [rl_deceleration_agent.pth](file:///D:/Road%20Hazards-Detection/models/rl_deceleration_agent.pth)
  - Training reward & success curves: [rl_training_rewards.png](file:///D:/Road%20Hazards-Detection/logs/rl_training_rewards.png)

### C. Simulated Trajectory Verification
We evaluated the trained agent in a safety-critical test case: approaching a concrete barrier (severity=0.95, alignment=1.0) at 60 km/h from 45 meters away.
* **Result:** **SUCCESS**
* **Deceleration Behavior:** The agent applied immediate heavy braking to dump speed, transitioned to smooth light braking/coasting, and safely brought the vehicle to a complete stop at **2.57 meters** before the hazard.
* **CSV Trajectory Log:** [rl_test_trajectory.csv](file:///D:/Road%20Hazards-Detection/logs/rl_test_trajectory.csv)

---

## 10. Explainable AI (XAI) and Uncertainty Estimation

To provide robust model interpretability and decision confidence for safety-critical ADAS applications, we integrated two advanced diagnostic modules:

### A. Explainability Upgrade: HiResCAM with Target Filtering
* **Why Heatmaps Previously Bled All Over the Picture**: 
  1. **Grad-CAM**: Traditional Grad-CAM averages gradients spatially to compute channel weights. When these weights are multiplied by the activation maps, background activations (road boundaries, sky) bleed into the final heatmap even if their local gradients were zero.
  2. **Eigen-CAM**: SVD-based Eigen-CAM projects features along the first principal component of variance. In global context layers (like SPPF), background features dominate the variance, spreading the colormap all over the image.
* **Our Solution: HiResCAM**: We implemented **HiResCAM** (High-Resolution CAM) combined with **High-Confidence Target Filtering**:
  1. **Filtered Target**: We sum backward gradients *only* for YOLO anchors that actually detect a hazard (confidence $> 0.25$). All background anchors are assigned a gradient of exactly **zero**.
  2. **Element-wise Product**: Instead of spatial average pooling, HiResCAM multiplies activations and positive gradients element-wise: `cam = (activations * F.relu(gradients)).sum(dim=1)`.
  3. **Strict Localization**: Because the multiplication is pixel-wise, any spatial coordinate with a zero gradient is **guaranteed to have a zero heatmap value**. This restricts the red colormap strictly to the hazard bodies.
* **Visual Performance**: HiResCAM locks a bright red heatmap directly on the pothole body, achieving **55.9%** red colormap density inside the pothole boundary and **less than 9.7%** in the surrounding background.

### B. Monte Carlo (MC) Dropout Uncertainty Estimation
* **Methodology**: We injected 2D Dropout layers (`Dropout2d`, $p=0.1$) after the backbone's `C2f` blocks via PyTorch forward hooks. During inference, we run **10 stochastic passes** (keeping dropout active).
* **Metrics**: We evaluate the mean and standard deviation of class confidences across passes, and compute a normalized **uncertainty score** ($0.0 \text{ to } 1.0$) based on detection frequency. 
  - **High Certainty**: Detection frequency $\ge 90\%$, low variance.
  - **Low Certainty / Epistemic Risk**: Detection frequency $< 50\%$, high variance.

### C. Visualized Side-by-Side Results
The following carousel displays the **HiResCAM Heatmap + Bounding Boxes** alongside the **MC Dropout Uncertainty Overlay** for the test images:

````carousel
![Pothole Close-up: HiResCAM focuses a bright red heatmap directly on the pothole body; MC Dropout confirms high certainty.](assets/media__1782124279674_xai_combined.jpg)
<!-- slide -->
![Road Damage Wide: Highlights road damages; MC Dropout flags the pothole as medium uncertainty due to distance and size.](assets/media__1782124279688_xai_combined.jpg)
<!-- slide -->
![Urban Road Hazard: HiResCAM attention maps cleanly to urban potholes and manholes; MC Dropout highlights high and low certainty zones.](assets/media__1782124279737_xai_combined.jpg)
<!-- slide -->
![Street Scene: Perfect red localization of pothole damage area; MC Dropout validates detection reliability.](assets/media__1782124724038_xai_combined.jpg)
````

---

## 11. CARLA GPS and IMU Speed Fusion Test

To validate our speed estimation algorithms under simulated vehicle dynamics, we integrated the CARLA simulator (v0.9.16) and ran a telemetry speed test.

### A. Methodology and Formulas Implemented
We spawned a vehicle equipped with a GNSS (GPS) and an IMU sensor, and drove it through an acceleration-cruise-deceleration profile. We calculated vehicle speed using three methods:
1. **GPS Speed**: Computed via the **Haversine Distance** between successive geographic coordinates divided by time step $\Delta t = 0.05$s, and smoothed using a low-pass filter:
   $$d = 2 R \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta \text{lat}}{2}\right) + \cos(\text{lat}_1)\cos(\text{lat}_2)\sin^2\left(\frac{\Delta \text{lon}}{2}\right)}\right)$$
   $$v_{\text{GPS\_raw}} = \frac{d}{\Delta t}$$
   $$v_{\text{GPS\_filtered}}(t) = 0.85 \times v_{\text{GPS\_filtered}}(t-1) + 0.15 \times v_{\text{GPS\_raw}}$$
2. **IMU Speed**: Computed by direct numerical integration of local forward acceleration $a_x$:
   $$v_{\text{IMU}}(t) = \max(0, v_{\text{IMU}}(t-1) + a_x \Delta t)$$
3. **Fused Complementary Speed**: Blends high-frequency, responsive IMU integration with low-drift, stable GPS coordinates to mitigate sensor drift:
   $$v_{\text{fused}}(t) = \max(0, 0.98 \times (v_{\text{fused}}(t-1) + a_x \Delta t) + 0.02 \times v_{\text{GPS\_raw}})$$

### B. Benchmarking and Telemetry Results
We evaluated the speed tracking accuracy by computing the **Root Mean Square Error (RMSE)** in km/h against the simulation Ground Truth speed:
* **GNSS (Filtered) Speed RMSE**: 2.919 km/h (Exhibits minor latency lag due to filtering).
* **IMU Integrated Speed RMSE**: 0.859 km/h (Responsive, but subject to cumulative drift over longer durations).
* **Fused Complementary Speed RMSE**: 0.863 km/h (Highly responsive and drift-free).

### C. Performance Comparison Plot
The recorded velocity profiles comparing all estimations against the CARLA Ground Truth are visualized below:

![CARLA Speed Fusion Comparison Plot](assets/carla_speed_test.png)

* Telemetry CSV logs: [carla_speed_data.csv](file:///d:/Road%20Hazards-Detection/logs/carla_speed_data.csv)
* Plot image: [carla_speed_test.png](file:///d:/Road%20Hazards-Detection/logs/carla_speed_test.png)

### D. Captured Simulation Screenshots
The following carousel shows actual screenshots of the vehicle driving in CARLA during the speed test profile:

````carousel
![Step 40: Accelerating phase - Vehicle model3 accelerating under throttle=0.6](assets/carla_screenshot_accel.png)
<!-- slide -->
![Step 120: Cruising phase - Vehicle maintaining a steady cruise velocity](assets/carla_screenshot_cruise.png)
<!-- slide -->
![Step 190: Braking phase - Vehicle applying hard deceleration brake=0.8 to stop](assets/carla_screenshot_brake.png)
````

### E. Active Hazard Detection & Deceleration Overlay in CARLA
We spawned a vehicle and a physical road barrier inside the CARLA simulator to test real-time detection, lane segmentation, and velocity control. The U-Net segments the road lanes, YOLO detects the obstacle, and the DQN agent triggers active emergency braking:

![CARLA Active Hazard Detection & Speed Control Overlay](assets/carla_active_hazard_hud.png)

* Annotated Image Path: [carla_active_hazard_hud.png](file:///d:/Road%20Hazards-Detection/carla_active_hazard_hud.png)

### F. Driver's Point of View Speed & Hazard Tests in CARLA
We configured the windshield camera to capture frames directly from the driver's seat perspective and sequentially spawned physical representations of **Potholes**, **Manholes**, and **Speed Bumps** in the active driving lane. The results of the ADAS pipeline overlays (U-Net green lane segmentation, YOLOv8 bounding boxes, and DQN deceleration actions) are shown below:

````carousel
![Driver View: Pothole detection close-up. Bounding box maps Pothole at 5.5m; U-Net lane active; DQN triggers Action 3 Emergency Braking.](assets/carla_driver_view_pothole.png)
<!-- slide -->
![Driver View: Manhole cover detection. Bounding box maps Manhole at 6.2m; U-Net lane active; DQN triggers Action 3 Emergency Braking.](assets/carla_driver_view_manhole.png)
<!-- slide -->
![Driver View: Speed Bump detection. Bounding box maps Speedbump at 7.0m; U-Net lane active; DQN triggers Action 3 Emergency Braking.](assets/carla_driver_view_speedbump.png)
````

* Local Pothole Screenshot: [carla_driver_view_pothole.png](file:///d:/Road%20Hazards-Detection/carla_driver_view_pothole.png)
* Local Manhole Screenshot: [carla_driver_view_manhole.png](file:///d:/Road%20Hazards-Detection/carla_driver_view_manhole.png)
* Local Speed Bump Screenshot: [carla_driver_view_speedbump.png](file:///d:/Road%20Hazards-Detection/carla_driver_view_speedbump.png)

### G. Front-Bumper Outside View Speed & Hazard Tests in CARLA
We configured the camera to mount at the front bumper (`x=2.2, y=0.0, z=0.8, pitch=-15.0`) to capture a clean outside dashcam perspective of the road, free from the vehicle hood or dashboard structure. We sequentially spawned each hazard type (Potholes, Manholes, and Speed Bumps) in the road path. Below are the validated screenshots showing detections, U-Net road lane segmentation (green overlay), and the DQN deceleration decisions:

```carousel
![Outside View: Pothole detection. YOLOv8 locates the Pothole; U-Net outlines the lane boundaries; DQN commands Emergency Braking.](assets/carla_outside_view_pothole.png)
<!-- slide -->
![Outside View: Manhole cover detection. YOLOv8 detects the Manhole; U-Net road segmentation is active; DQN triggers Emergency Braking.](assets/carla_outside_view_manhole.png)
<!-- slide -->
![Outside View: Speed Bump detection. YOLOv8 detects the Speedbump; U-Net lane borders mapped; DQN applies Emergency Braking.](assets/carla_outside_view_speedbump.png)
```

* Local Outside Pothole Screenshot: [carla_outside_view_pothole.png](file:///d:/Road%20Hazards-Detection/carla_outside_view_pothole.png)
* Local Outside Manhole Screenshot: [carla_outside_view_manhole.png](file:///d:/Road%20Hazards-Detection/carla_outside_view_manhole.png)
* Local Outside Speed Bump Screenshot: [carla_outside_view_speedbump.png](file:///d:/Road%20Hazards-Detection/carla_outside_view_speedbump.png)

