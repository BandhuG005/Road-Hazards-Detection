import time
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import src.config as config
from src.preprocess import preprocess_pipeline
from src.segmentation import RoadSegmenter
from src.yolo_detector import YOLODetector
from src.distance_risk import (
    estimate_distance, 
    calculate_ttc, 
    get_lane_alignment_factor, 
    compute_risk_score
)
from src.alerts_logging import (
    classify_risk, 
    trigger_audio_alert, 
    DataLogger, 
    draw_hud
)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class ADASPipeline:
    """
    Main ADAS Processing Pipeline.
    Integrates Preprocessing, Segmentation, YOLO Detection, Distance Estimation, 
    Risk Assessment, Alerts, and Data Logging.
    """
    def __init__(self, yolo_model_path=None, seg_model_path=None, ensemble=True, use_rl=True):
        self.segmenter = RoadSegmenter(seg_model_path)
        self.detector = YOLODetector(yolo_model_path, ensemble=ensemble)
        self.logger = DataLogger()
        self.use_rl = use_rl
        self.rl_agent = None
        self.current_speed_kmh = None
        
        if self.use_rl:
            rl_path = os.path.join(config.MODEL_DIR, "rl_deceleration_agent.pth")
            if os.path.exists(rl_path):
                try:
                    self.rl_agent = QNetwork(state_dim=4, action_dim=5)
                    self.rl_agent.load_state_dict(torch.load(rl_path, map_location="cpu", weights_only=True))
                    self.rl_agent.eval()
                    print("[+] Trained DRL Deceleration Agent loaded successfully into ADAS Pipeline!")
                except Exception as e:
                    print(f"[!] Warning: Failed to load DRL agent ({e}). Falling back to rule-based control.")
                    self.use_rl = False
            else:
                print("[!] DRL agent weights not found. Falling back to rule-based control.")
                self.use_rl = False
                
        print("[*] ADAS Pipeline initialized successfully!")

    def process_frame(self, frame_raw, speed_kmh=None, gps_coords=(12.9716, 77.5946)):
        """
        Processes a single input frame through the entire pipeline.
        Returns the annotated HUD frame and the list of detected hazards.
        """
        # Time step per frame (approx 100ms for real-time video simulation)
        dt = 0.1
        
        # Initialize current speed state variable
        target_speed = speed_kmh if speed_kmh is not None else config.DEFAULT_SPEED_KMH
        if self.current_speed_kmh is None:
            self.current_speed_kmh = target_speed
            
        # 1 & 2. Preprocess frame
        normalized_frame, enhanced_frame = preprocess_pipeline(frame_raw)
        
        # 3. Road Segmentation (Drivable Area Mask)
        seg_mask = self.segmenter.run_inference(enhanced_frame)
        
        # 4. YOLOv8 Hazard Detection with Slicing-Aided Hyper Inference (LH-SAHI)
        raw_detections = self.detector.detect_hazards_sahi(enhanced_frame)
        
        # Filter out Construction_Object and Traffic_Cone classes as requested
        raw_detections = [d for d in raw_detections if d["class_label"] not in ["Construction_Object", "Traffic_Cone"]]
        
        # Default acceleration if no hazards: accelerate back to target_speed
        min_accel = 1.5
        
        processed_detections = []
        for det in raw_detections:
            # 8. Distance and TTC estimation
            distance = estimate_distance(det["bbox"], det["class_label"])
            ttc = calculate_ttc(distance, self.current_speed_kmh)
            
            # 9. Risk Assessment
            lane_factor = get_lane_alignment_factor(det["bbox"], seg_mask)
            risk_score = compute_risk_score(det["class_label"], distance, self.current_speed_kmh, lane_factor)
            
            # 10. Decision Engine (Rule-based)
            risk_level = classify_risk(risk_score)
            
            # 10b. DRL Control Inference
            rl_action_name = "N/A"
            if self.use_rl and self.rl_agent is not None:
                severity_val = config.SEVERITY_WEIGHTS.get(det["class_label"], 0.5)
                speed_mps = self.current_speed_kmh / 3.6
                
                # State vector: [distance, speed_mps, severity, lane_factor]
                state_vec = np.array([distance, speed_mps, severity_val, lane_factor], dtype=np.float32)
                state_t = torch.FloatTensor(state_vec).unsqueeze(0)
                
                with torch.no_grad():
                    q_val = self.rl_agent(state_t)
                    action_idx = torch.argmax(q_val).item()
                    
                action_names = {
                    0: "Coast / Maintain",
                    1: "Light Brake",
                    2: "Medium Brake",
                    3: "Heavy Brake",
                    4: "Accelerate"
                }
                rl_action_name = action_names.get(action_idx, "Coast / Maintain")
                
                # Map action index to deceleration (m/s^2)
                accel_mapping = {
                    0: -0.2,   # Coast
                    1: -1.5,   # Light Brake
                    2: -4.0,   # Medium Brake
                    3: -7.5,   # Heavy Brake
                    4: 1.2     # Accelerate
                }
                accel_val = accel_mapping.get(action_idx, 0.0)
                
                # Only apply control deceleration if hazard lies in our path (lane_factor > 0.4)
                if lane_factor > 0.4:
                    min_accel = min(min_accel, accel_val)
            
            # Build finalized hazard info
            hazard_info = {
                "bbox": det["bbox"],
                "class_label": det["class_label"],
                "confidence_score": det["confidence_score"],
                "distance": distance,
                "ttc": ttc,
                "risk_score": risk_score,
                "risk_level": risk_level,
                "rl_action": rl_action_name
            }
            
            processed_detections.append(hazard_info)
            
            # 12. Data Logging (CSV & Snapshots for high-risk)
            self.logger.log_hazard(frame_raw, hazard_info, self.current_speed_kmh, gps_coords)
            
            # 11. Driver Alert System (Audio Warnings)
            trigger_audio_alert(det["class_label"], distance, risk_level)
            
        # Update speed state dynamically based on DRL agent commands
        if self.use_rl and self.rl_agent is not None:
            # If no hazards detected, accelerate gently back to target speed
            if not raw_detections:
                min_accel = 1.5
                
            self.current_speed_kmh = max(0.0, min(target_speed, self.current_speed_kmh + (min_accel * 3.6) * dt))
            
        # 11. Driver Alert System (Visual HUD Overlay)
        seg_color_mask = self.segmenter.get_color_mask(seg_mask)
        hud_frame = draw_hud(enhanced_frame, processed_detections, self.current_speed_kmh, seg_color_mask)
        
        return hud_frame, processed_detections

    def run_on_video(self, video_path, output_path=None, display=False, speed_kmh=config.DEFAULT_SPEED_KMH):
        """
        Runs the ADAS pipeline on a video file.
        Saves the resulting HUD overlay video if output_path is provided.
        """
        self.current_speed_kmh = None
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[!] Error: Could not open video file {video_path}")
            return
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[*] Processing video: {video_path} ({width}x{height} @ {fps} fps, {total_frames} frames)")
        
        out = None
        if output_path:
            # Dynamically select codec based on file extension
            if output_path.lower().endswith('.avi'):
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            else:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps if fps > 0 else 30.0, (config.IMG_SIZE, config.IMG_SIZE))
            print(f"[*] Saving output video to: {output_path}")

        frame_idx = 0
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            
            # Dynamically simulate small speed fluctuations depending on frame context
            # (e.g. speed decreases slowly if a hazard is detected to show ADAS behavior)
            hud_frame, detections = self.process_frame(frame, speed_kmh)
            
            # Write to output file
            if out:
                out.write(hud_frame)
                
            # Print progress
            if frame_idx % 30 == 0 or frame_idx == total_frames:
                elapsed = time.time() - start_time
                avg_fps = frame_idx / elapsed
                print(f"[*] Frame {frame_idx}/{total_frames} processed | Avg Speed: {avg_fps:.1f} FPS")
                
            if display:
                cv2.imshow("ADAS Pipeline Preview", hud_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        cap.release()
        if out:
            out.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print(f"[+] Video processing complete! Output saved to {output_path}")
