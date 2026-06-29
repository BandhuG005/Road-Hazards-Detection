#!/usr/bin/env python
"""
Processes a real pothole road hazard image through the ADAS Pipeline.
Demonstrates the U-Net road mask, YOLOv8 pothole detection box,
and the DRL DQN controller's deceleration action output on the HUD.
"""

import os
import sys
import cv2

# Ensure we can import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import ADASPipeline
import src.config as config


def main():
    brain_dir = os.path.join(config.LOG_DIR, "inputs")
    input_path = os.path.join(brain_dir, "media__1782124279674.jpg")
    output_path = os.path.join(config.LOG_DIR, "pothole_hud_implementation.png")
    
    if not os.path.exists(input_path):
        print(f"[!] Error: Test image not found at {input_path}")
        sys.exit(1)
        
    print("[*] Initializing ADAS Pipeline...")
    pipeline = ADASPipeline()
    
    print(f"[*] Processing pothole hazard image at 50.0 km/h speed...")
    frame = cv2.imread(input_path)
    if frame is None:
        print("[!] Error reading frame.")
        sys.exit(1)
        
    # Run the pipeline
    hud_frame, detections = pipeline.process_frame(frame, speed_kmh=50.0)
    
    # Save to workspace root
    cv2.imwrite(output_path, hud_frame)
    print(f"[+] Saved annotated HUD image to: {output_path}")
    
    # Save copy to brain folder
    # Optional brain copy
    if os.path.exists(brain_dir):
        brain_path = os.path.join(brain_dir, "pothole_hud_implementation.png")
        cv2.imwrite(brain_path, hud_frame)
        print(f"[+] Saved copy to: {brain_path}")
    
    print("\n=== Active Hazard Tracking ===")
    if detections:
        for idx, det in enumerate(detections):
            print(f"[{idx+1}] {det['class_label']} | Distance: {det['distance']}m | Alignment: {det.get('lane_alignment', 1.0)} | Risk Score: {det['risk_score']} ({det['risk_level']})")
    else:
        print("No hazards detected.")


if __name__ == "__main__":
    main()
