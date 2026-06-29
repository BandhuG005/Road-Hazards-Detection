#!/usr/bin/env python
"""
Runs the ADAS Pipeline on the captured CARLA screenshots.
Saves the annotated frames containing the green road lane segmentation,
YOLO detections (if any), and active telemetry HUD.
"""

import os
import sys
import cv2

# Ensure we can import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import ADASPipeline
import src.config as config


def main():
    print("[*] Initializing ADAS Pipeline...")
    try:
        pipeline = ADASPipeline()
        print("[+] Pipeline initialized successfully!")
    except Exception as e:
        print(f"[!] Error initializing pipeline: {e}")
        sys.exit(1)
        
    frames_to_process = [
        ("logs/carla_screenshot_accel.png", os.path.join(config.LOG_DIR, "carla_screenshot_accel_hud.png"), 20.0),
        ("logs/carla_screenshot_cruise.png", os.path.join(config.LOG_DIR, "carla_screenshot_cruise_hud.png"), 30.0),
        ("logs/carla_screenshot_brake.png", os.path.join(config.LOG_DIR, "carla_screenshot_brake_hud.png"), 10.0)
    ]
    
    brain_dir = r"C:\Users\user\.gemini\antigravity\brain\f6794dcd-3648-4e74-9f97-9cb90bdae7b6"
    if not os.path.exists(brain_dir):
        brain_dir = None
    
    for input_path, output_path, speed in frames_to_process:
        if not os.path.exists(input_path):
            print(f"[!] Warning: Input file not found: {input_path}")
            continue
            
        print(f"[*] Processing frame {input_path} at speed {speed} km/h...")
        frame = cv2.imread(input_path)
        if frame is None:
            print(f"[!] Error: Could not read image: {input_path}")
            continue
            
        # Run ADAS pipeline (which overlays U-Net segmentation and HUD)
        hud_frame, detections = pipeline.process_frame(frame, speed_kmh=speed)
        
        # Save to workspace root
        cv2.imwrite(output_path, hud_frame)
        print(f"[+] Saved annotated frame to: {output_path}")
        
        # Save copy to brain folder for display
        if brain_dir is not None and os.path.exists(brain_dir):
            brain_path = os.path.join(brain_dir, os.path.basename(output_path))
            cv2.imwrite(brain_path, hud_frame)
            print(f"[+] Saved copy to: {brain_path}")
        
        if detections:
            print(f"    Hazard detections present: {len(detections)}")
        else:
            print("    No hazards detected in this frame.")
            
    print("[+] All CARLA frames successfully processed!")


if __name__ == "__main__":
    main()
