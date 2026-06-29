import os
import sys
import argparse
import cv2
import time

# Ensure we can import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from src.pipeline import ADASPipeline
import src.config as config

def main():
    parser = argparse.ArgumentParser(description="Run ADAS Segmentation + Crop-Detection Hybrid Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Path to input image or video file")
    parser.add_argument("--output", type=str, default=None, help="Path to save output. If None, defaults to logs/output_...")
    parser.add_argument("--speed", type=float, default=50.0, help="Simulated vehicle speed in km/h")
    parser.add_argument("--weights", type=str, default=None, help="Path to custom YOLOv8 weights (e.g., models/best_1.pt). Defaults to models/best.pt")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"[!] Error: Input file not found: {args.input}")
        return
        
    # Initialize pipeline with specified weights
    pipeline = ADASPipeline(yolo_model_path=args.weights)
    
    # Check if input is video or image
    is_video = args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    
    # Create default output paths
    os.makedirs("logs", exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    
    if is_video:
        output_path = args.output if args.output else os.path.join("logs", f"{base_name}_hud.mp4")
        t0 = time.time()
        pipeline.run_on_video(args.input, output_path, display=False, speed_kmh=args.speed)
        print(f"[+] Video processing complete in {time.time() - t0:.1f}s.")
        print(f"[+] Output HUD video saved to: {output_path}")
    else:
        # Load single image
        frame = cv2.imread(args.input)
        if frame is None:
            print(f"[!] Error: Could not read image: {args.input}")
            return
            
        output_path = args.output if args.output else os.path.join("logs", f"{base_name}_hud.jpg")
        print(f"[*] Processing image: {args.input}...")
        
        t0 = time.time()
        hud_frame, detections = pipeline.process_frame(frame, speed_kmh=args.speed)
        print(f"[+] Image processed in {(time.time() - t0)*1000:.1f}ms.")
        
        cv2.imwrite(output_path, hud_frame)
        print(f"[+] Output HUD image saved to: {output_path}")
        print("\n=== Detected Hazards ===")
        if detections:
            for idx, det in enumerate(detections):
                print(f"[{idx+1}] {det['class_label']} | Confidence: {det['confidence_score']:.2f} | Distance: {det['distance']}m | Risk: {det['risk_score']} ({det['risk_level']})")
        else:
            print("No hazards detected.")

if __name__ == "__main__":
    main()
