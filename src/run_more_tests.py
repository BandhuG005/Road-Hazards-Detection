import os
import shutil
import cv2
from src.pipeline import ADASPipeline

def run_more_tests():
    source_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    inputs_dir = os.path.join(source_dir, "logs", "inputs")
    outputs_dir = os.path.join(source_dir, "logs", "outputs")
    
    os.makedirs(inputs_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    
    # 1. Define list of natural hazard test images
    test_cases = [
        ("dust_natural.jpg", "dust_natural_output.jpg", 55.0, "Dust (Sandstorm)"),
        ("fog_natural.jpg", "fog_natural_output.jpg", 45.0, "Fog (Low Visibility)"),
        ("night_natural.jpg", "night_natural_output.jpg", 40.0, "Night (Low Light)"),
        ("rain_natural.jpg", "rain_natural_output.jpg", 45.0, "Rain (Reflective Road)"),
        ("orig_natural.jpg", "daylight_natural_output.jpg", 60.0, "Daylight (Clear)")
    ]
    
    # 2. Initialize Pipeline (Loads YOLO baseline, U-Net, and DRL agent)
    print("\n[*] Initializing ADAS Pipeline with all three modules...")
    pipeline = ADASPipeline()
    
    results = []
    
    # 3. Process each test case
    for input_name, output_name, speed, domain in test_cases:
        input_path = os.path.join(inputs_dir, input_name)
        output_path = os.path.join(outputs_dir, output_name)
        
        if not os.path.exists(input_path):
            print(f"[!] Warning: Test case {input_name} not found.")
            continue
            
        print(f"[*] Processing {domain} split image...")
        frame = cv2.imread(input_path)
        
        # Reset speed state to the target speed for this static frame
        pipeline.current_speed_kmh = None
        
        hud_frame, detections = pipeline.process_frame(frame, speed_kmh=speed)
        
        # Save output image
        cv2.imwrite(output_path, hud_frame)
        print(f"  [+] Saved HUD visual output to: {output_path}")
        
        # Log detections details
        if detections:
            for det in detections:
                results.append({
                    "domain": domain,
                    "class": det["class_label"],
                    "conf": det["confidence_score"],
                    "dist": det["distance"],
                    "risk": det["risk_score"],
                    "level": det["risk_level"],
                    "action": det["rl_action"]
                })
        else:
            results.append({
                "domain": domain,
                "class": "None",
                "conf": 0.0,
                "dist": 0.0,
                "risk": 0.0,
                "level": "LOW",
                "action": "Coast / Maintain"
            })
            
    # 4. Copy outputs to clean GitHub folder also so user has them there!
    github_outputs_dir = os.path.join(source_dir, "logs", "github_outputs")
    os.makedirs(github_outputs_dir, exist_ok=True)
    for _, output_name, _, _ in test_cases:
        src = os.path.join(outputs_dir, output_name)
        dst = os.path.join(github_outputs_dir, output_name)
        if os.path.exists(src):
            shutil.copy(src, dst)
            
    print("\n" + "=" * 80)
    print("NATURAL HAZARDS EVALUATION RESULTS (ALL MODULES WORKING TOGETHER)")
    print("=" * 80)
    print(f"{'Domain Split':<20} | {'Detected':<15} | {'Dist (m)':<9} | {'Risk Score':<11} | {'DRL Control Action':<18}")
    print("-" * 80)
    for r in results:
        print(f"{r['domain']:<20} | {r['class']:<15} | {r['dist']:<9.1f} | {r['risk']:<11.1f} | {r['action']:<18}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    run_more_tests()
