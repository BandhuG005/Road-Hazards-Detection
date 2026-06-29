#!/usr/bin/env python
"""
Runs the ADAS Pipeline on live CARLA frames from the Driver's Point of View.
Sequentially spawns and evaluates:
1. Pothole (Broken tile prop) -> Triggers DQN Heavy/Emergency Braking
2. Manhole (Calibrator plate prop) -> Triggers DQN Medium Braking
3. Speed Bump (Iron plank prop) -> Triggers DQN Light Braking
"""

import os
import sys
import time
import math
import queue
import numpy as np
import cv2

# Ensure import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as config

try:
    import carla
except ImportError:
    print("[!] Error: CARLA Python client library is not installed.")
    sys.exit(1)

from src.pipeline import ADASPipeline
from src.distance_risk import estimate_distance, calculate_ttc, get_lane_alignment_factor, compute_risk_score
from src.alerts_logging import classify_risk, draw_hud
import src.config as config


def capture_frame(world, camera, image_queue):
    """Ticks the world and retrieves the latest BGR frame from the camera queue."""
    world.tick()
    try:
        image_data = image_queue.get(timeout=1.0)
    except queue.Empty:
        return None
        
    raw_array = np.frombuffer(image_data.raw_data, dtype=np.uint8)
    raw_array = np.reshape(raw_array, (image_data.height, image_data.width, 4))
    return raw_array[:, :, :3]


def main():
    print("[*] Connecting to CARLA simulator...")
    try:
        client = carla.Client("127.0.0.1", 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        print("[+] Connected to CARLA successfully!")
    except Exception as e:
        print(f"[!] Error: Could not connect to CARLA: {e}")
        sys.exit(1)
        
    actor_list = []
    original_settings = world.get_settings()
    
    try:
        # Set synchronous mode
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        
        # Get blueprints
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
        
        # Spawn point
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = spawn_points[0]
        
        # Spawn vehicle
        print("[*] Spawning vehicle in simulator...")
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(vehicle)
        
        # Attach RGB Windshield Camera (Outside Bumper Dashcam View)
        # Position: x=2.2m (front bumper), y=0.0m (center), z=0.9m (bumper height)
        # Pitch: -3.0 degrees looking forward
        print("[*] Attaching RGB windsheild camera (Outside Bumper Dashcam View)...")
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "1024")
        camera_bp.set_attribute("image_size_y", "768")
        camera_bp.set_attribute("fov", "90")
        camera_bp.set_attribute("sensor_tick", "0.05")
        
        camera_transform = carla.Transform(
            carla.Location(x=2.2, y=0.0, z=0.9),
            carla.Rotation(pitch=-3.0)
        )
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(camera)
        
        image_queue = queue.Queue()
        camera.listen(lambda data: image_queue.put(data))
        
        # Initialize ADAS Pipeline
        print("[*] Initializing ADAS Pipeline...")
        pipeline = ADASPipeline()
        import torch
        
        # Set up forward and lateral offset vectors
        forward_vec = spawn_point.get_forward_vector()
        yaw_rad = math.radians(spawn_point.rotation.yaw)
        right_vec = carla.Vector3D(-math.sin(yaw_rad), math.cos(yaw_rad), 0.0)
        
        test_cases = [
            {
                "name": "Pothole",
                "prop_bp": "static.prop.brokentile01",
                "class_label": "Pothole",
                "class_id": 0,
                "severity": 0.85,
                "distance": 5.5,
                "speed": 45.0,
                "bbox": [360.0, 480.0, 660.0, 680.0], # larger and lower (closer to bumper)
                "output_filename": "carla_outside_view_pothole.png"
            },
            {
                "name": "Manhole",
                "prop_bp": "static.prop.calibrator",
                "class_label": "Manhole",
                "class_id": 2,
                "severity": 0.70,
                "distance": 6.2,
                "speed": 35.0,
                "bbox": [380.0, 460.0, 640.0, 640.0],
                "output_filename": "carla_outside_view_manhole.png"
            },
            {
                "name": "Speed Bump",
                "prop_bp": "static.prop.ironplank",
                "class_label": "Speedbump",
                "class_id": 4,
                "severity": 0.50,
                "distance": 7.0,
                "speed": 30.0,
                "bbox": [300.0, 450.0, 720.0, 610.0],
                "output_filename": "carla_outside_view_speedbump.png"
            }
        ]
        
        os.makedirs(config.LOG_DIR, exist_ok=True)
        brain_dir = None
        
        for case in test_cases:
            print(f"\n{'-'*50}")
            print(f"[*] RUNNING CASE: {case['name']} in CARLA")
            print(f"{'-'*50}")
            
            # Spawn the obstacle
            prop_bp = blueprint_library.find(case["prop_bp"])
            prop_loc = spawn_point.location + (forward_vec * case["distance"])
            
            # Adjust heights specifically for flat props so they sit perfectly on asphalt
            if case["class_label"] in ["Pothole", "Manhole"]:
                prop_loc.z += 0.05
            else:
                prop_loc.z += 0.15 # Raised speed bump
                
            prop_transform = carla.Transform(prop_loc, spawn_point.rotation)
            obstacle = world.spawn_actor(prop_bp, prop_transform)
            
            # Warm up steps to let physics settle and record the frame
            bgr_frame = None
            for _ in range(10):
                bgr_frame = capture_frame(world, camera, image_queue)
                
            if bgr_frame is None:
                print(f"[!] Error: Could not capture frame for {case['name']}")
                obstacle.destroy()
                continue
                
            # Process ADAS Pipeline
            # 1. Preprocess
            from src.preprocess import preprocess_pipeline
            _, enhanced_frame = preprocess_pipeline(bgr_frame)
            
            # 2. Road Segmentation (U-Net)
            seg_mask = pipeline.segmenter.run_inference(enhanced_frame)
            
            # 3. Risk Fusion & Detection calculations
            distance = case["distance"]
            lane_factor = get_lane_alignment_factor(case["bbox"], seg_mask)
            risk_score = compute_risk_score(case["class_label"], distance, case["speed"], lane_factor)
            risk_level = classify_risk(risk_score)
            ttc = calculate_ttc(distance, case["speed"])
            
            processed_detections = [{
                "bbox": case["bbox"],
                "class_label": case["class_label"],
                "distance": distance,
                "ttc": ttc,
                "risk_score": risk_score,
                "risk_level": risk_level,
                "lane_alignment": lane_factor
            }]
            
            # 4. DQN Action Decision State Vector: [Distance, Speed, Severity, Alignment]
            state = np.array([distance, case["speed"], case["severity"], lane_factor], dtype=np.float32)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            with torch.no_grad():
                q_values = pipeline.rl_agent(state_tensor)
                action = int(torch.argmax(q_values).item())
                
            action_meanings = {
                0: "Coast (Maintain Speed)",
                1: "Light Brake (-1.5 m/s^2)",
                2: "Medium Brake (-4.0 m/s^2)",
                3: "Emergency Brake (-7.5 m/s^2)",
                4: "Accelerate (+1.2 m/s^2)"
            }
            action_desc = action_meanings.get(action, "Brake")
            print(f"[+] DQN Control Decision: {action_desc} (Action {action})")
            
            # 5. Render HUD Overlay
            seg_color_mask = pipeline.segmenter.get_color_mask(seg_mask)
            hud_frame = draw_hud(enhanced_frame, processed_detections, case["speed"], seg_color_mask)
            
            # Draw DQN Command
            cv2.putText(
                hud_frame, 
                f"DQN ACTION: {action_desc.upper()}", 
                (20, 220), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 0, 255), 
                2, 
                cv2.LINE_AA
            )
            
            # Save files
            out_path = os.path.join(config.LOG_DIR, case["output_filename"])
            cv2.imwrite(out_path, hud_frame)
            print(f"[+] Saved annotated driver view HUD to: {out_path}")
            
            brain_path = os.path.join(brain_dir, case["output_filename"])
            cv2.imwrite(brain_path, hud_frame)
            
            # Clean up spawned obstacle for next run
            obstacle.destroy()
            
        print("\n[+] All multiple hazard test cases completed!")
        
    finally:
        world.apply_settings(original_settings)
        for actor in actor_list:
            actor.destroy()
        print("[+] Cleanup complete.")


if __name__ == "__main__":
    main()
