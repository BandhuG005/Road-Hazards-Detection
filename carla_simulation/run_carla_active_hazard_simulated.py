#!/usr/bin/env python
"""
Runs the ADAS Pipeline on a live CARLA frame with a spawned street barrier.
Simulates a successful detection to show the full U-Net road segmentation,
YOLO detection box, and DQN deceleration action on the HUD.
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
        
        # Spawn a large street barrier at 8 meters directly in front of the vehicle
        print("[*] Spawning physical street barrier on the road...")
        forward_vec = spawn_point.get_forward_vector()
        barrier_bp = blueprint_library.find("static.prop.streetbarrier")
        barrier_loc = spawn_point.location + (forward_vec * 8.0)
        barrier_loc.z += 0.2
        barrier_transform = carla.Transform(barrier_loc, spawn_point.rotation)
        barrier = world.spawn_actor(barrier_bp, barrier_transform)
        actor_list.append(barrier)
        
        # Attach RGB Dashcam camera to vehicle
        print("[*] Attaching RGB Dashcam camera to vehicle...")
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "1024")
        camera_bp.set_attribute("image_size_y", "768")
        camera_bp.set_attribute("fov", "90")
        camera_bp.set_attribute("sensor_tick", "0.05")
        
        # Position camera inside vehicle windshield
        camera_transform = carla.Transform(
            carla.Location(x=1.6, z=1.4),
            carla.Rotation(pitch=-5.0)
        )
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(camera)
        
        # Queue to capture the image
        image_queue = queue.Queue()
        camera.listen(lambda data: image_queue.put(data))
        
        # Run simulation for a few ticks to stabilize and capture the frame
        print("[*] Ticking simulator and capturing camera frame...")
        for _ in range(15):
            world.tick()
            try:
                image_data = image_queue.get(timeout=1.0)
            except queue.Empty:
                pass
                
        # Save raw frame
        raw_array = np.frombuffer(image_data.raw_data, dtype=np.uint8)
        raw_array = np.reshape(raw_array, (image_data.height, image_data.width, 4))
        raw_bgr = raw_array[:, :, :3]
        
        # Initialize ADAS Pipeline
        print("[*] Initializing ADAS Pipeline...")
        pipeline = ADASPipeline()
        
        # We manually process the frame to inject the barrier's bounding box coordinates in the frame
        # This simulates the YOLO detection of the spawned barrier
        speed = 50.0 # Vehicle approaching at 50 km/h
        
        # 1. Preprocess
        normalized_frame, enhanced_frame = pipeline.process_frame_preprocess(raw_bgr) if hasattr(pipeline, 'process_frame_preprocess') else (None, raw_bgr)
        from src.preprocess import preprocess_pipeline
        normalized_frame, enhanced_frame = preprocess_pipeline(raw_bgr)
        
        # 2. Road Segmentation (U-Net)
        print("[*] Running U-Net Road Segmentation model...")
        seg_mask = pipeline.segmenter.run_inference(enhanced_frame)
        
        # 3. Simulate YOLO detection for the spawned street barrier
        # The barrier is in the center-bottom of the windshield view
        # Bbox: xmin=412, ymin=440, xmax=612, ymax=580
        simulated_detections = [{
            "bbox": [412.0, 440.0, 612.0, 580.0],
            "class_label": "Road_Barrier",
            "confidence_score": 0.92,
            "class_id": 6
        }]
        
        # 4. Run Risk Fusion
        print("[*] Running Spatial Risk Fusion & Decision Control...")
        processed_detections = []
        for det in simulated_detections:
            distance = estimate_distance(det["bbox"], det["class_label"])
            # Set distance to match spawned distance (approx 8.0 meters)
            distance = 8.0
            ttc = calculate_ttc(distance, speed)
            lane_factor = get_lane_alignment_factor(det["bbox"], seg_mask)
            risk_score = compute_risk_score(det["class_label"], distance, speed, lane_factor)
            risk_level = classify_risk(risk_score)
            
            processed_detections.append({
                "bbox": det["bbox"],
                "class_label": det["class_label"],
                "distance": distance,
                "ttc": ttc,
                "risk_score": risk_score,
                "risk_level": risk_level,
                "lane_alignment": lane_factor
            })
            
        # 5. Run DRL DQN Deceleration Agent
        # State vector: [Distance, Speed, Severity, Alignment]
        # Road_Barrier severity weight is 0.95
        import torch
        state = np.array([8.0, speed, 0.95, 1.0], dtype=np.float32)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            q_values = pipeline.rl_agent(state_tensor)
            action = int(torch.argmax(q_values).item())
            
        # Action mappings:
        action_meanings = {
            0: "Coast (Maintain Speed)",
            1: "Light Brake (-1.5 m/s^2)",
            2: "Medium Brake (-4.0 m/s^2)",
            3: "Emergency Brake (-7.5 m/s^2)",
            4: "Accelerate (+1.2 m/s^2)"
        }
        action_desc = action_meanings.get(action, "Brake")
        print(f"[+] DQN Agent Deceleration Command: {action_desc} (Action {action})")
        
        # 6. Render HUD Overlay
        # Draw green mask for road segmentation and overlay detection boxes
        seg_color_mask = pipeline.segmenter.get_color_mask(seg_mask)
        hud_frame = draw_hud(enhanced_frame, processed_detections, speed, seg_color_mask)
        
        # Overlay the DQN Action text on the HUD frame
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
        
        # Save output images
        output_path = os.path.join(config.LOG_DIR, "carla_active_hazard_hud.png")
        cv2.imwrite(output_path, hud_frame)
        print(f"[+] Saved annotated HUD image to: {output_path}")
        
        brain_path = None
        pass
        
    finally:
        # Restore settings and destroy actors
        world.apply_settings(original_settings)
        for actor in actor_list:
            actor.destroy()
        print("[+] Cleanup complete.")


if __name__ == "__main__":
    main()
