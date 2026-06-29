#!/usr/bin/env python
"""
Spawns a vehicle and physical road hazards (Traffic Cones / Barriers) in CARLA,
captures the camera view, runs the ADAS pipeline (YOLOv8 + U-Net + DQN),
and saves the annotated HUD output to demonstrate active detection and speed management.
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
        print(f"[*] Spawning vehicle...")
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(vehicle)
        
        # Spawn physical hazards (Barriers and Cones) closer to the vehicle (5-8m)
        # to ensure they are large and clearly visible in the dashcam view
        print("[*] Spawning physical road hazards (Street Barrier & Cones) in driving lane...")
        
        # Calculate forward vector and right vector for lateral offset
        forward_vec = spawn_point.get_forward_vector()
        yaw_rad = math.radians(spawn_point.rotation.yaw)
        right_vec = carla.Vector3D(-math.sin(yaw_rad), math.cos(yaw_rad), 0.0)
        
        # 1. Spawn a large street barrier at 8 meters
        barrier_bp = blueprint_library.find("static.prop.streetbarrier")
        barrier_loc = spawn_point.location + (forward_vec * 7.5)
        barrier_loc.z += 0.2
        barrier_transform = carla.Transform(barrier_loc, spawn_point.rotation)
        barrier = world.spawn_actor(barrier_bp, barrier_transform)
        actor_list.append(barrier)
        
        # 2. Spawn two construction cones flanking it
        cone_bp = blueprint_library.find("static.prop.constructioncone")
        
        cone_loc1 = spawn_point.location + (forward_vec * 6.0) + (right_vec * -0.6)
        cone_loc1.z += 0.15
        cone1 = world.spawn_actor(cone_bp, carla.Transform(cone_loc1, spawn_point.rotation))
        actor_list.append(cone1)
        
        cone_loc2 = spawn_point.location + (forward_vec * 6.2) + (right_vec * 0.6)
        cone_loc2.z += 0.15
        cone2 = world.spawn_actor(cone_bp, carla.Transform(cone_loc2, spawn_point.rotation))
        actor_list.append(cone2)
            
        # Attach RGB Camera positioned as a vehicle dashcam
        print("[*] Attaching RGB Dashcam camera to vehicle...")
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "1024")
        camera_bp.set_attribute("image_size_y", "768")
        camera_bp.set_attribute("fov", "90")
        camera_bp.set_attribute("sensor_tick", "0.05")
        
        # Position camera inside vehicle windshield (x=1.6m forward, z=1.4m height)
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
        print("[*] Capturing camera frame from simulator...")
        for _ in range(10):
            world.tick()
            try:
                image_data = image_queue.get(timeout=1.0)
            except queue.Empty:
                pass
                
        # Save raw frame
        raw_array = np.frombuffer(image_data.raw_data, dtype=np.uint8)
        raw_array = np.reshape(raw_array, (image_data.height, image_data.width, 4))
        raw_bgr = raw_array[:, :, :3]
        
        # 7. Initialize ADAS Pipeline and process the raw simulator frame
        print("[*] Running ADAS Pipeline (YOLOv8 + U-Net + DQN) on the CARLA camera frame...")
        pipeline = ADASPipeline()
        
        # Assume vehicle is approaching at 50 km/h speed
        speed = 50.0
        hud_frame, detections = pipeline.process_frame(raw_bgr, speed_kmh=speed)
        
        # Save outputs
        output_path = os.path.join(config.LOG_DIR, "carla_active_hazard_hud.png")
        cv2.imwrite(output_path, hud_frame)
        print(f"[+] Saved annotated HUD image to: {output_path}")
        
        brain_path = None
        pass
        
        print("\n=== CARLA Pipeline Detections ===")
        if detections:
            for idx, det in enumerate(detections):
                print(f"[{idx+1}] {det['class_label']} | Distance: {det['distance']}m | Risk: {det['risk_score']} ({det['risk_level']})")
        else:
            print("No hazards detected.")
            
    finally:
        world.apply_settings(original_settings)
        for actor in actor_list:
            actor.destroy()
        print("[+] Cleanup complete.")


if __name__ == "__main__":
    main()
