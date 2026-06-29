#!/usr/bin/env python
"""
CARLA Screenshot Capture Script.
Spawns a vehicle and an RGB camera, runs the speed profile,
and saves screenshots of the vehicle driving in Town10.
"""

import os
import sys
import time
import math
import queue
import numpy as np
import cv2

try:
    import carla
except ImportError:
    print("[!] Error: CARLA Python client library is not installed.")
    sys.exit(1)


def main():
    print("[*] Connecting to CARLA simulator...")
    try:
        client = carla.Client("127.0.0.1", 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        print("[+] Connected successfully!")
    except Exception as e:
        print(f"[!] Error connecting to CARLA: {e}")
        sys.exit(1)
        
    actor_list = []
    original_settings = world.get_settings()
    
    try:
        # Set synchronous mode
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        
        # Blueprints
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
        
        # Spawn point
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = spawn_points[0]
        
        # Spawn vehicle
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(vehicle)
        
        # Attach Camera
        print("[*] Attaching RGB Camera sensor for screenshots...")
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "1024")
        camera_bp.set_attribute("image_size_y", "768")
        camera_bp.set_attribute("fov", "90")
        camera_bp.set_attribute("sensor_tick", "0.05")
        
        # Position camera slightly behind and above the vehicle for a clear chase view
        camera_transform = carla.Transform(
            carla.Location(x=-8.0, z=3.5),
            carla.Rotation(pitch=-15.0)
        )
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(camera)
        
        # Set spectator to follow
        spectator = world.get_spectator()
        
        # Queue to hold images
        image_queue = queue.Queue()
        camera.listen(lambda image: image_queue.put(image))
        
        os.makedirs("logs", exist_ok=True)
        print("[*] Running simulation and capturing screenshots...")
        
        for step in range(250):
            # Apply control
            if step < 80:
                vehicle.apply_control(carla.VehicleControl(throttle=0.6, brake=0.0))
            elif step < 160:
                vehicle.apply_control(carla.VehicleControl(throttle=0.25, brake=0.0))
            elif step < 240:
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.8))
            else:
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                
            world.tick()
            
            # Follow with spectator
            v_trans = vehicle.get_transform()
            v_yaw = math.radians(v_trans.rotation.yaw)
            cam_loc = v_trans.location + carla.Location(
                x=-8.0 * math.cos(v_yaw),
                y=-8.0 * math.sin(v_yaw),
                z=3.5
            )
            spectator.set_transform(carla.Transform(cam_loc, carla.Rotation(pitch=-15.0, yaw=v_trans.rotation.yaw)))
            
            # Fetch image
            try:
                image_data = image_queue.get(timeout=1.0)
            except queue.Empty:
                continue
                
            # Save screenshots at key events
            # Step 40: Accelerating
            if step == 40:
                print("[-] Capturing Accelerating screenshot...")
                save_frame(image_data, "logs/carla_screenshot_accel.png")
            # Step 120: Cruising
            elif step == 120:
                print("[-] Capturing Cruising screenshot...")
                save_frame(image_data, "logs/carla_screenshot_cruise.png")
            # Step 190: Braking
            elif step == 190:
                print("[-] Capturing Braking screenshot...")
                save_frame(image_data, "logs/carla_screenshot_brake.png")
                
        print("[+] Screenshot capture completed!")
        
    finally:
        world.apply_settings(original_settings)
        for actor in actor_list:
            actor.destroy()
        print("[+] Cleanup complete.")


def save_frame(image, filename):
    # Convert raw image data to numpy BGRA array
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    # Drop alpha channel and write using OpenCV
    cv2.imwrite(filename, array[:, :, :3])
    print(f"[+] Saved screenshot to {filename}")


if __name__ == "__main__":
    main()
