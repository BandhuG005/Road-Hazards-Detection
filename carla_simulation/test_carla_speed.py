#!/usr/bin/env python
"""
CARLA GPS and IMU Speed Estimation and Sensor Fusion Test Script.

Attaches GNSS and IMU sensors to a spawned vehicle, drives it in a controlled
acceleration/deceleration profile, estimates speed from raw sensors, applies
Complementary Filter fusion, and compares everything against CARLA Ground Truth.
"""

import os
import sys
import time
import math
import queue
import csv
import numpy as np
import matplotlib.pyplot as plt

# Ensure CARLA library is available
try:
    import carla
except ImportError:
    print("[!] Error: CARLA Python client library is not installed.")
    print("Please install it using: pip install <path-to-wheel>")
    sys.exit(1)


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Computes the great-circle distance between two GPS coordinates in meters.
    """
    R = 6371000.0  # Earth's radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    
    a = (math.sin(dphi / 2.0) ** 2 + 
         math.cos(phi1) * math.cos(phi2) * (math.sin(dlambda / 2.0) ** 2))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def main():
    print("=" * 60)
    print("[*] CARLA GPS & IMU Speed Fusion Test")
    print("=" * 60)
    
    # 1. Connect to CARLA server
    try:
        print("[*] Connecting to CARLA simulator on localhost:2000...")
        client = carla.Client("127.0.0.1", 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        print("[+] Connected to CARLA successfully!")
    except Exception as e:
        print(f"[!] Error: Could not connect to CARLA: {e}")
        print("[!] Please verify that CarlaUE4.exe is running in another window.")
        sys.exit(1)
        
    actor_list = []
    original_settings = world.get_settings()
    
    try:
        # 2. Configure synchronous mode for stable tick-based sensor recording
        print("[*] Setting simulation to synchronous mode (dt = 0.05s)...")
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        
        # Get blueprints
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
        
        # Find spawn point
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points found on the current map.")
        spawn_point = spawn_points[0]
        
        # 3. Spawn vehicle
        print(f"[*] Spawning vehicle at: {spawn_point.location}...")
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(vehicle)
        
        # Set spectator camera to follow vehicle
        spectator = world.get_spectator()
        spec_transform = carla.Transform(
            spawn_point.location + carla.Location(x=-8.0, z=4.0),
            carla.Rotation(pitch=-15.0, yaw=spawn_point.rotation.yaw)
        )
        spectator.set_transform(spec_transform)
        
        # 4. Attach GNSS (GPS) Sensor
        print("[*] Attaching GNSS (GPS) sensor...")
        gnss_bp = blueprint_library.find("sensor.other.gnss")
        # Configure standard noise levels
        gnss_bp.set_attribute("noise_lat_stddev", "0.00000002")  # ~0.2m noise
        gnss_bp.set_attribute("noise_lon_stddev", "0.00000002")
        gnss_bp.set_attribute("sensor_tick", "0.05")
        gnss_transform = carla.Transform(carla.Location(x=1.0, z=1.8))
        gnss = world.spawn_actor(gnss_bp, gnss_transform, attach_to=vehicle)
        actor_list.append(gnss)
        
        # 5. Attach IMU Sensor
        print("[*] Attaching IMU sensor...")
        imu_bp = blueprint_library.find("sensor.other.imu")
        imu_bp.set_attribute("noise_accel_stddev_x", "0.05")  # ~0.05 m/s^2 noise
        imu_bp.set_attribute("sensor_tick", "0.05")
        imu_transform = carla.Transform(carla.Location(x=0.0, z=1.0))
        imu = world.spawn_actor(imu_bp, imu_transform, attach_to=vehicle)
        actor_list.append(imu)
        
        # 6. Set up sensor queues
        gnss_queue = queue.Queue()
        imu_queue = queue.Queue()
        gnss.listen(lambda data: gnss_queue.put(data))
        imu.listen(lambda data: imu_queue.put(data))
        
        # Warmup simulator steps
        print("[*] Warming up simulator...")
        for _ in range(10):
            world.tick()
            try:
                gnss_queue.get_nowait()
                imu_queue.get_nowait()
            except queue.Empty:
                pass
                
        # Data recording variables
        dt = 0.05
        steps = 300  # 15 seconds of test
        
        time_steps = []
        gt_speeds = []
        gps_speeds = []
        gps_speeds_filtered = []
        imu_speeds = []
        fused_speeds = []
        raw_accelerations = []
        
        # Filter / Integration states
        last_gnss = None
        current_imu_speed = 0.0
        current_fused_speed = 0.0
        filtered_gps_speed = 0.0
        
        print("\n[*] Starting Speed Test Profile...")
        print("    - Steps 0-80:   Accelerate (Throttle: 0.6)")
        print("    - Steps 80-160: Cruise (Throttle: 0.25)")
        print("    - Steps 160-240: Brake hard (Brake: 0.8)")
        print("    - Steps 240-300: Stationary (Brake: 1.0)\n")
        
        for step in range(steps):
            # Apply vehicle control based on step profile
            if step < 80:
                vehicle.apply_control(carla.VehicleControl(throttle=0.6, brake=0.0))
            elif step < 160:
                vehicle.apply_control(carla.VehicleControl(throttle=0.25, brake=0.0))
            elif step < 240:
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.8))
            else:
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                
            # Tick the simulator
            world.tick()
            
            # Position spectator behind the vehicle dynamically
            v_trans = vehicle.get_transform()
            v_yaw = math.radians(v_trans.rotation.yaw)
            cam_loc = v_trans.location + carla.Location(
                x=-8.0 * math.cos(v_yaw),
                y=-8.0 * math.sin(v_yaw),
                z=3.5
            )
            spectator.set_transform(carla.Transform(cam_loc, carla.Rotation(pitch=-15.0, yaw=v_trans.rotation.yaw)))
            
            # Retrieve sensor measurements
            try:
                gnss_data = gnss_queue.get(timeout=1.0)
                imu_data = imu_queue.get(timeout=1.0)
            except queue.Empty:
                print(f"[!] Warning: Missed sensor frames at step {step}")
                continue
                
            # A. Ground Truth Speed (Magnitude of velocity vector in m/s)
            vel = vehicle.get_velocity()
            speed_gt = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
            
            # B. GPS Speed (Haversine distance divided by dt)
            if last_gnss is not None:
                dist = haversine_distance(
                    last_gnss.latitude, last_gnss.longitude,
                    gnss_data.latitude, gnss_data.longitude
                )
                speed_gps = dist / dt
            else:
                speed_gps = 0.0
                
            # Apply low-pass filter to raw GPS speed to remove jitter
            filtered_gps_speed = 0.85 * filtered_gps_speed + 0.15 * speed_gps
            last_gnss = gnss_data
            
            # C. IMU Speed (Direct integration of forward acceleration)
            # local forward acceleration is in imu_data.accelerometer.x
            ax = imu_data.accelerometer.x
            current_imu_speed = max(0.0, current_imu_speed + ax * dt)
            
            # D. Complementary Sensor Fusion
            # Integrates high-frequency IMU acceleration but corrects drift using GNSS absolute speed
            alpha = 0.98
            current_fused_speed = max(0.0, alpha * (current_fused_speed + ax * dt) + (1.0 - alpha) * speed_gps)
            
            # Record telemetry
            time_steps.append(step * dt)
            gt_speeds.append(speed_gt)
            gps_speeds.append(speed_gps)
            gps_speeds_filtered.append(filtered_gps_speed)
            imu_speeds.append(current_imu_speed)
            fused_speeds.append(current_fused_speed)
            raw_accelerations.append(ax)
            
            if step % 50 == 0:
                print(f"Step {step:03d} | GT: {speed_gt*3.6:5.1f} km/h | GPS (Filt): {filtered_gps_speed*3.6:5.1f} km/h | IMU: {current_imu_speed*3.6:5.1f} km/h | Fused: {current_fused_speed*3.6:5.1f} km/h")
                
        print("\n[+] Speed profile completed successfully!")
        
        # 7. Calculate Error Metrics (RMSE)
        # Convert all to km/h for standard automotive comparison
        gt_arr = np.array(gt_speeds) * 3.6
        gps_arr = np.array(gps_speeds_filtered) * 3.6
        imu_arr = np.array(imu_speeds) * 3.6
        fused_arr = np.array(fused_speeds) * 3.6
        
        rmse_gps = np.sqrt(np.mean((gt_arr - gps_arr) ** 2))
        rmse_imu = np.sqrt(np.mean((gt_arr - imu_arr) ** 2))
        rmse_fused = np.sqrt(np.mean((gt_arr - fused_arr) ** 2))
        
        print("=" * 60)
        print("SPEED ESTIMATION ACCURACY (RMSE against Ground Truth in km/h)")
        print("=" * 60)
        print(f"GNSS (Filtered) Speed RMSE:  {rmse_gps:.3f} km/h")
        print(f"IMU Integrated Speed RMSE:  {rmse_imu:.3f} km/h (Drift active!)")
        print(f"Fused Complementary RMSE:    {rmse_fused:.3f} km/h")
        print("=" * 60)
        
        # 8. Save CSV telemetry logs
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        csv_path = os.path.join(log_dir, "carla_speed_data.csv")
        print(f"[*] Saving telemetry data to {csv_path}...")
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Time (s)", "Ground Truth (km/h)", "GPS Filtered (km/h)", "IMU Integrated (km/h)", "Fused Speed (km/h)", "Accel X (m/s^2)"])
            for i in range(len(time_steps)):
                writer.writerow([
                    f"{time_steps[i]:.2f}",
                    f"{gt_speeds[i]*3.6:.2f}",
                    f"{gps_speeds_filtered[i]*3.6:.2f}",
                    f"{imu_speeds[i]*3.6:.2f}",
                    f"{fused_speeds[i]*3.6:.2f}",
                    f"{raw_accelerations[i]:.3f}"
                ])
                
        # 9. Plot results
        chart_path = os.path.join(log_dir, "carla_speed_test.png")
        print(f"[*] Saving comparison chart to {chart_path}...")
        
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        
        # Subplot 1: GT vs Fused
        axs[0, 0].plot(time_steps, gt_arr, label="Ground Truth", color="#10b981", linewidth=2.5)
        axs[0, 0].plot(time_steps, fused_arr, label="Fused (Complementary)", color="#3b82f6", linestyle="--", linewidth=2.0)
        axs[0, 0].set_title(f"Fused Speed vs Ground Truth (RMSE: {rmse_fused:.2f} km/h)")
        axs[0, 0].set_ylabel("Speed (km/h)")
        axs[0, 0].grid(True, linestyle="--", alpha=0.5)
        axs[0, 0].legend()
        
        # Subplot 2: GT vs GPS
        axs[0, 1].plot(time_steps, gt_arr, label="Ground Truth", color="#10b981", linewidth=2.5)
        axs[0, 1].plot(time_steps, gps_arr, label="GPS Filtered", color="#f59e0b", linestyle="-.", linewidth=2.0)
        axs[0, 1].set_title(f"GPS Filtered vs Ground Truth (RMSE: {rmse_gps:.2f} km/h)")
        axs[0, 1].set_ylabel("Speed (km/h)")
        axs[0, 1].grid(True, linestyle="--", alpha=0.5)
        axs[0, 1].legend()
        
        # Subplot 3: GT vs IMU
        axs[1, 0].plot(time_steps, gt_arr, label="Ground Truth", color="#10b981", linewidth=2.5)
        axs[1, 0].plot(time_steps, imu_arr, label="IMU Integrated", color="#ef4444", linestyle=":", linewidth=2.0)
        axs[1, 0].set_title(f"IMU Integrated vs Ground Truth (RMSE: {rmse_imu:.2f} km/h)")
        axs[1, 0].set_xlabel("Time (seconds)")
        axs[1, 0].set_ylabel("Speed (km/h)")
        axs[1, 0].grid(True, linestyle="--", alpha=0.5)
        axs[1, 0].legend()
        
        # Subplot 4: Accel X
        axs[1, 1].plot(time_steps, raw_accelerations, label="Longitudinal Accel X", color="#8b5cf6", linewidth=1.5)
        axs[1, 1].set_title("Raw IMU Accelerometer X")
        axs[1, 1].set_xlabel("Time (seconds)")
        axs[1, 1].set_ylabel("Acceleration (m/s^2)")
        axs[1, 1].grid(True, linestyle="--", alpha=0.5)
        axs[1, 1].legend()
        
        plt.suptitle("CARLA Telemetry Speed Estimation: Individual Sensor Behavior", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(chart_path, dpi=150)
        plt.close()
        print("[+] Benchmark plotting completed successfully!")
        
    finally:
        # Restore settings and clean up spawned actors
        print("[*] Restoring original server settings...")
        world.apply_settings(original_settings)
        print(f"[*] Destroying {len(actor_list)} spawned actors...")
        for actor in actor_list:
            actor.destroy()
        print("[+] Cleanup complete. Exiting.")


if __name__ == "__main__":
    main()
