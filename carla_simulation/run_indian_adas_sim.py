import os
import sys
import time
import math
import queue
import numpy as np
import cv2
import pygame
import torch

# Add root path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import carla
except ImportError:
    print("[!] CARLA library missing.")
    sys.exit(1)

from src.pipeline import ADASPipeline
from src.distance_risk import get_lane_alignment_factor, compute_risk_score, calculate_ttc
from src.alerts_logging import classify_risk, draw_hud

def setup_indian_weather(world):
    """Sets up a heavy monsoon rain weather profile (water-logged roads, low visibility)."""
    weather = carla.WeatherParameters(
        cloudiness=85.0,
        precipitation=80.0,            # Heavy Monsoon rain
        precipitation_deposits=75.0,   # High puddle accumulation on roads
        wind_intensity=40.0,
        sun_azimuth_angle=0.0,
        sun_altitude_angle=20.0        # Gloomy overcast lighting
    )
    world.set_weather(weather)
    print("[+] Monsoon weather profile loaded.")

def main():
    pygame.init()
    display = pygame.display.set_mode((1024, 768), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("Indian Road ADAS Simulator (Manual WASD Drive)")
    
    print("[*] Connecting to CARLA simulator...")
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(30.0)
    world = client.get_world()
    
    # Load Town10HD map for urban traffic testing
    print("[*] Loading Town10HD Map...")
    world = client.load_world("Town10HD")
    
    # Setup Synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    
    setup_indian_weather(world)
    
    actor_list = []
    
    try:
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
        
        # Spawn point on the left lane (Left-Hand Traffic mapping)
        spawn_points = world.get_map().get_spawn_points()
        left_spawn = spawn_points[1] # Choose a straight road point (no offset to prevent wall stucks)
        
        print("[*] Spawning manual vehicle...")
        vehicle = world.spawn_actor(vehicle_bp, left_spawn)
        actor_list.append(vehicle)
        
        # Attach RGB Windshield Camera (Outside Bumper Dashcam View)
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "1024")
        camera_bp.set_attribute("image_size_y", "768")
        camera_bp.set_attribute("fov", "90")
        camera_transform = carla.Transform(
            carla.Location(x=2.2, y=0.0, z=0.9),
            carla.Rotation(pitch=-3.0)
        )
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(camera)
        
        image_queue = queue.Queue()
        camera.listen(lambda data: image_queue.put(data))
        
        # Spawn potholes and speed bumps directly in the left lane
        print("[*] Spawning road hazards (Potholes, Speed Bumps) in the left lane...")
        forward_vec = left_spawn.get_forward_vector()
        
        # Pothole (Broken tile prop) - Spawned 45 meters ahead, flush with road to prevent wheel trapping
        pothole_bp = blueprint_library.find("static.prop.brokentile01")
        pothole_loc = left_spawn.location + (forward_vec * 45.0) 
        pothole_loc.z -= 0.02  # Sunk flush with asphalt
        pothole = world.spawn_actor(pothole_bp, carla.Transform(pothole_loc, left_spawn.rotation))
        actor_list.append(pothole)
        
        # Speed Bump (raised plank prop) - Spawned 80 meters ahead, low profile
        bump_bp = blueprint_library.find("static.prop.ironplank")
        bump_loc = left_spawn.location + (forward_vec * 80.0) 
        bump_loc.z += 0.02  # Low profile to allow wheels to roll over
        bump = world.spawn_actor(bump_bp, carla.Transform(bump_loc, left_spawn.rotation))
        actor_list.append(bump)
        
        # Initialize ADAS pipeline
        pipeline = ADASPipeline()
        clock = pygame.time.Clock()
        
        # Disable autopilot - Manual drive only
        vehicle.set_autopilot(False)
        
        print("\n" + "="*50)
        print("MANUAL DRIVING MODE ENABLED:")
        print("  - [W] Throttle (Accelerate)")
        print("  - [S] Brake (when moving) / Reverse (when stopped)")
        print("  - [A] Steer Left - [D] Steer Right")
        print("  - [SPACE] Toggle ADAS Automatic Braking Override")
        print("="*50 + "\n")
        
        # Initialize control states
        throttle = 0.0
        steer = 0.0
        adas_active = True
        
        while True:
            # Handle manual keyboard controls
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        adas_active = not adas_active
                        print(f"[*] Toggle ADAS Override: {adas_active}")
            
            # Read steering and throttle keys
            keys = pygame.key.get_pressed()
            throttle = 0.8 if keys[pygame.K_w] else 0.0
            if keys[pygame.K_s]:
                throttle = -0.5
            steer = 0.0
            if keys[pygame.K_a]:
                steer = -0.3
            if keys[pygame.K_d]:
                steer = 0.3
            
            # Tick simulator
            world.tick()
            try:
                image_data = image_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            
            # Format image to OpenCV BGR
            raw_array = np.frombuffer(image_data.raw_data, dtype=np.uint8)
            raw_array = np.reshape(raw_array, (image_data.height, image_data.width, 4))
            bgr_frame = raw_array[:, :, :3]
            
            # Get current speed in km/h
            velocity = vehicle.get_velocity()
            speed_kmh = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            
            # Run ADAS Inference Pipeline
            from src.preprocess import preprocess_pipeline
            _, enhanced_frame = preprocess_pipeline(bgr_frame)
            seg_mask = pipeline.segmenter.run_inference(enhanced_frame)
            detections = pipeline.detector.detect_hazards_sahi(enhanced_frame, conf_threshold=0.25)
            
            # Process closest hazard detection
            processed_detections = []
            dqn_action = 0
            dqn_desc = "Coast (Maintain)"
            
            # Filter detections to safety-critical hazards only
            safety_critical_classes = {"Pothole", "Speedbump", "Road_Barrier", "Construction_Object", "Manhole"}
            hazard_detections = [d for d in detections if d["class_label"] in safety_critical_classes]
            
            if len(hazard_detections) > 0:
                closest_det = min(hazard_detections, key=lambda d: d["bbox"][3])
                distance = max(2.0, 15.0 - (closest_det["bbox"][3] - 400) * 0.05)
                
                # Only apply override if the hazard is within 15 meters
                if distance <= 15.0:
                    lane_factor = get_lane_alignment_factor(closest_det["bbox"], seg_mask)
                    risk_score = compute_risk_score(closest_det["class_label"], distance, speed_kmh, lane_factor)
                    risk_level = classify_risk(risk_score)
                    ttc = calculate_ttc(distance, speed_kmh)
                    
                    processed_detections.append({
                        "bbox": closest_det["bbox"],
                        "class_label": closest_det["class_label"],
                        "distance": distance,
                        "ttc": ttc,
                        "risk_score": risk_score,
                        "risk_level": risk_level,
                        "lane_alignment": lane_factor
                    })
                    
                    # DQN Deceleration Trigger
                    severity = 0.85 if closest_det["class_label"] == "Pothole" else 0.50
                    state = np.array([distance, speed_kmh, severity, lane_factor], dtype=np.float32)
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    
                    with torch.no_grad():
                        q_values = pipeline.rl_agent(state_tensor)
                        dqn_action = int(torch.argmax(q_values).item())
                    
                    action_meanings = {
                        0: "Coast (Maintain)",
                        1: "Light Brake (-1.5 m/s^2)",
                        2: "Medium Brake (-4.0 m/s^2)",
                        3: "Emergency Brake (-7.5 m/s^2)",
                        4: "Accelerate (+1.2 m/s^2)"
                    }
                    dqn_desc = action_meanings.get(dqn_action, "Brake")
            
            # Draw HUD
            seg_color_mask = pipeline.segmenter.get_color_mask(seg_mask)
            hud_frame = draw_hud(enhanced_frame, processed_detections, speed_kmh, seg_color_mask)
            
            # HUD text overlays
            cv2.putText(hud_frame, "MODE: MANUAL DRIVE (WASD)", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            if adas_active:
                cv2.putText(hud_frame, f"DQN ACTION OVERRIDE: {dqn_desc.upper()}", (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(hud_frame, "ADAS OVERRIDE: DISABLED (MANUAL DRIVE)", (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
            cv2.putText(hud_frame, f"Steer: {steer:.2f} | Throttle: {throttle:.2f}", (20, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Manual WASD controls
            control = carla.VehicleControl()
            control.steer = steer
            control.manual_gear_shift = False
            control.hand_brake = False
            
            # Evasion & Low-Speed Suppression: Release brakes if steering evasively OR crawling/stopped
            is_steering_evasively = (abs(steer) > 0.01)
            is_crawling_or_stopped = (speed_kmh < 2.0)
            
            if adas_active and dqn_action == 3 and not is_steering_evasively and not is_crawling_or_stopped:
                control.throttle = 0.0
                control.brake = 1.0
                print("[!] ADAS OVERRIDE: Emergency braking engaged!")
            elif adas_active and dqn_action == 2 and not is_steering_evasively and not is_crawling_or_stopped:
                control.throttle = 0.0
                control.brake = 0.6
            elif adas_active and dqn_action == 1 and not is_steering_evasively and not is_crawling_or_stopped:
                control.throttle = 0.0
                control.brake = 0.2
            else:
                if throttle >= 0:
                    control.throttle = throttle
                    control.reverse = False
                    control.brake = 0.0
                else:
                    if speed_kmh < 1.0:
                        control.throttle = abs(throttle)
                        control.reverse = True
                        control.brake = 0.0
                    else:
                        control.throttle = 0.0
                        control.reverse = False
                        control.brake = abs(throttle)
            
            vehicle.apply_control(control)
            
            # Display image in Pygame window
            surface_frame = cv2.cvtColor(hud_frame, cv2.COLOR_BGR2RGB)
            surface_frame = np.rot90(surface_frame)
            surface_frame = np.flipud(surface_frame)
            pygame_surface = pygame.surfarray.make_surface(surface_frame)
            display.blit(pygame_surface, (0, 0))
            pygame.display.flip()
            clock.tick(20)
            
    finally:
        pygame.quit()
        # Restore settings
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print("[*] Cleaning up spawned simulator actors...")
        for actor in actor_list:
            if actor is not None and actor.is_alive:
                actor.destroy()
        print("[+] Cleanup complete. Simulation closed.")

if __name__ == "__main__":
    main()
