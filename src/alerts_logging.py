import os
import time
import csv
import threading
import winsound
import cv2
import src.config as config

# Threading lock for TTS speech to prevent overlapping voices
speech_lock = threading.Lock()
last_speech_time = 0

def _speak_text(text):
    """Worker function to run TTS in a separate thread."""
    global last_speech_time
    with speech_lock:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            # Set speed and volume
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 1.0)
            engine.say(text)
            engine.runAndWait()
            # Clean up
            del engine
        except Exception as e:
            print(f"[!] TTS error: {e}")

def trigger_audio_alert(hazard_class, distance, risk_level):
    """
    Step 11: Driver Alert System (Audio).
    Triggers voice warnings (TTS) and tones based on risk level.
    Runs asynchronously to avoid blocking the real-time video stream.
    """
    global last_speech_time
    current_time = time.time()
    
    # Limit alert frequency to prevent voice cluttering (max once every 2.5 seconds)
    if current_time - last_speech_time < 2.5:
        return
        
    if risk_level == "LOW":
        # Low risk: no audio, just visual
        return
        
    elif risk_level == "MEDIUM":
        # Medium risk: simple tone alert (440Hz beep for 100ms)
        last_speech_time = current_time
        threading.Thread(target=lambda: winsound.Beep(500, 150), daemon=True).start()
        
    elif risk_level == "HIGH":
        # High risk: TTS warning
        last_speech_time = current_time
        msg = f"Warning. {hazard_class} ahead in {int(distance)} meters."
        # Run TTS and a caution beep
        def play_alert():
            winsound.Beep(800, 200)
            _speak_text(msg)
        threading.Thread(target=play_alert, daemon=True).start()
        
    elif risk_level == "CRITICAL":
        # Critical risk: Urgent alarm + urgent TTS warning
        last_speech_time = current_time
        msg = f"Critical. Brake. {hazard_class} in {int(distance)} meters."
        def play_critical_alert():
            # Urgent double beep
            winsound.Beep(1000, 150)
            winsound.Beep(1000, 150)
            _speak_text(msg)
        threading.Thread(target=play_critical_alert, daemon=True).start()

class DataLogger:
    """
    Step 12: Data Logging.
    Logs telemetry and saves JPEG snapshots of high-risk hazards.
    """
    def __init__(self):
        self.log_file = os.path.join(config.LOG_DIR, "telemetry_log.csv")
        self.snapshots_dir = os.path.join(config.LOG_DIR, "snapshots")
        os.makedirs(self.snapshots_dir, exist_ok=True)
        
        # Write CSV header if file does not exist
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Timestamp", "GPS_Lat", "GPS_Lon", "Vehicle_Speed_KMH", 
                    "Hazard_Class", "Confidence", "Distance_M", "TTC_Sec", 
                    "Risk_Score", "Risk_Level"
                ])
                
    def log_hazard(self, frame_raw, hazard_info, speed_kmh, gps_coords=(12.9716, 77.5946)):
        """Logs a single hazard event to CSV and saves snapshot if high risk."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        lat, lon = gps_coords
        
        # Log to CSV
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, lat, lon, speed_kmh,
                hazard_info["class_label"], 
                round(hazard_info["confidence_score"], 2),
                hazard_info["distance"],
                hazard_info["ttc"],
                hazard_info["risk_score"],
                hazard_info["risk_level"]
            ])
            
        # Save snapshot if Risk is HIGH or CRITICAL (risk_score >= 50.0)
        if hazard_info["risk_score"] >= 50.0:
            filename = f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}_{hazard_info['class_label']}_{int(hazard_info['risk_score'])}.jpg"
            save_path = os.path.join(self.snapshots_dir, filename)
            
            # Draw overlay on the saved snapshot
            snapshot_img = frame_raw.copy()
            xmin, ymin, xmax, ymax = map(int, hazard_info["bbox"])
            
            # Color-coded box
            color = (0, 0, 255) if hazard_info["risk_level"] == "CRITICAL" else (0, 165, 255)
            cv2.rectangle(snapshot_img, (xmin, ymin), (xmax, ymax), color, 3)
            
            # Label
            label = f"{hazard_info['class_label']} (Risk: {hazard_info['risk_score']})"
            cv2.putText(snapshot_img, label, (xmin, ymin - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Save the annotated image
            cv2.imwrite(save_path, snapshot_img)

def classify_risk(risk_score):
    """
    Step 10: Decision Engine.
    Categorizes the hazard risk score into Alert Levels.
    """
    if risk_score < config.RISK_THRESHOLD_LOW:
        return "LOW"
    elif risk_score < config.RISK_THRESHOLD_MEDIUM:
        return "MEDIUM"
    elif risk_score < config.RISK_THRESHOLD_HIGH:
        return "HIGH"
    else:
        return "CRITICAL"

def draw_hud(frame, detections, speed_kmh, segmentation_color_mask=None):
    """
    Step 11: Driver Alert System (Visual HUD Overlay).
    Draws road segmentation masks, hazard bounding boxes, stats, and warning banners.
    """
    hud_frame = frame.copy()
    
    # 1. Overlay Road Segmentation Mask with transparency
    if segmentation_color_mask is not None:
        cv2.addWeighted(segmentation_color_mask, 0.35, hud_frame, 0.65, 0, dst=hud_frame)
        
    # 2. Draw hazard detections
    h_img, w_img = frame.shape[:2]
    max_risk_detection = None
    
    for det in detections:
        xmin, ymin, xmax, ymax = map(int, det["bbox"])
        risk_level = det["risk_level"]
        risk_score = det["risk_score"]
        dist = det["distance"]
        ttc = det["ttc"]
        class_label = det["class_label"]
        
        # Color mapping (BGR)
        if risk_level == "LOW":
            color = (0, 255, 0)       # Green
        elif risk_level == "MEDIUM":
            color = (0, 255, 255)     # Yellow
        elif risk_level == "HIGH":
            color = (0, 165, 255)     # Orange
        else: # CRITICAL
            color = (0, 0, 255)       # Red
            
        # Draw bounding box
        cv2.rectangle(hud_frame, (xmin, ymin), (xmax, ymax), color, 2)
        
        # Draw label card
        lbl_text = f"{class_label}: {dist}m | TTC: {ttc}s"
        cv2.putText(hud_frame, lbl_text, (xmin, max(15, ymin - 8)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        
        # Track maximum risk to display in the main alert banner
        if max_risk_detection is None or risk_score > max_risk_detection["risk_score"]:
            max_risk_detection = det
            
    # 3. Draw Speedometer HUD
    speed_text = f"SPEED: {int(speed_kmh)} km/h"
    cv2.rectangle(hud_frame, (10, 10), (220, 50), (0, 0, 0), -1)
    cv2.putText(hud_frame, speed_text, (20, 38), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    # 4. Draw RL Agent Control Status Card (Top Right)
    rl_actions_active = [det["rl_action"] for det in detections if "rl_action" in det and det["rl_action"] != "N/A"]
    if rl_actions_active:
        priority = {"Heavy Brake": 4, "Medium Brake": 3, "Light Brake": 2, "Coast / Maintain": 1, "Accelerate": 0}
        critical_action = max(rl_actions_active, key=lambda a: priority.get(a, 0))
        
        # Color coding
        if "Heavy" in critical_action:
            action_color = (0, 0, 255)      # Red
        elif "Medium" in critical_action:
            action_color = (0, 100, 255)     # Orange-Red
        elif "Light" in critical_action:
            action_color = (0, 255, 255)     # Yellow
        elif "Accelerate" in critical_action:
            action_color = (0, 255, 0)       # Green
        else:
            action_color = (255, 255, 255)   # White
            
        action_text = f"RL ACTION: {critical_action.upper()}"
        cv2.rectangle(hud_frame, (w_img - 280, 10), (w_img - 10, 50), (0, 0, 0), -1)
        cv2.putText(hud_frame, action_text, (w_img - 270, 38), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, action_color, 2, cv2.LINE_AA)
                    
    # 5. Draw Alert Warning Banner for High/Critical Risks
    if max_risk_detection is not None and max_risk_detection["risk_level"] in ["HIGH", "CRITICAL"]:
        det = max_risk_detection
        color = (0, 0, 255) if det["risk_level"] == "CRITICAL" else (0, 165, 255)
        banner_text = f"{det['risk_level']} ALERT: {det['class_label'].upper()} AHEAD!"
        
        # Flashing effect based on milliseconds
        flash = int(time.time() * 5) % 2
        if det["risk_level"] == "HIGH" or flash == 0:
            # Banner background
            cv2.rectangle(hud_frame, (w_img // 2 - 250, 15), (w_img // 2 + 250, 60), color, -1)
            # Text
            cv2.putText(hud_frame, banner_text, (w_img // 2 - 230, 47), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
        # Draw haptic warning text
        if det["risk_level"] == "CRITICAL":
            cv2.putText(hud_frame, "HAPTIC FEEDBACK ACTIVE", (w_img // 2 - 120, h_img - 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
            
    return hud_frame
