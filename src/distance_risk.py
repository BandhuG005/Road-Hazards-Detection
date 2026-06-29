import numpy as np
import src.config as config

def estimate_distance(bbox, class_label):
    """
    Step 8: Distance Estimation.
    Estimates the distance of a hazard from the camera in meters.
    Formula: distance = (focal_length * real_height) / bbox_height_pixels
    """
    xmin, ymin, xmax, ymax = bbox
    bbox_height = max(1.0, ymax - ymin)
    
    # Get physical height of the hazard
    real_height = config.REAL_HEIGHTS.get(class_label, 0.5)
    
    # Calculate distance
    distance = (config.FOCAL_LENGTH * real_height) / bbox_height
    return round(distance, 2)

def calculate_ttc(distance, speed_kmh):
    """
    Step 8: Time to Collision (TTC) Estimation.
    Formula: TTC = distance / speed (in m/s)
    """
    # Convert speed from km/h to m/s
    speed_mps = speed_kmh / 3.6
    
    if speed_mps <= 0.1:
        # Vehicle is stationary, no collision risk from forward motion
        return 999.0
        
    ttc = distance / speed_mps
    return round(ttc, 2)

def get_lane_alignment_factor(bbox, segmentation_mask):
    """
    Determines the lane alignment factor based on where the hazard lies in the segmentation mask.
    If the mask has very few road pixels (due to bad weather/night model failure),
    falls back to a geometric center lane estimation.
    Returns:
    - 1.0 if hazard is in Drivable Direct (our lane, class 1)
    - 0.4 if hazard is in Drivable Alt (adjacent lanes, class 2)
    - 0.7 if hazard is in a Waterlogged area (class 3)
    - 0.5 if hazard is in an Unpaved shoulder (class 4)
    - 0.1 if hazard is off-road (class 0)
    """
    h_mask, w_mask = segmentation_mask.shape
    xmin, ymin, xmax, ymax = bbox
    
    # Clip coordinates to mask bounds
    xmin_c = max(0, int(xmin))
    ymin_c = max(0, int(ymin))
    xmax_c = min(w_mask, int(xmax))
    ymax_c = min(h_mask, int(ymax))
    
    if xmax_c <= xmin_c or ymax_c <= ymin_c:
        return 0.1
        
    # Check if the mask is reliable (total road pixels > 5000)
    road_pixels = np.sum(np.isin(segmentation_mask, [1, 2, 3, 4]))
    if road_pixels < 5000:
        # Fallback to geometric lane factor (direct lane if center of hazard is in the middle 40% of the screen)
        cx = (xmin + xmax) / 2.0
        if 192.0 <= cx <= 448.0:
            return 1.0  # Direct lane default
        else:
            return 0.4  # Side lane default
            
    # Crop the segmentation mask for the bounding box region
    bbox_mask_crop = segmentation_mask[ymin_c:ymax_c, xmin_c:xmax_c]
    
    # Count occurrence of each class in the bounding box
    counts = np.bincount(bbox_mask_crop.flatten(), minlength=5)
    
    # Exclude background (class 0) if other classes are present
    active_counts = counts[1:]
    if np.sum(active_counts) > 0:
        dominant_class = np.argmax(active_counts) + 1  # Shift index back to 1-4
    else:
        dominant_class = 0
        
    # Map dominant class to alignment factor
    alignment_mapping = {
        1: 1.0,  # Drivable Direct (directly in front)
        2: 0.4,  # Drivable Alt (less danger, adjacent lane)
        3: 0.7,  # Waterlogged (on road, hazardous)
        4: 0.5,  # Unpaved (shoulder)
        0: 0.1   # Off-road
    }
    
    return alignment_mapping.get(dominant_class, 0.1)

def compute_risk_score(class_label, distance, speed_kmh, lane_factor):
    """
    Step 9: Risk Assessment.
    Weighted risk formula:
    score = severity * proximity_factor * speed_factor * lane_factor * 100
    Returns risk score in [0, 100]
    """
    severity = config.SEVERITY_WEIGHTS.get(class_label, 0.5)
    
    # Proximity Factor: closer means higher risk
    # Linear decay model up to 40 meters, where risk approaches 0
    proximity_factor = max(0.05, 1.0 - (distance / 40.0))
    
    # Speed Factor: higher speed means higher risk
    # Bounded between 0.5 (stationary/slow) and 1.5 (high speed)
    speed_factor = 0.5 + 0.5 * (speed_kmh / 80.0)
    speed_factor = min(1.5, max(0.5, speed_factor))
    
    # Combine terms
    raw_score = severity * proximity_factor * speed_factor * lane_factor
    
    # Scale to 0-100 range
    risk_score = raw_score * 100.0
    
    # Cap between 0 and 100
    risk_score = min(100.0, max(0.0, risk_score))
    
    return round(risk_score, 1)
