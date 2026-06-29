import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ZIP = os.path.join(BASE_DIR, "Augmented DATASET - 8 classes.zip")
DATASET_DIR = "C:/augmented_dataset"
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# YOLOv8 settings
IMG_SIZE = 640
YOLO_MODEL_NAME = "yolov8s"  # We will fine-tune YOLOv8s for high accuracy
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")

# Hazard Classes (8 classes)
CLASSES = {
    0: "Pothole",
    1: "Crack",
    2: "Manhole",
    3: "Waterlogging",
    4: "Speedbump",
    5: "Traffic_Cone",
    6: "Road_Barrier",
    7: "Construction_Object"
}

# Severity weights (0.0 to 1.0)
SEVERITY_WEIGHTS = {
    "Road_Barrier": 0.95,
    "Pothole": 0.85,
    "Construction_Object": 0.80,
    "Manhole": 0.70,
    "Waterlogging": 0.60,
    "Speedbump": 0.50,
    "Traffic_Cone": 0.40,
    "Crack": 0.30
}

# Estimated real-world height of classes (in meters) for distance estimation
# distance = (focal_length * real_height) / bbox_height_pixels
REAL_HEIGHTS = {
    "Road_Barrier": 1.0,           # Standard concrete or plastic barrier is ~1.0m tall
    "Pothole": 0.35,               # Physical diameter/longitudinal size represented by bbox ~0.35m
    "Construction_Object": 1.2,    # Barrels, signs, etc. average ~1.2m
    "Manhole": 0.60,               # Manhole cover diameter ~0.6m
    "Waterlogging": 0.50,          # Depth/size represented by bbox
    "Speedbump": 0.12,             # Height of speed bump
    "Traffic_Cone": 0.70,          # Standard traffic cone height is ~0.7m
    "Crack": 0.20                  # Physical scale represented by bbox
}

# Camera Calibration Parameters
# For 640x640 resized frame, focal length (fx) can be approximated if not calibrated.
# Standard value fx = 800 is a reasonable default for typical dashcam FOV.
FOCAL_LENGTH = 800.0  # Pixels
CAMERA_HEIGHT = 1.5   # Meters (mounting height on the vehicle)

# Telemetry defaults
DEFAULT_SPEED_KMH = 50.0  # Default vehicle speed in km/h

# Decision Engine Risk Thresholds
RISK_THRESHOLD_LOW = 30.0
RISK_THRESHOLD_MEDIUM = 65.0
RISK_THRESHOLD_HIGH = 85.0
# LOW: 0 - 29
# MEDIUM: 30 - 64
# HIGH: 65 - 84
# CRITICAL: 85 - 100
