import os
import cv2
import numpy as np
import src.config as config

try:
    import torch
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from src.segmentation_model import get_model
    HAS_TORCH_MODEL = True
except Exception as e:
    import sys
    # Local fallback to resolve dependencies using the existing internship virtual environment
    print(f"[!] Warning: Failed to import segmentation dependencies ({e}). Using mock segmenter.")
    HAS_TORCH_MODEL = False

class RoadSegmenter:
    """
    Wrapper for the road segmentation model.
    Loads the trained SMP U-Net model from models/best_model_augmented.pth if available,
    otherwise falls back to a geometrically realistic mock mode.
    """
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(config.MODEL_DIR, "best_model_augmented.pth")
        self.model_path = model_path
        self.is_mock = not HAS_TORCH_MODEL or not os.path.exists(self.model_path)
        self.device = None
        self.model = None
        
        if not self.is_mock:
            self.load_model()
        else:
            print("[*] RoadSegmenter initialized in Mock Perspective Fallback mode.")
            
    def load_model(self):
        """Loads the pre-trained SMP U-Net ResNet-34 road segmentation model."""
        print(f"[*] Loading PyTorch U-Net road segmentation model from {self.model_path}...")
        try:
            checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
            model_name = checkpoint.get('model_name', 'unet')
            encoder_name = checkpoint.get('encoder_name', 'resnet34')
            
            self.model = get_model(model_name=model_name, encoder_name=encoder_name, num_classes=4)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            self.model.eval()
            print(f"[+] U-Net Road Segmentation model loaded successfully on device: {self.device}")
        except Exception as e:
            print(f"[!] Error loading road segmentation checkpoint: {e}")
            print("[*] Falling back to Mock Perspective mode.")
            self.is_mock = True

    def run_inference(self, frame_enhanced):
        """
        Runs segmentation inference on the enhanced frame.
        Returns a single-channel segmentation mask of size 640x640:
        0: Non-Drivable / Background
        1: Drivable Direct (Current lane)
        2: Drivable Alt (Adjacent lane)
        4: Unpaved (Road shoulders)
        """
        if self.is_mock:
            return self._generate_mock_mask(frame_enhanced)
        
        try:
            # U-Net expects RGB image for PyTorch model
            rgb_frame = cv2.cvtColor(frame_enhanced, cv2.COLOR_BGR2RGB)
            
            # Albumentations transforms matching original training pipeline
            transform = A.Compose([
                A.Resize(640, 640),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
            transformed = transform(image=rgb_frame)
            input_tensor = transformed['image'].unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(input_tensor)
                # Output prediction mask of shape (640, 640)
                pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                
            # Map U-Net class 3 (Unpaved) to pipeline class 4 (Unpaved)
            pred_mask[pred_mask == 3] = 4
            return pred_mask
        except Exception as e:
            print(f"[!] Error running segmentation inference: {e}. Falling back to mock.")
            return self._generate_mock_mask(frame_enhanced)

    def _generate_mock_mask(self, frame):
        """
        Generates a geometrically realistic drivable area mask based on camera perspectives.
        Uses a trapezoidal region centered at the bottom of the frame for the current lane.
        """
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)  # Default: 0 (Non-Drivable)
        
        # Define trapezoid vertices for Drivable Direct (current lane)
        drivable_direct_pts = np.array([
            [w * 0.40, h * 0.55],  # Top left
            [w * 0.60, h * 0.55],  # Top right
            [w * 0.85, h * 1.0],   # Bottom right
            [w * 0.15, h * 1.0]    # Bottom left
        ], dtype=np.int32)
        
        # Fill Drivable Direct with class 1
        cv2.fillPoly(mask, [drivable_direct_pts], 1)
        
        # Define trapezoid for Drivable Alt (left adjacent lane)
        drivable_alt_left_pts = np.array([
            [w * 0.20, h * 0.55],
            [w * 0.40, h * 0.55],
            [w * 0.15, h * 1.0],
            [w * 0.0, h * 1.0]
        ], dtype=np.int32)
        
        # Define trapezoid for Drivable Alt (right adjacent lane)
        drivable_alt_right_pts = np.array([
            [w * 0.60, h * 0.55],
            [w * 0.80, h * 0.55],
            [w * 1.0, h * 1.0],
            [w * 0.85, h * 1.0]
        ], dtype=np.int32)
        
        # Fill Drivable Alt with class 2
        cv2.fillPoly(mask, [drivable_alt_left_pts], 2)
        cv2.fillPoly(mask, [drivable_alt_right_pts], 2)
        
        # Add unpaved shoulders on the far left/right
        unpaved_left_pts = np.array([
            [w * 0.05, h * 0.55],
            [w * 0.20, h * 0.55],
            [w * 0.0, h * 1.0],
            [w * 0.0, h * 0.8]
        ], dtype=np.int32)
        cv2.fillPoly(mask, [unpaved_left_pts], 4)
        
        # Optionally simulate a small waterlogged region on the road (class 3) for demo
        water_center = (int(w * 0.70), int(h * 0.85))
        cv2.circle(mask, water_center, 40, 3, -1)
        
        return mask

    def get_color_mask(self, mask):
        """
        Creates a colored visualization of the segmentation mask.
        Green: Drivable Direct
        Blue: Drivable Alt
        Yellow: Waterlogged
        Purple: Unpaved
        """
        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        
        # Color definitions (BGR)
        color_mask[mask == 1] = [0, 255, 0]    # Drivable Direct -> Green
        color_mask[mask == 2] = [255, 0, 0]    # Drivable Alt -> Blue
        color_mask[mask == 3] = [0, 255, 255]  # Waterlogged -> Yellow
        color_mask[mask == 4] = [255, 0, 255]  # Unpaved -> Purple
        
        return color_mask

