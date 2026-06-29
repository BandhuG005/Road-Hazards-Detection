import cv2
import numpy as np
import src.config as config

def resize_frame(frame, width=config.IMG_SIZE, height=config.IMG_SIZE):
    """Resizes the frame to the target size."""
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

def denoise_frame(frame):
    """Applies a fast Gaussian filter to denoise the image while maintaining edges."""
    return cv2.GaussianBlur(frame, (3, 3), 0)

def apply_clahe(frame):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    in the LAB color space to enhance local contrast without distorting colors.
    """
    # Convert BGR to LAB
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge channels and convert back to BGR
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def defog_frame(frame):
    """
    Defogs the image using a fast local contrast stretching/adaptive histogram approach.
    For real-time ADAS, standard dark channel defogging is computationally heavy.
    This function uses an adaptive color balance and channel stretching to defog efficiently.
    """
    # Split channels
    b, g, r = cv2.split(frame)
    
    # Simple contrast stretching per channel (minimum and maximum stretching)
    def stretch_channel(channel):
        min_val = np.percentile(channel, 1)
        max_val = np.percentile(channel, 99)
        if max_val > min_val:
            stretched = np.clip((channel - min_val) * 255.0 / (max_val - min_val), 0, 255)
            return stretched.astype(np.uint8)
        return channel

    sb = stretch_channel(b)
    sg = stretch_channel(g)
    sr = stretch_channel(r)
    
    stretched_frame = cv2.merge((sb, sg, sr))
    # Blend with original to avoid over-saturation in non-foggy areas
    return cv2.addWeighted(frame, 0.4, stretched_frame, 0.6, 0)

def normalize_frame(frame):
    """Normalizes the image pixels to [0, 1]."""
    return frame.astype(np.float32) / 255.0

def preprocess_pipeline(frame):
    """
    Executes the full preprocessing pipeline:
    Resize -> Denoise -> CLAHE -> Defog -> Normalize
    Returns both the preprocessed normalized frame (for models) and the enhanced BGR frame (for HUD display).
    """
    # 1. Resize
    resized = resize_frame(frame)
    
    # 2. Denoise
    denoised = denoise_frame(resized)
    
    # 3. CLAHE
    clahe = apply_clahe(denoised)
    
    # 4. Defogging
    enhanced = defog_frame(clahe)
    
    # 5. Normalization (for model inputs)
    normalized = normalize_frame(enhanced)
    
    return normalized, enhanced
