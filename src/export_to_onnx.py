"""
ONNX Model Export Utility for ADAS Road Hazard Warning System.

Exports both the YOLOv8 Hazard Detector and the EfficientNet-B0 U-Net Road Segmenter
to optimized ONNX formats, enabling high-performance deployment on edge devices
using ONNX Runtime or TensorRT.
"""

import os
import torch
import src.config as config
from ultralytics import YOLO
from src.segmentation_model import get_model


def export_yolo(weights_path, output_dir):
    """Export the YOLOv8 model to ONNX format."""
    print(f"\n{'='*60}")
    print(f"[*] Starting YOLOv8 ONNX Export...")
    print(f"{'='*60}")
    
    if not os.path.exists(weights_path):
        print(f"[!] Error: YOLO weights file not found at {weights_path}")
        return
        
    try:
        model = YOLO(weights_path)
        # Export the model using Ultralytics native API
        # half=True option can be added for FP16 optimization if running on GPU
        exported_path = model.export(format="onnx", imgsz=640, dynamic=True)
        print(f"[+] YOLOv8 ONNX export successful!")
        print(f"[+] Exported model saved at: {exported_path}")
    except Exception as e:
        print(f"[!] Error exporting YOLOv8 to ONNX: {e}")


def export_unet(weights_path, output_dir):
    """Export the PyTorch U-Net segmentation model to ONNX format."""
    print(f"\n{'='*60}")
    print(f"[*] Starting U-Net Road Segmenter ONNX Export...")
    print(f"{'='*60}")
    
    if not os.path.exists(weights_path):
        print(f"[!] Error: U-Net weights file not found at {weights_path}")
        return
        
    try:
        print(f"[*] Loading PyTorch model weights from {weights_path}...")
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        model_name = checkpoint.get('model_name', 'unet')
        encoder_name = checkpoint.get('encoder_name', 'efficientnet-b0')
        
        # Instantiate model architecture
        model = get_model(model_name=model_name, encoder_name=encoder_name, num_classes=4)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Create a dummy input matching the model input size (Batch size 1, 3 channels, 640x640)
        dummy_input = torch.randn(1, 3, 640, 640, requires_grad=False)
        
        output_onnx_path = os.path.join(output_dir, "unet_road_segmenter.onnx")
        
        # Export PyTorch model to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_onnx_path,
            export_params=True,        # Store the trained parameter weights inside the model file
            opset_version=12,          # ONNX operator set version
            do_constant_folding=True,  # Execute constant folding optimization
            input_names=['input'],     # Define input node name
            output_names=['output'],   # Define output node name
            dynamic_axes={             # Enable dynamic batch sizing for flexible deployment
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"[+] U-Net ONNX export successful!")
        print(f"[+] Exported model saved at: {output_onnx_path}")
    except Exception as e:
        print(f"[!] Error exporting U-Net to ONNX: {e}")


def main():
    project_dir = config.BASE_DIR
    models_dir = config.MODEL_DIR
    
    yolo_weights = os.path.join(models_dir, "best.pt")
    unet_weights = os.path.join(models_dir, "best_model_augmented.pth")
    
    # Export YOLOv8s
    export_yolo(yolo_weights, models_dir)
    
    # Export U-Net Road Segmenter
    export_unet(unet_weights, models_dir)


if __name__ == "__main__":
    main()
