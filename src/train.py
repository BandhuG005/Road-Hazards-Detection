import os
import shutil
import yaml
import torch
import argparse
from ultralytics import YOLO
import src.config as config

def update_dataset_yaml():
    """Updates the path variable in dataset.yaml to point to the local absolute path."""
    yaml_path = os.path.join(config.DATASET_DIR, "dataset.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"dataset.yaml not found at {yaml_path}. Is the dataset fully extracted?")
        
    print(f"[*] Updating {yaml_path} with local paths...")
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
        
    # Update path to local absolute directory
    data['path'] = config.DATASET_DIR.replace("\\", "/")
    
    with open(yaml_path, 'w') as f:
        yaml.safe_dump(data, f, default_flow_style=False)
    print(f"[+] dataset.yaml updated successfully. Path set to: {data['path']}")

def check_gpu():
    """Checks and prints GPU information."""
    print("=" * 50)
    print("HARDWARE STATUS CHECK")
    print("=" * 50)
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Device Capability: {torch.cuda.get_device_capability(0)}")
    else:
        print("[!] WARNING: CUDA is NOT available. Training will run on CPU and be EXTREMELY SLOW.")
    print("=" * 50)
    return cuda_available

def run_training(epochs=30, batch_size=16, optimizer="SGD", lr0=0.01, patience=15, use_subset=True, weights="yolov8s.pt", data_yaml=None, imgsz=None):
    """
    Main training function.
    Loads YOLOv8 model and fine-tunes it on the hazard dataset.
    """
    # 1. Verify GPU
    cuda_available = check_gpu()
    device_arg = 0 if cuda_available else "cpu"
    
    # 2. Update yaml configuration for full dataset
    update_dataset_yaml()
    
    # 3. Load YOLOv8 model from specified weights
    print(f"[*] Loading model weights from: {weights}...")
    model = YOLO(weights)
    
    # 4. Check dataset config
    if data_yaml:
        yaml_path = data_yaml
        if not os.path.isabs(yaml_path):
            yaml_path = os.path.join(config.DATASET_DIR, yaml_path)
    else:
        if use_subset:
            yaml_name = "dataset_subset.yaml"
        else:
            yaml_name = "dataset.yaml"
        yaml_path = os.path.join(config.DATASET_DIR, yaml_name)
        
    if not os.path.exists(yaml_path):
        if not data_yaml and use_subset:
            print(f"[*] {yaml_name} not found. Running subset generation first...")
            from src.create_subset import create_dataset_subset
            create_dataset_subset(sample_ratio=0.10)
        else:
            raise FileNotFoundError(f"Dataset configuration not found at {yaml_path}")
            
    print(f"[*] Starting YOLOv8 fine-tuning on: {yaml_path}")
    print(f"[*] Training parameters: epochs={epochs}, batch={batch_size}, optimizer={optimizer}, lr0={lr0}, patience={patience}, cos_lr=True")
    
    import pathlib
    weights_stem = pathlib.Path(weights).stem
    run_name = f"{weights_stem}_hazard_detect"
    training_imgsz = imgsz if imgsz is not None else config.IMG_SIZE

    results = model.train(
        data=yaml_path.replace("\\", "/"),
        epochs=epochs,
        batch=batch_size,
        imgsz=training_imgsz,
        device=device_arg,
        optimizer=optimizer,
        lr0=lr0,
        cos_lr=True,          # Cosine learning rate scheduler
        patience=patience,     # Early stopping patience
        val=True,             # Run validation at the end of each epoch
        save=True,            # Save checkpoints
        project=config.MODEL_DIR.replace("\\", "/"),
        name=run_name,
        workers=4,
        amp=True              # Mixed precision training
    )
    
    print("[+] Training complete!")
    
    # 5. Locate and copy the best weights to models/best.pt
    # Find the most recently modified directory matching the name
    dirs = [d for d in os.listdir(config.MODEL_DIR) if d.startswith(run_name)]
    best_weights_src = None
    if dirs:
        latest_dir = max([os.path.join(config.MODEL_DIR, d) for d in dirs], key=os.path.getmtime)
        best_weights_src = os.path.join(latest_dir, "weights", "best.pt")
            
    if best_weights_src and os.path.exists(best_weights_src):
        shutil.copy(best_weights_src, config.BEST_MODEL_PATH)
        print(f"[+] Best weights copied to target path: {config.BEST_MODEL_PATH}")
    else:
        print(f"[!] Error: Could not locate best.pt weights")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 on the Road Hazards dataset.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--optimizer", type=str, default="SGD", choices=["SGD", "Adam", "AdamW"], help="Optimizer to use")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate (e.g. 0.01 for SGD, 0.001 for AdamW)")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--no_subset", action="store_true", help="Train on the full dataset instead of a 10% subset")
    parser.add_argument("--weights", type=str, default="yolov8s.pt", help="Initial weights file to load")
    parser.add_argument("--data", type=str, default=None, help="Custom dataset config YAML file name or path")
    parser.add_argument("--imgsz", type=int, default=None, help="Training image size resolution")
    
    args = parser.parse_args()
    
    # Set default learning rate based on optimizer if not customized by user
    lr_val = args.lr0
    if args.optimizer in ["Adam", "AdamW"] and args.lr0 == 0.01:
        lr_val = 0.001
        print(f"[*] Adjusted default learning rate to {lr_val} for {args.optimizer} optimizer.")
        
    run_training(
        epochs=args.epochs,
        batch_size=args.batch,
        optimizer=args.optimizer,
        lr0=lr_val,
        patience=args.patience,
        use_subset=not args.no_subset,
        weights=args.weights,
        data_yaml=args.data,
        imgsz=args.imgsz
    )
