import os
import shutil
import time
import torch
import src.config as config
from src.train import run_training
from src.evaluate import run_test_validation

def check_gpu():
    print("=" * 60)
    print("GPU ACCELERATION CHECK")
    print("=" * 60)
    if not torch.cuda.is_available():
        print("[!] ERROR: CUDA is not available. High-resolution training requires a GPU.")
        return False
    
    device_name = torch.cuda.get_device_name(0)
    vram_bytes = torch.cuda.get_device_properties(0).total_memory
    vram_gb = vram_bytes / (1024 ** 3)
    print(f"[+] CUDA Device: {device_name}")
    print(f"[+] Total VRAM: {vram_gb:.2f} GB")
    print("=" * 60)
    return True

def main():
    if not check_gpu():
        return
        
    t_start = time.time()
    
    # 1. Back up current best.pt if it exists and best_baseline.pt does not yet exist
    best_path = os.path.join(config.MODEL_DIR, "best.pt")
    baseline_backup = os.path.join(config.MODEL_DIR, "best_baseline.pt")
    if os.path.exists(best_path):
        if not os.path.exists(baseline_backup):
            print(f"[*] Backing up baseline models/best.pt to models/best_baseline.pt...")
            shutil.copy(best_path, baseline_backup)
        else:
            print(f"[*] Baseline models/best_baseline.pt already exists.")
    else:
        # If best.pt doesn't exist, we can copy best_final.pt or do nothing
        final_weights = os.path.join(config.MODEL_DIR, "best_final.pt")
        if os.path.exists(final_weights):
            print(f"[*] Copying models/best_final.pt to models/best_baseline.pt...")
            shutil.copy(final_weights, baseline_backup)

    # 2. Make sure representative subset is generated
    subset_yaml = os.path.join(config.DATASET_DIR, "dataset_subset.yaml")
    if not os.path.exists(subset_yaml):
        print("[*] Generating 10% representative dataset subset...")
        from src.create_subset import create_dataset_subset
        create_dataset_subset(sample_ratio=0.10)
    else:
        print("[*] 10% representative dataset subset files already exist.")

    # 3. Launch high-resolution YOLOv8l training
    print("\n" + "#" * 80)
    print("LAUNCHING HIGH-ACCURACY TRAINING RUN (YOLOv8l, imgsz=960, 10% subset)")
    print("#" * 80 + "\n")
    
    # Training with batch=4 to prevent VRAM paging on 8GB RTX 3050 GPU
    run_training(
        epochs=30,
        batch_size=4,
        optimizer="SGD",
        lr0=0.01,
        patience=10,
        use_subset=True,
        weights="yolov8l.pt",
        imgsz=960
    )
    
    # 4. Check for newly trained weights and save as best_large.pt
    large_weights_target = os.path.join(config.MODEL_DIR, "best_large.pt")
    if os.path.exists(best_path):
        print(f"[*] Saving newly trained YOLOv8l weights to: {large_weights_target}")
        shutil.copy(best_path, large_weights_target)
        
        # Restore the baseline model to best.pt
        if os.path.exists(baseline_backup):
            print(f"[*] Restoring baseline model back to: {best_path}")
            shutil.copy(baseline_backup, best_path)
    else:
        print("[!] ERROR: Could not locate newly trained weights at models/best.pt!")
        return

    # 5. Run evaluation on the test subset at resolution 960
    print("\n" + "=" * 80)
    print("EVALUATING YOLOv8l MODEL ON TEST SUBSET AT imgsz=960")
    print("=" * 80 + "\n")
    
    # Run evaluation using the customized evaluate module
    # We validate on the test subset first for speed, or we can use the full test set
    # Let's run on test split of dataset.yaml (which evaluates the full test split of 43k images)
    # to compare fairly with our baseline. Evaluating 43k images at 960 will take ~20-30 mins.
    # To keep validation quick but complete, we can run on the full test split.
    print("[*] Starting validation on full test set (43,363 images) at resolution 960...")
    run_test_validation(model_path=large_weights_target, imgsz=960)
    
    elapsed = (time.time() - t_start) / 3600
    print("\n" + "=" * 80)
    print(f"[+] YOLOv8l High-Resolution Training and Evaluation completed in {elapsed:.2f} hours!")
    print(f"[+] Large weights saved at: {large_weights_target}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
