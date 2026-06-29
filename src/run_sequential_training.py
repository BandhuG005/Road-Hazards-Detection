import os
import shutil
import time
from src.train import run_training
from src.evaluate import run_test_validation
from src.verify_pipeline import verify_pipeline_end_to_end

def main():
    print("=" * 70)
    print("STARTING AUTONOMOUS OVERNIGHT SEQUENTIAL TRAINING ENGINE")
    print("=" * 70)
    
    # Define weights paths (absolute paths to prevent resolution issues)
    weights_phase1 = "D:/Road Hazards-Detection/models/best_phase1.pt"
    weights_phase2 = "D:/Road Hazards-Detection/models/best_phase2.pt"
    weights_phase3 = "D:/Road Hazards-Detection/models/best_phase3.pt"
    weights_final = "D:/Road Hazards-Detection/models/best_final.pt"
    
    # -------------------------------------------------------------------------
    # PHASE 2: Train Split B + Buffer A
    # -------------------------------------------------------------------------
    print("\n" + "#" * 70)
    print("LAUNCHING PHASE 2: Split B + Buffer A (~35.7k images)")
    print("#" * 70)
    t0 = time.time()
    
    run_training(
        epochs=15,
        batch_size=16,
        optimizer="SGD",
        lr0=0.003,          # Reduced learning rate to protect Phase 1 features
        patience=15,
        use_subset=False,   # Override with custom data YAML
        weights=weights_phase1,
        data_yaml="dataset_seq_phase2.yaml"
    )
    
    best_weights_src = "D:/Road Hazards-Detection/models/best.pt"
    if os.path.exists(best_weights_src):
        shutil.copy(best_weights_src, weights_phase2)
        print(f"\n[+] Phase 2 complete in {(time.time() - t0)/3600:.2f} hours.")
        print(f"[+] Saved weights to: {weights_phase2}")
    else:
        raise FileNotFoundError("Could not locate best.pt weights after Phase 2 training!")

    # -------------------------------------------------------------------------
    # PHASE 3: Train Split C + Buffer AB
    # -------------------------------------------------------------------------
    print("\n" + "#" * 70)
    print("LAUNCHING PHASE 3: Split C + Buffer AB (~39.0k images)")
    print("#" * 70)
    t0 = time.time()
    
    run_training(
        epochs=15,
        batch_size=16,
        optimizer="SGD",
        lr0=0.002,          # Further decayed learning rate
        patience=15,
        use_subset=False,
        weights=weights_phase2,
        data_yaml="dataset_seq_phase3.yaml"
    )
    
    if os.path.exists(best_weights_src):
        shutil.copy(best_weights_src, weights_phase3)
        print(f"\n[+] Phase 3 complete in {(time.time() - t0)/3600:.2f} hours.")
        print(f"[+] Saved weights to: {weights_phase3}")
    else:
        raise FileNotFoundError("Could not locate best.pt weights after Phase 3 training!")

    # -------------------------------------------------------------------------
    # PHASE 4: Train Split D + Buffer ABC
    # -------------------------------------------------------------------------
    print("\n" + "#" * 70)
    print("LAUNCHING PHASE 4 (FINAL PHASE): Split D + Buffer ABC (~42.2k images)")
    print("#" * 70)
    t0 = time.time()
    
    run_training(
        epochs=15,
        batch_size=16,
        optimizer="SGD",
        lr0=0.001,          # Final decayed learning rate
        patience=15,
        use_subset=False,
        weights=weights_phase3,
        data_yaml="dataset_seq_phase4.yaml"
    )
    
    if os.path.exists(best_weights_src):
        shutil.copy(best_weights_src, weights_final)
        print(f"\n[+] Phase 4 complete in {(time.time() - t0)/3600:.2f} hours.")
        print(f"[+] Saved final weights to: {weights_final}")
    else:
        raise FileNotFoundError("Could not locate best.pt weights after Phase 4 training!")

    # -------------------------------------------------------------------------
    # EVALUATION & PIPELINE VERIFICATION
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RUNNING FINAL TEST SPLIT EVALUATION (43,363 images)")
    print("=" * 70)
    run_test_validation(model_path=best_weights_src)
    
    print("\n" + "=" * 70)
    print("RUNNING END-TO-END ADAS PIPELINE VERIFICATION")
    print("=" * 70)
    verify_pipeline_end_to_end()
    
    print("\n" + "=" * 70)
    print("[+] AUTONOMOUS SEQUENTIAL TRAINING RUN SUCCESSFULLY COMPLETED!")
    print("=" * 70)

if __name__ == "__main__":
    main()
