import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import src.config as config
from src.pipeline import ADASPipeline
from src.distance_risk import (
    estimate_distance, 
    calculate_ttc, 
    get_lane_alignment_factor, 
    compute_risk_score
)
from src.alerts_logging import (
    classify_risk, 
    draw_hud
)

def run_test_validation(model_path=None, imgsz=None):
    """
    Evaluates the fine-tuned model on the test partition.
    Prints mAP@0.5 and other performance metrics.
    """
    model_path = model_path if model_path else config.BEST_MODEL_PATH
    if not os.path.exists(model_path):
        print(f"[!] Warning: Custom weights not found at {model_path}. Using yolov8s.pt instead.")
        model_path = "yolov8s.pt"
        
    print(f"[*] Validating model {model_path} on test dataset...")
    model = YOLO(model_path)
    
    # Run validation on test set
    yaml_path = os.path.join(config.DATASET_DIR, "dataset.yaml")
    results = model.val(
        data=yaml_path.replace("\\", "/"),
        split="test",
        imgsz=imgsz if imgsz is not None else config.IMG_SIZE,
        device=0 if torch.cuda.is_available() else "cpu",
        verbose=True
    )
    
    # Extract key metrics
    map50 = results.results_dict.get("metrics/mAP50(B)", 0.0)
    map50_95 = results.results_dict.get("metrics/mAP50-95(B)", 0.0)
    precision = results.results_dict.get("metrics/precision(B)", 0.0)
    recall = results.results_dict.get("metrics/recall(B)", 0.0)
    
    print("=" * 50)
    print("TEST EVALUATION METRICS")
    print("=" * 50)
    print(f"Precision:         {precision:.4f}")
    print(f"Recall:            {recall:.4f}")
    print(f"mAP@0.5:           {map50:.4f} (Target: >= 0.80)")
    print(f"mAP@0.5:0.95:      {map50_95:.4f}")
    print("=" * 50)
    
    if map50 >= 0.80:
        print("[+] SUCCESS: Model achieved or exceeded the target mAP@0.5 of 0.80!")
    else:
        print("[!] NOTICE: Model did not meet the 0.80 mAP target. Consider training for more epochs.")
        
    return results

def profile_pipeline_latency(num_frames=100):
    """
    Profiles the frame-by-frame processing latency of each component in the pipeline.
    Prints a latency breakdown table and generates a bar chart.
    """
    print(f"[*] Starting latency profiling over {num_frames} simulated frames...")
    
    pipeline = ADASPipeline()
    
    # Create a dummy frame (640x640)
    dummy_frame = np.random.randint(0, 256, (config.IMG_SIZE, config.IMG_SIZE, 3), dtype=np.uint8)
    
    # Warmup runs to initialize models on GPU
    for _ in range(5):
        _, _ = pipeline.process_frame(dummy_frame)
        
    # Latency accumulation lists (in ms)
    latencies = {
        "Preprocessing": [],
        "Segmentation": [],
        "ROI Masking": [],
        "YOLO Detection": [],
        "Risk & Distance": [],
        "HUD & Alerts": []
    }
    
    for i in range(num_frames):
        # 1. Preprocessing
        t0 = time.perf_counter()
        normalized_frame, enhanced_frame = pipeline.process_frame_preprocess(dummy_frame) if hasattr(pipeline, 'process_frame_preprocess') else (None, dummy_frame)
        from src.preprocess import preprocess_pipeline
        normalized_frame, enhanced_frame = preprocess_pipeline(dummy_frame)
        t1 = time.perf_counter()
        latencies["Preprocessing"].append((t1 - t0) * 1000)
        
        # 2. Segmentation
        t0 = time.perf_counter()
        seg_mask = pipeline.segmenter.run_inference(enhanced_frame)
        t1 = time.perf_counter()
        latencies["Segmentation"].append((t1 - t0) * 1000)
        
        # 3. ROI Masking
        t0 = time.perf_counter()
        roi_frame, binary_drivable_mask = pipeline.detector.extract_roi(enhanced_frame, seg_mask)
        t1 = time.perf_counter()
        latencies["ROI Masking"].append((t1 - t0) * 1000)
        
        # 4. YOLO Detection
        t0 = time.perf_counter()
        raw_detections = pipeline.detector.detect_hazards(roi_frame)
        t1 = time.perf_counter()
        latencies["YOLO Detection"].append((t1 - t0) * 1000)
        
        # 5. Risk & Distance
        t0 = time.perf_counter()
        processed_detections = []
        for det in raw_detections:
            distance = estimate_distance(det["bbox"], det["class_label"])
            ttc = calculate_ttc(distance, config.DEFAULT_SPEED_KMH)
            lane_factor = get_lane_alignment_factor(det["bbox"], seg_mask)
            risk_score = compute_risk_score(det["class_label"], distance, config.DEFAULT_SPEED_KMH, lane_factor)
            risk_level = classify_risk(risk_score)
            processed_detections.append({
                "bbox": det["bbox"],
                "class_label": det["class_label"],
                "distance": distance,
                "ttc": ttc,
                "risk_score": risk_score,
                "risk_level": risk_level
            })
        t1 = time.perf_counter()
        latencies["Risk & Distance"].append((t1 - t0) * 1000)
        
        # 6. HUD & Alerts
        t0 = time.perf_counter()
        seg_color_mask = pipeline.segmenter.get_color_mask(seg_mask)
        _ = draw_hud(enhanced_frame, processed_detections, config.DEFAULT_SPEED_KMH, seg_color_mask)
        # Suppress sound in benchmarking to avoid system beeps
        # (sound trigger would normally go here)
        t1 = time.perf_counter()
        latencies["HUD & Alerts"].append((t1 - t0) * 1000)

    # Compute statistics
    print("\n" + "=" * 60)
    print(f"{'Pipeline Component':<25} | {'Mean Latency (ms)':<18} | {'Std Dev (ms)':<12}")
    print("=" * 60)
    
    means = []
    names = []
    total_mean = 0.0
    
    for key, values in latencies.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        means.append(mean_val)
        names.append(key)
        total_mean += mean_val
        print(f"{key:<25} | {mean_val:<18.2f} | {std_val:<12.2f}")
        
    print("-" * 60)
    print(f"{'TOTAL LATENCY':<25} | {total_mean:<18.2f} | N/A")
    print(f"{'FPS (Equivalent)':<25} | {1000.0/total_mean:<18.2f} | N/A")
    print("=" * 60)
    
    # Latency Budget Target Comparison (From flow chart budget ~31ms on Jetson Orin NX)
    # Target: Pre ~2ms, Seg ~12ms, YOLO ~10ms, Fusion ~3ms, Risk ~1ms, Alert ~3ms
    target_latencies = {
        "Preprocessing": 2.0,
        "Segmentation": 12.0,
        "ROI Masking": 0.5,
        "YOLO Detection": 10.0,
        "Risk & Distance": 4.0, # Fusion + Risk
        "HUD & Alerts": 3.0     # Alert
    }
    targets = [target_latencies.get(n, 0.0) for n in names]
    
    # Generate bar chart comparison
    x = np.arange(len(names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, means, width, label='Measured (Local RTX 3050)', color='#3b82f6')
    rects2 = ax.bar(x + width/2, targets, width, label='Target Budget (Jetson Orin NX)', color='#f59e0b', alpha=0.8)
    
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Pipeline Component Latency Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}ms',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
                        
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    chart_path = os.path.join(config.LOG_DIR, "latency_breakdown.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"[+] Latency breakdown visualization saved to {chart_path}")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 model on Road Hazards dataset.")
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights file")
    parser.add_argument("--imgsz", type=int, default=None, help="Inference image size resolution")
    parser.add_argument("--profile", action="store_true", help="Profile pipeline component latency")
    args = parser.parse_args()
    
    if args.profile:
        profile_pipeline_latency()
    else:
        run_test_validation(model_path=args.weights, imgsz=args.imgsz)
