import os
import random
import yaml
import argparse
import src.config as config

def create_dataset_subset(sample_ratio=0.10, seed=42):
    """
    Creates a representative subset of the dataset.
    Generates text files listing subset image paths and a dataset_subset.yaml.
    Does not copy any image/label files, saving disk space and time.
    Uses in-memory set lookups for ultra-fast performance.
    """
    print("=" * 60)
    print(f"CREATING REPRESENTATIVE DATASET SUBSET (Ratio: {sample_ratio:.2%})")
    print("=" * 60)
    
    random.seed(seed)
    dataset_dir = config.DATASET_DIR
    
    splits = ["train", "val", "test"]
    sampled_counts = {}
    
    for split in splits:
        img_dir = os.path.join(dataset_dir, "images", split)
        lbl_dir = os.path.join(dataset_dir, "labels", split)
        
        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            print(f"[!] Warning: Directory for split '{split}' not found. Skipping.")
            continue
            
        # List all images and labels (in-memory optimization)
        all_files = os.listdir(img_dir)
        img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        all_images = [f for f in all_files if f.lower().endswith(img_extensions)]
        
        # Load all label filenames into a set for O(1) in-memory lookups
        all_labels = set(os.listdir(lbl_dir))
        
        valid_pairs = []
        for img in all_images:
            base_name, _ = os.path.splitext(img)
            lbl_filename = base_name + ".txt"
            if lbl_filename in all_labels:
                # Use absolute paths to prevent any path resolution bugs in YOLOv8
                abs_img_path = os.path.abspath(os.path.join(img_dir, img)).replace("\\", "/")
                valid_pairs.append(abs_img_path)
                
        total_valid = len(valid_pairs)
        if total_valid == 0:
            print(f"[!] No valid image-label pairs found in split '{split}'.")
            continue
            
        # Sample subset
        sample_size = max(1, int(total_valid * sample_ratio))
        sampled_pairs = random.sample(valid_pairs, sample_size)
        sampled_counts[split] = (sample_size, total_valid)
        
        # Write to txt file
        txt_filename = f"{split}_subset.txt"
        txt_path = os.path.join(dataset_dir, txt_filename)
        with open(txt_path, 'w') as f:
            f.write('\n'.join(sampled_pairs) + '\n')
            
        print(f"[+] Sampled {sample_size:,} / {total_valid:,} ({sample_size/total_valid:.2%}) files for split '{split}'.")
        print(f"    Saved list to: {txt_path}")
        
    # Generate dataset_subset.yaml
    original_yaml_path = os.path.join(dataset_dir, "dataset.yaml")
    if os.path.exists(original_yaml_path):
        with open(original_yaml_path, 'r') as f:
            data = yaml.safe_load(f)
    else:
        data = {
            'names': {i: name for i, name in config.CLASSES.items()},
            'nc': len(config.CLASSES),
            'path': dataset_dir.replace("\\", "/")
        }
        
    # Update paths to the generated txt files
    data['train'] = "train_subset.txt"
    data['val'] = "val_subset.txt"
    data['test'] = "test_subset.txt"
    data['path'] = dataset_dir.replace("\\", "/")
    
    subset_yaml_path = os.path.join(dataset_dir, "dataset_subset.yaml")
    with open(subset_yaml_path, 'w') as f:
        yaml.safe_dump(data, f, default_flow_style=False)
        
    print("-" * 60)
    print(f"[+] Generated dataset_subset.yaml at: {subset_yaml_path}")
    print("=" * 60)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a representative subset of the road hazards dataset.")
    parser.add_argument("--ratio", type=float, default=0.10, help="Sampling ratio (0.0 to 1.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    create_dataset_subset(sample_ratio=args.ratio, seed=args.seed)
