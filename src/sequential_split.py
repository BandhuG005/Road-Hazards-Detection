import os
import random
import yaml
import src.config as config

def create_sequential_splits(seed=42):
    """
    Splits the SSD dataset into 4 equal chunks and prepares the training path lists
    and replay buffers for sequential training.
    """
    print("=" * 60)
    print("GENERATING SEQUENTIAL MEMORY-BUFFERED TRAINING SPLITS")
    print("=" * 60)
    
    random.seed(seed)
    dataset_dir = config.DATASET_DIR
    
    # 1. Scan and find all valid train image-label pairs on the SSD
    img_dir = os.path.join(dataset_dir, "images", "train")
    lbl_dir = os.path.join(dataset_dir, "labels", "train")
    
    if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
        raise FileNotFoundError(f"Train directory not found at {img_dir}")
        
    all_files = os.listdir(img_dir)
    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    all_images = [f for f in all_files if f.lower().endswith(img_extensions)]
    all_labels = set(os.listdir(lbl_dir))
    
    valid_pairs = []
    for img in all_images:
        base_name, _ = os.path.splitext(img)
        lbl_filename = base_name + ".txt"
        if lbl_filename in all_labels:
            abs_img_path = os.path.abspath(os.path.join(img_dir, img)).replace("\\", "/")
            valid_pairs.append(abs_img_path)
            
    total_train = len(valid_pairs)
    print(f"[*] Found {total_train:,} valid train pairs on SSD.")
    
    # Shuffle and divide into 4 equal splits
    random.shuffle(valid_pairs)
    chunk_size = total_train // 4
    
    split_A = valid_pairs[0:chunk_size]
    split_B = valid_pairs[chunk_size:2*chunk_size]
    split_C = valid_pairs[2*chunk_size:3*chunk_size]
    split_D = valid_pairs[3*chunk_size:]
    
    print(f"    Split A: {len(split_A):,} images")
    print(f"    Split B: {len(split_B):,} images")
    print(f"    Split C: {len(split_C):,} images")
    print(f"    Split D: {len(split_D):,} images")
    
    # 2. Generate Memory Replay Buffers (10% random sample from each split)
    buffer_A = random.sample(split_A, max(1, int(len(split_A) * 0.10)))
    buffer_B = random.sample(split_B, max(1, int(len(split_B) * 0.10)))
    buffer_C = random.sample(split_C, max(1, int(len(split_C) * 0.10)))
    
    # 3. Create training sets for each phase
    # Phase 1: Only Split A
    phase1_list = split_A
    # Phase 2: Split B + Buffer A
    phase2_list = split_B + buffer_A
    # Phase 3: Split C + Buffer A + Buffer B
    phase3_list = split_C + buffer_A + buffer_B
    # Phase 4: Split D + Buffer A + Buffer B + Buffer C
    phase4_list = split_D + buffer_A + buffer_B + buffer_C
    
    lists = [phase1_list, phase2_list, phase3_list, phase4_list]
    phase_names = ["phase1", "phase2", "phase3", "phase4"]
    
    # Get base yaml config info
    original_yaml_path = os.path.join(dataset_dir, "dataset.yaml")
    if os.path.exists(original_yaml_path):
        with open(original_yaml_path, 'r') as f:
            base_data = yaml.safe_load(f)
    else:
        base_data = {
            'names': {i: name for i, name in config.CLASSES.items()},
            'nc': len(config.CLASSES),
            'path': dataset_dir.replace("\\", "/")
        }
        
    for i, phase_list in enumerate(lists):
        phase_name = phase_names[i]
        
        # Write list file
        txt_filename = f"train_seq_{phase_name}.txt"
        txt_path = os.path.join(dataset_dir, txt_filename)
        with open(txt_path, 'w') as f:
            f.write('\n'.join(phase_list) + '\n')
            
        print(f"[+] Saved {phase_name} training list ({len(phase_list):,} images) to: {txt_path}")
        
        # Write YAML config
        phase_data = base_data.copy()
        phase_data['train'] = txt_filename
        # Keep validation and testing mapped to the full validation split on SSD
        phase_data['val'] = "val_subset.txt"
        phase_data['test'] = "test_subset.txt"
        phase_data['path'] = dataset_dir.replace("\\", "/")
        
        yaml_filename = f"dataset_seq_{phase_name}.yaml"
        yaml_path = os.path.join(dataset_dir, yaml_filename)
        with open(yaml_path, 'w') as f:
            yaml.safe_dump(phase_data, f, default_flow_style=False)
            
        print(f"    Saved {phase_name} YAML configuration to: {yaml_path}")
        
    print("=" * 60)
    print("[+] All sequential splits and configurations generated successfully!")
    print("=" * 60)

if __name__ == "__main__":
    create_sequential_splits()
