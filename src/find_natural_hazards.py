import os
import src.config as config
import shutil

def find_natural_hazards():
    dataset_dir = "C:/augmented_dataset"
    labels_dir = os.path.join(dataset_dir, "labels", "test")
    images_dir = os.path.join(dataset_dir, "images", "test")
    
    if not os.path.exists(labels_dir):
        print(f"[!] Error: Labels directory not found at: {labels_dir}")
        return
        
    print("[*] Searching dataset for natural hazards (Pothole, Crack, Manhole, Waterlogging, Speedbump, Road_Barrier)...")
    
    # Natural hazard class IDs: 0 (Pothole), 1 (Crack), 2 (Manhole), 3 (Waterlogging), 4 (Speedbump), 6 (Road_Barrier)
    natural_classes = {0, 1, 2, 3, 4, 6}
    class_names = {
        0: "Pothole",
        1: "Crack",
        2: "Manhole",
        3: "Waterlogging",
        4: "Speedbump",
        6: "Road_Barrier"
    }
    
    # We want to find one good image for each environmental domain:
    # original/daylight (orig_), dust (dust_), fog (fog_), night (night_), rain (rain_)
    domains = ["orig", "dust", "fog", "night", "rain"]
    found_cases = {}
    
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    
    for filename in label_files:
        # Determine the domain from the filename prefix
        matched_domain = None
        for d in domains:
            if filename.startswith(f"{d}_"):
                matched_domain = d
                break
                
        if not matched_domain or matched_domain in found_cases:
            continue
            
        label_path = os.path.join(labels_dir, filename)
        
        # Read labels
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        detected_natural = []
        for line in lines:
            parts = line.strip().split()
            if parts:
                class_id = int(parts[0])
                if class_id in natural_classes:
                    detected_natural.append(class_id)
                    
        # If the image has at least one natural hazard, save it
        if detected_natural:
            base_name = os.path.splitext(filename)[0]
            img_filename = f"{base_name}.jpg"
            img_path = os.path.join(images_dir, img_filename)
            
            if os.path.exists(img_path):
                found_cases[matched_domain] = {
                    "image_name": img_filename,
                    "image_path": img_path,
                    "classes": [class_names[cid] for cid in detected_natural]
                }
                print(f"  [+] Found {matched_domain} case: {img_filename} containing {found_cases[matched_domain]['classes']}")
                
        # Break early if we have all 5 domains
        if len(found_cases) == len(domains):
            break
            
    print("\n[*] Copying found natural hazard images to working directory...")
    target_inputs_dir = os.path.join(config.LOG_DIR, "inputs")
    os.makedirs(target_inputs_dir, exist_ok=True)
    
    for domain, data in found_cases.items():
        src = data["image_path"]
        # Rename to a standard test name: daylight_natural.jpg, rain_natural.jpg, etc.
        target_name = f"{domain}_natural.jpg"
        dst = os.path.join(target_inputs_dir, target_name)
        shutil.copy(src, dst)
        print(f"  [+] Copied {src} to {dst}")
        
    print("[+] Natural hazard search and copy complete!")

if __name__ == "__main__":
    find_natural_hazards()
