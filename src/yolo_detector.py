import cv2
import numpy as np
import src.config as config
# Ultralytics will be loaded dynamically to avoid import crashes if not yet fully installed
# but we write the import here as it will be run when executing the pipeline.

class YOLODetector:
    """
    Wrapper for YOLOv8 Hazard Detection.
    Handles model loading, ROI masking, and inference post-processing.
    """
    def __init__(self, model_path=None, ensemble=True):
        from ultralytics import YOLO
        import os
        self.model_path = model_path if model_path else config.BEST_MODEL_PATH
        print(f"[*] Loading primary YOLOv8 model from {self.model_path}...")
        
        # Load the model. If best.pt does not exist, we fall back to a pretrained yolov8s.pt
        try:
            self.model = YOLO(self.model_path)
        except Exception as e:
            print(f"[!] Warning: Could not load weights from {self.model_path} ({e}).")
            print("[*] Falling back to standard yolov8s.pt...")
            self.model = YOLO("yolov8s.pt")
            
        self.secondary_model = None
        self.ensemble = ensemble
        
        if self.ensemble:
            base_dir = os.path.dirname(self.model_path)
            large_path = os.path.join(base_dir, "best_large.pt")
            if os.path.exists(large_path) and os.path.abspath(large_path) != os.path.abspath(self.model_path):
                print(f"[*] Ensembling enabled: Loading secondary model from {large_path}...")
                try:
                    self.secondary_model = YOLO(large_path)
                except Exception as e:
                    print(f"[!] Warning: Could not load secondary weights ({e}).")

    def extract_roi(self, frame_enhanced, segmentation_mask):
        """
        Step 5: ROI Extraction.
        Masks the drivable region onto the frame and suppresses the non-road background.
        Drivable region is where segmentation mask is 1 (Direct), 2 (Alt), 3 (Waterlogged), or 4 (Unpaved).
        """
        # Create binary mask for drivable regions
        # 1: Drivable Direct, 2: Drivable Alt, 3: Waterlogged, 4: Unpaved
        drivable_mask = np.isin(segmentation_mask, [1, 2, 3, 4]).astype(np.uint8)
        
        # Apply binary dilation to slightly expand the mask edges (to capture hazards right at the edge of the road)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        drivable_mask_dilated = cv2.dilate(drivable_mask, kernel, iterations=2)
        
        # Mask the frame (suppress non-road background by multiplying)
        roi_frame = cv2.bitwise_and(frame_enhanced, frame_enhanced, mask=drivable_mask_dilated)
        return roi_frame, drivable_mask_dilated

    def detect_hazards(self, roi_frame, conf_threshold=0.25):
        """
        Step 6: YOLOv8 Hazard Detection.
        Runs fine-tuned YOLOv8 on the masked ROI.
        """
        results = self.model(roi_frame, verbose=False, conf=conf_threshold)[0]
        
        detections = []
        boxes = results.boxes
        
        for box in boxes:
            # Get box coordinates, conf, and class index
            xyxy = box.xyxy[0].cpu().numpy().tolist()  # [xmin, ymin, xmax, ymax]
            conf = float(box.conf[0].cpu().numpy())
            cls_idx = int(box.cls[0].cpu().numpy())
            
            class_name = config.CLASSES.get(cls_idx, f"Unknown_{cls_idx}")
            
            detections.append({
                "bbox": xyxy,
                "class_label": class_name,
                "confidence_score": conf,
                "class_id": cls_idx
            })
            
        # Run secondary model if ensembling is active and secondary model is loaded
        if self.ensemble and self.secondary_model is not None:
            results_sec = self.secondary_model(roi_frame, verbose=False, conf=conf_threshold)[0]
            boxes_sec = results_sec.boxes
            
            sec_detections = []
            for box in boxes_sec:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().numpy())
                cls_idx = int(box.cls[0].cpu().numpy())
                
                class_name = config.CLASSES.get(cls_idx, f"Unknown_{cls_idx}")
                
                # Check if this overlaps significantly with any primary detection
                has_overlap = False
                for prim in detections:
                    iou = self.calculate_iou(xyxy, prim["bbox"])
                    if iou > 0.15:  # Overlap threshold
                        has_overlap = True
                        break
                        
                if not has_overlap:
                    sec_detections.append({
                        "bbox": xyxy,
                        "class_label": class_name,
                        "confidence_score": conf,
                        "class_id": cls_idx
                    })
                
            if sec_detections:
                # We append non-overlapping secondary detections directly to the primary detections
                # and do NOT run global NMS so that primary detections are never suppressed.
                detections.extend(sec_detections)
            
        return detections

    def calculate_iou(self, boxA, boxB):
        ixmin = max(boxA[0], boxB[0])
        iymin = max(boxA[1], boxB[1])
        ixmax = min(boxA[2], boxB[2])
        iymax = min(boxA[3], boxB[3])
        
        i_w = max(0, ixmax - ixmin)
        i_h = max(0, iymax - iymin)
        intersection = i_w * i_h
        
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        union = areaA + areaB - intersection
        return (intersection / union) if union > 0 else 0

    def nms(self, detections, iou_threshold=0.40):
        if not detections:
            return []
            
        # Group detections by class_id
        class_groups = {}
        for det in detections:
            cls_id = det["class_id"]
            if cls_id not in class_groups:
                class_groups[cls_id] = []
            class_groups[cls_id].append(det)
            
        keep_all = []
        for cls_id, cls_dets in class_groups.items():
            # Sort detections by confidence score descending
            sorted_dets = sorted(cls_dets, key=lambda x: x["confidence_score"], reverse=True)
            keep_cls = []
            
            while sorted_dets:
                curr = sorted_dets.pop(0)
                keep_cls.append(curr)
                
                filtered = []
                for det in sorted_dets:
                    iou = self.calculate_iou(curr["bbox"], det["bbox"])
                    if iou < iou_threshold:
                        filtered.append(det)
                sorted_dets = filtered
                
            keep_all.extend(keep_cls)
            
        return keep_all

    def detect_hazards_sahi(self, enhanced_frame, conf_threshold=0.25, use_sahi=True):
        """
        Runs YOLOv8 hazard detection with Slicing-Aided Hyper Inference (SAHI)
        to magnify and detect small objects (Cracks, Traffic Cones) in the lower road region.
        """
        # 1. Full Frame Detection
        full_dets = self.detect_hazards(enhanced_frame, conf_threshold)
        if not use_sahi:
            return full_dets
            
        # 2. Slice lower road region (y in [240, 640], height = 400)
        # Slices are: Left [0, 400] and Right [240, 640]
        y_start, y_end = 240, 640
        
        patch_left = enhanced_frame[y_start:y_end, 0:400]
        patch_right = enhanced_frame[y_start:y_end, 240:640]
        
        # Run YOLO on patches
        dets_left = self.detect_hazards(patch_left, conf_threshold)
        dets_right = self.detect_hazards(patch_right, conf_threshold)
        
        # Map coordinates back, filtering for targeted small object classes:
        # 0: Pothole, 1: Crack, 5: Traffic_Cone
        mapped_dets = list(full_dets)
        target_classes = {"Crack", "Traffic_Cone", "Pothole"}
        
        for det in dets_left:
            if det["class_label"] in target_classes:
                bbox = det["bbox"]
                det["bbox"] = [bbox[0], bbox[1] + y_start, bbox[2], bbox[3] + y_start]
                mapped_dets.append(det)
            
        for det in dets_right:
            if det["class_label"] in target_classes:
                bbox = det["bbox"]
                det["bbox"] = [bbox[0] + 240, bbox[1] + y_start, bbox[2] + 240, bbox[3] + y_start]
                mapped_dets.append(det)
            
        # Apply Non-Maximum Suppression (NMS) to merge overlapping detections
        merged_detections = self.nms(mapped_dets, iou_threshold=0.40)
        return merged_detections
