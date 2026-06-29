"""
Monte Carlo Dropout Uncertainty Estimation for YOLOv8.

Injects Dropout2d layers into the YOLOv8 backbone via forward hooks and
runs multiple stochastic forward passes to estimate prediction uncertainty.
High uncertainty indicates the model is unsure about a detection, which is
critical for safety in ADAS applications.

Reference: Gal & Ghahramani, "Dropout as a Bayesian Approximation:
           Representing Model Uncertainty in Deep Learning", ICML 2016.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import src.config as config


class MCDropoutEstimator:
    """
    Estimates detection uncertainty using Monte Carlo Dropout.

    Dropout2d layers are injected after C2f backbone blocks via forward hooks.
    Running N stochastic forward passes produces a distribution of predictions,
    whose variance quantifies the model's epistemic uncertainty.
    """

    def __init__(self, yolo_model, n_passes=10, dropout_rate=0.1):
        """
        Args:
            yolo_model: Ultralytics YOLO model instance
            n_passes: Number of stochastic forward passes (default 10)
            dropout_rate: Dropout probability (default 0.1 = 10% of features dropped)
        """
        self.model = yolo_model
        self.n_passes = n_passes
        self.dropout_rate = dropout_rate
        self.hooks = []
        self.dropout_modules = []

        self.device = next(self.model.model.parameters()).device
        self._inject_dropout()

    def _inject_dropout(self):
        """
        Inject Dropout2d after C2f blocks (layers 2, 4, 6, 8) in the backbone.

        Uses forward hooks so the original model architecture is not modified.
        Dropout modules are kept in train() mode so they remain stochastic
        even when the rest of the model is in eval() mode.
        """
        target_indices = [2, 4, 6, 8]  # C2f block indices in YOLOv8s backbone

        for idx in target_indices:
            layer = self.model.model.model[idx]
            dropout = nn.Dropout2d(p=self.dropout_rate).to(self.device)
            dropout.train()  # ALWAYS keep dropout active (stochastic)

            def make_hook(d):
                def hook(module, input, output):
                    d.train()  # Re-enforce train mode each call
                    return d(output)
                return hook

            h = layer.register_forward_hook(make_hook(dropout))
            self.hooks.append(h)
            self.dropout_modules.append(dropout)

        print(f"[+] MC Dropout: Injected Dropout2d(p={self.dropout_rate}) "
              f"after {len(target_indices)} C2f backbone blocks")

    def _calculate_iou(self, boxA, boxB):
        """Calculate Intersection over Union between two [xmin, ymin, xmax, ymax] boxes."""
        ixmin = max(boxA[0], boxB[0])
        iymin = max(boxA[1], boxB[1])
        ixmax = min(boxA[2], boxB[2])
        iymax = min(boxA[3], boxB[3])

        inter = max(0, ixmax - ixmin) * max(0, iymax - iymin)
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        union = areaA + areaB - inter

        return inter / union if union > 0 else 0.0

    def estimate_uncertainty(self, image, conf_threshold=0.25):
        """
        Run N stochastic forward passes and estimate uncertainty for each detection.

        For each base detection (from pass 1), we check how many of the N passes
        also detect the same object (IoU > 0.5, same class). The detection frequency
        and confidence variance together quantify uncertainty.

        Args:
            image: Input image (BGR numpy array or path)
            conf_threshold: Minimum confidence for YOLO detection

        Returns:
            List of detection dicts, each augmented with:
                - 'uncertainty': float 0.0 (certain) to 1.0 (highly uncertain)
                - 'confidence_mean': mean confidence across passes
                - 'confidence_std': standard deviation of confidence
                - 'detection_frequency': fraction of passes that detected this object
                - 'confidence_values': list of per-pass confidence scores
                - 'certainty_level': string label (HIGH / MEDIUM / LOW)
        """
        # Ensure dropout is active
        for d in self.dropout_modules:
            d.train()

        # Run N stochastic forward passes
        all_pass_detections = []

        for pass_idx in range(self.n_passes):
            results = self.model(image, verbose=False, conf=conf_threshold)[0]

            pass_dets = []
            for box in results.boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                conf_val = float(box.conf[0].cpu().numpy())
                cls_idx = int(box.cls[0].cpu().numpy())

                pass_dets.append({
                    "bbox": xyxy,
                    "confidence": conf_val,
                    "class_id": cls_idx,
                })

            all_pass_detections.append(pass_dets)

        # Find the pass with the most detections as the "base" reference
        base_pass_idx = max(range(len(all_pass_detections)),
                           key=lambda i: len(all_pass_detections[i]))
        base_dets = all_pass_detections[base_pass_idx]

        if not base_dets:
            return []

        # For each base detection, track consistency across all passes
        results_with_uncertainty = []

        for base_det in base_dets:
            base_box = base_det["bbox"]
            base_cls = base_det["class_id"]
            class_name = config.CLASSES.get(base_cls, f"Unknown_{base_cls}")

            # Skip construction objects
            if class_name in ["Construction_Object", "Traffic_Cone"]:
                continue

            confidence_values = []
            detection_count = 0

            for pass_dets in all_pass_detections:
                # Find best matching detection in this pass
                best_conf = None
                for pd in pass_dets:
                    if pd["class_id"] == base_cls:
                        iou = self._calculate_iou(base_box, pd["bbox"])
                        if iou > 0.5:
                            if best_conf is None or pd["confidence"] > best_conf:
                                best_conf = pd["confidence"]

                if best_conf is not None:
                    detection_count += 1
                    confidence_values.append(best_conf)

            detection_frequency = detection_count / self.n_passes

            if confidence_values:
                conf_mean = float(np.mean(confidence_values))
                conf_std = float(np.std(confidence_values))
            else:
                conf_mean = base_det["confidence"]
                conf_std = 0.0

            # Uncertainty = 1 - detection_frequency
            # 0.0 = detected in every pass (certain)
            # 1.0 = detected in zero passes (highly uncertain)
            uncertainty = 1.0 - detection_frequency

            # Classify certainty level
            if uncertainty <= 0.2:
                certainty_level = "HIGH"
            elif uncertainty <= 0.5:
                certainty_level = "MEDIUM"
            else:
                certainty_level = "LOW"

            results_with_uncertainty.append({
                "bbox": base_box,
                "class_label": class_name,
                "class_id": base_cls,
                "confidence_mean": conf_mean,
                "confidence_std": conf_std,
                "detection_frequency": detection_frequency,
                "uncertainty": uncertainty,
                "certainty_level": certainty_level,
                "confidence_values": confidence_values,
            })

        return results_with_uncertainty

    def draw_uncertainty_overlay(self, image, detections_with_uncertainty, img_size=640):
        """
        Draw detection boxes color-coded by uncertainty level.

        Green = HIGH certainty (reliable detection)
        Yellow = MEDIUM certainty (moderate confidence)
        Red = LOW certainty (unreliable, flag for review)

        Args:
            image: Input image (BGR)
            detections_with_uncertainty: Output from estimate_uncertainty()
            img_size: Target display size

        Returns:
            annotated: Image with color-coded boxes and uncertainty labels
        """
        annotated = cv2.resize(image.copy(), (img_size, img_size))

        certainty_colors = {
            "HIGH": (0, 200, 0),      # Green — reliable
            "MEDIUM": (0, 200, 255),   # Yellow/Orange — moderate
            "LOW": (0, 0, 255),        # Red — unreliable
        }

        for det in detections_with_uncertainty:
            bbox = det["bbox"]
            cls_name = det["class_label"]
            certainty = det["certainty_level"]
            freq = det["detection_frequency"]
            conf_mean = det["confidence_mean"]
            conf_std = det["confidence_std"]
            uncertainty = det["uncertainty"]

            color = certainty_colors.get(certainty, (200, 200, 200))

            xmin, ymin, xmax, ymax = map(int, bbox)
            cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), color, 2)

            # Main label: class + mean confidence
            label1 = f"{cls_name} {conf_mean:.2f}"
            (w1, h1), _ = cv2.getTextSize(label1, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (xmin, ymin - 34), (xmin + max(w1, 150), ymin), color, -1)
            cv2.putText(annotated, label1, (xmin + 2, ymin - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            # Uncertainty label: frequency + certainty level
            label2 = f"Unc:{uncertainty:.2f} Freq:{freq:.0%} [{certainty}]"
            cv2.putText(annotated, label2, (xmin + 2, ymin - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        # Add legend in top-right corner
        legend_x = img_size - 220
        cv2.rectangle(annotated, (legend_x - 10, 5), (img_size - 5, 75), (40, 40, 40), -1)
        cv2.putText(annotated, "MC Dropout Uncertainty", (legend_x, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        for i, (level, col) in enumerate(certainty_colors.items()):
            y = 35 + i * 15
            cv2.circle(annotated, (legend_x + 5, y), 4, col, -1)
            cv2.putText(annotated, f"{level} Certainty", (legend_x + 15, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        return annotated

    def cleanup(self):
        """Remove injected dropout hooks and free memory."""
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.dropout_modules = []
        print("[+] MC Dropout hooks removed.")
