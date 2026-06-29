"""
Explainability Module for YOLOv8 Hazard Detection.

Provides visual heatmaps showing which image regions the YOLOv8 model
focuses on. Supports four methods:
1. HiResCAM (Element-wise activations multiplied by positive gradients; default, strictly localized)
2. EigenGrad-CAM (SVD/PCA on gradient-weighted activations)
3. Eigen-CAM (SVD/PCA of activations; gradient-free)
4. Feature Activation Mapping (FAM; simple average activation; gradient-free)
5. Grad-CAM (gradient-weighted class activation mapping; traditional)

Reference: Draelos & Liny, "Use HiResCAM instead of Grad-CAM for faithful
           explanations of convolutional neural networks", arXiv:2011.08891, 2020.
           Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
           via Gradient-based Localization", ICCV 2017.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class YOLOExplainer:
    """
    Generates explainability heatmaps for YOLOv8.
    Supports HiResCAM, EigenGrad-CAM, Eigen-CAM, FAM, and Grad-CAM.
    """

    def __init__(self, yolo_model, method='hirescam', target_layer_idx=9):
        """
        Args:
            yolo_model: Ultralytics YOLO model instance
            method: 'hirescam', 'eigengradcam', 'eigencam', 'fam', or 'gradcam' (default: 'hirescam')
            target_layer_idx: Index of target layer in model.model.model (default: 9 = SPPF)
        """
        self.model = yolo_model
        self.method = method.lower()
        self.activations = None
        self.gradients = None

        # Target layer: SPPF (layer 9) — last layer of CSPDarknet backbone
        self.target_layer = self.model.model.model[target_layer_idx]

        # Register hooks
        self._fwd_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self._bwd_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Explicitly move the underlying DetectionModel to the target device
        self.model.model.to(self.device)

    def _forward_hook(self, module, input, output):
        """Capture feature map activations during forward pass."""
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        """Capture gradients during backward pass."""
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, image, img_size=640, method=None):
        """
        Generate an explainability heatmap for the given image.

        Args:
            image: Input image (BGR, numpy array)
            img_size: Target size for model input (default 640)
            method: Override default method ('hirescam', 'eigengradcam', 'eigencam', 'fam', 'gradcam')

        Returns:
            cam: Normalized heatmap array (0.0–1.0), shape (img_size, img_size)
            heatmap_colored: JET-colored heatmap (BGR uint8), shape (img_size, img_size, 3)
            overlay: Heatmap blended with original image (BGR uint8)
        """
        if method is None:
            method = self.method
        method = method.lower()

        # Prepare input tensor
        img_resized = cv2.resize(image, (img_size, img_size))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # Clear hooks/state
        self.activations = None
        self.gradients = None
        self.model.model.eval()

        if method == 'hirescam':
            # HiResCAM: Element-wise product of activations and positive gradients
            img_tensor.requires_grad_(True)
            try:
                with torch.enable_grad():
                    preds = self.model.model(img_tensor)
                    pred_tensor = preds[0] if isinstance(preds, (list, tuple)) else preds
                    pred_tensor = pred_tensor.to(self.device)

                    # Filter class scores to target only actual hazard detections
                    class_scores = pred_tensor[0, 4:, :]  # [num_classes, num_preds]
                    max_scores, _ = class_scores.max(dim=0)
                    
                    high_conf_mask = max_scores > 0.25
                    if high_conf_mask.sum() > 0:
                        target = max_scores[high_conf_mask].sum()
                    else:
                        target = max_scores.max()

                    self.model.model.zero_grad()
                    target.backward()

                if self.gradients is not None and self.activations is not None:
                    # HiResCAM: Element-wise product ensures strict localization to gradient region
                    cam_tensor = (self.activations * F.relu(self.gradients)).sum(dim=1, keepdim=True)
                    cam = cam_tensor.squeeze().cpu().numpy()
                else:
                    print("[!] No gradients captured. Falling back to FAM.")
                    cam = self.activations.squeeze(0).abs().mean(dim=0).cpu().numpy()

            except Exception as e:
                print(f"[!] HiResCAM failed: {e}. Falling back to FAM.")
                cam = self.activations.squeeze(0).abs().mean(dim=0).cpu().numpy()

        elif method == 'eigengradcam':
            # EigenGrad-CAM: SVD on activations weighted by positive gradients
            img_tensor.requires_grad_(True)
            try:
                with torch.enable_grad():
                    preds = self.model.model(img_tensor)
                    pred_tensor = preds[0] if isinstance(preds, (list, tuple)) else preds
                    pred_tensor = pred_tensor.to(self.device)

                    class_scores = pred_tensor[0, 4:, :]
                    max_scores, _ = class_scores.max(dim=0)
                    
                    high_conf_mask = max_scores > 0.25
                    if high_conf_mask.sum() > 0:
                        target = max_scores[high_conf_mask].sum()
                    else:
                        target = max_scores.max()

                    self.model.model.zero_grad()
                    target.backward()

                if self.gradients is not None and self.activations is not None:
                    weighted_act = self.activations * F.relu(self.gradients)
                    cam = self._compute_svd(weighted_act)
                else:
                    print("[!] No gradients captured. Falling back to Eigen-CAM.")
                    cam = self._compute_svd(self.activations)

            except Exception as e:
                print(f"[!] EigenGrad-CAM failed: {e}. Falling back to Eigen-CAM.")
                with torch.no_grad():
                    _ = self.model.model(img_tensor)
                cam = self._compute_svd(self.activations)

        elif method == 'gradcam':
            # Grad-CAM: Gradient-weighted activation mapping with high-conf target filtering
            img_tensor.requires_grad_(True)
            try:
                with torch.enable_grad():
                    preds = self.model.model(img_tensor)
                    pred_tensor = preds[0] if isinstance(preds, (list, tuple)) else preds
                    pred_tensor = pred_tensor.to(self.device)

                    class_scores = pred_tensor[0, 4:, :]
                    max_scores, _ = class_scores.max(dim=0)
                    
                    high_conf_mask = max_scores > 0.25
                    if high_conf_mask.sum() > 0:
                        target = max_scores[high_conf_mask].sum()
                    else:
                        target = max_scores.max()

                    self.model.model.zero_grad()
                    target.backward()

                if self.gradients is not None and self.activations is not None:
                    weights = self.gradients.mean(dim=(2, 3), keepdim=True)
                    cam_tensor = (weights * self.activations).sum(dim=1, keepdim=True)
                    cam_tensor = F.relu(cam_tensor)
                    cam = cam_tensor.squeeze().cpu().numpy()
                else:
                    print("[!] Grad-CAM failed (no gradients). Falling back to FAM.")
                    cam = self.activations.squeeze(0).abs().mean(dim=0).cpu().numpy()

            except Exception as e:
                print(f"[!] Grad-CAM failed: {e}. Falling back to FAM.")
                cam = self.activations.squeeze(0).abs().mean(dim=0).cpu().numpy()

        elif method == 'eigencam':
            # Eigen-CAM: Principal Component Analysis (SVD) of activations
            with torch.no_grad():
                _ = self.model.model(img_tensor)

            if self.activations is not None:
                cam = self._compute_svd(self.activations)
            else:
                print("[!] No activations captured. Using empty heatmap.")
                cam = np.zeros((img_size, img_size), dtype=np.float32)

        elif method == 'fam':
            # FAM: Simple feature activation mapping (mean absolute value)
            with torch.no_grad():
                _ = self.model.model(img_tensor)

            if self.activations is not None:
                cam = self.activations.squeeze(0).abs().mean(dim=0).cpu().numpy()
            else:
                print("[!] No activations captured. Using empty heatmap.")
                cam = np.zeros((img_size, img_size), dtype=np.float32)
        else:
            raise ValueError(f"Unknown explainability method: {method}")

        # Normalize heatmap to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        # Resize to input dimensions
        cam = cv2.resize(cam, (img_size, img_size))

        # Create colored heatmap (JET colormap)
        heatmap_uint8 = np.uint8(255 * cam)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Overlay: blend heatmap with original image
        overlay = cv2.addWeighted(img_resized, 0.55, heatmap_colored, 0.45, 0)

        return cam, heatmap_colored, overlay

    def _compute_svd(self, tensor):
        """Compute the first principal component of the given activation tensor."""
        act = tensor.squeeze(0)  # [C, H, W]
        C, H, W = act.shape

        # Reshape to [C, H*W] and transpose to [H*W, C]
        reshaped = act.reshape(C, H * W).T  # [H*W, C]

        # Center the data
        reshaped_centered = reshaped - reshaped.mean(dim=0, keepdim=True)

        try:
            # Perform SVD natively on active device
            U, S, Vh = torch.linalg.svd(reshaped_centered, full_matrices=False)
            
            # First principal component (eigenvector) corresponds to Vh[0] (shape [C])
            v = Vh[0]

            # Project raw activations onto the first principal component
            projection = reshaped @ v  # [H*W]
            cam = projection.reshape(H, W)

            # Resolve SVD sign ambiguity: ensure positive correlation with mean activation map
            mean_act = act.mean(dim=0)
            dot_product = (cam * mean_act).sum()
            if dot_product < 0:
                cam = -cam

        except Exception as e:
            print(f"[!] SVD failed: {e}. Falling back to mean map.")
            return act.abs().mean(dim=0).detach().cpu().numpy()

        return cam.detach().cpu().numpy()

    def generate_detection_overlay(self, image, detections, img_size=640, method=None):
        """
        Generate explainability heatmap with detection bounding boxes overlaid.

        Args:
            image: Input image (BGR)
            detections: List of dicts with 'bbox', 'class_label', 'confidence_score'
            img_size: Target size
            method: Explainability method override

        Returns:
            overlay: Image with heatmap + detection boxes drawn
        """
        _, _, overlay = self.generate_heatmap(image, img_size, method)

        colors = {
            "Pothole": (0, 0, 255),
            "Crack": (0, 165, 255),
            "Manhole": (255, 0, 0),
            "Waterlogging": (255, 255, 0),
            "Speedbump": (0, 255, 255),
            "Road_Barrier": (255, 0, 255),
        }

        for det in detections:
            if det["class_label"] in ["Construction_Object", "Traffic_Cone"]:
                continue
            bbox = det["bbox"]
            cls_name = det["class_label"]
            conf = det["confidence_score"]
            color = colors.get(cls_name, (0, 255, 0))

            xmin, ymin, xmax, ymax = map(int, bbox)
            cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), color, 2)

            label = f"{cls_name} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(overlay, (xmin, ymin - 18), (xmin + w + 4, ymin), color, -1)
            cv2.putText(overlay, label, (xmin + 2, ymin - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return overlay

    def cleanup(self):
        """Remove registered hooks to free memory."""
        self._fwd_handle.remove()
        self._bwd_handle.remove()


# Backward-compatible alias for existing imports
GradCAMExplainer = YOLOExplainer
