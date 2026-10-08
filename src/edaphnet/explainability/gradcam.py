"""Grad-CAM (Gradient-weighted Class Activation Mapping) implementation for EDAPHNET.

Generates visual explanations for CNN decisions (Soil & Plant Disease classifiers)
by computing gradients of the target class score with respect to feature maps of the
final convolutional layer.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger(__name__)


class GradCAM:
    """Computes Grad-CAM heatmaps for PyTorch vision models."""

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        """Initialize GradCAM hook on the target convolutional layer.
        
        Args:
            model: PyTorch model (e.g. EfficientNet-B0).
            target_layer: The target convolutional layer. If None, automatically
                          selects the last convolutional layer.
        """
        self.model = model
        self.model.eval()

        self.target_layer = target_layer or self._find_target_layer(model)
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        self._register_hooks()

    def _find_target_layer(self, model: nn.Module) -> nn.Module:
        """Find the final Conv2d layer in the model."""
        target = None
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                target = module
        if target is None:
            # Fallback to last child module
            modules = list(model.children())
            target = modules[-1] if modules else model
        return target

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on the target layer."""
        def forward_hook(module: nn.Module, input: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            self.activations = output.detach()

        def backward_hook(module: nn.Module, grad_input: Tuple[torch.Tensor, ...], grad_output: Tuple[torch.Tensor, ...]) -> None:
            if grad_output and grad_output[0] is not None:
                self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """Generate normalized 2D Grad-CAM heatmap for the target class.
        
        Args:
            input_tensor: Preprocessed image tensor [1, C, H, W].
            target_class_idx: Index of class to explain. If None, uses top predicted class.
            
        Returns:
            Normalized 2D float array in [0, 1] of shape (H, W).
        """
        self.model.zero_grad()
        logits = self.model(input_tensor)

        if target_class_idx is None:
            target_class_idx = int(torch.argmax(logits, dim=1).item())

        score = logits[0, target_class_idx]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            logger.warning("Gradients or activations not captured. Returning uniform heatmap.")
            h, w = input_tensor.shape[2], input_tensor.shape[3]
            return np.zeros((h, w), dtype=np.float32)

        # Global average pooling on gradients across spatial dimensions
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # Linear combination of forward activation maps weighted by gradients
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # [1, 1, H', W']
        cam = torch.relu(cam)  # ReLU to focus on positive contributions

        cam_np = cam.squeeze().cpu().numpy()

        # Handle degenerate case of all zeros
        if np.max(cam_np) > np.min(cam_np):
            cam_np = (cam_np - np.min(cam_np)) / (np.max(cam_np) - np.min(cam_np) + 1e-8)
        else:
            cam_np = np.zeros_like(cam_np)

        # Resize heatmap to match original input tensor dimensions
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam_img = Image.fromarray((cam_np * 255).astype(np.uint8))
        cam_resized = cam_img.resize((w, h), resample=Image.Resampling.BILINEAR)
        return np.array(cam_resized, dtype=np.float32) / 255.0

    @staticmethod
    def overlay_heatmap(
        original_image: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: str = "jet",
    ) -> Image.Image:
        """Overlay normalized heatmap onto the original PIL image.
        
        Args:
            original_image: Source PIL image.
            heatmap: 2D float array in [0, 1] matching image size.
            alpha: Heatmap blend factor (0.0 to 1.0).
            colormap: Pseudocolor mapping scheme.
            
        Returns:
            Blended PIL Image showing visual explanation.
        """
        img_rgb = original_image.convert("RGB")
        w, h = img_rgb.size

        # Resize heatmap if dimensions mismatch
        if heatmap.shape != (h, w):
            heat_img = Image.fromarray((heatmap * 255).astype(np.uint8))
            heatmap = np.array(heat_img.resize((w, h), resample=Image.Resampling.BILINEAR), dtype=np.float32) / 255.0

        # Apply simple color map (JET approximation in pure Python/NumPy)
        # Red = High activation, Yellow/Green = Medium, Blue = Low
        r = np.clip(1.5 - np.abs(heatmap * 4.0 - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(heatmap * 4.0 - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(heatmap * 4.0 - 1.0), 0.0, 1.0)

        color_map = np.stack([r, g, b], axis=-1) * 255.0
        color_img = Image.fromarray(color_map.astype(np.uint8), mode="RGB")

        # Blend
        blended = Image.blend(img_rgb, color_img, alpha=alpha)
        return blended
