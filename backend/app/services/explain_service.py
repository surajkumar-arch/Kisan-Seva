"""Explainable AI: Grad-CAM heatmaps and top-k predictions for the plant and soil models."""

import base64
import io
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from backend.app.models.plant_net import get_plant_class_names, get_plant_model
from backend.app.models.soil_net import get_soil_class_names, get_soil_model
from backend.app.utils.image_utils import preprocess_image


def _jet_colormap(cam: np.ndarray) -> np.ndarray:
    """Map a [0,1] array to an RGB jet-like colormap (no matplotlib dependency)."""
    r = np.clip(1.5 - np.abs(4.0 * cam - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4.0 * cam - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4.0 * cam - 1.0), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def _grad_cam(model: torch.nn.Module, target_layer: torch.nn.Module, tensor: torch.Tensor):
    """Return (cam[H,W] in 0..1, probs[C]) for the top predicted class."""
    activations: List[torch.Tensor] = []
    gradients: List[torch.Tensor] = []

    h1 = target_layer.register_forward_hook(lambda _m, _i, out: activations.append(out))
    h2 = target_layer.register_full_backward_hook(lambda _m, _gi, go: gradients.append(go[0]))
    try:
        model.eval()
        x = tensor.clone().requires_grad_(True)
        logits = model(x)
        probs = F.softmax(logits, dim=1).detach().squeeze(0).numpy()
        model.zero_grad()
        logits[0, int(logits.argmax(dim=1))].backward()
    finally:
        h1.remove()
        h2.remove()

    act, grad = activations[0].detach(), gradients[0].detach()
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * act).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=tensor.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam.squeeze().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam, probs


def _overlay_b64(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> str:
    base = np.asarray(image.convert("RGB").resize((224, 224)), dtype=np.float32)
    heat = _jet_colormap(cam).astype(np.float32)
    blended = np.clip((1 - alpha) * base + alpha * heat, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(blended).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def explain_image(image: Image.Image, kind: str, top_k: int = 3) -> Dict[str, Any]:
    """Grad-CAM explanation for 'plant' or 'soil' images."""
    if kind == "plant":
        model, names = get_plant_model(), get_plant_class_names()
        target_layer = model.backbone.features[-1]
    elif kind == "soil":
        model, names = get_soil_model(), get_soil_class_names()
        target_layer = model.backbone.features[-1]
    else:
        raise ValueError("kind must be 'plant' or 'soil'")

    arr = preprocess_image(image, target_size=(224, 224), normalize_mode="imagenet")
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()

    cam, probs = _grad_cam(model, target_layer, tensor)
    order = np.argsort(-probs)[:top_k]
    focus = float((cam > 0.6).mean())  # fraction of the image the model relies on

    return {
        "kind": kind,
        "top_k": [{"label": names[i], "probability": round(float(probs[i]), 4)} for i in order],
        "heatmap_overlay_png_b64": _overlay_b64(image, cam),
        "focus_area_fraction": round(focus, 4),
        "interpretation": (
            f"The model is mostly looking at {focus * 100:.0f}% of the image (red/yellow areas). "
            + ("Attention is concentrated on a small region, typical of localized lesions."
               if focus < 0.25 else
               "Attention is spread widely; confirm the photo is a close, single-subject shot.")
        ),
    }
