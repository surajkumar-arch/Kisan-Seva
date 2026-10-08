"""PyTorch U-Net Architecture for Leaf Disease Lesion Segmentation & Affected Area Calculation."""

from typing import Tuple, Optional
import io
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

from src.edaphnet.domain.models import SegmentationResult


class DoubleConv(nn.Module):
    """Two successive [Conv2D -> BatchNorm -> ReLU] blocks."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNetDiseaseSegmenter(nn.Module):
    """U-Net Encoder-Decoder for Foliar Necrotic Lesion Segmentation."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))

        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128, 64)

        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(64, 32)

        self.outc = nn.Conv2d(32, out_channels, kernel_size=1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv3(x)

        logits = self.outc(x)
        return torch.sigmoid(logits)

    def segment_leaf(self, pil_image: Image.Image) -> Tuple[SegmentationResult, Image.Image]:
        """Perform segmentation, compute affected lesion area %, and produce overlay."""
        self.eval()
        orig_w, orig_h = pil_image.size
        img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred_mask = self.forward(img_tensor).squeeze().cpu().numpy()

        # Resize mask to original image dimensions
        mask_uint8 = (pred_mask > 0.45).astype(np.uint8) * 255
        mask_pil = Image.fromarray(mask_uint8).resize((orig_w, orig_h), Image.NEAREST)
        mask_arr = np.array(mask_pil) > 128

        # Segment leaf body vs background using color intensity
        img_arr = np.array(pil_image)
        # Background is typically white/black/neutral
        is_leaf_pixel = (img_arr[:, :, 1] > 20) | (img_arr[:, :, 0] > 20)
        total_leaf_pixels = int(np.sum(is_leaf_pixel))
        if total_leaf_pixels == 0:
            total_leaf_pixels = orig_w * orig_h

        lesion_pixels = int(np.sum(mask_arr & is_leaf_pixel))
        affected_pct = round((lesion_pixels / max(1, total_leaf_pixels)) * 100.0, 2)

        # Generate red semi-transparent overlay
        overlay = img_arr.copy()
        overlay[mask_arr, 0] = np.clip(overlay[mask_arr, 0].astype(int) + 120, 0, 255)  # Boost red on lesions
        overlay[mask_arr, 1] = np.clip(overlay[mask_arr, 1].astype(int) // 2, 0, 255)
        overlay[mask_arr, 2] = np.clip(overlay[mask_arr, 2].astype(int) // 2, 0, 255)
        overlay_pil = Image.fromarray(overlay)

        result = SegmentationResult(
            segmented=True,
            affected_area_percentage=min(100.0, affected_pct),
            total_leaf_pixels=total_leaf_pixels,
            lesion_pixels=lesion_pixels,
            model_version="unet-1.0.0",
        )

        return result, overlay_pil
