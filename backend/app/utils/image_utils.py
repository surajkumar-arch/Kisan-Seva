"""Image validation, storage, and preprocessing utilities for Phase 1.

Reusable across Phase 2 (Soil Health) and Phase 3 (Plant Health).
"""

import io
import os
import shutil
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from PIL import Image, UnidentifiedImageError
import numpy as np

# Constraints specified in Phase 1
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_PIL_FORMATS = {"JPEG", "PNG"}
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB
MIN_RESOLUTION = (224, 224)  # Width, Height


class ImageValidationError(Exception):
    """Raised when an uploaded file violates image validation constraints."""
    def __init__(self, message: str, errors: Optional[Dict[str, str]] = None):
        super().__init__(message)
        self.message = message
        self.errors = errors or {"image": message}


def validate_image_file(file_bytes: bytes, filename: str, field_name: str = "image") -> Image.Image:
    """Validate uploaded file for extension, size, actual image integrity, and minimum resolution.

    Args:
        file_bytes: Raw bytes of the uploaded file.
        filename: Original filename of the upload.
        field_name: Identifying field name (e.g., 'soil_image' or 'plant_image').

    Returns:
        PIL.Image.Image: Validated PIL Image instance.

    Raises:
        ImageValidationError: If any constraint is violated.
    """
    if not file_bytes:
        raise ImageValidationError(
            f"Uploaded {field_name} is empty.",
            errors={field_name: "File is empty or missing."}
        )

    # 1. Check file size
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        size_mb = len(file_bytes) / (1024 * 1024)
        raise ImageValidationError(
            f"Image exceeds maximum size of 5 MB ({size_mb:.2f} MB uploaded).",
            errors={field_name: f"File size ({size_mb:.2f} MB) exceeds maximum allowed limit of 5 MB."}
        )

    # 2. Check extension
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ImageValidationError(
            f"Invalid file extension '{ext}'. Only .jpg, .jpeg, and .png are allowed.",
            errors={field_name: f"Unsupported file extension '{ext}'. Allowed extensions are: {', '.join(sorted(ALLOWED_EXTENSIONS))}."}
        )

    # 3. Verify actual image content using PIL
    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.verify()  # Verifies file integrity / detects corrupt files
    except Exception as exc:
        raise ImageValidationError(
            f"Corrupted or invalid image file for {field_name}.",
            errors={field_name: "The uploaded file is not a valid or readable image."}
        ) from exc

    # Re-open after verify() because verify() empties image state
    try:
        image = Image.open(io.BytesIO(file_bytes))
    except Exception as exc:
        raise ImageValidationError(
            f"Unable to process image {field_name}.",
            errors={field_name: "Failed to read image data."}
        ) from exc

    # 4. Check format matches accepted formats
    if image.format not in ALLOWED_PIL_FORMATS:
        raise ImageValidationError(
            f"Image format '{image.format}' is not supported. Allowed formats: JPEG, PNG.",
            errors={field_name: f"Format '{image.format}' is rejected. Only JPEG and PNG are allowed."}
        )

    # 5. Check minimum resolution
    width, height = image.size
    min_w, min_h = MIN_RESOLUTION
    if width < min_w or height < min_h:
        raise ImageValidationError(
            f"Image resolution {width}x{height} is smaller than required minimum {min_w}x{min_h}.",
            errors={field_name: f"Resolution ({width}x{height}) is below minimum requirement of {min_w}x{min_h} pixels."}
        )

    return image


def preprocess_image(
    image: Image.Image,
    target_size: Tuple[int, int] = (224, 224),
    normalize_mode: str = "scale"
) -> np.ndarray:
    """Preprocess image for future model inference (Phase 2 & Phase 3).

    Pipeline:
        1. Convert to RGB (handles RGBA, Grayscale, etc.)
        2. Resize to target size (224x224 with high-quality resampling)
        3. Convert to float numpy array
        4. Normalize:
           - 'scale': normalize pixel values to [0.0, 1.0]
           - 'imagenet': standardize with ImageNet mean & std (standard for EfficientNet)

    Args:
        image: Validated PIL Image.
        target_size: Tuple (width, height), default (224, 224).
        normalize_mode: 'scale' or 'imagenet'.

    Returns:
        np.ndarray: Preprocessed array of shape (target_height, target_width, 3).
    """
    # 1. Convert to RGB
    if image.mode != "RGB":
        rgb_image = image.convert("RGB")
    else:
        rgb_image = image

    # 2. Resize to target resolution (224 x 224)
    resized_image = rgb_image.resize(target_size, resample=Image.Resampling.BILINEAR)

    # 3. Convert to NumPy float array
    img_array = np.asarray(resized_image, dtype=np.float32)

    # 4. Normalize
    if normalize_mode == "scale":
        # Scale pixel values [0, 255] -> [0.0, 1.0]
        normalized = img_array / 255.0
    elif normalize_mode == "imagenet":
        # Standard EfficientNet / PyTorch ImageNet normalization
        scaled = img_array / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (scaled - mean) / std
    else:
        normalized = img_array / 255.0

    return normalized


def save_uploaded_image(
    file_bytes: bytes,
    filename: str,
    target_dir: str,
    prefix: str = ""
) -> Tuple[str, str]:
    """Save raw uploaded bytes safely to target directory.

    Args:
        file_bytes: Raw bytes to persist.
        filename: Original filename.
        target_dir: Target folder path.
        prefix: Optional prefix for filename uniqueness.

    Returns:
        Tuple[str, str]: (saved_filename, full_saved_path).
    """
    os.makedirs(target_dir, exist_ok=True)
    clean_name = Path(filename).name.replace(" ", "_")
    saved_filename = f"{prefix}_{clean_name}" if prefix else clean_name
    save_path = os.path.join(target_dir, saved_filename)

    with open(save_path, "wb") as f:
        f.write(file_bytes)

    # Return normalized relative-safe or absolute path
    return saved_filename, os.path.abspath(save_path)
