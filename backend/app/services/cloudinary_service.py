"""Cloudinary Cloud Asset Storage Service for KisanSeva.

Uploads leaf foliage photos, topsoil photographs, and diagnostic visual assets
directly to Cloudinary CDN with responsive transformations.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("CloudinaryService")

CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME", "stg42sga")
API_KEY = os.getenv("CLOUDINARY_API_KEY", "648173328782197")
API_SECRET = os.getenv("CLOUDINARY_API_SECRET", "OAZNGsjBpZkB8LZa-gLe6Ejggug")

_IS_CONFIGURED = False
if CLOUD_NAME and API_KEY and API_SECRET:
    try:
        cloudinary.config(
            cloud_name=CLOUD_NAME,
            api_key=API_KEY,
            api_secret=API_SECRET,
            secure=True,
        )
        _IS_CONFIGURED = True
        logger.info(f"Cloudinary initialized successfully for cloud '{CLOUD_NAME}'.")
    except Exception as exc:
        logger.warning(f"Failed to initialize Cloudinary: {exc}")


def is_cloudinary_configured() -> bool:
    """Return True if Cloudinary credentials are valid and active."""
    return _IS_CONFIGURED


def upload_to_cloudinary(
    file_source: Union[bytes, str, Path],
    folder: str = "kisanseva/plant",
    public_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Upload image bytes or file to Cloudinary and return CDN asset metadata.

    Gracefully returns None if offline or upload fails.
    """
    if not _IS_CONFIGURED:
        return None

    try:
        options = {
            "folder": folder,
            "resource_type": "image",
            "overwrite": True,
        }
        if public_id:
            options["public_id"] = public_id

        # Perform upload
        upload_result = cloudinary.uploader.upload(file_source, **options)

        return {
            "secure_url": upload_result.get("secure_url"),
            "url": upload_result.get("url"),
            "public_id": upload_result.get("public_id"),
            "format": upload_result.get("format"),
            "bytes": upload_result.get("bytes"),
            "width": upload_result.get("width"),
            "height": upload_result.get("height"),
        }
    except Exception as exc:
        logger.warning(f"Cloudinary upload failed (falling back to local): {exc}")
        return None
