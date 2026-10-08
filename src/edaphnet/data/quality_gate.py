"""Production-Grade Image Quality Gate & Domain Validation Service."""

import io
from typing import Optional, Tuple
import numpy as np
from PIL import Image

from src.edaphnet.domain.models import ImageQualityReport, QualityIssue
from src.edaphnet.config.settings import settings


class ImageQualityAssessor:
    """Pre-inference Quality and Domain Gate.

    Rejects corrupted, blurry, underexposed, overexposed, or out-of-domain images
    before triggering expensive GPU/CPU neural inference.
    """

    def __init__(
        self,
        blur_threshold: float = settings.quality_blur_threshold,
        min_brightness: float = settings.quality_min_brightness,
        max_brightness: float = settings.quality_max_brightness,
        min_resolution: int = settings.quality_min_resolution,
    ):
        self.blur_threshold = blur_threshold
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_resolution = min_resolution

    def _compute_laplacian_variance(self, gray_array: np.ndarray) -> float:
        """Estimate sharpness using discrete Laplacian variance."""
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        # Pad image
        padded = np.pad(gray_array.astype(np.float32), 1, mode="edge")
        # Convolution
        h, w = gray_array.shape
        laplacian = (
            padded[0:h, 1:w+1]
            + padded[2:h+2, 1:w+1]
            + padded[1:h+1, 0:w]
            + padded[1:h+1, 2:w+2]
            - 4.0 * padded[1:h+1, 1:w+1]
        )
        return float(np.var(laplacian))

    def evaluate_image(
        self,
        image_bytes: bytes,
        input_domain: Optional[str] = None,  # "leaf", "soil", or None
    ) -> Tuple[ImageQualityReport, Optional[Image.Image]]:
        issues = []

        # 1. Check Corruption & Format
        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
            pil_image.verify()  # verify integrity
            # Reopen for pixel access after verify()
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            report = ImageQualityReport(
                passed=False,
                blur_score=0.0,
                brightness=0.0,
                contrast=0.0,
                aspect_ratio=1.0,
                resolution=[0, 0],
                issues=[QualityIssue(check_name="file_integrity", severity="CRITICAL", details=f"Corrupted or invalid image: {str(exc)}")],
                recommendation="REJECT",
            )
            return report, None

        width, height = pil_image.size
        aspect_ratio = width / max(1, height)

        # 2. Resolution Check
        if width < self.min_resolution or height < self.min_resolution:
            issues.append(
                QualityIssue(
                    check_name="resolution",
                    severity="CRITICAL",
                    details=f"Resolution ({width}x{height}) is below minimum {self.min_resolution}x{self.min_resolution}px.",
                )
            )

        # 3. Aspect Ratio Check
        if aspect_ratio > 4.0 or aspect_ratio < 0.25:
            issues.append(
                QualityIssue(
                    check_name="aspect_ratio",
                    severity="WARNING",
                    details=f"Extreme aspect ratio ({aspect_ratio:.2f}) may cause severe distortion in 224x224 normalization.",
                )
            )

        # Convert to numpy array for optical analysis
        arr = np.array(pil_image)
        gray = np.mean(arr, axis=2)

        # 4. Blur / Focus Check
        blur_score = self._compute_laplacian_variance(gray)
        if blur_score < self.blur_threshold:
            issues.append(
                QualityIssue(
                    check_name="sharpness",
                    severity="CRITICAL",
                    details=f"Image is excessively blurred (sharpness {blur_score:.1f} < threshold {self.blur_threshold}). Please hold camera steady.",
                )
            )

        # 5. Brightness / Darkness / Exposure Check
        mean_brightness = float(np.mean(gray))
        contrast = float(np.std(gray))

        if mean_brightness < self.min_brightness:
            issues.append(
                QualityIssue(
                    check_name="underexposure",
                    severity="CRITICAL",
                    details=f"Image is too dark (mean luminance {mean_brightness:.1f} < {self.min_brightness}). Increase ambient lighting.",
                )
            )
        elif mean_brightness > self.max_brightness:
            issues.append(
                QualityIssue(
                    check_name="overexposure",
                    severity="CRITICAL",
                    details=f"Image is overexposed/washed out (mean luminance {mean_brightness:.1f} > {self.max_brightness}). Avoid direct flash glare.",
                )
            )

        # 6. Domain Validation (Leaf vs Soil Domain Sanity Checks)
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        if input_domain == "leaf":
            # Check that there is vegetative foliage signal (vegetation index or green channel presence)
            green_dominance = np.mean((g > r) & (g > b))
            if green_dominance < 0.05 and mean_brightness > 50:
                issues.append(
                    QualityIssue(
                        check_name="domain_mismatch",
                        severity="WARNING",
                        details="Foliage domain check flagged low vegetative presence. Ensure the leaf is centered in the frame.",
                    )
                )
        elif input_domain == "soil":
            # Soil typically has earthy tones (red/brown/gray where R >= B usually)
            earth_tone_ratio = np.mean(r >= b * 0.8)
            if earth_tone_ratio < 0.30:
                issues.append(
                    QualityIssue(
                        check_name="domain_mismatch",
                        severity="WARNING",
                        details="Soil domain check flagged unusual color spectrum. Ensure topsoil surface is visible.",
                    )
                )

        # Critical issues fail the gate
        has_critical = any(issue.severity == "CRITICAL" for issue in issues)
        passed = not has_critical
        recommendation = "PROCEED" if passed else "RETAKE_IMAGE"

        report = ImageQualityReport(
            passed=passed,
            blur_score=blur_score,
            brightness=mean_brightness,
            contrast=contrast,
            aspect_ratio=aspect_ratio,
            resolution=[width, height],
            issues=issues,
            recommendation=recommendation,
        )

        return report, pil_image

    assess = evaluate_image

    def validate_soil_image(self, image_bytes: bytes) -> ImageQualityReport:
        """Validate an uploaded soil photograph."""
        report, _ = self.evaluate_image(image_bytes, input_domain="soil")
        return report

    def validate_leaf_image(self, image_bytes: bytes) -> ImageQualityReport:
        """Validate an uploaded foliage photograph."""
        report, _ = self.evaluate_image(image_bytes, input_domain="leaf")
        return report



# Production alias
ImageQualityGate = ImageQualityAssessor

