"""Typed Configuration Management for EDAPHNET Platform."""

import os
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.edaphnet.domain.enums import EnvironmentType

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Environment
    app_env: EnvironmentType = Field(EnvironmentType.DEVELOPMENT, alias="APP_ENV")
    app_name: str = "EDAPHNET - Smart Agriculture Intelligence Platform"
    app_version: str = "1.0.0"
    debug: bool = False

    # Server Config
    host: str = Field("0.0.0.0", alias="BACKEND_HOST")
    port: int = Field(8000, alias="BACKEND_PORT")
    workers: int = Field(2, alias="WORKERS")

    # Security
    secret_key: str = Field("edaphnet_production_secure_secret_key_change_me", alias="SECRET_KEY")
    max_upload_size_mb: int = Field(15, alias="MAX_UPLOAD_SIZE_MB")
    allowed_hosts: list[str] = Field(default_factory=lambda: ["*"])

    @property
    def environment(self) -> str:
        return self.app_env.value if hasattr(self.app_env, "value") else str(self.app_env)


    # Database
    database_url: str = Field(
        f"sqlite:///{BASE_DIR / 'data' / 'edaphnet.db'}",
        alias="DATABASE_URL",
    )
    mongodb_uri: str = Field(
        "mongodb+srv://tusharmishraop1_db_user:ECCJYHOUbSBbh9jW@setup.tdbbq7v.mongodb.net/?appName=Setup",
        alias="MONGODB_URI",
    )
    mongodb_db_name: str = Field("edaphnet", alias="MONGODB_DB_NAME")

    # Cloudinary Object Storage
    cloudinary_cloud_name: str = Field("stg42sga", alias="CLOUDINARY_CLOUD_NAME")
    cloudinary_api_key: str = Field("648173328782197", alias="CLOUDINARY_API_KEY")
    cloudinary_api_secret: str = Field("OAZNGsjBpZkB8LZa-gLe6Ejggug", alias="CLOUDINARY_API_SECRET")
    cloudinary_url: Optional[str] = Field(None, alias="CLOUDINARY_URL")

    # Redis Task Queue
    redis_url: str = Field("redis://localhost:6379/0", alias="REDIS_URL")

    # Model Registry & Storage
    models_dir: Path = Field(BASE_DIR / "models", alias="MODELS_DIR")
    soil_model_version: str = "1.3.0"
    disease_model_version: str = "1.3.0"
    localization_model_version: str = "yolov8-1.0.0"
    segmentation_model_version: str = "unet-1.0.0"
    recommendation_model_version: str = "rf-1.0.0"

    # Quality Gate Thresholds
    quality_blur_threshold: float = Field(40.0, description="Laplacian variance threshold below which image is blurry")
    quality_min_brightness: float = Field(25.0, description="Minimum mean luminance (0-255)")
    quality_max_brightness: float = Field(235.0, description="Maximum mean luminance (0-255)")
    quality_min_resolution: int = Field(128, description="Minimum allowed dimension in pixels")

    # Calibration & Reliability Thresholds
    ood_mahalanobis_threshold: float = Field(16.5, description="Mahalanobis distance cutoff for OOD detection")
    abstain_confidence_threshold: float = Field(0.40, description="Minimum confidence below which model abstains")


settings = Settings()
