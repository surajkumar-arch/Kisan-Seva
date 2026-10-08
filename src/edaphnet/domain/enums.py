"""Domain Enumerations for EDAPHNET Platform."""

from enum import Enum


class ReliabilityLevel(str, Enum):
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    ABSTAIN = "ABSTAIN"


class AbstainReason(str, Enum):
    OOD_DETECTED = "OOD_DETECTED"
    POOR_IMAGE_QUALITY = "POOR_IMAGE_QUALITY"
    IMAGE_QUALITY_REJECTED = "IMAGE_QUALITY_REJECTED"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    MISSING_CRITICAL_FEATURES = "MISSING_CRITICAL_FEATURES"
    CORRUPTED_INPUT = "CORRUPTED_INPUT"
    DOMAIN_MISMATCH = "DOMAIN_MISMATCH"
    NONE = "NONE"


class MeasurementStatus(str, Enum):
    KNOWN = "KNOWN"
    MISSING = "MISSING"
    ESTIMATED = "ESTIMATED"
    UNAVAILABLE = "UNAVAILABLE"
    INVALID = "INVALID"


class ModelStatus(str, Enum):
    READY = "READY"
    PRODUCTION = "PRODUCTION"
    ACTIVE = "ACTIVE"
    TRAINING_REQUIRED = "TRAINING_REQUIRED"
    PLACEHOLDER = "PLACEHOLDER"
    DEPRECATED = "DEPRECATED"



class EnvironmentType(str, Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
