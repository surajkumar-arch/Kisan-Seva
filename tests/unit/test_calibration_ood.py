"""Unit tests for Temperature Scaling Calibration and Mahalanobis OOD Detection."""

import numpy as np
import pytest
import torch
from src.edaphnet.domain.enums import ReliabilityLevel
from src.edaphnet.ml.calibration import TemperatureScaler
from src.edaphnet.ml.ood import MahalanobisOODDetector


def test_temperature_scaling_initialization():
    scaler = TemperatureScaler(initial_temperature=1.5)
    assert round(scaler.temperature.item(), 2) == 1.5

    # Scale logits
    logits = torch.tensor([[2.0, 1.0, 0.0]])
    scaled = scaler.forward(logits)
    assert torch.allclose(scaled, logits / 1.5)


def test_temperature_scaler_fit():
    scaler = TemperatureScaler()
    logits = torch.randn(50, 5) * 5.0  # Overconfident logits
    labels = torch.randint(0, 5, (50,))
    final_temp = scaler.fit(logits, labels)
    assert final_temp > 0.1
    assert isinstance(final_temp, float)


def test_mahalanobis_ood_detector():
    dim = 16
    num_classes = 4
    detector = MahalanobisOODDetector(embedding_dim=dim, num_classes=num_classes)

    # Synthetic reference embeddings
    rng = np.random.RandomState(42)
    embs = []
    labels = []
    for c in range(num_classes):
        center = rng.randn(dim) * 2.0
        class_embs = center + rng.randn(30, dim) * 0.2
        embs.append(class_embs)
        labels.extend([c] * 30)

    embs = np.vstack(embs)
    labels = np.array(labels)

    detector.fit(embs, labels)

    # In-distribution sample
    in_sample = embs[0] + rng.randn(dim) * 0.05
    dist_in, p_in = detector.compute_mahalanobis_distance(in_sample)
    rep_in = detector.evaluate_reliability(in_sample, calibrated_confidence=0.95)
    assert rep_in.level in [ReliabilityLevel.HIGH, ReliabilityLevel.MODERATE]

    # Out-of-distribution sample (very far away)
    out_sample = rng.randn(dim) * 50.0
    dist_out, p_out = detector.compute_mahalanobis_distance(out_sample)
    assert dist_out > dist_in

    rep_out = detector.evaluate_reliability(out_sample, calibrated_confidence=0.99)
    assert rep_out.ood_detected
