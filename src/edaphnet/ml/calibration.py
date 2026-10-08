"""Post-Hoc Confidence Calibration using Temperature Scaling."""

from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim


class TemperatureScaler(nn.Module):
    """Post-hoc probability calibration layer.

    Scales unnormalized network logits by a learned scalar temperature parameter T:
        p_calibrated = softmax(logits / T)

    Optimized via Negative Log-Likelihood (NLL) on validation logits.
    """

    def __init__(self, initial_temperature: float = 1.15):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * initial_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by temperature."""
        temp = self.temperature.clamp(min=1e-2)
        return logits / temp

    def get_calibrated_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """Return temperature-calibrated softmax probabilities."""
        scaled_logits = self.forward(logits)
        return torch.softmax(scaled_logits, dim=-1)

    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor, max_iter: int = 50) -> float:
        """Optimize temperature on validation logits using L-BFGS."""
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(val_logits), val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        return float(self.temperature.item())
