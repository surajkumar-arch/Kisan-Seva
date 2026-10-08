"""PyTorch Gated Multi-Modal Fusion Neural Network for Phase 4."""

from pathlib import Path
from typing import Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_WEIGHTS_PATH = BASE_DIR / "models" / "fusion" / "gated_fusion_net.pt"


class GatedMultiModalFusionNet(nn.Module):
    """Deep Gated Multi-Modal Fusion Network.

    Fuses Soil (10d), Plant Pathology (8d), and Climate (3d) features into
    a unified 128-dimensional latent embedding vector with learned attention gating.
    """

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim

        # 1. Domain Specific Feature Encoders
        self.soil_encoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
        )

        self.plant_encoder = nn.Sequential(
            nn.Linear(8, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
        )

        self.climate_encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 64),
        )

        # 2. Modality Gating Mechanism (Cross-Modal Attention)
        self.gate_network = nn.Sequential(
            nn.Linear(64 * 3, 64),
            nn.GELU(),
            nn.Linear(64, 3),  # 3 modalities: soil, plant, climate
        )

        # 3. Unified Latent Projection Head
        self.latent_projector = nn.Sequential(
            nn.Linear(64 * 3, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Tanh(),  # Normalized bounded embedding space [-1, 1]
        )

        # 4. Auxiliary Task Heads
        # Farm Vigor Index Head (0 to 100)
        self.fvi_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Domain Stress Head (3 dimensions: soil stress, plant risk, climate stress)
        self.stress_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 3),
            nn.Sigmoid(),
        )

    def forward(
        self,
        soil_feats: torch.Tensor,
        plant_feats: torch.Tensor,
        climate_feats: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through Gated Fusion Network.

        Returns:
            latent_embedding: (B, 128)
            fvi_score: (B, 1) in [0, 100]
            stress_scores: (B, 3) in [0, 1]
            modality_weights: (B, 3) softmax weights
        """
        # Encode modalities into 64-dim hidden representations
        h_s = self.soil_encoder(soil_feats)
        h_p = self.plant_encoder(plant_feats)
        h_c = self.climate_encoder(climate_feats)

        # Compute gating weights
        concat_h = torch.cat([h_s, h_p, h_c], dim=-1)
        gate_logits = self.gate_network(concat_h)
        modality_weights = F.softmax(gate_logits, dim=-1)  # (B, 3)

        w_s = modality_weights[:, 0:1]
        w_p = modality_weights[:, 1:2]
        w_c = modality_weights[:, 2:3]

        # Apply gated attention weighting
        gated_fused = torch.cat([h_s * w_s, h_p * w_p, h_c * w_c], dim=-1)

        # Project to 128-dimensional latent space
        latent_embedding = self.latent_projector(gated_fused)

        # Calculate Farm Vigor Index (0-100) and Stress Scores
        fvi_score = self.fvi_head(latent_embedding) * 100.0
        stress_scores = self.stress_head(latent_embedding)

        return latent_embedding, fvi_score, stress_scores, modality_weights


_FUSION_MODEL_INSTANCE: Optional[GatedMultiModalFusionNet] = None


def get_fusion_model(weights_path: Optional[Path] = None) -> GatedMultiModalFusionNet:
    """Retrieve or initialize singleton instance of GatedMultiModalFusionNet."""
    global _FUSION_MODEL_INSTANCE
    if _FUSION_MODEL_INSTANCE is not None:
        return _FUSION_MODEL_INSTANCE

    target_weights = weights_path or DEFAULT_WEIGHTS_PATH
    model = GatedMultiModalFusionNet(latent_dim=128)
    model.eval()

    if target_weights.exists():
        try:
            state_dict = torch.load(target_weights, map_location=torch.device("cpu"))
            model.load_state_dict(state_dict)
        except Exception:
            pass

    _FUSION_MODEL_INSTANCE = model
    return _FUSION_MODEL_INSTANCE
