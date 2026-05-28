
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .encoders import DualStreamEncoder
from .pair_conditioned import PairConditionedCrossAttention


class PredictionHead(nn.Module):
  


    def __init__(
        self,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        num_layers: int = 3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        layers = []
        current_dim = hidden_dim

        for i in range(num_layers - 1):
            next_dim = current_dim // 2
            layers.extend([
                nn.Linear(current_dim, next_dim),
                nn.BatchNorm1d(next_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            current_dim = next_dim

        # 最终回归层
        layers.append(nn.Linear(current_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, h_pair: torch.Tensor) -> torch.Tensor:

        return self.mlp(h_pair)


class PICDTA(nn.Module):

    def __init__(
        self,
        drug_model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
        protein_model_name: str = "facebook/esm2_t33_650M_UR50D",
        hidden_dim: int = 256,
        num_heads: int = 4,
        rank: int = 8,
        dropout: float = 0.1,
        drug_freeze: bool = False,
        protein_freeze: bool = True,
        num_mlp_layers: int = 3,
    ):

        super().__init__()
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.num_heads = num_heads

        self.encoder = DualStreamEncoder(
            drug_model_name=drug_model_name,
            protein_model_name=protein_model_name,
            hidden_dim=hidden_dim,
            drug_freeze=drug_freeze,
            protein_freeze=protein_freeze,
        )


        self.interaction = PairConditionedCrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            rank=rank,
            dropout=dropout,
        )


        self.predictor = PredictionHead(
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_layers=num_mlp_layers,
        )

        self.interaction_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        drug_input_ids: torch.Tensor,
        drug_attention_mask: torch.Tensor,
        protein_input_ids: torch.Tensor,
        protein_attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        encoded = self.encoder(
            drug_input_ids=drug_input_ids,
            drug_attention_mask=drug_attention_mask,
            protein_input_ids=protein_input_ids,
            protein_attention_mask=protein_attention_mask,
        )

        h_d = encoded["h_d"]  # (batch, d)
        h_p = encoded["h_p"]  # (batch, d)
        H_d = encoded["H_d"]  # (batch, L, d)
        H_p = encoded["H_p"]  # (batch, M, d)


        H_d_interact, H_p_interact, h_pair = self.interaction(
            H_d=H_d,
            H_p=H_p,
            h_d=h_d,
            h_p=h_p,
            drug_mask=drug_attention_mask,
            protein_mask=protein_attention_mask,
        )


        h_combined = torch.cat([h_d, h_p, h_pair], dim=-1)  # (batch, 3d)
        h_enhanced = self.interaction_mlp(h_combined)       # (batch, d)
        h_final = h_enhanced + h_pair  # 残差连接


        affinity = self.predictor(h_final)  # (batch, 1)

        return {
            "affinity": affinity,
            "h_d": h_d,
            "h_p": h_p,
            "h_pair": h_pair,
        }

    def count_parameters(self) -> Dict[str, int]:

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        }

    def get_conditional_params(self) -> int:
        return sum(
            p.numel() for p in self.interaction.parameters()
        )


def create_model(config: dict) -> PICDTA:
    model_cfg = config.get("model", {})
    encoder_cfg = config.get("encoders", {})

    model = PICDTA(
        drug_model_name=encoder_cfg.get(
            "drug", {}
        ).get("name", "seyonec/ChemBERTa-zinc-base-v1"),
        protein_model_name=encoder_cfg.get(
            "protein", {}
        ).get("name", "facebook/esm2_t33_650M_UR50D"),
        hidden_dim=model_cfg.get("hidden_dim", 256),
        num_heads=model_cfg.get("num_attention_heads", 4),
        rank=model_cfg.get("low_rank", 8),
        dropout=model_cfg.get("dropout", 0.1),
        drug_freeze=encoder_cfg.get("drug", {}).get("freeze", False),
        protein_freeze=encoder_cfg.get("protein", {}).get("freeze", True),
    )

    return model
