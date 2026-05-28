
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from transformers import AutoModel, AutoConfig


class DrugEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
        freeze: bool = False,
        output_dim: int = 256,
    ):
        super().__init__()
        self.model_name = model_name
        self.output_dim = output_dim


        self.chemberta = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.chemberta.config.hidden_size


        if freeze:
            for param in self.chemberta.parameters():
                param.requires_grad = False

        self.projection = nn.Linear(self.hidden_size, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        outputs = self.chemberta(
            input_ids=input_ids, attention_mask=attention_mask
        )


        sequence_output = outputs.last_hidden_state

        cls_output = sequence_output[:, 0, :]

        cls_projected = self.projection(cls_output)
        cls_projected = self.layer_norm(cls_projected)
        cls_projected = self.dropout(cls_projected)

        seq_projected = self.projection(sequence_output)
        seq_projected = self.layer_norm(seq_projected)

        return cls_projected, seq_projected


class ProteinEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/esm2_t33_650M_UR50D",
        freeze: bool = True,
        output_dim: int = 256,
    ):
        super().__init__()
        self.model_name = model_name
        self.output_dim = output_dim

        self.esm = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.esm.config.hidden_size

        if freeze:
            for param in self.esm.parameters():
                param.requires_grad = False

        self.projection = nn.Linear(self.hidden_size, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        outputs = self.esm(
            input_ids=input_ids, attention_mask=attention_mask
        )

        sequence_output = outputs.last_hidden_state

        cls_output = sequence_output[:, 0, :]

        cls_projected = self.projection(cls_output)
        cls_projected = self.layer_norm(cls_projected)
        cls_projected = self.dropout(cls_projected)

        seq_projected = self.projection(sequence_output)
        seq_projected = self.layer_norm(seq_projected)

        return cls_projected, seq_projected


class DualStreamEncoder(nn.Module):

    def __init__(
        self,
        drug_model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
        protein_model_name: str = "facebook/esm2_t33_650M_UR50D",
        hidden_dim: int = 256,
        drug_freeze: bool = False,
        protein_freeze: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.drug_encoder = DrugEncoder(
            model_name=drug_model_name,
            freeze=drug_freeze,
            output_dim=hidden_dim,
        )
        self.protein_encoder = ProteinEncoder(
            model_name=protein_model_name,
            freeze=protein_freeze,
            output_dim=hidden_dim,
        )

    def forward(
        self,
        drug_input_ids: torch.Tensor,
        drug_attention_mask: torch.Tensor,
        protein_input_ids: torch.Tensor,
        protein_attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        h_d, H_d = self.drug_encoder(drug_input_ids, drug_attention_mask)
        h_p, H_p = self.protein_encoder(
            protein_input_ids, protein_attention_mask
        )

        return {
            "h_d": h_d,
            "H_d": H_d,
            "h_p": h_p,
            "H_p": H_p,
        }
