import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class LowRankConditionalGenerator(nn.Module):


    def __init__(
        self,
        hidden_dim: int = 256,
        rank: int = 8,
        dropout: float = 0.1,
    ):

        )

        self.W_base = nn.Parameter(
            torch.empty(hidden_dim, hidden_dim)
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W_base, a=math.sqrt(5))

    def forward(
        self, c_pair: torch.Tensor
    ) -> torch.Tensor:

        batch_size = c_pair.size(0)
        d = self.hidden_dim
        r = self.rank

        params = self.context_mlp(c_pair) 

        A_params = params[:, : d * r * 1].view(batch_size, d, r)
        B_params = params[:, d * r * 1 : d * r * 2].view(batch_size, r, d)

        delta_W = torch.bmm(A_params, B_params)

        W = self.W_base.unsqueeze(0) + delta_W

        return W


class PairConditionedCrossAttention(nn.Module):


    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 4,
        rank: int = 8,
        dropout: float = 0.1,
    ):

        super().__init__()
        assert hidden_dim % num_heads == 0

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.rank = rank


        self.pair_context_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )


        self.drug_cond_q = LowRankConditionalGenerator(
            hidden_dim, rank, dropout
        )
        self.drug_cond_k = LowRankConditionalGenerator(
            hidden_dim, rank, dropout
        )
        self.drug_cond_v = LowRankConditionalGenerator(
            hidden_dim, rank, dropout
        )
        self.protein_cond_q = LowRankConditionalGenerator(
            hidden_dim, rank, dropout
        )
        self.protein_cond_k = LowRankConditionalGenerator(
            hidden_dim, rank, dropout
        )
        self.protein_cond_v = LowRankConditionalGenerator(
            hidden_dim, rank, dropout
        )

        self.out_proj_drug = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj_protein = nn.Linear(hidden_dim, hidden_dim)


        self.fusion_gate_drug = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.fusion_gate_protein = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm_drug = nn.LayerNorm(hidden_dim)
        self.layer_norm_protein = nn.LayerNorm(hidden_dim)

    def _generate_pair_context(
        self, h_d: torch.Tensor, h_p: torch.Tensor
    ) -> torch.Tensor:

        # 拼接
        c_pair_raw = torch.cat([h_d, h_p], dim=-1)  # (batch, 2d)
        c_pair = self.pair_context_mlp(c_pair_raw)  # (batch, 2d)
        return c_pair

    def _conditional_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        W_q: torch.Tensor,
        W_k: torch.Tensor,
        W_v: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        key_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
       
        batch_size, L_q, d = query.size()
        _, L_k, _ = key.size()
        n_h = self.num_heads
        h_d = self.head_dim


        Q = torch.bmm(query, W_q.transpose(1, 2))  # (batch, L_q, d)
        K = torch.bmm(key, W_k.transpose(1, 2))    # (batch, L_k, d)
        V = torch.bmm(value, W_v.transpose(1, 2))   # (batch, L_v, d)


        Q = Q.view(batch_size, L_q, n_h, h_d).transpose(1, 2)  # (batch, n_h, L_q, h_d)
        K = K.view(batch_size, L_k, n_h, h_d).transpose(1, 2)  # (batch, n_h, L_k, h_d)
        V = V.view(batch_size, L_k, n_h, h_d).transpose(1, 2)  # (batch, n_h, L_v, h_d)


        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch, n_h, L_q, L_k)


        if key_mask is not None:
            # key_mask: (batch, L_k) -> (batch, 1, 1, L_k)
            expanded_mask = key_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(
                expanded_mask == 0, float("-inf")
            )


        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)


        attended = torch.matmul(attn_weights, V)  # (batch, n_h, L_q, h_d)
        attended = attended.transpose(1, 2).contiguous()  # (batch, L_q, n_h, h_d)
        attended = attended.view(batch_size, L_q, d)  # (batch, L_q, d)

        return attended

    def forward(
        self,
        H_d: torch.Tensor,
        H_p: torch.Tensor,
        h_d: torch.Tensor,
        h_p: torch.Tensor,
        drug_mask: Optional[torch.Tensor] = None,
        protein_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        batch_size = H_d.size(0)


        c_pair = self._generate_pair_context(h_d, h_p)  # (batch, 2d)


        W_q_d = self.drug_cond_q(c_pair)  # (batch, d, d)
        W_k_d = self.drug_cond_k(c_pair)
        W_v_d = self.drug_cond_v(c_pair)


        W_q_p = self.protein_cond_q(c_pair)  # (batch, d, d)
        W_k_p = self.protein_cond_k(c_pair)
        W_v_p = self.protein_cond_v(c_pair)

        H_d_attended = self._conditional_attention(
            query=H_d,  
            key=H_p,   
            value=H_p, 
            W_q=W_q_d,
            W_k=W_k_d,
            W_v=W_v_d,
            key_mask=protein_mask,
        )


        H_p_attended = self._conditional_attention(
            query=H_p,  
            key=H_d,  
            value=H_d,
            W_q=W_q_p,
            W_k=W_k_p,
            W_v=W_v_p,
            key_mask=drug_mask,
        )


        H_d_out = self.layer_norm_drug(H_d + self.out_proj_drug(H_d_attended))
        H_p_out = self.layer_norm_protein(H_p + self.out_proj_protein(H_p_attended))


        drug_fusion_input = torch.cat([H_d_out, H_d_attended], dim=-1)
        drug_gate = self.fusion_gate_drug(drug_fusion_input)  # (batch, L, d)
        H_d_fused = drug_gate * H_d_out + (1 - drug_gate) * H_d_attended


        prot_fusion_input = torch.cat([H_p_out, H_p_attended], dim=-1)
        prot_gate = self.fusion_gate_protein(prot_fusion_input)  # (batch, M, d)
        H_p_fused = prot_gate * H_p_out + (1 - prot_gate) * H_p_attended


        if drug_mask is not None:
            mask_expanded = drug_mask.unsqueeze(-1).float()  # (batch, L, 1)
            h_drug_interact = (H_d_fused * mask_expanded).sum(dim=1) / (
                mask_expanded.sum(dim=1) + 1e-8
            )
        else:
            h_drug_interact = H_d_fused.mean(dim=1)

        if protein_mask is not None:
            mask_expanded = protein_mask.unsqueeze(-1).float()  # (batch, M, 1)
            h_prot_interact = (H_p_fused * mask_expanded).sum(dim=1) / (
                mask_expanded.sum(dim=1) + 1e-8
            )
        else:
            h_prot_interact = H_p_fused.mean(dim=1)

        h_pair = (h_drug_interact + h_prot_interact) / 2.0  # (batch, d)

        return H_d_fused, H_p_fused, h_pair
