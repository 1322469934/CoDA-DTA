# CoDA-DTA: Context-Driven Adaptive Interaction Learning for Cold-Start Drug–Target Affinity Prediction
a Pair-Conditioned Interaction Modeling for Cold-Start Drug–Target Affinity Prediction, particularly robust under cold-start scenarios (unseen drugs or unseen targets).
## 📄 Paper Overview

Predicting drug–target binding affinity is a fundamental task in computational drug discovery. Most existing DTA methods use a **globally shared interaction function** for all drug–target pairs, which is insufficient to capture the diverse, pair-specific nature of molecular interactions.

PIC-DTA addresses this limitation by introducing a **Pair-Conditioned Interaction Function (PCIF)** that dynamically generates pair-specific interaction projections using a low-rank conditional parameterization scheme.

### Architecture

```
┌──────────────────────────────────────────────────────────┐
│                       PIC-DTA                            │
│                                                          │
│  SMILES ──► ChemBERTa-2 ──► H_d, h_d ──┐               │
│                                          │               │
│  AA Seq ──► ESM-2 (frozen) ──► H_p, h_p─┤               │
│                                          │               │
│                    ┌─────────────────────▼──────────────┐ │
│                    │  Pair-Conditioned Cross-Attention   │ │
│                    │                                    │ │
│                    │  c_pair = [h_d || h_p]             │ │
│                    │  ΔW(c) = A(c)·B(c)    (low-rank)   │ │
│                    │  W(c) = W_base + ΔW(c)             │ │
│                    │                                    │ │
│                    │  Drug ↔ Protein (bi-directional)   │ │
│                    └─────────────────────┬──────────────┘ │
│                                          │               │
│                    ┌─────────────────────▼──────────────┐ │
│                    │  Mean Pooling + MLP Predictor      │ │
│                    │         ŷ = f(h_pair)              │ │
│                    └────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

### Key Features

- **Pair-Conditioned Interaction**: Dynamically adjusts interaction patterns based on pair-level contextual information, replacing fixed global fusion functions
- **Low-Rank Conditional Parameterization**: `ΔW(c) = A(c)·B(c)` with rank `r << d`, achieving adaptive modeling with modest parameter overhead
- **Dual Pre-trained Encoders**: ChemBERTa-2 for drugs + frozen ESM-2 for proteins, leveraging complementary molecular representations
- **Cold-Start Robustness**: Superior generalization on unseen drugs and unseen targets (Davis & KIBA benchmarks)

## 🔬 Cold-Start Split Strategies

PIC-DTA supports three data split modes:

| Mode | Description | Use Case |
|------|-------------|----------|
| `warm` | Random split across all pairs | Standard generalization |
| `cold_drug` | Split by drug — test drugs unseen during training | New drug candidate screening |
| `cold_target` | Split by target — test targets unseen during training | Novel target identification |

## 📊 Model Architecture Details

### Dual-Stream Encoders

**Drug Encoder (ChemBERTa-2)**
- Input: SMILES string → tokenized with ChemBERTa tokenizer
- Output: `H_d ∈ R^{L×d}` (sequence), `h_d = H_d[CLS]` (global)
- Configurable fine-tuning

**Protein Encoder (ESM-2, Frozen)**
- Input: Amino acid sequence → spaced tokens for ESM tokenizer
- Output: `H_p ∈ R^{M×d}` (sequence), `h_p = H_p[0]` (global)
- Frozen to preserve pre-trained representations

### PCIF: Pair-Conditioned Interaction Function

1. **Pair Context Generation**: `c_pair = MLP([h_d || h_p]) ∈ R^{2d}`
2. **Low-Rank Conditional Parameterization**: Generate `ΔW_Q, ΔW_K, ΔW_V` for both drug→protein and protein→drug cross-attention via `ΔW(c) = A(c)·B(c)` with `A ∈ R^{d×r}, B ∈ R^{r×d}`
3. **Bi-directional Cross-Attention**: Drug attends to protein features and vice versa
4. **Gated Fusion**: Learnable gates control information flow from each direction
5. **Mean Pooling + MLP**: Pooled pair representation → multi-layer MLP → affinity prediction

### Parameter Efficiency

The low-rank decomposition (`r = 8` for `d = 256`) means each dynamic projection matrix is parameterized by only `d × r + r × d = 2dr = 4,096` additional parameters instead of `d² = 65,536` for full matrices — a **16× reduction**.
