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
## Installation

Create a Python environment and install the required libraries:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Linux/macOS, use:

```bash
source .venv/bin/activate
```

Python 3.9 or newer is recommended. The code can run on CPU. A CUDA-capable GPU is recommended for training or large-scale prediction.


## Requirements

The required Python libraries are listed in `requirements.txt`:

- `torch`
- `transformers`
- `pandas`
- `numpy`

Install them with:

```bash
pip install -r requirements.txt
```

## Data format

Training, validation, and test CSV files should use the following columns:

```text
drug_id,smiles,target_id,protein_sequence,affinity
```

Prediction CSV files should use:

```text
drug_id,smiles,target_id,protein_sequence
```

`smiles` is the molecular SMILES representation of the drug. `protein_sequence` is the amino-acid sequence of the target protein. `affinity` should be a numeric value on one consistent scale within a dataset.

## Dataset preparation

Davis, KIBA, or user-provided data should be converted to the unified CSV format above before training or evaluation. If the original Davis/KIBA files cannot be redistributed, provide instructions or scripts to convert locally downloaded raw files into this CSV format.

For a CSV that is already in the unified format, split it with:

```bash
python prepare_dataset.py --input data/all_pairs.csv --output_dir data/processed/davis_warm --split_type warm --seed 42
```

This creates:

- `train.csv`
- `valid.csv`
- `test.csv`
- `split_info.json`

## Data splitting

The helper script supports three split types:

- `warm`: random split of drug-target interaction pairs.
- `cold_drug`: test drugs do not appear in the training set.
- `cold_target`: test targets do not appear in the training set.

A random/warm split may contain the same drug or target in both training and test sets. This can overestimate cold-start performance. Cold-start evaluation should therefore use `cold_drug` or `cold_target`.

Examples:

```bash
python prepare_dataset.py --input data/all_pairs.csv --output_dir data/processed/davis_warm --split_type warm
python prepare_dataset.py --input data/all_pairs.csv --output_dir data/processed/davis_cold_drug --split_type cold_drug
python prepare_dataset.py --input data/all_pairs.csv --output_dir data/processed/davis_cold_target --split_type cold_target
```

Default ratios are 70% training, 10% validation, and 20% test. They can be changed:

```bash
python prepare_dataset.py --input data/all_pairs.csv --output_dir data/processed/custom --split_type cold_drug --train_ratio 0.7 --valid_ratio 0.1 --test_ratio 0.2 --seed 42
```

`split_info.json` records the split type, random seed, sample counts, drug overlap, and target overlap.

## Training command

Train with prepared CSV files:

```bash
python train.py --train_csv data/processed/davis_warm/train.csv --valid_csv data/processed/davis_warm/valid.csv --output checkpoints/davis_warm_model.pt --device cpu
```

For GPU training, use:

```bash
python train.py --train_csv data/processed/davis_warm/train.csv --valid_csv data/processed/davis_warm/valid.csv --output checkpoints/davis_warm_model.pt --device cuda
```

This command saves the best validation checkpoint to the path specified by `--output`.

## Evaluation command

Evaluate a trained checkpoint on a test CSV:

```bash
python evaluate.py --test_csv data/processed/davis_warm/test.csv --checkpoint checkpoints/davis_warm_model.pt --output results/davis_warm_metrics.csv --device cpu
```

The output CSV contains:

- `MSE`
- `CI`
- `R2`

These correspond to the metrics reported in the manuscript.

## Prediction command

Predict affinity for new drug-target pairs:

```bash
python predict.py --input example_inference.csv --checkpoint checkpoints/model.pt --output predictions.csv --device cpu
```

The input CSV must contain:

```text
drug_id,smiles,target_id,protein_sequence
```

The output CSV contains:

```text
drug_id,smiles,target_id,predicted_affinity
```

The script checks required input columns and creates the output directory if it does not exist.

