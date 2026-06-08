import argparse
from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem
from transformers import AutoTokenizer

from pic_dta import PICDTA


REQUIRED_COLUMNS = ["drug_id", "smiles", "target_id", "protein_sequence"]


def parse_args():
    parser = argparse.ArgumentParser(description="Predict drug-target affinity using a trained CoDA-DTA checkpoint.")
    parser.add_argument("--input", required=True, help="Input CSV with drug_id,smiles,target_id,protein_sequence.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint, for example checkpoints/model.pt.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device for prediction.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_drug_len", type=int, default=256)
    parser.add_argument("--max_protein_len", type=int, default=1024)
    return parser.parse_args()


def check_input(df):
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required column(s): {', '.join(missing)}")
    for idx, row in df.iterrows():
        row_no = idx + 2
        if not isinstance(row["smiles"], str) or Chem.MolFromSmiles(row["smiles"]) is None:
            raise ValueError(f"Invalid SMILES at row {row_no}: {row['smiles']!r}")
        if not isinstance(row["protein_sequence"], str) or not row["protein_sequence"].strip():
            raise ValueError(f"Empty protein_sequence at row {row_no}")


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict):
        for key in ["model_state_dict", "state_dict", "model"]:
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def tokenize_batch(df, drug_tokenizer, protein_tokenizer, max_drug_len, max_protein_len):
    drug_encoded = drug_tokenizer(
        df["smiles"].tolist(),
        max_length=max_drug_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    proteins = [" ".join(list(seq.strip())) for seq in df["protein_sequence"].tolist()]
    protein_encoded = protein_tokenizer(
        proteins,
        max_length=max_protein_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return {
        "drug_input_ids": drug_encoded["input_ids"],
        "drug_attention_mask": drug_encoded["attention_mask"],
        "protein_input_ids": protein_encoded["input_ids"],
        "protein_attention_mask": protein_encoded["attention_mask"],
    }


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Use --device cpu.")
    device = torch.device(args.device)

    df = pd.read_csv(args.input)
    check_input(df)

    model = PICDTA()
    model.load_state_dict(load_checkpoint(args.checkpoint, device), strict=True)
    model.to(device)
    model.eval()

    drug_tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    protein_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    predictions = []
    with torch.no_grad():
        for start in range(0, len(df), args.batch_size):
            batch_df = df.iloc[start : start + args.batch_size]
            batch = tokenize_batch(batch_df, drug_tokenizer, protein_tokenizer, args.max_drug_len, args.max_protein_len)
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)["affinity"].squeeze(-1)
            predictions.extend(output.cpu().tolist())

    out = df[["drug_id", "smiles", "target_id"]].copy()
    out["predicted_affinity"] = predictions
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
