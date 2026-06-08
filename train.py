import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from pic_dta import PICDTA
from predict import tokenize_batch


REQUIRED_COLUMNS = ["drug_id", "smiles", "target_id", "protein_sequence", "affinity"]


class CsvDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        missing = [col for col in REQUIRED_COLUMNS if col not in self.df.columns]
        if missing:
            raise ValueError(f"{csv_path} is missing required column(s): {', '.join(missing)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx].to_dict()


def collate_fn(rows, drug_tokenizer, protein_tokenizer):
    df = pd.DataFrame(rows)
    batch = tokenize_batch(df, drug_tokenizer, protein_tokenizer, 256, 1024)
    batch["affinity"] = torch.tensor(df["affinity"].astype(float).to_numpy(), dtype=torch.float32)
    return batch


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal CoDA-DTA training entry point.")
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--valid_csv", required=True)
    parser.add_argument("--output", required=True, help="Output checkpoint path.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    return parser.parse_args()


def run_validation(model, loader, device):
    criterion = nn.MSELoss()
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            y = batch.pop("affinity").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            pred = model(**batch)["affinity"].squeeze(-1)
            total_loss += criterion(pred, y).item() * len(y)
            count += len(y)
    return total_loss / max(count, 1)


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Use --device cpu.")
    device = torch.device(args.device)

    drug_tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    protein_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    collate = lambda rows: collate_fn(rows, drug_tokenizer, protein_tokenizer)
    train_set = CsvDataset(args.train_csv)
    valid_set = CsvDataset(args.valid_csv)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate, drop_last=len(train_set) > args.batch_size)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = PICDTA().to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    best_valid_mse = float("inf")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            y = batch.pop("affinity").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            pred = model(**batch)["affinity"].squeeze(-1)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
        valid_mse = run_validation(model, valid_loader, device)
        print(f"epoch={epoch} valid_mse={valid_mse:.6f}")
        if valid_mse < best_valid_mse:
            best_valid_mse = valid_mse
            torch.save({"model_state_dict": model.state_dict(), "valid_mse": valid_mse, "epoch": epoch}, output)
            print(f"Saved checkpoint to {output}")


if __name__ == "__main__":
    main()
