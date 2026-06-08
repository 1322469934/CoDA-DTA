import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


REQUIRED_COLUMNS = ["drug_id", "smiles", "target_id", "protein_sequence", "affinity"]


def parse_args():
    parser = argparse.ArgumentParser(description="Split a unified CoDA-DTA CSV into train/validation/test files.")
    parser.add_argument("--input", required=True, help="CSV with drug_id,smiles,target_id,protein_sequence,affinity.")
    parser.add_argument("--output_dir", required=True, help="Directory for train.csv, valid.csv, test.csv, split_info.json.")
    parser.add_argument("--split_type", default="warm", choices=["warm", "cold_drug", "cold_target"])
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def check_columns(df):
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required column(s): {', '.join(missing)}")


def check_ratios(args):
    total = args.train_ratio + args.valid_ratio + args.test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")


def split_warm(df, args):
    train_valid, test = train_test_split(df, test_size=args.test_ratio, random_state=args.seed, shuffle=True)
    valid_fraction = args.valid_ratio / (args.train_ratio + args.valid_ratio)
    train, valid = train_test_split(train_valid, test_size=valid_fraction, random_state=args.seed, shuffle=True)
    return train, valid, test


def split_cold(df, column, args):
    values = np.array(sorted(df[column].astype(str).unique()))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(values)
    n_test = max(1, int(round(len(values) * args.test_ratio)))
    n_valid = max(1, int(round(len(values) * args.valid_ratio)))
    if len(values) - n_test - n_valid < 1:
        raise ValueError(f"Not enough unique {column} values for a cold split.")
    test_values = set(values[:n_test])
    valid_values = set(values[n_test : n_test + n_valid])
    train_values = set(values[n_test + n_valid :])
    train = df[df[column].astype(str).isin(train_values)]
    valid = df[df[column].astype(str).isin(valid_values)]
    test = df[df[column].astype(str).isin(test_values)]
    return train, valid, test


def compute_overlap(train, valid, test, column):
    train_set = set(train[column].astype(str))
    valid_set = set(valid[column].astype(str))
    test_set = set(test[column].astype(str))
    return {
        "train_valid": len(train_set & valid_set),
        "train_test": len(train_set & test_set),
        "valid_test": len(valid_set & test_set),
    }


def main():
    args = parse_args()
    check_ratios(args)
    df = pd.read_csv(args.input)
    check_columns(df)

    if args.split_type == "warm":
        train, valid, test = split_warm(df, args)
    elif args.split_type == "cold_drug":
        train, valid, test = split_cold(df, "drug_id", args)
    else:
        train, valid, test = split_cold(df, "target_id", args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / "train.csv", index=False)
    valid.to_csv(output_dir / "valid.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)

    split_info = {
        "split_type": args.split_type,
        "seed": args.seed,
        "train_count": int(len(train)),
        "valid_count": int(len(valid)),
        "test_count": int(len(test)),
        "drug_overlap": compute_overlap(train, valid, test, "drug_id"),
        "target_overlap": compute_overlap(train, valid, test, "target_id"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(output_dir / "split_info.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)
    print(json.dumps(split_info, indent=2))


if __name__ == "__main__":
    main()
