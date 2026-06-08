import argparse
from pathlib import Path


from pic_dta import PICDTA
from predict import tokenize_batch


REQUIRED_COLUMNS = ["drug_id", "smiles", "target_id", "protein_sequence", "affinity"]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a CoDA-DTA checkpoint on a test CSV.")
    parser.add_argument("--test_csv", required=True, help="CSV with drug_id,smiles,target_id,protein_sequence,affinity.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--output", required=True, help="Output metrics CSV.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()


def concordance_index(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = 0
    h_sum = 0.0
    for i in range(len(y_true)):
        for j in range(i + 1, len(y_true)):
            if y_true[i] == y_true[j]:
                continue
            n += 1
            if (y_pred[i] - y_pred[j]) * (y_true[i] - y_true[j]) > 0:
                h_sum += 1
            elif y_pred[i] == y_pred[j]:
                h_sum += 0.5
    return float(h_sum / n) if n else float("nan")


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict):
        for key in ["model_state_dict", "state_dict", "model"]:
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Use --device cpu.")
    device = torch.device(args.device)
    df = pd.read_csv(args.test_csv)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Test CSV is missing required column(s): {', '.join(missing)}")

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
            batch = tokenize_batch(batch_df, drug_tokenizer, protein_tokenizer, 256, 1024)
            batch = {key: value.to(device) for key, value in batch.items()}
            predictions.extend(model(**batch)["affinity"].squeeze(-1).cpu().tolist())

    y_true = df["affinity"].astype(float).to_numpy()
    metrics = {
        "MSE": mean_squared_error(y_true, predictions),
        "CI": concordance_index(y_true, predictions),
        "R2": r2_score(y_true, predictions),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(output, index=False)
    print(metrics)


if __name__ == "__main__":
    main()
