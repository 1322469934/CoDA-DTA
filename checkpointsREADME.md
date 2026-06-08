# Checkpoints

Place trained CoDA-DTA model parameter files in this directory.

Suggested names:

- `davis_cold_drug_model.pt`
- `davis_cold_target_model.pt`
- `kiba_cold_drug_model.pt`
- `kiba_cold_target_model.pt`

Use a checkpoint with:

```bash
python predict.py --input example_inference.csv --checkpoint checkpoints/model.pt --output predictions.csv
python evaluate.py --test_csv data/processed/test.csv --checkpoint checkpoints/model.pt --output results/test_metrics.csv
```

If checkpoint files are too large for GitHub, provide them through the article supplementary material, Zenodo, Figshare, Google Drive, or another stable public link, and document the download link in the README.
