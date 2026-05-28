
import os
import json
import pickle
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split


class DTADataset(Dataset):
    

    def __init__(
        self,
        drugs: List[str],
        proteins: List[str],
        affinities: List[float],
        drug_tokenizer: AutoTokenizer,
        protein_tokenizer: AutoTokenizer,
        max_drug_len: int = 256,
        max_protein_len: int = 1024,
        mode: str = "train",
    ):
      
        self.drugs = drugs
        self.proteins = proteins
        self.affinities = torch.tensor(affinities, dtype=torch.float32)
        self.drug_tokenizer = drug_tokenizer
        self.protein_tokenizer = protein_tokenizer
        self.max_drug_len = max_drug_len
        self.max_protein_len = max_protein_len
        self.mode = mode

    def __len__(self) -> int:
        return len(self.drugs)

    def _tokenize_drug(self, smiles: str) -> Dict[str, torch.Tensor]:
      
        # ChemBERTa-2 使用标准 tokenizer
        encoded = self.drug_tokenizer(
            smiles,
            max_length=self.max_drug_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "drug_input_ids": encoded["input_ids"].squeeze(0),
            "drug_attention_mask": encoded["attention_mask"].squeeze(0),
        }

    def _tokenize_protein(self, sequence: str) -> Dict[str, torch.Tensor]:
       
        spaced_seq = " ".join(list(sequence))
        encoded = self.protein_tokenizer(
            spaced_seq,
            max_length=self.max_protein_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "protein_input_ids": encoded["input_ids"].squeeze(0),
            "protein_attention_mask": encoded["attention_mask"].squeeze(0),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {}
        item.update(self._tokenize_drug(self.drugs[idx]))
        item.update(self._tokenize_protein(self.proteins[idx]))
        item["affinity"] = self.affinities[idx]
        return item


class DataProcessor:
    def __init__(
        self,
        data_dir: str = "./data/raw",
        processed_dir: str = "./data/processed",
        seed: int = 42,
    ):
        self.data_dir = data_dir
        self.processed_dir = processed_dir
        self.seed = seed
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)

    def load_davis(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        affinity_path = os.path.join(self.processed_dir, "davis_affinity.csv")
        drug_path = os.path.join(self.processed_dir, "davis_drugs.csv")
        target_path = os.path.join(self.processed_dir, "davis_targets.csv")

        if os.path.exists(affinity_path):
            return (
                pd.read_csv(affinity_path),
                pd.read_csv(drug_path),
                pd.read_csv(target_path),
            )

        raise FileNotFoundError(
            "Davis dataset not found. Please download it first using "
            "scripts/download_data.py or place the files in data/raw/."
        )

    def load_kiba(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        affinity_path = os.path.join(self.processed_dir, "kiba_affinity.csv")

        if os.path.exists(affinity_path):
            return (
                pd.read_csv(affinity_path),
                pd.read_csv(os.path.join(self.processed_dir, "kiba_drugs.csv")),
                pd.read_csv(os.path.join(self.processed_dir, "kiba_targets.csv")),
            )

        raise FileNotFoundError(
            "KIBA dataset not found. Please download it first."
        )

    def prepare_data(
        self, dataset_name: str = "davis"
    ) -> Tuple[List[str], List[str], np.ndarray]:

        if dataset_name == "davis":
            affinity_df, drug_df, target_df = self.load_davis()
        elif dataset_name == "kiba":
            affinity_df, drug_df, target_df = self.load_kiba()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        drug_to_smiles = dict(zip(drug_df["drug_id"], drug_df["SMILES"]))
        target_to_seq = dict(zip(target_df["target_id"], target_df["sequence"]))

        drugs = []
        proteins = []
        affinities = []

        for _, row in affinity_df.iterrows():
            drug_id = row["drug_id"]
            target_id = row["target_id"]

            if drug_id in drug_to_smiles and target_id in target_to_seq:
                drugs.append(drug_to_smiles[drug_id])
                proteins.append(target_to_seq[target_id])

                if "pKd" in row:
                    affinities.append(row["pKd"])
                elif "kiba_score" in row:
                    affinities.append(row["kiba_score"])
                else:
                    raise ValueError(
                        f"Unknown affinity column in {dataset_name} data"
                    )

        affinities = np.array(affinities, dtype=np.float32)

        print(
            f"Prepared {dataset_name.upper()} data: "
            f"{len(drugs)} pairs, "
            f"{len(drug_to_smiles)} unique drugs, "
            f"{len(target_to_seq)} unique targets"
        )

        return drugs, proteins, affinities

    @staticmethod
    def normalize_affinity(affinities: np.ndarray) -> np.ndarray:
        mean = affinities.mean()
        std = affinities.std()
        if std < 1e-8:
            return affinities - mean
        return (affinities - mean) / std

    @staticmethod
    def denormalize_affinity(
        normalized: np.ndarray, mean: float, std: float
    ) -> np.ndarray:
        return normalized * std + mean

    def split_dataset(
        self,
        drugs: List[str],
        proteins: List[str],
        affinities: np.ndarray,
        split_mode: str = "warm",
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> Dict:
        
        n_samples = len(drugs)
        indices = np.arange(n_samples)
        np.random.seed(self.seed)

        if split_mode == "warm":
            train_val_idx, test_idx = train_test_split(
                indices, test_size=test_ratio, random_state=self.seed
            )
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=val_ratio / (1 - test_ratio),
                random_state=self.seed,
            )

        elif split_mode == "cold_drug":
            unique_drugs = list(set(drugs))
            np.random.shuffle(unique_drugs)
            n_test_drugs = max(1, int(len(unique_drugs) * test_ratio))
            n_val_drugs = max(1, int(len(unique_drugs) * val_ratio))
            n_train_drugs = len(unique_drugs) - n_test_drugs - n_val_drugs

            test_drugs = set(unique_drugs[:n_test_drugs])
            val_drugs = set(unique_drugs[n_test_drugs : n_test_drugs + n_val_drugs])
            train_drugs = set(unique_drugs[n_test_drugs + n_val_drugs:])

            test_idx = np.array(
                [i for i, d in enumerate(drugs) if d in test_drugs]
            )
            val_idx = np.array(
                [i for i, d in enumerate(drugs) if d in val_drugs]
            )
            train_idx = np.array(
                [i for i, d in enumerate(drugs) if d in train_drugs]
            )

        elif split_mode == "cold_target":
            unique_proteins = list(set(proteins))
            np.random.shuffle(unique_proteins)
            n_test_prot = max(1, int(len(unique_proteins) * test_ratio))
            n_val_prot = max(1, int(len(unique_proteins) * val_ratio))
            n_train_prot = len(unique_proteins) - n_test_prot - n_val_prot

            test_proteins = set(unique_proteins[:n_test_prot])
            val_proteins = set(
                unique_proteins[n_test_prot : n_test_prot + n_val_prot]
            )
            train_proteins = set(unique_proteins[n_test_prot + n_val_prot:])

            test_idx = np.array(
                [i for i, p in enumerate(proteins) if p in test_proteins]
            )
            val_idx = np.array(
                [i for i, p in enumerate(proteins) if p in val_proteins]
            )
            train_idx = np.array(
                [i for i, p in enumerate(proteins) if p in train_proteins]
            )

        else:
            raise ValueError(f"Unknown split_mode: {split_mode}")

        return {
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
            "split_mode": split_mode,
        }

    def create_dataloaders(
        self,
        drugs: List[str],
        proteins: List[str],
        affinities: np.ndarray,
        drug_tokenizer: AutoTokenizer,
        protein_tokenizer: AutoTokenizer,
        split_info: Dict,
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True,
        max_drug_len: int = 256,
        max_protein_len: int = 1024,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        datasets = {}
        for mode, idx in [
            ("train", split_info["train_idx"]),
            ("val", split_info["val_idx"]),
            ("test", split_info["test_idx"]),
        ]:
            datasets[mode] = DTADataset(
                drugs=[drugs[i] for i in idx],
                proteins=[proteins[i] for i in idx],
                affinities=affinities[idx].tolist(),
                drug_tokenizer=drug_tokenizer,
                protein_tokenizer=protein_tokenizer,
                max_drug_len=max_drug_len,
                max_protein_len=max_protein_len,
                mode=mode,
            )

        train_loader = DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
        val_loader = DataLoader(
            datasets["val"],
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            datasets["test"],
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return train_loader, val_loader, test_loader


def load_tokenizers(
    drug_model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
    protein_model_name: str = "facebook/esm2_t33_650M_UR50D",
) -> Tuple[AutoTokenizer, AutoTokenizer]:
    drug_tokenizer = AutoTokenizer.from_pretrained(drug_model_name)
    protein_tokenizer = AutoTokenizer.from_pretrained(protein_model_name)
    return drug_tokenizer, protein_tokenizer
