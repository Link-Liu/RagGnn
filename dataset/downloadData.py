"""
Download and prepare real molecular datasets for experiments.

Uses PyTorch Geometric's MoleculeNet to download Tox21, SIDER, BBBP, etc.
Exports them to a unified CSV format (smiles, label) for the experiment pipeline.
"""

import os
import sys
import io
import pandas as pd
import numpy as np
from pathlib import Path

from torch_geometric.datasets import MoleculeNet
from rdkit import Chem


def _load_moleculenet(data_dir: str, name: str) -> "MoleculeNet":
    """Load a MoleculeNet dataset via PyG (downloads if needed)."""
    raw_root = os.path.join(data_dir, "_raw")
    return MoleculeNet(root=raw_root, name=name)


def _dataset_to_df(dataset, task_index: int = 0) -> pd.DataFrame:
    """
    Convert a PyG MoleculeNet dataset to a DataFrame with (smiles, label).
    Uses the specified task column; drops rows where the label is NaN.
    """
    rows = []
    for data in dataset:
        smi = data.smiles if hasattr(data, 'smiles') else None
        if smi is None:
            continue
        # Validate SMILES
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        y = data.y.numpy().flatten()
        # Use the specified task column
        if task_index < len(y) and not np.isnan(y[task_index]):
            label = int(y[task_index])
        else:
            # Fallback: use first non-NaN label
            label = None
            for val in y:
                if not np.isnan(val):
                    label = int(val)
                    break
            if label is None:
                continue

        rows.append({'smiles': smi, 'label': label})

    return pd.DataFrame(rows)


def download_tox21(data_dir: str = "data", task_index: int = 2) -> str:
    """
    Download Tox21 dataset. Default task: NR-AhR (index 2).
    Tox21 has 12 tasks: NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase, NR-ER,
    NR-ER-LBD, NR-PPAR-gamma, SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53

    Returns path to saved CSV.
    """
    output_path = os.path.join(data_dir, "tox21.csv")
    print(f"[Tox21] Downloading / loading dataset (task_index={task_index}) ...")
    dataset = _load_moleculenet(data_dir, "Tox21")
    df = _dataset_to_df(dataset, task_index=task_index)
    os.makedirs(data_dir, exist_ok=True)
    df.to_csv(output_path, index=False)
    pos = int(df['label'].sum())
    neg = len(df) - pos
    print(f"[Tox21] Done: {len(df)} molecules (pos={pos}, neg={neg}) -> {output_path}")
    return output_path


def download_sider(data_dir: str = "data", task_index: int = 0) -> str:
    """
    Download SIDER dataset. Default task: first side-effect category.
    Returns path to saved CSV.
    """
    output_path = os.path.join(data_dir, "sider.csv")
    print(f"[SIDER] Downloading / loading dataset (task_index={task_index}) ...")
    dataset = _load_moleculenet(data_dir, "SIDER")
    df = _dataset_to_df(dataset, task_index=task_index)
    os.makedirs(data_dir, exist_ok=True)
    df.to_csv(output_path, index=False)
    pos = int(df['label'].sum())
    neg = len(df) - pos
    print(f"[SIDER] Done: {len(df)} molecules (pos={pos}, neg={neg}) -> {output_path}")
    return output_path


def download_bbbp(data_dir: str = "data") -> str:
    """
    Download BBBP (Blood-Brain Barrier Penetration) dataset.
    Returns path to saved CSV.
    """
    output_path = os.path.join(data_dir, "bbbp.csv")
    print("[BBBP] Downloading / loading dataset ...")
    dataset = _load_moleculenet(data_dir, "BBBP")
    df = _dataset_to_df(dataset, task_index=0)
    os.makedirs(data_dir, exist_ok=True)
    df.to_csv(output_path, index=False)
    pos = int(df['label'].sum())
    neg = len(df) - pos
    print(f"[BBBP] Done: {len(df)} molecules (pos={pos}, neg={neg}) -> {output_path}")
    return output_path


def create_source_domain(data_dir: str = "data",
                         source_size: int = 500,
                         seed: int = 42) -> str:
    """
    Create source domain molecules from BBBP for RAG retrieval.
    BBBP (blood-brain barrier penetration) is used as source domain
    because it is a DIFFERENT task from the test datasets (Tox21, SIDER),
    ensuring NO data leakage between source and target domains.

    Returns path to saved CSV.
    """
    output_path = os.path.join(data_dir, "source_molecules.csv")
    bbbp_path = os.path.join(data_dir, "bbbp.csv")

    if not os.path.exists(bbbp_path):
        download_bbbp(data_dir)

    df = pd.read_csv(bbbp_path)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(df), size=min(source_size, len(df)), replace=False)
    source_df = df.iloc[idx].reset_index(drop=True)
    source_df['source_domain'] = 'BBBP'
    source_df['name'] = [f'source_mol_{i}' for i in range(len(source_df))]
    source_df['description'] = 'Source domain molecule from BBBP'

    source_df.to_csv(output_path, index=False)
    print(f"[Source] Done: {len(source_df)} molecules from BBBP -> {output_path}")
    return output_path


def download_all_datasets(data_dir: str = "data"):
    """Download all datasets required for the experiment."""
    os.makedirs(data_dir, exist_ok=True)
    print("=" * 60)
    print(" Downloading real molecular datasets")
    print("=" * 60)

    download_tox21(data_dir)
    download_sider(data_dir)
    download_bbbp(data_dir)
    create_source_domain(data_dir)

    print("\n" + "=" * 60)
    print("All datasets downloaded successfully!")
    print("=" * 60)

    # Summary
    print("\nDataset summary:")
    for csv_file in sorted(Path(data_dir).glob("*.csv")):
        df = pd.read_csv(csv_file)
        print(f"  {csv_file.name}: {len(df)} rows, columns={list(df.columns)}")


if __name__ == "__main__":
    download_all_datasets("data")
