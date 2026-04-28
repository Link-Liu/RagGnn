"""
Download and prepare graph classification datasets for transfer learning experiments.

Uses PyTorch Geometric's TUDataset to download PROTEINS, DD, COX2, COX2_MD, BZR, BZR_MD.
These are graph-level classification datasets used for cross-domain transfer learning.

Transfer learning pairs:
  - PROTEINS <-> DD        (蛋白质结构图)
  - COX2     <-> COX2_MD   (环氧合酶-2抑制剂)
  - BZR      <-> BZR_MD    (苯二氮卓受体配体)
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple, List

from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data


# ---------------------------------------------------------------------------
# 数据集元信息
# ---------------------------------------------------------------------------

DATASET_INFO = {
    'PROTEINS': {
        'description': '蛋白质图数据集 — 节点为氨基酸(二级结构元素)，边为相邻关系',
        'task': '判断蛋白质是否为酶 (enzyme vs non-enzyme)',
        'num_classes': 2,
        'transfer_partner': 'DD',
    },
    'DD': {
        'description': '蛋白质图数据集 — 节点为氨基酸，边为空间接触',
        'task': '判断蛋白质是否为酶 (enzyme vs non-enzyme)',
        'num_classes': 2,
        'transfer_partner': 'PROTEINS',
    },
    'COX2': {
        'description': '环氧合酶-2(COX-2)抑制剂活性数据集',
        'task': '判断分子是否为COX-2活性抑制剂',
        'num_classes': 2,
        'transfer_partner': 'COX2_MD',
    },
    'COX2_MD': {
        'description': 'COX-2抑制剂分子动力学描述符数据集',
        'task': '判断分子是否为COX-2活性抑制剂 (基于MD描述符)',
        'num_classes': 2,
        'transfer_partner': 'COX2',
    },
    'BZR': {
        'description': '苯二氮卓受体(BZR)配体活性数据集',
        'task': '判断分子是否为BZR活性配体',
        'num_classes': 2,
        'transfer_partner': 'BZR_MD',
    },
    'BZR_MD': {
        'description': 'BZR配体分子动力学描述符数据集',
        'task': '判断分子是否为BZR活性配体 (基于MD描述符)',
        'num_classes': 2,
        'transfer_partner': 'BZR',
    },
}

# 迁移学习实验对
TRANSFER_PAIRS = [
    ('PROTEINS', 'DD'),      # P -> D
    ('DD', 'PROTEINS'),      # D -> P
    ('COX2', 'COX2_MD'),     # C -> CM
    ('COX2_MD', 'COX2'),     # CM -> C
    ('BZR', 'BZR_MD'),       # B -> BM
    ('BZR_MD', 'BZR'),       # BM -> B
]


def load_tu_dataset(data_dir: str, name: str) -> TUDataset:
    """
    Load a TUDataset by name. Downloads automatically if not present.

    Args:
        data_dir: root directory for storing datasets
        name:     dataset name (e.g. 'PROTEINS', 'DD', 'COX2', etc.)

    Returns:
        TUDataset object
    """
    raw_root = os.path.join(data_dir, "_raw_tu")
    print(f"[TUDataset] Loading {name} from {raw_root} ...")
    dataset = TUDataset(root=raw_root, name=name, use_node_attr=True)
    print(f"[TUDataset] {name}: {len(dataset)} graphs, "
          f"num_node_features={dataset.num_node_features}, "
          f"num_edge_features={dataset.num_edge_features}, "
          f"num_classes={dataset.num_classes}")
    return dataset


def get_dataset_stats(dataset: TUDataset, name: str) -> Dict:
    """Compute basic statistics for a TUDataset."""
    num_graphs = len(dataset)
    labels = []
    num_nodes_list = []
    num_edges_list = []

    for data in dataset:
        labels.append(data.y.item())
        num_nodes_list.append(data.num_nodes)
        num_edges_list.append(data.num_edges)

    labels = np.array(labels)
    unique, counts = np.unique(labels, return_counts=True)

    stats = {
        'name': name,
        'num_graphs': num_graphs,
        'num_node_features': dataset.num_node_features,
        'num_edge_features': dataset.num_edge_features,
        'num_classes': dataset.num_classes,
        'class_distribution': {int(u): int(c) for u, c in zip(unique, counts)},
        'avg_nodes': float(np.mean(num_nodes_list)),
        'avg_edges': float(np.mean(num_edges_list)),
        'min_nodes': int(np.min(num_nodes_list)),
        'max_nodes': int(np.max(num_nodes_list)),
    }
    return stats


def ensure_node_features(dataset: TUDataset, name: str) -> int:
    """
    确保数据集有节点特征。如果没有，则使用节点度数作为特征。

    某些 TUDataset（如 DD）可能没有节点属性，需要手动添加。

    Returns:
        实际的节点特征维度
    """
    if dataset.num_node_features > 0:
        return dataset.num_node_features

    print(f"[{name}] No node features found. Using degree as node feature ...")
    max_degree = 0
    for data in dataset:
        if data.edge_index.numel() > 0:
            deg = torch.zeros(data.num_nodes, dtype=torch.long)
            row = data.edge_index[0]
            for node_idx in row:
                deg[node_idx] += 1
            max_degree = max(max_degree, deg.max().item())

    # 使用 one-hot 编码度数
    feat_dim = max_degree + 1
    for data in dataset:
        deg = torch.zeros(data.num_nodes, dtype=torch.long)
        if data.edge_index.numel() > 0:
            row = data.edge_index[0]
            for node_idx in row:
                deg[node_idx] += 1
        # one-hot 编码
        x = torch.zeros(data.num_nodes, feat_dim)
        for i, d in enumerate(deg):
            if d < feat_dim:
                x[i, d] = 1.0
        data.x = x

    print(f"[{name}] Added degree-based features, dim={feat_dim}")
    return feat_dim


def download_all_datasets(data_dir: str = "data"):
    """Download all datasets required for the transfer learning experiments."""
    os.makedirs(data_dir, exist_ok=True)
    print("=" * 60)
    print(" Downloading graph classification datasets for transfer learning")
    print("=" * 60)

    all_stats = {}
    for name in DATASET_INFO:
        try:
            dataset = load_tu_dataset(data_dir, name)
            feat_dim = ensure_node_features(dataset, name)
            stats = get_dataset_stats(dataset, name)
            stats['effective_node_features'] = feat_dim
            all_stats[name] = stats
        except Exception as e:
            print(f"[ERROR] Failed to load {name}: {e}")
            all_stats[name] = {'name': name, 'error': str(e)}

    print("\n" + "=" * 60)
    print(" Dataset Summary")
    print("=" * 60)
    for name, stats in all_stats.items():
        if 'error' in stats:
            print(f"  {name}: FAILED — {stats['error']}")
        else:
            print(f"  {name}: {stats['num_graphs']} graphs, "
                  f"node_feat={stats.get('effective_node_features', stats['num_node_features'])}, "
                  f"classes={stats['class_distribution']}, "
                  f"avg_nodes={stats['avg_nodes']:.1f}")

    print("\n Transfer Pairs:")
    for src, tgt in TRANSFER_PAIRS:
        print(f"  {src} -> {tgt}")

    return all_stats


if __name__ == "__main__":
    download_all_datasets("data")
