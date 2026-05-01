"""
Graph Data Utilities for TUDataset-based Transfer Learning

为 PROTEINS, DD, COX2, COX2_MD, BZR, BZR_MD 等 TUDataset 图分类数据集
提供统一的加载、特征处理和数据分割接口。

这些数据集的图数据由 PyG 的 TUDataset 直接提供：
  - 节点特征 (x) — 由数据集自带或通过度数编码补充
  - 边索引 (edge_index) — 图结构
  - 标签 (y) — 图级别的分类标签
  - 无需 SMILES 或 RDKit 转换
"""

import torch
import numpy as np
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from typing import Optional, List, Tuple, Dict
import os


def load_dataset(data_dir: str, name: str, use_node_attr: bool = True) -> TUDataset:
    """
    加载 TUDataset 图分类数据集。

    Args:
        data_dir:      数据根目录
        name:          数据集名称 (PROTEINS, DD, COX2, COX2_MD, BZR, BZR_MD)
        use_node_attr: 是否使用节点属性

    Returns:
        TUDataset 对象
    """
    raw_root = os.path.join(data_dir, "_raw_tu")
    dataset = TUDataset(root=raw_root, name=name, use_node_attr=use_node_attr)
    return dataset


def ensure_node_features(dataset: TUDataset, name: str = "") -> int:
    """
    确保数据集中所有图都有节点特征。

    如果数据集没有节点属性 (x 为 None 或 num_node_features==0)，
    则使用节点度数的 one-hot 编码作为特征。

    Args:
        dataset: TUDataset 对象
        name:    数据集名称（仅用于日志）

    Returns:
        实际的节点特征维度
    """
    # 检查第一个图是否有节点特征
    sample = dataset[0]
    if sample.x is not None and sample.x.shape[1] > 0:
        return sample.x.shape[1]

    print(f"[{name}] No node features. Computing degree-based features ...")

    # 第一遍：找到最大度数
    max_degree = 0
    for data in dataset:
        if data.edge_index is not None and data.edge_index.numel() > 0:
            deg = torch.bincount(data.edge_index[0],
                                 minlength=data.num_nodes)
            max_degree = max(max_degree, deg.max().item())

    feat_dim = min(max_degree + 1, 100)  # 防止过大

    # 第二遍：赋值 one-hot 度数特征
    for data in dataset:
        if data.edge_index is not None and data.edge_index.numel() > 0:
            deg = torch.bincount(data.edge_index[0],
                                 minlength=data.num_nodes)
        else:
            deg = torch.zeros(data.num_nodes, dtype=torch.long)
        # clamp 度数
        deg = deg.clamp(max=feat_dim - 1)
        x = torch.zeros(data.num_nodes, feat_dim)
        x[torch.arange(data.num_nodes), deg] = 1.0
        data.x = x

    print(f"[{name}] Degree-based features added, dim={feat_dim}")
    return feat_dim


def get_num_node_features(dataset: TUDataset) -> int:
    """返回数据集的节点特征维度。"""
    sample = dataset[0]
    if sample.x is not None:
        return sample.x.shape[1]
    return 0


def get_num_edge_features(dataset: TUDataset) -> int:
    """返回数据集的边特征维度。"""
    sample = dataset[0]
    if sample.edge_attr is not None:
        return sample.edge_attr.shape[1]
    return 0


def dataset_to_list(dataset) -> List[Data]:
    """
    将 TUDataset 转为 Data 列表。

    TUDataset.__getitem__ 每次返回新副本，不适合 in-place 修改。
    转为列表后可以安全地修改每个 Data 对象。
    """
    return [dataset[i] for i in range(len(dataset))]


def pad_features(data_list: List[Data], target_dim: int) -> List[Data]:
    """
    将数据列表中所有图的节点特征 zero-pad 到 target_dim。
    """
    for data in data_list:
        if data.x is not None:
            current_dim = data.x.shape[1]
            if current_dim < target_dim:
                padding = torch.zeros(data.x.shape[0], target_dim - current_dim)
                data.x = torch.cat([data.x, padding], dim=1)
    return data_list


def unify_feature_dim_lists(source_list: List[Data],
                            target_list: List[Data],
                            source_name: str = "source",
                            target_name: str = "target") -> int:
    """
    统一源域和目标域数据列表的节点特征维度（in-place 修改）。

    Args:
        source_list: 源域 Data 列表
        target_list: 目标域 Data 列表

    Returns:
        统一后的特征维度
    """
    src_dim = source_list[0].x.shape[1] if source_list and source_list[0].x is not None else 0
    tgt_dim = target_list[0].x.shape[1] if target_list and target_list[0].x is not None else 0

    if src_dim == tgt_dim:
        print(f"[Feature] {source_name} and {target_name} have same dim={src_dim}")
        return src_dim

    max_dim = max(src_dim, tgt_dim)
    print(f"[Feature] Unifying dims: {source_name}={src_dim}, "
          f"{target_name}={tgt_dim} -> {max_dim}")

    if src_dim < max_dim:
        print(f"[Feature] Padding {source_name} with {max_dim - src_dim} zeros")
        pad_features(source_list, max_dim)
    if tgt_dim < max_dim:
        print(f"[Feature] Padding {target_name} with {max_dim - tgt_dim} zeros")
        pad_features(target_list, max_dim)

    return max_dim


# 保留旧函数签名的兼容版本
def unify_feature_dim(source_dataset, target_dataset,
                      source_name="source", target_name="target"):
    """兼容版本 — 不推荐直接用于 TUDataset，请使用 unify_feature_dim_lists。"""
    src_dim = get_num_node_features(source_dataset)
    tgt_dim = get_num_node_features(target_dataset)
    return max(src_dim, tgt_dim)


def split_dataset(dataset: TUDataset,
                  train_ratio: float = 0.8,
                  seed: int = 42) -> Tuple[List[Data], List[Data]]:
    """
    将数据集按比例划分为训练集和测试集。

    Args:
        dataset:     TUDataset 对象
        train_ratio: 训练集比例
        seed:        随机种子

    Returns:
        (train_data_list, test_data_list)
    """
    rng = np.random.RandomState(seed)
    n = len(dataset)
    indices = rng.permutation(n)
    split = int(n * train_ratio)

    train_idx = indices[:split]
    test_idx = indices[split:]

    train_data = [dataset[int(i)] for i in train_idx]
    test_data = [dataset[int(i)] for i in test_idx]

    print(f"[Split] Total={n}, Train={len(train_data)}, Test={len(test_data)}")
    return train_data, test_data


def create_dataloader(data_list: List[Data],
                      batch_size: int = 32,
                      shuffle: bool = False,
                      num_workers: int = 0,
                      pin_memory: bool = True) -> DataLoader:
    """创建 PyG DataLoader。"""
    return DataLoader(
        data_list, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )



def dataset_summary(dataset: TUDataset, name: str = "") -> Dict:
    """返回数据集的摘要统计信息。"""
    labels = []
    num_nodes = []
    num_edges = []
    for data in dataset:
        labels.append(data.y.item())
        num_nodes.append(data.num_nodes)
        num_edges.append(data.num_edges)

    labels = np.array(labels)
    unique, counts = np.unique(labels, return_counts=True)

    return {
        'name': name,
        'num_graphs': len(dataset),
        'num_node_features': get_num_node_features(dataset),
        'num_edge_features': get_num_edge_features(dataset),
        'num_classes': int(len(unique)),
        'class_distribution': {int(u): int(c) for u, c in zip(unique, counts)},
        'avg_nodes': float(np.mean(num_nodes)),
        'avg_edges': float(np.mean(num_edges)),
    }


if __name__ == "__main__":
    # 快速测试
    for name in ['PROTEINS', 'DD', 'COX2', 'BZR']:
        try:
            ds = load_dataset("data", name)
            feat_dim = ensure_node_features(ds, name)
            stats = dataset_summary(ds, name)
            print(f"  {name}: {stats['num_graphs']} graphs, "
                  f"node_feat={feat_dim}, "
                  f"classes={stats['class_distribution']}")
        except Exception as e:
            print(f"  {name}: ERROR — {e}")
