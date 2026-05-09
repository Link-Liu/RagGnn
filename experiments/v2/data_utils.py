"""
V2 数据工具：加载 TUDataset + 统一特征 + 8维拓扑特征 + WL 子树标签
"""
import os, sys, torch, numpy as np
from typing import List, Tuple

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dataset.mol_graph_utils import (
    load_dataset, ensure_node_features, dataset_to_list, unify_feature_dim_lists
)


# ================================================================
# 8 维通用拓扑特征
# ================================================================
def add_topo_features(data_list: List, name: str) -> int:
    """为每个图添加 8 维域无关拓扑特征。"""
    import networkx as nx
    for data in data_list:
        N = data.num_nodes
        ei = data.edge_index
        # 邻接表
        adj = [set() for _ in range(N)]
        if ei.numel() > 0:
            src, dst = ei[0].tolist(), ei[1].tolist()
            for s, d in zip(src, dst):
                adj[s].add(d); adj[d].add(s)
        # 1. 归一化度数
        deg = torch.tensor([len(adj[i]) for i in range(N)], dtype=torch.float32)
        norm_deg = deg / deg.max().clamp(min=1)
        # 2. log 度数
        log_deg = torch.log1p(deg)
        log_deg = log_deg / log_deg.max().clamp(min=1)
        # 3. 图密度
        n_e = ei.shape[1] if ei.numel() > 0 else 0
        density = n_e / (N * (N - 1) + 1e-6)
        # 4. 图大小
        size_f = torch.log1p(torch.tensor(float(N))) / 10.0
        # 5. 聚类系数
        cc = torch.zeros(N)
        for i in range(N):
            nb = adj[i]; k = len(nb)
            if k < 2: continue
            tri = sum(1 for u in nb for v in nb if u < v and v in adj[u])
            cc[i] = 2.0 * tri / (k * (k - 1))
        # 6. 邻域度数标准差
        dstd = torch.zeros(N)
        for i in range(N):
            if adj[i]:
                nd = torch.tensor([len(adj[j]) for j in adj[i]], dtype=torch.float32)
                dstd[i] = nd.std() if len(nd) > 1 else 0.0
        dstd = dstd / dstd.max().clamp(min=1)
        # 7. 二跳邻居
        hop2 = torch.zeros(N)
        for i in range(N):
            h2 = set()
            for j in adj[i]: h2.update(adj[j])
            h2.discard(i); h2 -= adj[i]
            hop2[i] = len(h2)
        hop2 = hop2 / hop2.max().clamp(min=1)
        # 8. k-core
        G = nx.Graph(); G.add_nodes_from(range(N))
        if ei.numel() > 0:
            G.add_edges_from(zip(src, dst))
        cn = nx.core_number(G)
        kc = torch.tensor([cn.get(i, 0) for i in range(N)], dtype=torch.float32)
        kc = kc / kc.max().clamp(min=1)

        feat = torch.stack([norm_deg, log_deg, torch.full((N,), density),
                            torch.full((N,), size_f.item()), cc, dstd, hop2, kc], dim=-1)
        data.x = torch.cat([data.x.float(), feat], dim=-1)
    dim = data_list[0].x.shape[1]
    print(f"[Topo] +8 features → {name} dim={dim}")
    return dim


# ================================================================
# WL 子树标签（哈希到固定 bins）
# ================================================================
def add_wl_features(data_list: List, name: str, num_iters: int = 2, num_bins: int = 16) -> int:
    """为每个图添加 WL 子树标签的 one-hot 特征。"""
    for data in data_list:
        N = data.num_nodes
        ei = data.edge_index
        adj = [[] for _ in range(N)]
        if ei.numel() > 0:
            for s, d in zip(ei[0].tolist(), ei[1].tolist()):
                adj[s].append(d)
        # 初始标签 = 度数
        labels = [len(adj[i]) for i in range(N)]
        for _ in range(num_iters):
            new_labels = []
            for i in range(N):
                nb_labels = sorted([labels[j] for j in adj[i]])
                new_labels.append(hash((labels[i], tuple(nb_labels))) % num_bins)
            labels = new_labels
        # one-hot [N, num_bins]
        wl_onehot = torch.zeros(N, num_bins, dtype=torch.float32)
        for i, lbl in enumerate(labels):
            wl_onehot[i, lbl] = 1.0
        data.x = torch.cat([data.x.float(), wl_onehot], dim=-1)
    dim = data_list[0].x.shape[1]
    print(f"[WL] +{num_bins} WL features → {name} dim={dim}")
    return dim


# ================================================================
# 完整数据准备流程
# ================================================================
def load_and_prepare(data_dir: str, source_name: str, target_name: str,
                     wl_iters: int = 2, wl_bins: int = 16,
                     universal_only: bool = True) -> Tuple:
    """
    加载数据 + 特征工程。

    universal_only=True（推荐）：
      只用拓扑特征(8) + WL标签(16) = 24维
      不使用原始特征，不需要零填充 → 消除域迁移的特征分布偏移

    universal_only=False（传统）：
      原始特征 + 零填充统一 + 拓扑 + WL = 113维
    """
    src_ds = load_dataset(data_dir, source_name)
    tgt_ds = load_dataset(data_dir, target_name)
    ensure_node_features(src_ds, source_name)
    ensure_node_features(tgt_ds, target_name)
    src_list = dataset_to_list(src_ds)
    tgt_list = dataset_to_list(tgt_ds)

    if universal_only:
        # 只用通用特征：先把 x 置为空，再加 topo + WL
        for data in src_list:
            data.x = torch.zeros(data.num_nodes, 0, dtype=torch.float32)
        for data in tgt_list:
            data.x = torch.zeros(data.num_nodes, 0, dtype=torch.float32)
        print(f"[Data] Universal-only mode: dropped original features")
    else:
        # 传统模式：零填充统一原始特征
        unify_feature_dim_lists(src_list, tgt_list, source_name, target_name)

    # 拓扑特征（8维）
    add_topo_features(src_list, source_name)
    add_topo_features(tgt_list, target_name)
    # WL 标签（16维）
    unified_dim = add_wl_features(src_list, source_name, wl_iters, wl_bins)
    add_wl_features(tgt_list, target_name, wl_iters, wl_bins)

    print(f"[Data] Final feature dim = {unified_dim}")
    return src_list, tgt_list, unified_dim
