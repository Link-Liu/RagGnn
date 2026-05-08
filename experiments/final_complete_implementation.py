#!/usr/bin/env python3
"""
Domain-Aware RAG-MDA — Graph Classification Transfer Learning

Pipeline:
  1. Load TUDataset graph datasets (PROTEINS/DD, COX2/COX2_MD, BZR/BZR_MD)
  2. Train GNN (GIN) on source domain with supervised graph classification
  3. Use trained GNN encoder to produce graph-level embeddings
  4. Build RAG retrieval index from source-domain embeddings
  5. Retrieve similar graphs for each target graph
  6. Construct prompts with GNN graph tokens (soft embedding injection)
    7. Run inference with frozen ModelScope LLM
  8. Evaluate predictions against ground truth

LLM 后端：
    默认使用 ModelScope 加载 LLM-Research/Meta-Llama-3.1-8B
    设置环境变量 MODELSCOPE_MODEL_ID / MODELSCOPE_CACHE_DIR

Transfer pairs:
  PROTEINS <-> DD, COX2 <-> COX2_MD, BZR <-> BZR_MD
"""
import logging
import sys
import os
import json
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from pathlib import Path
from typing import List, Dict, Tuple, Optional

warnings.filterwarnings('ignore')

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.local_llm_interface import LocalLLMInterface
from retrieval.domain_aware_retriever import GraphRetriever
from prompting.prompt_template import create_detailed_prompt, create_no_rag_prompt
from dataset.mol_graph_utils import (
    load_dataset, ensure_node_features, get_num_node_features,
    get_num_edge_features, unify_feature_dim_lists, split_dataset,
    dataset_to_list, pad_features,
)

# 默认使用 ModelScope 模型 ID，可通过环境变量覆盖
# 使用 Instruct 模型 + Chat Template，同时采用 GraphPrompter 的 concat 注入方式
MODELSCOPE_MODEL_ID = os.getenv(
    'MODELSCOPE_MODEL_ID',
    'LLM-Research/Meta-Llama-3.1-8B-Instruct',
)

# 默认下载缓存目录放在项目 checkpoints/modelscope
_DEFAULT_MS_CACHE_DIR = str(Path(__file__).resolve().parents[1] / 'checkpoints' / 'modelscope')
MODELSCOPE_CACHE_DIR = os.getenv('MODELSCOPE_CACHE_DIR', _DEFAULT_MS_CACHE_DIR)
MODELSCOPE_REVISION = os.getenv('MODELSCOPE_REVISION', 'master')





# ---------------------------------------------------------------------------
# 图信息提取（供 Prompt 使用）
# ---------------------------------------------------------------------------

def extract_graph_info(data, graph_id: str = "", dataset_name: str = "") -> Dict:
    """从 PyG Data 对象提取 Prompt 需要的图信息（增强版：含结构统计量）。"""
    num_nodes = data.num_nodes
    num_edges = data.num_edges
    max_edges = num_nodes * (num_nodes - 1) if num_nodes > 1 else 1
    density = num_edges / max_edges if max_edges > 0 else 0
    avg_degree = (2 * num_edges / num_nodes) if num_nodes > 0 else 0

    # 度分布统计
    edge_index = data.edge_index
    if edge_index is not None and edge_index.numel() > 0:
        from torch_geometric.utils import degree
        deg = degree(edge_index[0], num_nodes=num_nodes).cpu().numpy()
        max_degree = int(deg.max()) if len(deg) > 0 else 0
        degree_std = float(deg.std()) if len(deg) > 0 else 0
    else:
        max_degree = 0
        degree_std = 0.0

    info = {
        'graph_id': graph_id,
        'dataset': dataset_name,
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'density': round(density, 4),
        'avg_degree': round(avg_degree, 2),
        'max_degree': max_degree,
        'degree_std': round(degree_std, 2),
    }

    if data.x is not None:
        feat_mean = data.x.cpu().mean(dim=0).numpy()
        top_feats = np.argsort(feat_mean)[-3:][::-1]
        info['feature_summary'] = f"top_feat_indices={list(top_feats)}, feat_dim={data.x.shape[1]}"

    return info


# ---------------------------------------------------------------------------
# 辅助函数：封装 LocalLLMInterface 的图编码操作
# ---------------------------------------------------------------------------

def _make_single_pyg_batch(data):
    """将单个 PyG Data 包装为 batch（batch 向量全为 0）。"""
    from torch_geometric.data import Batch
    return Batch.from_data_list([data])


@torch.no_grad()
def _encode_single_graph(llm_interface, data, device) -> np.ndarray:
    """用 LocalLLMInterface 内置的 GNN 编码单个图，返回 L2 归一化 numpy 向量。"""
    llm_interface.gnn.eval()
    batch = _make_single_pyg_batch(data)
    x = batch.x.to(device)
    edge_index = batch.edge_index.to(device)
    batch_vec = batch.batch.to(device)
    emb = llm_interface.gnn(x, edge_index, batch_vec).cpu().numpy().flatten()
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else emb


@torch.no_grad()
def _encode_dataset_with_llm(llm_interface, dataset, device):
    """用 LocalLLMInterface 内置的 GNN 编码整个数据集。"""
    llm_interface.gnn.eval()
    embeddings, labels = [], []
    for data in dataset:
        emb = _encode_single_graph(llm_interface, data, device)
        embeddings.append(emb)
        labels.append(data.y.item())
    return embeddings, labels


# ---------------------------------------------------------------------------
# Transfer Learning Experiment
# ---------------------------------------------------------------------------

class TransferExperiment:
    """
    跨域迁移学习实验。

    流程:
      1. 在源域数据集上训练 GNN
      2. 用训练好的 GNN 编码源域和目标域图
      3. 构建 RAG 索引
      4. 对目标域每个图：检索 → 构建 Prompt → LLM 预测
      5. 评估
    """

    TRANSFER_PAIRS = [
        ('PROTEINS', 'DD'),
        ('DD', 'PROTEINS'),
        ('COX2', 'COX2_MD'),
        ('COX2_MD', 'COX2'),
        ('BZR', 'BZR_MD'),
        ('BZR_MD', 'BZR'),
    ]

    PROP_DESC = {
        'proteins': 'enzyme classification',
        'dd': 'enzyme classification',
        'cox2': 'COX-2 inhibitor activity',
        'cox2_md': 'COX-2 inhibitor activity (MD)',
        'bzr': 'benzodiazepine receptor ligand activity',
        'bzr_md': 'benzodiazepine receptor ligand activity (MD)',
    }

    def __init__(self, data_dir: str = "data", hidden_dim: int = 128,
                 gnn_epochs: int = 30, gnn_batch_size: int = 64,
                 llm_path: str = MODELSCOPE_MODEL_ID,
                 modelscope_cache_dir: str = MODELSCOPE_CACHE_DIR,
                 modelscope_revision: str = MODELSCOPE_REVISION,
                 num_graph_tokens: int = 8,
                 load_in_8bit: bool = False):
        self.data_dir = data_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hidden_dim = hidden_dim
        self.gnn_epochs = gnn_epochs
        self.gnn_batch_size = gnn_batch_size
        self.num_graph_tokens = num_graph_tokens
        print(f"[Init] device={self.device}, llm={llm_path}")

        # 延迟初始化 LLM（需知道 num_node_features 后才能创建 GNN）
        self._llm_path = llm_path
        self._ms_cache_dir = modelscope_cache_dir
        self._ms_revision = modelscope_revision
        self._load_in_8bit = load_in_8bit
        self.llm: Optional[LocalLLMInterface] = None

    @staticmethod
    def _add_universal_features(data_list: List, name: str):
        """
        为每个图添加域无关的拓扑特征（8 维），拼接到已有特征后面。
        这些特征在所有数据集上语义一致，是跨域迁移的关键信号。
        """
        import networkx as nx
        for data in data_list:
            N = data.num_nodes
            edge_index = data.edge_index

            # 构建邻接表（用于聚类系数和 k-core）
            adj = [set() for _ in range(N)]
            if edge_index.numel() > 0:
                src, dst = edge_index[0].tolist(), edge_index[1].tolist()
                for s, d in zip(src, dst):
                    adj[s].add(d)
                    adj[d].add(s)

            # 1. 归一化度数
            deg = torch.tensor([len(adj[i]) for i in range(N)], dtype=torch.float32)
            max_deg = deg.max().clamp(min=1)
            norm_deg = deg / max_deg

            # 2. log 度数
            log_deg = torch.log1p(deg)
            log_deg = log_deg / log_deg.max().clamp(min=1)

            # 3. 图密度（broadcast 到所有节点）
            n_edges = edge_index.shape[1] if edge_index.numel() > 0 else 0
            density = n_edges / (N * (N - 1) + 1e-6)

            # 4. 相对图大小
            size_feat = torch.log1p(torch.tensor(float(N))) / 10.0

            # 5. 局部聚类系数（三角形密度）
            clustering = torch.zeros(N, dtype=torch.float32)
            for i in range(N):
                neighbors = adj[i]
                k = len(neighbors)
                if k < 2:
                    continue
                # 邻居之间的边数
                triangles = sum(1 for u in neighbors for v in neighbors if u < v and v in adj[u])
                clustering[i] = 2.0 * triangles / (k * (k - 1))

            # 6. 邻域度数标准差（邻域多样性）
            deg_std = torch.zeros(N, dtype=torch.float32)
            for i in range(N):
                if len(adj[i]) > 0:
                    neighbor_degs = torch.tensor([len(adj[j]) for j in adj[i]], dtype=torch.float32)
                    deg_std[i] = neighbor_degs.std() if len(neighbor_degs) > 1 else 0.0
            deg_std_max = deg_std.max().clamp(min=1)
            deg_std = deg_std / deg_std_max

            # 7. 归一化二跳邻居数量
            two_hop = torch.zeros(N, dtype=torch.float32)
            for i in range(N):
                hop2 = set()
                for j in adj[i]:
                    hop2.update(adj[j])
                hop2.discard(i)
                hop2 -= adj[i]  # 排除一跳邻居
                two_hop[i] = len(hop2)
            two_hop_max = two_hop.max().clamp(min=1)
            two_hop = two_hop / two_hop_max

            # 8. k-core 数（用 networkx）
            G = nx.Graph()
            G.add_nodes_from(range(N))
            if edge_index.numel() > 0:
                edges = list(zip(src, dst))
                G.add_edges_from(edges)
            core_numbers = nx.core_number(G)
            kcore = torch.tensor([core_numbers.get(i, 0) for i in range(N)], dtype=torch.float32)
            kcore_max = kcore.max().clamp(min=1)
            kcore = kcore / kcore_max

            # 拼接 [N, 8]
            uni_feat = torch.stack([
                norm_deg, log_deg,
                torch.full((N,), density),
                torch.full((N,), size_feat.item()),
                clustering, deg_std, two_hop, kcore,
            ], dim=-1)
            data.x = torch.cat([data.x.float(), uni_feat], dim=-1)
        new_dim = data_list[0].x.shape[1]
        print(f"[Feature] Added 8 universal topo features to {name} -> dim={new_dim}")
        return new_dim

    def _load_and_prepare(self, source_name: str, target_name: str):
        """加载源域和目标域数据集，统一特征维度 + 添加通用拓扑特征。"""
        src_ds = load_dataset(self.data_dir, source_name)
        tgt_ds = load_dataset(self.data_dir, target_name)

        ensure_node_features(src_ds, source_name)
        ensure_node_features(tgt_ds, target_name)

        # 转为列表以便 in-place 修改特征
        src_list = dataset_to_list(src_ds)
        tgt_list = dataset_to_list(tgt_ds)

        unified_dim = unify_feature_dim_lists(src_list, tgt_list, source_name, target_name)

        # 添加域无关的拓扑特征（度数、密度等）→ 跨域迁移的关键信号
        unified_dim = self._add_universal_features(src_list, source_name)
        self._add_universal_features(tgt_list, target_name)

        # 获取边特征维度
        edge_dim = 0
        if src_list and src_list[0].edge_attr is not None:
            edge_dim = src_list[0].edge_attr.shape[1]

        return src_list, tgt_list, unified_dim, edge_dim

    # ------------------------------------------------------------------
    # Phase 2: 联合训练 GNN + Projector（纯源域数据，不使用目标域标签）
    # ------------------------------------------------------------------
    def _joint_train(
        self,
        src_list: List,
        tgt_list: List,
        source_name: str,
        target_name: str,
        ckpt_proj: str,
        train_epochs: int = 50,
        train_batch_size: int = 8,
        lr_gnn: float = 5e-5,
        lr_proj: float = 2e-4,
        grad_accum_steps: int = 2,
        train_ratio: float = 0.8,
    ):
        """
        用源域的标签数据联合训练 GNN + Projector。
        
        纯跨域迁移设定（UDA）：
          - 训练数据 100% 来自源域
          - 目标域完全没有标签，不参与训练
          - 训练时不使用 RAG 提示词，强迫模型通过 GNN soft tokens 学习图表示
          - RAG 仅在评估阶段作为跨域迁移辅助手段
        """
        import random
        from torch_geometric.data import Batch as PyGBatch

        print(f"\n{'='*60}")
        print(f"  [Joint Train] GNN + Projector — SOURCE DOMAIN ONLY")
        print(f"  {source_name} -> {target_name}")
        print(f"  Train epochs={train_epochs}, lr_gnn={lr_gnn}, lr_proj={lr_proj}")
        print(f"{'='*60}")

        prop_desc = self.PROP_DESC.get(source_name.lower(), source_name)

        # ---- 划分源域数据：80% 训练 / 20% 验证 ----
        indices = list(range(len(src_list)))
        random.seed(42)
        random.shuffle(indices)
        n_train = int(train_ratio * len(indices))
        train_indices = indices[:n_train]
        val_indices   = indices[n_train:]
        print(f"  Source domain: {len(src_list)} graphs total")
        print(f"  Train: {len(train_indices)}, Val: {len(val_indices)}")
        print(f"  Target domain: {len(tgt_list)} graphs (unlabeled, for domain adversarial)")
        print(f"  Prompt: No-RAG (train GNN soft tokens without retrieval shortcuts)")

        # ---- 封装 Dataset（源域数据，无 RAG） ----
        class SourceDataset(torch.utils.data.Dataset):
            def __init__(self, indices):
                self.indices = indices
                self.labels = [int(src_list[idx].y.item()) for idx in indices]
            def __len__(self): return len(self.indices)
            def __getitem__(self, i):
                idx = self.indices[i]
                return src_list[idx]

        def collate_src(batch_samples):
            return PyGBatch.from_data_list(batch_samples)

        # ---- 平衡采样 ----
        train_dataset = SourceDataset(train_indices)
        labels = train_dataset.labels
        class_sample_count = np.array([len(np.where(labels == t)[0]) for t in np.unique(labels)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in labels])
        samples_weight = torch.from_numpy(samples_weight)
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))

        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=train_batch_size, sampler=sampler, collate_fn=collate_src,
            num_workers=0, pin_memory=(self.device == 'cuda')
        )
        val_batch_size = 8  # 验证用更大 batch，稳定 val loss 估计
        val_loader = torch.utils.data.DataLoader(
            SourceDataset(val_indices), 
            batch_size=val_batch_size, shuffle=False, collate_fn=collate_src,
            num_workers=0, pin_memory=(self.device == 'cuda')
        )

        # ---- 目标域 DataLoader（无标签，仅用于域对抗）----
        class TargetDataset(torch.utils.data.Dataset):
            def __init__(self, data_list):
                self.data_list = data_list
            def __len__(self): return len(self.data_list)
            def __getitem__(self, i): return self.data_list[i]

        tgt_loader = torch.utils.data.DataLoader(
            TargetDataset(tgt_list),
            batch_size=train_batch_size, shuffle=True, collate_fn=collate_src,
            num_workers=0, pin_memory=(self.device == 'cuda'), drop_last=True,
        )

        # ---- 优化器（base_tokens 用更高 lr，需要快速收敛到 LLM 能理解的空间）----
        optimizer = torch.optim.AdamW([
            {"params": self.llm.gnn.parameters(), "lr": lr_gnn},
            {"params": [self.llm.projector.base_tokens], "lr": 5e-4},
            {"params": self.llm.projector.delta_shared.parameters(), "lr": lr_proj},
            {"params": self.llm.projector.delta_token_gate.parameters(), "lr": lr_proj},
            {"params": self.llm.projector.delta_classifier.parameters(), "lr": lr_proj},
        ], weight_decay=1e-5)

        # warmup 步数基于优化器实际更新步数（而非 batch 步数）
        steps_per_epoch = max(len(train_loader) // grad_accum_steps, 1)
        total_update_steps = train_epochs * steps_per_epoch
        warmup_steps = max(total_update_steps // 5, 1)
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(warmup_steps, 1))
            progress = float(step - warmup_steps) / float(max(total_update_steps - warmup_steps, 1))
            return max(0.1, 0.5 * (1.0 + np.cos(np.pi * progress)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        best_val_auc = 0.0
        best_state = None
        patience_counter = 0
        patience_limit = 10
        use_amp = self.device == "cuda"

        # ---- 训练循环 ----
        for epoch in range(1, train_epochs + 1):
            self.llm.gnn.train()
            self.llm.projector.train()
            optimizer.zero_grad()
            epoch_loss, n_steps = 0.0, 0
            train_correct, train_total = 0, 0

            for batch in train_loader:
                batch = batch.to(self.device)
                B = batch.num_graphs
                prompts = []
                for i in range(B):
                    info = extract_graph_info(batch[i], "train", source_name)
                    p = create_no_rag_prompt(
                        target_graph_info=info,
                        property_description=prop_desc,
                        target_dataset=source_name.lower(),
                    )
                    prompts.append(p)
                
                labels_text = [str(int(y.item())) for y in batch.y]
                with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
                    loss, acc = self.llm.compute_loss(
                        batch, prompts, labels_text,
                    )
                    loss = loss / grad_accum_steps

                loss.backward()
                n_steps += 1
                epoch_loss += loss.item() * grad_accum_steps
                train_correct += int(acc * B)
                train_total += B
                if n_steps % grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.llm.trainable_parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            if n_steps % grad_accum_steps != 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            avg_train_loss = epoch_loss / max(n_steps, 1)
            train_acc = train_correct / max(train_total, 1)

            # ---- 验证（源域验证集，同样无 RAG） ----
            self.llm.gnn.eval()
            self.llm.projector.eval()
            val_loss, val_steps = 0.0, 0
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    B = batch.num_graphs
                    prompts = []
                    for i in range(B):
                        info = extract_graph_info(batch[i], "val", source_name)
                        p = create_no_rag_prompt(
                            target_graph_info=info,
                            property_description=prop_desc,
                            target_dataset=source_name.lower(),
                        )
                        prompts.append(p)
                    
                    labels_text = [str(int(y.item())) for y in batch.y]
                    with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
                        loss, acc = self.llm.compute_loss(
                            batch, prompts, labels_text,
                        )
                        val_loss += loss.item()
                        val_steps += 1
                        val_correct += int(acc * B)
                        val_total += B

            avg_val_loss = val_loss / max(val_steps, 1)
            val_acc = val_correct / max(val_total, 1)

            # ---- 计算验证集 AUC（用 LLM logits 推理）----
            val_auc = float('nan')
            try:
                from torch_geometric.data import Batch as PyGBatch
                val_probs, val_trues = [], []
                for batch in val_loader:
                    batch = batch.to(self.device)
                    B_val = batch.num_graphs
                    val_prompts = []
                    for vi in range(B_val):
                        info = extract_graph_info(batch[vi], "val", source_name)
                        p = create_no_rag_prompt(
                            target_graph_info=info,
                            property_description=prop_desc,
                            target_dataset=source_name.lower(),
                        )
                        val_prompts.append(p)
                    results = self.llm.predict_with_llm_logits(batch, val_prompts)
                    for vi, res in enumerate(results):
                        val_probs.append(res.get('prob_1', 0.5))
                        val_trues.append(int(batch[vi].y.item()))
                if len(set(val_trues)) > 1:
                    from sklearn.metrics import roc_auc_score
                    val_auc = roc_auc_score(val_trues, val_probs)
            except Exception as e:
                print(f"\n  [Val-AUC ERROR] {e}")

            print(f"  Epoch {epoch}/{train_epochs}  "
                  f"train_loss={avg_train_loss:.4f} train_acc={train_acc:.4f}  "
                  f"val_loss={avg_val_loss:.4f} val_auc={val_auc:.4f}", end="")

            # Early stopping 基于 val_auc（越大越好）
            if not np.isnan(val_auc) and val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                best_state = {
                    "gnn": {k: v.cpu().clone() for k, v in self.llm.gnn.state_dict().items()},
                    "projector": {k: v.cpu().clone() for k, v in self.llm.projector.state_dict().items()},
                }
                print(f"  ← best ✓")
            else:
                patience_counter += 1
                print(f"  (patience {patience_counter}/{patience_limit})")
                if patience_counter >= patience_limit:
                    print(f"  [Early Stopping] No improvement for {patience_limit} epochs, stopping.")
                    break

        # ---- 恢复最佳权重并保存 ----
        if best_state:
            self.llm.gnn.load_state_dict(best_state["gnn"])
            self.llm.projector.load_state_dict(best_state["projector"])
            self.llm.gnn.to(self.device)
            self.llm.projector.to(self.device)
            Path(ckpt_proj).parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, ckpt_proj)
            print(f"  [Joint Train] Best checkpoint saved -> {ckpt_proj} (val_auc={best_val_auc:.4f})")

        self.llm.gnn.eval()
        self.llm.projector.eval()

    def run_transfer(self, source_name: str, target_name: str,
                     sample_size: int = 50, eval_batch_size: int = 4,
                     infer_method: str = 'llm_logits') -> Dict:
        """
        运行单个迁移学习实验: source_name -> target_name
        """
        print(f"\n{'='*60}")
        print(f"  Transfer: {source_name} -> {target_name}  (sample={sample_size}, batch={eval_batch_size})")
        print(f"{'='*60}")

        # 1. 加载数据
        src_list, tgt_list, unified_dim, edge_dim = self._load_and_prepare(source_name, target_name)
        ckpt_gnn  = f"checkpoints/gnn_{source_name.lower()}.pt"
        ckpt_proj = f"checkpoints/proj_{source_name.lower()}_{target_name.lower()}.pt"

        # 2. 延迟初始化 LocalLLMInterface（首次运行才加载 LLM）
        if self.llm is None or self.llm.gnn.input_proj[0].in_features != unified_dim:
            # 显式释放旧 LLM 的 GPU 显存，防止 CUDA OOM
            if self.llm is not None:
                print(f"[LLM] Feature dim changed, releasing old LLM ...")
                self.llm.release()
                self.llm = None
            print(f"[LLM] Initializing LocalLLMInterface (node_feat={unified_dim}) ...")
            self.llm = LocalLLMInterface(
                model_name_or_path=self._llm_path,
                num_node_features=unified_dim,
                gnn_hidden_dim=self.hidden_dim,
                num_graph_tokens=self.num_graph_tokens,
                load_in_8bit=self._load_in_8bit,
                modelscope_cache_dir=self._ms_cache_dir,
                modelscope_revision=self._ms_revision,
            )

        # 3. GNN 预训练 / 加载
        if not self.llm.load_checkpoint(ckpt_proj):
            # 先用源域数据有监督预训练 GNN
            from experiments.pretrain_gnn import pretrain_gnn_standalone
            pretrain_gnn_standalone(
                self.llm.gnn, src_list, device=self.device,
                epochs=self.gnn_epochs, batch_size=self.gnn_batch_size,
                ckpt=ckpt_gnn,
            )

        # 4. ========== 联合训练 GNN + Projector（纯源域数据） ==========
        if not Path(ckpt_proj).exists():
            self._joint_train(
                src_list, tgt_list, source_name, target_name, ckpt_proj
            )

        # 5. 编码源域 → 构建 RAG 索引
        print(f"[RAG] Encoding {len(src_list)} source graphs ...")
        src_embs, src_labels = _encode_dataset_with_llm(self.llm, src_list, self.device)
        src_infos = []
        for i, (data, emb) in enumerate(zip(src_list, src_embs)):
            info = extract_graph_info(data, f"{source_name}_{i}", source_name)
            info['source_domain'] = source_name.lower()
            src_infos.append(info)

        retriever = GraphRetriever(embedding_dim=self.hidden_dim)
        retriever.add_source_graphs(src_embs, src_infos, src_labels)
        print(f"[RAG] Indexed {len(src_embs)} source graphs")

        # 5b. ========== 源域验证集评估（诊断：模型在源域学得如何）==========
        print(f"\n[Source-Eval] Evaluating on source domain validation set ({source_name}) ...")
        # pyrefly: ignore [missing-import]
        from torch_geometric.data import Batch as PyGBatch
        # 用固定种子打乱，确保验证集包含两个类别（TUDataset 通常按标签排序）
        import random
        src_indices = list(range(len(src_list)))
        random.Random(42).shuffle(src_indices)
        val_split = int(len(src_list) * 0.8)
        src_val_list = [src_list[i] for i in src_indices[val_split:]]
        src_val_preds, src_val_trues, src_val_probs = [], [], []
        prop_desc_src = self.PROP_DESC.get(source_name.lower(), source_name)
        for start_pos in range(0, len(src_val_list), eval_batch_size):
            end_pos = min(start_pos + eval_batch_size, len(src_val_list))
            batch_data = src_val_list[start_pos:end_pos]
            try:
                pyg_batch = PyGBatch.from_data_list(batch_data).to(self.device)
                batch_prompts = []
                for data in batch_data:
                    info = extract_graph_info(data, "src_val", source_name)
                    p = create_no_rag_prompt(
                        target_graph_info=info,
                        property_description=prop_desc_src,
                        target_dataset=source_name.lower(),
                    )
                    batch_prompts.append(p)
                results = self.llm.predict_with_llm_logits(pyg_batch, batch_prompts)
                for j, res in enumerate(results):
                    true_label = int(batch_data[j].y.item())
                    pred = int(res.get('prediction', 0))
                    src_val_preds.append(pred)
                    src_val_trues.append(true_label)
                    src_val_probs.append(res.get('prob_1', float(pred)))
            except Exception as e:
                print(f"  [Source-Eval ERROR] {e}")
        
        src_metrics = self._compute_metrics(src_val_preds, src_val_trues, src_val_probs)
        if src_metrics:
            self._print_metrics(f"[Source] {source_name} val", src_metrics,
                                len(src_val_preds), 0)

        # 6. 在目标域全量数据上评估（纯跨域：目标域未参与任何训练）
        n_target = len(tgt_list)
        target_indices = list(range(n_target))
        print(f"[Eval] Evaluating on ALL {n_target} target domain graphs (none used in training)")

        prop_desc = self.PROP_DESC.get(target_name.lower(), target_name)
        predictions, true_labels, prob_scores, details = [], [], [], []
        failed = 0
        # B1: 改为按 Batch 推理
        # pyrefly: ignore [missing-import]
        from torch_geometric.data import Batch as PyGBatch
        for start_pos in range(0, n_target, eval_batch_size):
            end_pos = min(start_pos + eval_batch_size, n_target)
            batch_indices = target_indices[start_pos:end_pos]
            batch_data = [tgt_list[int(idx)] for idx in batch_indices]
            B_eval = len(batch_data)



            try:
                # 向量化 GNN 编码整个 Batch
                pyg_batch = PyGBatch.from_data_list(batch_data).to(self.device)
                with torch.no_grad():
                    # 调用 LocalLLMInterface 的编码方法（需要批量支持）
                    batch_embs = self.llm.gnn(pyg_batch.x.float(), pyg_batch.edge_index, pyg_batch.batch)
                    batch_embs = batch_embs.cpu().numpy()
                    # L2 归一化
                    norms = np.linalg.norm(batch_embs, axis=1, keepdims=True)
                    batch_embs = np.divide(batch_embs, norms, out=np.zeros_like(batch_embs), where=norms!=0)

                batch_prompts = []
                for i, data in enumerate(batch_data):
                    graph_info = extract_graph_info(data, f"{target_name}_{batch_indices[i]}", target_name)
                    emb = batch_embs[i]
                    
                    # RAG 检索
                    retrieved = retriever.retrieve_similar_graphs(emb, graph_info, k=5)

                    # 图结构信息通过 soft tokens concat 注入，prompt 只放 LLM 能理解的文本
                    prompt = create_detailed_prompt(
                        target_graph_info=graph_info,
                        retrieved_examples=retrieved,
                        property_description=prop_desc,
                        target_dataset=target_name.lower(),
                        source_dataset=source_name.lower(),
                    )
                    batch_prompts.append(prompt)

                # 根据 infer_method 选择推理方式
                if infer_method == 'llm_logits':
                    results = self.llm.predict_with_llm_logits(pyg_batch, batch_prompts)
                else:  # 'generate'
                    results = self.llm.predict_batch(pyg_batch, batch_prompts)

                for i, res in enumerate(results):
                    data = batch_data[i]
                    pred = res['prediction']
                    true_label = data.y.item()
                    
                    predictions.append(pred)
                    true_labels.append(true_label)
                    prob_scores.append(res.get('prob_1', float(pred)))
                    detail = {
                        'graph_id': f"{target_name}_{batch_indices[i]}",
                        'true_label': true_label,
                        'prediction': pred,
                        'llm_response': res['response'][:200],
                    }
                    # 记录概率信息（如果有）
                    if 'prob_0' in res:
                        detail['prob_0'] = res['prob_0']
                        detail['prob_1'] = res['prob_1']
                    details.append(detail)



            except Exception as e:
                failed += len(batch_indices)
                print(f"  [ERROR] batch {start_pos+1}-{end_pos}: {e}")

        # 5. 计算指标
        metrics = self._compute_metrics(predictions, true_labels, prob_scores)
        if metrics:
            self._print_metrics(f"{source_name}->{target_name}", metrics,
                                len(predictions), failed)

        return {
            'source': source_name, 'target': target_name,
            'infer_method': infer_method,
            'total_target': n_target, 'total_predicted': len(predictions),
            'failed': failed, 'metrics': metrics, 'details': details,
            'llm_calls': self.llm.call_count if self.llm is not None else 0,
        }

    def run_transfer_no_rag(self, source_name: str, target_name: str,
                            sample_size: int = 50, eval_batch_size: int = 4,
                            infer_method: str = 'llm_logits') -> Dict:
        """消融实验：无 RAG 对照组。"""
        print(f"\n{'='*60}")
        print(f"  [No-RAG] {source_name} -> {target_name}  (sample={sample_size}, batch={eval_batch_size})")
        print(f"{'='*60}")

        _, tgt_list, unified_dim, _ = self._load_and_prepare(source_name, target_name)

        # No-RAG 也需要 LLM（用于 soft token 推理），若尚未初始化则在此完成
        if self.llm is None or self.llm.gnn.input_proj[0].in_features != unified_dim:
            # 显式释放旧 LLM 的 GPU 显存，防止 CUDA OOM
            if self.llm is not None:
                print(f"[LLM] Feature dim changed, releasing old LLM ...")
                self.llm.release()
                self.llm = None
            print(f"[LLM] Initializing LocalLLMInterface for No-RAG "
                  f"(node_feat={unified_dim}) ...")
            self.llm = LocalLLMInterface(
                model_name_or_path=self._llm_path,
                num_node_features=unified_dim,
                gnn_hidden_dim=self.hidden_dim,
                num_graph_tokens=self.num_graph_tokens,
                load_in_8bit=self._load_in_8bit,
                modelscope_cache_dir=self._ms_cache_dir,
                modelscope_revision=self._ms_revision,
            )
            # 尝试加载已有联合训练 checkpoint
            ckpt_proj = f"checkpoints/proj_{source_name.lower()}_{target_name.lower()}.pt"
            ckpt_gnn  = f"checkpoints/gnn_{source_name.lower()}.pt"
            if not self.llm.load_checkpoint(ckpt_proj):
                # 若无联合训练权重，至少加载 GNN 预训练权重
                if Path(ckpt_gnn).exists():
                    self.llm.gnn.load_state_dict(
                        torch.load(ckpt_gnn, map_location=self.device)
                    )
                    print(f"[LLM] GNN weights loaded from {ckpt_gnn}")
                else:
                    print("[WARNING] No GNN checkpoint found, using random weights.")

        # 纯跨域：在目标域全量数据上评估（目标域未参与任何训练）
        n_target = len(tgt_list)
        target_indices = list(range(n_target))
        print(f"[No-RAG] Evaluating on ALL {n_target} target domain graphs (none used in training)")

        prop_desc = self.PROP_DESC.get(target_name.lower(), target_name)
        predictions, true_labels, prob_scores, details = [], [], [], []
        failed = 0

        # B1: 改为按 Batch 推理
        # pyrefly: ignore [missing-import]
        from torch_geometric.data import Batch as PyGBatch
        for start_pos in range(0, n_target, eval_batch_size):
            end_pos = min(start_pos + eval_batch_size, n_target)
            batch_indices = target_indices[start_pos:end_pos]
            batch_data = [tgt_list[int(idx)] for idx in batch_indices]
            B_eval = len(batch_data)



            try:
                batch_prompts = []
                for bi, data in enumerate(batch_data):
                    graph_info = extract_graph_info(data, f"{target_name}_{batch_indices[bi]}", target_name)
                    prompt = create_no_rag_prompt(
                        target_graph_info=graph_info,
                        property_description=prop_desc,
                        target_dataset=target_name.lower(),
                    )
                    batch_prompts.append(prompt)

                pyg_batch = PyGBatch.from_data_list(batch_data).to(self.device)

                # 根据 infer_method 选择推理方式
                if infer_method == 'llm_logits':
                    results = self.llm.predict_with_llm_logits(pyg_batch, batch_prompts)
                else:  # 'generate'
                    results = self.llm.predict_batch(pyg_batch, batch_prompts)

                for i, res in enumerate(results):
                    data = batch_data[i]
                    pred = res['prediction']
                    true_label = data.y.item()
                    
                    predictions.append(pred)
                    true_labels.append(true_label)
                    prob_scores.append(res.get('prob_1', float(pred)))
                    detail = {
                        'graph_id': f"{target_name}_{batch_indices[i]}",
                        'true_label': true_label,
                        'prediction': pred,
                        'retrieved_count': 0,
                        'llm_response': res['response'][:200],
                    }
                    if 'prob_0' in res:
                        detail['prob_0'] = res['prob_0']
                        detail['prob_1'] = res['prob_1']
                    details.append(detail)



            except Exception as e:
                failed += len(batch_indices)
                print(f"  [ERROR] batch {start_pos+1}-{end_pos}: {e}")

        metrics = self._compute_metrics(predictions, true_labels, prob_scores)
        if metrics:
            self._print_metrics(f"[No-RAG] {source_name}->{target_name}",
                                metrics, len(predictions), failed)

        return {
            'source': source_name, 'target': target_name,
            'condition': 'no_rag',
            'total_target': n_target, 'total_predicted': len(predictions),
            'failed': failed, 'metrics': metrics, 'details': details,
            'llm_calls': self.llm.call_count if self.llm is not None else 0,
        }

    def _compute_metrics(self, predictions, true_labels, prob_scores=None) -> Dict:
        if not predictions:
            return {}
        metrics = {
            'accuracy': float(accuracy_score(true_labels, predictions)),
            'f1': float(f1_score(true_labels, predictions, zero_division=0)),
            'precision': float(precision_score(true_labels, predictions, zero_division=0)),
            'recall': float(recall_score(true_labels, predictions, zero_division=0)),
        }
        if len(set(true_labels)) > 1:
            # 使用概率值计算 AUC（正确做法），回退到离散预测
            auc_scores = prob_scores if prob_scores else predictions
            metrics['auc'] = float(roc_auc_score(true_labels, auc_scores))
        else:
            metrics['auc'] = float('nan')
        return metrics

    def _print_metrics(self, name, m, n_pred, n_fail):
        correct = int(m['accuracy'] * n_pred)
        print(f"\n[Results] {name}:")
        print(f"  Accuracy : {m['accuracy']:.4f}  ({correct}/{n_pred})")
        print(f"  F1       : {m['f1']:.4f}")
        print(f"  Precision: {m['precision']:.4f}")
        print(f"  Recall   : {m['recall']:.4f}")
        print(f"  AUC      : {m['auc']:.4f}")
        print(f"  Failed   : {n_fail}")

    # ------------------------------------------------------------------
    # Full experiment suite
    # ------------------------------------------------------------------

    def run_all_transfers(self, sample_size: int = 50,
                          pairs: Optional[List[Tuple]] = None,
                          eval_batch_size: int = 8) -> Dict:
        """运行所有迁移学习对。"""
        if pairs is None:
            pairs = self.TRANSFER_PAIRS
        all_results = {}
        for src, tgt in pairs:
            key = f"{src}->{tgt}"
            try:
                all_results[key] = self.run_transfer(src, tgt, sample_size, eval_batch_size)
            except Exception as e:
                print(f"[ERROR] {key}: {e}")
                all_results[key] = {'error': str(e), 'status': 'FAILED'}
        return all_results

    def run_ablation_suite(self, sample_size: int = 50,
                           pairs: Optional[List[Tuple]] = None,
                           eval_batch_size: int = 8,
                           infer_method: str = 'llm_logits') -> Dict:
        """运行消融实验套件：每个迁移对同时运行 Full-RAG 和 No-RAG。"""
        import gc
        if pairs is None:
            pairs = self.TRANSFER_PAIRS
        ablation = {}
        for src, tgt in pairs:
            key = f"{src}->{tgt}"
            ablation[key] = {}
            try:
                ablation[key]['full_rag'] = self.run_transfer(
                    src, tgt, sample_size, eval_batch_size, infer_method=infer_method)
            except Exception as e:
                print(f"[ERROR] Full-RAG {key}: {e}")
                ablation[key]['full_rag'] = {'error': str(e)}
            try:
                ablation[key]['no_rag'] = self.run_transfer_no_rag(
                    src, tgt, sample_size, eval_batch_size, infer_method=infer_method)
            except Exception as e:
                print(f"[ERROR] No-RAG {key}: {e}")
                ablation[key]['no_rag'] = {'error': str(e)}

            # 每对实验结束后清理 GPU 缓存，防止累积导致 OOM
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                mem_alloc = torch.cuda.memory_allocated() / 1024**3
                mem_reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"[GPU] After {key}: allocated={mem_alloc:.1f}GB, reserved={mem_reserved:.1f}GB")

        return ablation

    def run_full_ablation(self, sample_size: int = 50,
                          pairs: Optional[List[Tuple]] = None,
                          eval_batch_size: int = 8) -> Dict:
        """
        完整消融实验：对每个迁移对，运行 2 种推理方式 × 2 种条件（RAG/NoRAG）。
        
        推理方式:
          - llm_logits: LLM next-token logits 分类
          - generate:   原始 LLM generate + regex 提取
        """
        import gc
        if pairs is None:
            pairs = self.TRANSFER_PAIRS

        infer_methods = ['llm_logits', 'generate']
        all_results = {}

        for src, tgt in pairs:
            key = f"{src}->{tgt}"
            all_results[key] = {}

            for method in infer_methods:
                # Full-RAG
                rag_key = f"full_rag_{method}"
                try:
                    print(f"\n>>> {key} | {rag_key} <<<")
                    all_results[key][rag_key] = self.run_transfer(
                        src, tgt, sample_size, eval_batch_size, infer_method=method)
                except Exception as e:
                    print(f"[ERROR] {rag_key} {key}: {e}")
                    all_results[key][rag_key] = {'error': str(e)}

                # No-RAG
                norag_key = f"no_rag_{method}"
                try:
                    print(f"\n>>> {key} | {norag_key} <<<")
                    all_results[key][norag_key] = self.run_transfer_no_rag(
                        src, tgt, sample_size, eval_batch_size, infer_method=method)
                except Exception as e:
                    print(f"[ERROR] {norag_key} {key}: {e}")
                    all_results[key][norag_key] = {'error': str(e)}

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return all_results

    def save_results(self, results: Dict, output_dir: str = "real_experiment_results"):
        Path(output_dir).mkdir(exist_ok=True)
        with open(f"{output_dir}/detailed_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str, ensure_ascii=False)
        print(f"\n[Save] Results saved to {output_dir}/")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_ablation_comparison(ablation_results: Dict):
    """打印 Full-RAG vs No-RAG 对比表。"""
    print("\n" + "=" * 75)
    print("  ABLATION STUDY: Full-RAG  vs  No-RAG")
    print("=" * 75)
    header = f"{'Transfer':<20} {'Cond':<12} {'Acc':>6} {'F1':>6} {'AUC':>6} {'#Pred':>6}"
    print(header)
    print("-" * 75)
    for key, conditions in ablation_results.items():
        for ck, cl in [('full_rag', 'Full-RAG'), ('no_rag', 'No-RAG')]:
            res = conditions.get(ck, {})
            m = res.get('metrics', {})
            if m:
                print(f"{key:<20} {cl:<12} "
                      f"{m.get('accuracy', float('nan')):>6.4f} "
                      f"{m.get('f1', float('nan')):>6.4f} "
                      f"{m.get('auc', float('nan')):>6.4f} "
                      f"{res.get('total_predicted', 0):>6}")
            else:
                print(f"{key:<20} {cl:<12}  FAILED")
        # Delta
        rag_m = conditions.get('full_rag', {}).get('metrics', {})
        norag_m = conditions.get('no_rag', {}).get('metrics', {})
        if rag_m and norag_m:
            da = rag_m.get('accuracy', 0) - norag_m.get('accuracy', 0)
            df = rag_m.get('f1', 0) - norag_m.get('f1', 0)
            print(f"{'':20} {'Δ(RAG)':<12} {da:>+6.4f} {df:>+6.4f}")
        print("-" * 75)
    print("=" * 75)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Domain-Aware RAG-MDA — Graph Transfer Learning + Ablation")
    print("  LLM Backend: ModelScope (frozen, soft graph token injection)")
    print("  Inference: llm_logits | Training: source domain only")
    print("=" * 60)

    experiment = TransferExperiment(
        data_dir="data",
        hidden_dim=128,
        gnn_epochs=100,
        gnn_batch_size=128,
        llm_path=MODELSCOPE_MODEL_ID,
        modelscope_cache_dir=MODELSCOPE_CACHE_DIR,
        modelscope_revision=MODELSCOPE_REVISION,
        num_graph_tokens=32,
        load_in_8bit=True,
    )

    pairs = [
        ('PROTEINS', 'DD'),
        ('DD', 'PROTEINS'),
        ('COX2', 'COX2_MD'),
        ('COX2_MD', 'COX2'),
        ('BZR', 'BZR_MD'),
        ('BZR_MD', 'BZR'),
    ]

    # 运行消融实验（Full-RAG vs No-RAG，使用 LLM logits 推理）
    ablation = experiment.run_ablation_suite(
        sample_size=9999,       # 全量评估（目标域全部数据）
        eval_batch_size=2,
        pairs=pairs,
        infer_method='llm_logits',
    )

    # 保存结果
    experiment.save_results(ablation)
    print_ablation_comparison(ablation)

    print("\nExperiment completed.")


if __name__ == "__main__":
    main()
