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
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from pathlib import Path
from typing import List, Dict, Tuple, Optional

warnings.filterwarnings('ignore')

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.gnn_encoder import GNNEncoder
from models.local_llm_interface import LocalLLMInterface, GRAPH_TOKEN
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
# GNN Embedding Engine (图级别嵌入)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Graph Token 序列化：将 GNN 嵌入转换为提示词中的文本 token
# ---------------------------------------------------------------------------

# 每个图在提示词中保留的嵌入维度数（取最显著的前 N 维）
GRAPH_TOKEN_TOP_K = 16


def serialize_graph_tokens(embedding: np.ndarray, top_k: int = GRAPH_TOKEN_TOP_K) -> str:
    """
    将 L2 归一化的 GNN 嵌入向量序列化为文本 token 字符串，
    只取绝对值最大的 top_k 维，格式为:
      [idx:val, idx:val, ...]
        供冻结 LLM 在文本提示词中理解图的连续特征表示。
    """
    top_k = min(top_k, len(embedding))
    top_indices = np.argsort(np.abs(embedding))[::-1][:top_k]
    top_indices_sorted = np.sort(top_indices)
    tokens = [f"{i}:{embedding[i]:+.3f}" for i in top_indices_sorted]
    return "[" + ", ".join(tokens) + "]"


class GNNEmbeddingEngine:
    """
    使用 GINEncoder 为图数据集产生图级别嵌入，
    并支持将嵌入序列化为提示词 graph token 文本。
    """

    def __init__(self, num_node_features: int, num_edge_features: int = 0,
                 hidden_dim: int = 64, num_layers: int = 3, device: str = 'cpu'):
        self.device = torch.device(device)
        self.hidden_dim = hidden_dim
        self.num_node_features = num_node_features

        self.encoder = GNNEncoder(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        ).to(self.device)
        self.encoder.eval()
        print(f"[GNN] Initialized: node_feat={num_node_features}, "
              f"edge_feat={num_edge_features}, hidden={hidden_dim}, layers={num_layers}")

    def train_on_dataset(self, dataset, epochs: int = 30,
                         batch_size: int = 32, lr: float = 1e-3,
                         checkpoint_path: Optional[str] = None) -> None:
        """在数据集上有监督地训练 GNN encoder（图分类任务）。LLM 天然冻结。"""
        print(f"[GNN-Train] Training on {len(dataset)} graphs, {epochs} epochs ...")
        classifier = nn.Linear(self.hidden_dim, 1).to(self.device)
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(classifier.parameters()),
            lr=lr, weight_decay=1e-5
        )
        criterion = nn.BCEWithLogitsLoss()
        
        # A5: DataLoader 参数调优
        # 兼容 device 为字符串或 torch.device 对象
        use_cuda = (self.device == 'cuda' or (isinstance(self.device, torch.device) and self.device.type == 'cuda'))
        loader = DataLoader(
            list(dataset), 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0, # Windows 下 num_workers > 0 易出错，保持 0 但开启 pin_memory
            pin_memory=use_cuda
        )

        self.encoder.train()
        best_loss = float('inf')
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            count = 0
            for batch in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                emb = self.encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                logit = classifier(emb).squeeze(-1)
                loss = criterion(logit, batch.y.float())
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch.num_graphs
                count += batch.num_graphs
            avg_loss = total_loss / max(count, 1)
            if avg_loss < best_loss:
                best_loss = avg_loss
            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}  best={best_loss:.4f}")

        self.encoder.eval()
        del classifier
        print(f"[GNN-Train] Done. Best loss={best_loss:.4f}")

        if checkpoint_path:
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.encoder.state_dict(), checkpoint_path)
            print(f"[GNN-Train] Saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        p = Path(checkpoint_path)
        if not p.exists():
            return False
        self.encoder.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device)
        )
        self.encoder.eval()
        print(f"[GNN] Loaded weights from {checkpoint_path}")
        return True

    @torch.no_grad()
    def encode_graph(self, data) -> np.ndarray:
        """编码单个图，返回 L2 归一化嵌入。"""
        data = data.to(self.device)
        emb = self.encoder(data.x, data.edge_index, data.edge_attr)
        emb = emb.cpu().numpy().flatten()
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

    @torch.no_grad()
    def encode_dataset(self, dataset) -> Tuple[List[np.ndarray], List[int]]:
        """编码整个数据集，返回 (embeddings, labels)。"""
        embeddings = []
        labels = []
        for data in dataset:
            emb = self.encode_graph(data)
            embeddings.append(emb)
            labels.append(data.y.item())
        return embeddings, labels

    def get_graph_token_text(self, data) -> str:
        """编码单个图并返回序列化 graph token 文本，用于插入提示词。"""
        emb = self.encode_graph(data)
        return serialize_graph_tokens(emb)


# ---------------------------------------------------------------------------
# 图信息提取（供 Prompt 使用）
# ---------------------------------------------------------------------------

def extract_graph_info(data, graph_id: str = "", dataset_name: str = "") -> Dict:
    """从 PyG Data 对象提取 Prompt 需要的图信息。"""
    num_nodes = data.num_nodes
    num_edges = data.num_edges
    max_edges = num_nodes * (num_nodes - 1) if num_nodes > 1 else 1
    density = num_edges / max_edges if max_edges > 0 else 0
    avg_degree = (2 * num_edges / num_nodes) if num_nodes > 0 else 0

    info = {
        'graph_id': graph_id,
        'dataset': dataset_name,
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'density': density,
        'avg_degree': round(avg_degree, 2),
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

    def _load_and_prepare(self, source_name: str, target_name: str):
        """加载源域和目标域数据集，统一特征维度，返回 Data 列表。"""
        src_ds = load_dataset(self.data_dir, source_name)
        tgt_ds = load_dataset(self.data_dir, target_name)

        ensure_node_features(src_ds, source_name)
        ensure_node_features(tgt_ds, target_name)

        # 转为列表以便 in-place 修改特征
        src_list = dataset_to_list(src_ds)
        tgt_list = dataset_to_list(tgt_ds)

        unified_dim = unify_feature_dim_lists(src_list, tgt_list, source_name, target_name)

        # 获取边特征维度
        edge_dim = 0
        if src_list and src_list[0].edge_attr is not None:
            edge_dim = src_list[0].edge_attr.shape[1]

        return src_list, tgt_list, unified_dim, edge_dim

    # ------------------------------------------------------------------
    # Phase 2: 联合训练 GNN + Projector（通过冻结 LLM 的 CE loss）
    # ------------------------------------------------------------------
    def _joint_train(
        self,
        tgt_list: List,
        src_list: List,
        target_name: str,
        source_name: str,
        ckpt_proj: str,
        train_epochs: int = 50,       # 增加训练轮数，让 projector 充分学习
        train_batch_size: int = 1,    # 显存紧张，单样本前向
        lr_gnn: float = 5e-5,
        lr_proj: float = 2e-4,
        grad_accum_steps: int = 16,   # 等效 batch_size = 1*16 = 16
        train_ratio: float = 0.64,
        val_ratio: float = 0.16,
    ) -> List[int]:
        """
        用目标域的标签数据，联合训练 GNN + Projector。
        训练时引入 RAG 例子以对齐推理阶段。
        """
        import random
        from torch_geometric.data import Batch as PyGBatch

        print(f"\n{'='*60}")
        print(f"  [Joint Train] GNN + Projector (RAG-Augmented) ({source_name} -> {target_name})")
        print(f"  Train epochs={train_epochs}, lr_gnn={lr_gnn}, lr_proj={lr_proj}")
        print(f"{'='*60}")

        prop_desc = self.PROP_DESC.get(target_name.lower(), target_name)

        # ---- 划分数据集 ----
        indices = list(range(len(tgt_list)))
        random.seed(42)
        random.shuffle(indices)
        n_train = int(train_ratio * len(indices))
        n_val   = int(val_ratio * len(indices))
        train_indices = indices[:n_train]
        val_indices   = indices[n_train: n_train + n_val]
        test_indices  = indices[n_train + n_val:]
        print(f"  Train: {len(train_indices)}, Val: {len(val_indices)}, "
              f"Test: {len(test_indices)} (held-out for evaluation)")

        # ---- 构建 RAG 预检索索引（使用当前预训练 GNN） ----
        print(f"  [RAG-Setup] Building temporary retrieval index from {source_name} ...")
        self.llm.gnn.eval()
        src_embs, src_labels = _encode_dataset_with_llm(self.llm, src_list, self.device)
        src_infos = []
        for i, (data, emb) in enumerate(zip(src_list, src_embs)):
            info = extract_graph_info(data, f"{source_name}_{i}", source_name)
            info['source_domain'] = source_name.lower()
            info['graph_tokens_text'] = serialize_graph_tokens(emb)
            src_infos.append(info)
        
        retriever = GraphRetriever(embedding_dim=self.hidden_dim)
        retriever.add_source_graphs(src_embs, src_infos, src_labels)

        # ---- 为训练/验证集预先检索例子 ----
        def pre_retrieve(dataset_indices):
            print(f"  [RAG-Setup] Pre-retrieving for {len(dataset_indices)} samples ...")
            results = {}
            for idx in dataset_indices:
                data = tgt_list[idx]
                emb = _encode_single_graph(self.llm, data, self.device)
                info = extract_graph_info(data, f"{target_name}_{idx}", target_name)
                retrieved = retriever.retrieve_similar_graphs(emb, info, k=5)
                results[idx] = retrieved
            return results

        train_rag_cache = pre_retrieve(train_indices)
        val_rag_cache = pre_retrieve(val_indices)

        # ---- 封装带 RAG 的自定义 Dataset ----
        class RAGDataset(torch.utils.data.Dataset):
            def __init__(self, indices, rag_cache):
                self.indices = indices
                self.rag_cache = rag_cache
                self.labels = [int(tgt_list[idx].y.item()) for idx in indices]
            def __len__(self): return len(self.indices)
            def __getitem__(self, i):
                idx = self.indices[i]
                return tgt_list[idx], self.rag_cache[idx]

        def collate_rag(batch_samples):
            data_list, rag_list = zip(*batch_samples)
            return PyGBatch.from_data_list(data_list), rag_list

        # ---- 平衡采样逻辑 (A7: 解决 F1 低的问题) ----
        train_dataset = RAGDataset(train_indices, train_rag_cache)
        labels = train_dataset.labels
        class_sample_count = np.array([len(np.where(labels == t)[0]) for t in np.unique(labels)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in labels])
        samples_weight = torch.from_numpy(samples_weight)
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))

        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=train_batch_size, sampler=sampler, collate_fn=collate_rag,
            num_workers=0, pin_memory=(self.device == 'cuda')
        )
        val_loader = torch.utils.data.DataLoader(
            RAGDataset(val_indices, val_rag_cache), 
            batch_size=train_batch_size, shuffle=False, collate_fn=collate_rag,
            num_workers=0, pin_memory=(self.device == 'cuda')
        )

        # ---- 优化器 ----
        optimizer = torch.optim.AdamW([
            {"params": self.llm.gnn.parameters(), "lr": lr_gnn},
            {"params": self.llm.projector.parameters(), "lr": lr_proj},
        ], weight_decay=1e-5)

        # ---- 学习率调度器（Cosine Annealing + Warmup）----
        total_steps = train_epochs * max(len(train_loader), 1)
        warmup_steps = max(total_steps // 5, 1)
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(warmup_steps, 1))
            progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
            return max(0.1, 0.5 * (1.0 + np.cos(np.pi * progress)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        patience_limit = 10  # Early stopping patience（50 epoch 需要更大 patience）
        use_amp = self.device == "cuda"

        # ---- 训练循环 ----
        for epoch in range(1, train_epochs + 1):
            self.llm.gnn.train()
            self.llm.projector.train()
            optimizer.zero_grad()
            epoch_loss, n_steps = 0.0, 0

            for batch, rag_examples_list in train_loader:
                batch = batch.to(self.device)
                B = batch.num_graphs
                prompts = []
                for i in range(B):
                    info = extract_graph_info(batch[i], "train", target_name)
                    p = create_detailed_prompt(
                        target_graph_info=info,
                        retrieved_examples=rag_examples_list[i],
                        property_description=prop_desc,
                        target_dataset=target_name.lower(),
                        source_dataset=source_name.lower(),
                    )
                    prompts.append(p)
                
                labels_text = [str(int(y.item())) for y in batch.y]
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
                    loss = self.llm.compute_loss(batch, prompts, labels_text)
                    loss = loss / grad_accum_steps

                loss.backward()
                n_steps += 1
                epoch_loss += loss.item() * grad_accum_steps
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

            # ---- 验证阶段 ----
            self.llm.gnn.eval()
            self.llm.projector.eval()
            val_loss, val_steps = 0.0, 0
            with torch.no_grad():
                for batch, rag_examples_list in val_loader:
                    batch = batch.to(self.device)
                    B = batch.num_graphs
                    prompts = []
                    for i in range(B):
                        info = extract_graph_info(batch[i], "val", target_name)
                        p = create_detailed_prompt(
                            target_graph_info=info,
                            retrieved_examples=rag_examples_list[i],
                            property_description=prop_desc,
                            target_dataset=target_name.lower(),
                            source_dataset=source_name.lower(),
                        )
                        prompts.append(p)
                    
                    labels_text = [str(int(y.item())) for y in batch.y]
                    with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
                        loss = self.llm.compute_loss(batch, prompts, labels_text)
                        val_loss += loss.item()
                        val_steps += 1

            avg_val_loss = val_loss / max(val_steps, 1)
            print(f"  Epoch {epoch}/{train_epochs}  train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}", end="")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
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
            best_state["test_indices"] = test_indices
            torch.save(best_state, ckpt_proj)
            print(f"  [Joint Train] Best checkpoint saved -> {ckpt_proj} (val_loss={best_val_loss:.4f})")

        self.llm.gnn.eval()
        self.llm.projector.eval()
        return test_indices

    def run_transfer(self, source_name: str, target_name: str,
                     sample_size: int = 50, eval_batch_size: int = 4) -> Dict:
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
        if self.llm is None or self.llm.gnn.convs[0].nn[0].in_features != unified_dim:
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

        # 4. ========== 联合训练 GNN + Projector（通过冻结 LLM 的 CE loss）==========
        #    返回 test_indices（从未参与训练/验证的样本，用于评估）
        if not Path(ckpt_proj).exists():
            test_indices = self._joint_train(
                tgt_list, src_list, target_name, source_name, ckpt_proj
            )
        else:
            # 从已保存的 checkpoint 恢复 test_indices
            ckpt_data = torch.load(ckpt_proj, map_location='cpu')
            test_indices = ckpt_data.get('test_indices', None)
            if test_indices is None:
                # 兼容旧 checkpoint：用相同 seed 重新划分
                import random
                indices = list(range(len(tgt_list)))
                random.seed(42)
                random.shuffle(indices)
                n_train = int(0.64 * len(indices))
                n_val   = int(0.16 * len(indices))
                test_indices = indices[n_train + n_val:]
                print(f"[WARN] test_indices not in checkpoint, re-derived {len(test_indices)} test samples")

        print(f"[Eval] Test set: {len(test_indices)} graphs (never seen during training)")

        # 5. 编码源域 → 构建 RAG 索引
        print(f"[RAG] Encoding {len(src_list)} source graphs ...")
        src_embs, src_labels = _encode_dataset_with_llm(self.llm, src_list, self.device)
        src_infos = []
        for i, (data, emb) in enumerate(zip(src_list, src_embs)):
            info = extract_graph_info(data, f"{source_name}_{i}", source_name)
            info['source_domain'] = source_name.lower()
            info['graph_tokens_text'] = serialize_graph_tokens(emb)
            src_infos.append(info)

        retriever = GraphRetriever(embedding_dim=self.hidden_dim)
        retriever.add_source_graphs(src_embs, src_infos, src_labels)
        print(f"[RAG] Indexed {len(src_embs)} source graphs")

        # 6. 仅在 test_indices（held-out）上评估
        n_target = len(test_indices)
        target_indices = test_indices

        prop_desc = self.PROP_DESC.get(target_name.lower(), target_name)
        predictions, true_labels, details = [], [], []
        failed = 0
        # B1: 改为按 Batch 推理
        from torch_geometric.data import Batch as PyGBatch
        for start_pos in range(0, n_target, eval_batch_size):
            end_pos = min(start_pos + eval_batch_size, n_target)
            batch_indices = target_indices[start_pos:end_pos]
            batch_data = [tgt_list[int(idx)] for idx in batch_indices]
            B_eval = len(batch_data)

            # 打印进度
            print(f"  [{start_pos+1}-{end_pos}/{n_target}] processing batch...", end="", flush=True)

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
                    graph_info = extract_graph_info(data, f"{target_name}_batch", target_name)
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

                # LLM 推理（已经是批量）
                results = self.llm.predict_batch(pyg_batch, batch_prompts)

                for i, res in enumerate(results):
                    data = batch_data[i]
                    pred = res['prediction']
                    true_label = data.y.item()
                    
                    predictions.append(pred)
                    true_labels.append(true_label)
                    details.append({
                        'graph_id': f"{target_name}_{batch_indices[i]}",
                        'true_label': true_label,
                        'prediction': pred,
                        'llm_response': res['response'][:200],
                    })

                print(f" done.")

            except Exception as e:
                failed += len(batch_indices)
                print(f"  ERROR in batch: {e}")

        # 5. 计算指标
        metrics = self._compute_metrics(predictions, true_labels)
        if metrics:
            self._print_metrics(f"{source_name}->{target_name}", metrics,
                                len(predictions), failed)

        return {
            'source': source_name, 'target': target_name,
            'total_target': n_target, 'total_predicted': len(predictions),
            'failed': failed, 'metrics': metrics, 'details': details,
            'llm_calls': self.llm.call_count if self.llm is not None else 0,
        }

    def run_transfer_no_rag(self, source_name: str, target_name: str,
                            sample_size: int = 50, eval_batch_size: int = 4) -> Dict:
        """消融实验：无 RAG 对照组。"""
        print(f"\n{'='*60}")
        print(f"  [No-RAG] {source_name} -> {target_name}  (sample={sample_size}, batch={eval_batch_size})")
        print(f"{'='*60}")

        _, tgt_list, unified_dim, _ = self._load_and_prepare(source_name, target_name)

        # No-RAG 也需要 LLM（用于 soft token 推理），若尚未初始化则在此完成
        if self.llm is None or self.llm.gnn.convs[0].nn[0].in_features != unified_dim:
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

        # 使用与 Full-RAG 相同的 test_indices 保证消融对比公平性
        ckpt_proj = f"checkpoints/proj_{source_name.lower()}_{target_name.lower()}.pt"
        if Path(ckpt_proj).exists():
            ckpt_data = torch.load(ckpt_proj, map_location='cpu')
            test_indices = ckpt_data.get('test_indices', None)
            if test_indices is not None:
                target_indices = test_indices
                n_target = len(target_indices)
                print(f"[No-RAG] Using same test_indices as Full-RAG: {n_target} graphs")
            else:
                rng = np.random.RandomState(42)
                n_target = min(sample_size, len(tgt_list))
                target_indices = rng.choice(len(tgt_list), n_target, replace=False)
                print(f"[No-RAG] No test_indices in checkpoint, random sampling {n_target} graphs")
        else:
            rng = np.random.RandomState(42)
            n_target = min(sample_size, len(tgt_list))
            target_indices = rng.choice(len(tgt_list), n_target, replace=False)
            print(f"[No-RAG] No checkpoint found, random sampling {n_target} graphs")

        prop_desc = self.PROP_DESC.get(target_name.lower(), target_name)
        predictions, true_labels, details = [], [], []
        failed = 0

        # B1: 改为按 Batch 推理
        from torch_geometric.data import Batch as PyGBatch
        for start_pos in range(0, n_target, eval_batch_size):
            end_pos = min(start_pos + eval_batch_size, n_target)
            batch_indices = target_indices[start_pos:end_pos]
            batch_data = [tgt_list[int(idx)] for idx in batch_indices]
            B_eval = len(batch_data)

            print(f"  [{start_pos+1}-{end_pos}/{n_target}] processing batch...", end="", flush=True)

            try:
                batch_prompts = []
                for data in batch_data:
                    graph_info = extract_graph_info(data, f"{target_name}_batch", target_name)
                    prompt = create_no_rag_prompt(
                        target_graph_info=graph_info,
                        property_description=prop_desc,
                        target_dataset=target_name.lower(),
                    )
                    batch_prompts.append(prompt)

                pyg_batch = PyGBatch.from_data_list(batch_data).to(self.device)
                results = self.llm.predict_batch(pyg_batch, batch_prompts)

                for i, res in enumerate(results):
                    data = batch_data[i]
                    pred = res['prediction']
                    true_label = data.y.item()
                    
                    predictions.append(pred)
                    true_labels.append(true_label)
                    details.append({
                        'graph_id': f"{target_name}_{batch_indices[i]}",
                        'true_label': true_label,
                        'prediction': pred,
                        'retrieved_count': 0,
                        'llm_response': res['response'][:200],
                    })

                print(f" done.")

            except Exception as e:
                failed += len(batch_indices)
                print(f"  ERROR in batch: {e}")

        metrics = self._compute_metrics(predictions, true_labels)
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

    def _compute_metrics(self, predictions, true_labels) -> Dict:
        if not predictions:
            return {}
        metrics = {
            'accuracy': float(accuracy_score(true_labels, predictions)),
            'f1': float(f1_score(true_labels, predictions, zero_division=0)),
            'precision': float(precision_score(true_labels, predictions, zero_division=0)),
            'recall': float(recall_score(true_labels, predictions, zero_division=0)),
        }
        if len(set(true_labels)) > 1:
            metrics['auc'] = float(roc_auc_score(true_labels, predictions))
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
                           eval_batch_size: int = 8) -> Dict:
        """运行消融实验套件：每个迁移对同时运行 Full-RAG 和 No-RAG。"""
        import gc
        if pairs is None:
            pairs = self.TRANSFER_PAIRS
        ablation = {}
        for src, tgt in pairs:
            key = f"{src}->{tgt}"
            ablation[key] = {}
            try:
                ablation[key]['full_rag'] = self.run_transfer(src, tgt, sample_size, eval_batch_size)
            except Exception as e:
                print(f"[ERROR] Full-RAG {key}: {e}")
                ablation[key]['full_rag'] = {'error': str(e)}
            try:
                ablation[key]['no_rag'] = self.run_transfer_no_rag(src, tgt, sample_size, eval_batch_size)
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
    print("=" * 60)

    # ModelScope 模型与缓存路径可通过环境变量覆盖
    # MODELSCOPE_MODEL_ID=LLM-Research/Meta-Llama-3.1-8B
    # MODELSCOPE_CACHE_DIR=<project>/checkpoints/modelscope
    experiment = TransferExperiment(
        data_dir="data",
        hidden_dim=512,            # 增大 GNN 维度，减少信息瓶颈（原 128，GraphPrompter 用 1024）
        gnn_epochs=100,            # 增加 GNN 预训练轮数（原 50）
        gnn_batch_size=128,
        llm_path=MODELSCOPE_MODEL_ID,
        modelscope_cache_dir=MODELSCOPE_CACHE_DIR,
        modelscope_revision=MODELSCOPE_REVISION,
        num_graph_tokens=1,        # 减少到 1（参考 GraphPrompter），减轻 projector 负担
        load_in_8bit=True,         # 8-bit 量化：LLM 显存 ~16GB→~8GB，防止溢出到共享内存
    )

    # LLM 天然冻结，无需检查服务状态

    # 运行消融实验
    ablation = experiment.run_ablation_suite(
        sample_size=200,
        eval_batch_size=2,         # 显存紧张，减至 2 降低 KV cache 占用
        pairs=[
            ('PROTEINS', 'DD'),
            ('DD', 'PROTEINS'),
            ('COX2', 'COX2_MD'),
            ('COX2_MD', 'COX2'),
            ('BZR', 'BZR_MD'),
            ('BZR_MD', 'BZR'),
        ],
    )

    # 保存结果
    experiment.save_results(ablation)

    # 打印对比表
    print_ablation_comparison(ablation)

    print("\nExperiment completed.")


if __name__ == "__main__":
    main()