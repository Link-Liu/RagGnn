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
MODELSCOPE_MODEL_ID = os.getenv(
    'MODELSCOPE_MODEL_ID',
    'LLM-Research/Meta-Llama-3.1-8B',
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
        loader = DataLoader(list(dataset), batch_size=batch_size, shuffle=True)

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
                 gnn_epochs: int = 30, gnn_batch_size: int = 32,
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
        target_name: str,
        source_name: str,
        ckpt_proj: str,
        train_epochs: int = 5,
        train_batch_size: int = 4,
        lr_gnn: float = 1e-4,
        lr_proj: float = 3e-4,
        grad_accum_steps: int = 4,
        train_ratio: float = 0.64,
        val_ratio: float = 0.16,
    ) -> List[int]:
        """
        用目标域的标签数据，联合训练 GNN + Projector。

        划分比例：train 64% / val 16% / test 20%（test 从不参与训练）
        返回 test_indices，供 run_transfer 做评估。

        训练数据流：
            目标图 → GNN(可训练) → Projector(可训练) → soft tokens
            → 替换 prompt 中 <graph_token> 位置 → 冻结 LLM → CE loss
            → 反向传播 → 只更新 GNN + Projector

        Returns:
            test_indices: 测试集索引（未被训练/验证使用的样本）
        """
        import random
        from torch_geometric.data import Batch as PyGBatch

        print(f"\n{'='*60}")
        print(f"  [Joint Train] GNN + Projector  ({source_name} -> {target_name})")
        print(f"  Train epochs={train_epochs}, lr_gnn={lr_gnn}, lr_proj={lr_proj}")
        print(f"{'='*60}")

        prop_desc = self.PROP_DESC.get(target_name.lower(), target_name)

        # ---- Train / Val / Test 三段划分 ----
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

        # ---- 优化器：只优化 GNN + Projector ----
        optimizer = torch.optim.AdamW([
            {"params": self.llm.gnn.parameters(), "lr": lr_gnn},
            {"params": self.llm.projector.parameters(), "lr": lr_proj},
        ], weight_decay=1e-5)

        # ---- 训练 prompt（与推理格式对齐）----
        graph_token_str = " ".join([GRAPH_TOKEN] * self.num_graph_tokens)
        train_prompt = (
            f"Graph representation: {graph_token_str}\n"
            f"Task: {prop_desc}\n"
            f"Predict the label (0 or 1).\n"
            f"Answer:"
        )

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(1, train_epochs + 1):
            # ---- 训练 ----
            self.llm.gnn.train()
            self.llm.projector.train()
            optimizer.zero_grad()

            epoch_loss, n_steps = 0.0, 0
            random.shuffle(train_indices)

            for start in range(0, len(train_indices), train_batch_size):
                batch_idx = train_indices[start: start + train_batch_size]
                batch_data = [tgt_list[i] for i in batch_idx]
                B = len(batch_data)

                pyg_batch = PyGBatch.from_data_list(batch_data)
                prompts = [train_prompt] * B
                labels_text = [str(int(d.y.item())) for d in batch_data]

                loss = self.llm.compute_loss(pyg_batch, prompts, labels_text)
                loss = loss / grad_accum_steps
                loss.backward()

                n_steps += 1
                epoch_loss += loss.item() * grad_accum_steps

                if n_steps % grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.llm.trainable_parameters(), 1.0
                    )
                    optimizer.step()
                    optimizer.zero_grad()

            if n_steps % grad_accum_steps != 0:
                torch.nn.utils.clip_grad_norm_(
                    self.llm.trainable_parameters(), 1.0
                )
                optimizer.step()
                optimizer.zero_grad()

            avg_train_loss = epoch_loss / max(n_steps, 1)

            # ---- 验证 ----
            self.llm.gnn.eval()
            self.llm.projector.eval()
            val_loss, val_steps = 0.0, 0

            with torch.no_grad():
                for start in range(0, len(val_indices), train_batch_size):
                    batch_idx = val_indices[start: start + train_batch_size]
                    batch_data = [tgt_list[i] for i in batch_idx]
                    B = len(batch_data)

                    pyg_batch = PyGBatch.from_data_list(batch_data)
                    prompts = [train_prompt] * B
                    labels_text = [str(int(d.y.item())) for d in batch_data]

                    loss = self.llm.compute_loss(pyg_batch, prompts, labels_text)
                    val_loss += loss.item()
                    val_steps += 1

            avg_val_loss = val_loss / max(val_steps, 1)

            print(f"  Epoch {epoch}/{train_epochs}  "
                  f"train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}",
                  end="")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = {
                    "gnn": {k: v.cpu().clone() for k, v in self.llm.gnn.state_dict().items()},
                    "projector": {k: v.cpu().clone() for k, v in self.llm.projector.state_dict().items()},
                }
                print(f"  ← best ✓")
            else:
                print()

        # ---- 恢复最佳权重并保存 ----
        if best_state:
            self.llm.gnn.load_state_dict(best_state["gnn"])
            self.llm.projector.load_state_dict(best_state["projector"])
            self.llm.gnn.to(self.device)
            self.llm.projector.to(self.device)

            Path(ckpt_proj).parent.mkdir(parents=True, exist_ok=True)
            # 同时保存 test_indices 以便复现
            best_state["test_indices"] = test_indices
            torch.save(best_state, ckpt_proj)
            print(f"  [Joint Train] Best checkpoint saved -> {ckpt_proj} "
                  f"(val_loss={best_val_loss:.4f})")

        self.llm.gnn.eval()
        self.llm.projector.eval()
        return test_indices

    def run_transfer(self, source_name: str, target_name: str,
                     sample_size: int = 50) -> Dict:
        """
        运行单个迁移学习实验: source_name -> target_name
        """
        print(f"\n{'='*60}")
        print(f"  Transfer: {source_name} -> {target_name}  (sample={sample_size})")
        print(f"{'='*60}")

        # 1. 加载数据
        src_list, tgt_list, unified_dim, edge_dim = self._load_and_prepare(source_name, target_name)
        ckpt_gnn  = f"checkpoints/gnn_{source_name.lower()}.pt"
        ckpt_proj = f"checkpoints/proj_{source_name.lower()}_{target_name.lower()}.pt"

        # 2. 延迟初始化 LocalLLMInterface（首次运行才加载 LLM）
        if self.llm is None or self.llm.gnn.convs[0].nn[0].in_features != unified_dim:
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
                tgt_list, target_name, source_name, ckpt_proj
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

        for pos, idx in enumerate(target_indices):
            data = tgt_list[int(idx)]
            true_label = data.y.item()
            graph_id = f"{target_name}_{idx}"
            graph_info = extract_graph_info(data, graph_id, target_name)

            print(f"  [{pos+1}/{n_target}] {graph_id}", end="")

            try:
                emb = _encode_single_graph(self.llm, data, self.device)
                retrieved = retriever.retrieve_similar_graphs(emb, graph_info, k=5)
                graph_tokens_text = serialize_graph_tokens(emb)

                prompt = create_detailed_prompt(
                    target_graph_info=graph_info,
                    retrieved_examples=retrieved,
                    property_description=prop_desc,
                    target_dataset=target_name.lower(),
                    source_dataset=source_name.lower(),
                    graph_tokens_text=graph_tokens_text,
                )

                # 使用 soft token 注入进行推理
                pyg_batch = _make_single_pyg_batch(data)
                result = self.llm.predict(pyg_batch, prompt)
                pred = result['prediction']
                predictions.append(pred)
                true_labels.append(true_label)

                details.append({
                    'graph_id': graph_id,
                    'true_label': true_label,
                    'prediction': pred,
                    'num_nodes': data.num_nodes,
                    'num_edges': data.num_edges,
                    'retrieved_count': len(retrieved),
                    'llm_response': result['response'][:200],
                })

                marker = 'OK' if pred == true_label else 'WRONG'
                print(f"  pred={pred} truth={true_label} [{marker}]")

            except Exception as e:
                failed += 1
                print(f"  ERROR: {e}")

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
                            sample_size: int = 50) -> Dict:
        """消融实验：无 RAG 对照组。"""
        print(f"\n{'='*60}")
        print(f"  [No-RAG] {source_name} -> {target_name}  (sample={sample_size})")
        print(f"{'='*60}")

        _, tgt_list, unified_dim, _ = self._load_and_prepare(source_name, target_name)

        # No-RAG 也需要 LLM（用于 soft token 推理），若尚未初始化则在此完成
        if self.llm is None or self.llm.gnn.convs[0].nn[0].in_features != unified_dim:
            print(f"[LLM] Initializing LocalLLMInterface for No-RAG "
                  f"(node_feat={unified_dim}) ...")
            self.llm = LocalLLMInterface(
                model_name_or_path=self._llm_path,
                num_node_features=unified_dim,
                gnn_hidden_dim=self.hidden_dim,
                num_graph_tokens=self.num_graph_tokens,
                load_in_8bit=self._load_in_8bit,
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

        rng = np.random.RandomState(42)
        n_target = min(sample_size, len(tgt_list))
        target_indices = rng.choice(len(tgt_list), n_target, replace=False)

        prop_desc = self.PROP_DESC.get(target_name.lower(), target_name)
        predictions, true_labels, details = [], [], []
        failed = 0

        for pos, idx in enumerate(target_indices):
            data = tgt_list[int(idx)]
            true_label = data.y.item()
            graph_id = f"{target_name}_{idx}"
            graph_info = extract_graph_info(data, graph_id, target_name)

            print(f"  [{pos+1}/{n_target}] {graph_id}", end="")

            try:
                prompt = create_no_rag_prompt(
                    target_graph_info=graph_info,
                    property_description=prop_desc,
                    target_dataset=target_name.lower(),
                )
                pyg_batch = _make_single_pyg_batch(data)
                result = self.llm.predict(pyg_batch, prompt)
                pred = result['prediction']
                predictions.append(pred)
                true_labels.append(true_label)

                details.append({
                    'graph_id': graph_id,
                    'true_label': true_label,
                    'prediction': pred,
                    'retrieved_count': 0,
                    'llm_response': result['response'][:200],
                })
                marker = 'OK' if pred == true_label else 'WRONG'
                print(f"  pred={pred} truth={true_label} [{marker}]")

            except Exception as e:
                failed += 1
                print(f"  ERROR: {e}")

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
                          pairs: Optional[List[Tuple]] = None) -> Dict:
        """运行所有迁移学习对。"""
        if pairs is None:
            pairs = self.TRANSFER_PAIRS
        all_results = {}
        for src, tgt in pairs:
            key = f"{src}->{tgt}"
            try:
                all_results[key] = self.run_transfer(src, tgt, sample_size)
            except Exception as e:
                print(f"[ERROR] {key}: {e}")
                all_results[key] = {'error': str(e), 'status': 'FAILED'}
        return all_results

    def run_ablation_suite(self, sample_size: int = 50,
                           pairs: Optional[List[Tuple]] = None) -> Dict:
        """运行消融实验套件：每个迁移对同时运行 Full-RAG 和 No-RAG。"""
        if pairs is None:
            pairs = self.TRANSFER_PAIRS
        ablation = {}
        for src, tgt in pairs:
            key = f"{src}->{tgt}"
            ablation[key] = {}
            try:
                ablation[key]['full_rag'] = self.run_transfer(src, tgt, sample_size)
            except Exception as e:
                print(f"[ERROR] Full-RAG {key}: {e}")
                ablation[key]['full_rag'] = {'error': str(e)}
            try:
                ablation[key]['no_rag'] = self.run_transfer_no_rag(src, tgt, sample_size)
            except Exception as e:
                print(f"[ERROR] No-RAG {key}: {e}")
                ablation[key]['no_rag'] = {'error': str(e)}
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
        hidden_dim=128,
        gnn_epochs=30,
        gnn_batch_size=32,
        llm_path=MODELSCOPE_MODEL_ID,
        modelscope_cache_dir=MODELSCOPE_CACHE_DIR,
        modelscope_revision=MODELSCOPE_REVISION,
        num_graph_tokens=8,
        load_in_8bit=False,        # 显存不足时设为 True
    )

    # LLM 天然冻结，无需检查服务状态

    # 运行消融实验
    ablation = experiment.run_ablation_suite(
        sample_size=50,
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