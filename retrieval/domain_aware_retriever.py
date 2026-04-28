"""
Domain-Aware Retrieval-Augmented Generation (RAG) for Graph Classification

从源域图数据中检索与目标图结构相似的图样本。

检索通路:
  A (Embedding):   GNN 嵌入内积相似度（FAISS IndexFlatIP，L2归一化向量等价于余弦）
  B (Structural):  图统计量相似度 — 基于节点数、边数、密度等结构特征的相似度

不再依赖 SMILES/MACCS 指纹，适用于 TUDataset 图分类数据集。
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Optional
import pickle
import faiss


class GraphRetriever:
    """
    图数据的域感知检索系统。

    使用两条互补通路检索相似源域图:
      Pathway A — GNN Embedding (FAISS IndexFlatIP):
          捕获全局拓扑和节点特征层面的相似性。
          Weight: 0.6

      Pathway B — Graph Statistics Similarity:
          基于图级别统计量（节点数、边数、密度、平均度等）的相似度。
          补偿 GNN 嵌入可能忽略的尺度/密度差异。
          Weight: 0.4
    """

    WEIGHT_EMBEDDING = 0.6
    WEIGHT_STRUCTURAL = 0.4

    def __init__(self, embedding_dim: int = 64, use_faiss: bool = True):
        """
        Args:
            embedding_dim: GNN embedding 的维度
            use_faiss:     是否使用 FAISS 加速
        """
        self.embedding_dim = embedding_dim
        self.use_faiss = use_faiss

        self.source_embeddings: List[np.ndarray] = []
        self.source_graphs: List[Dict] = []      # 每个 dict 含 graph_id, source_domain, num_nodes 等
        self.source_labels: List[Any] = []
        self.source_stats: List[Dict] = []        # 图统计量

        self.faiss_index = None

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def add_source_graphs(self, embeddings: List[np.ndarray],
                          graph_infos: List[Dict], labels: List[Any]):
        """
        将源域图加入检索库。

        Args:
            embeddings:   L2 归一化后的 GNN 嵌入向量列表
            graph_infos:  图元数据字典列表（需含 'graph_id'、'source_domain'、'num_nodes'、'num_edges'）
            labels:       每个图的原始标签
        """
        self.source_embeddings.extend(embeddings)
        self.source_graphs.extend(graph_infos)
        self.source_labels.extend(labels)

        # 提取图统计量用于通路 B
        for info in graph_infos:
            num_nodes = info.get('num_nodes', 1)
            num_edges = info.get('num_edges', 0)
            max_edges = num_nodes * (num_nodes - 1) if num_nodes > 1 else 1
            density = num_edges / max_edges if max_edges > 0 else 0
            avg_degree = (2 * num_edges / num_nodes) if num_nodes > 0 else 0
            self.source_stats.append({
                'num_nodes': num_nodes,
                'num_edges': num_edges,
                'density': density,
                'avg_degree': avg_degree,
            })

        if self.use_faiss and self.source_embeddings:
            self._build_faiss_index()

    def _build_faiss_index(self):
        """构建 FAISS IndexFlatIP 内积索引。"""
        if not self.source_embeddings:
            return
        arr = np.array(self.source_embeddings, dtype=np.float32)
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        self.faiss_index.add(arr)

    # ------------------------------------------------------------------
    # Structural similarity (Pathway B)
    # ------------------------------------------------------------------

    def _compute_structural_similarity(self, target_info: Dict) -> np.ndarray:
        """
        计算目标图与所有源域图的结构相似度。

        使用归一化后的图统计量（节点数、边数、密度、平均度）之间的欧氏距离，
        转换为相似度分数。
        """
        n = len(self.source_stats)
        if n == 0:
            return np.zeros(0, dtype=np.float32)

        num_nodes = target_info.get('num_nodes', 1)
        num_edges = target_info.get('num_edges', 0)
        max_edges = num_nodes * (num_nodes - 1) if num_nodes > 1 else 1
        target_density = num_edges / max_edges if max_edges > 0 else 0
        target_avg_deg = (2 * num_edges / num_nodes) if num_nodes > 0 else 0

        # 收集统计量矩阵
        target_vec = np.array([num_nodes, num_edges, target_density, target_avg_deg])
        source_vecs = np.array([
            [s['num_nodes'], s['num_edges'], s['density'], s['avg_degree']]
            for s in self.source_stats
        ])

        # Min-max 归一化 (防止节点数量级差异主导距离)
        all_vecs = np.vstack([source_vecs, target_vec.reshape(1, -1)])
        mins = all_vecs.min(axis=0)
        maxs = all_vecs.max(axis=0)
        ranges = maxs - mins
        ranges[ranges == 0] = 1.0  # 防除零

        norm_target = (target_vec - mins) / ranges
        norm_source = (source_vecs - mins) / ranges

        # 欧氏距离 → 相似度 (exp(-d))
        dists = np.linalg.norm(norm_source - norm_target, axis=1)
        similarities = np.exp(-dists)

        return similarities.astype(np.float32)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve_similar_graphs(self,
                                target_embedding: np.ndarray,
                                target_info: Dict,
                                k: int = 5,
                                balance_labels: bool = True) -> List[Dict]:
        """
        检索 top-k 相似源域图，融合两条通路的得分。

        Args:
            target_embedding: 目标图的 L2 归一化 GNN 嵌入
            target_info:      目标图的结构信息字典
            k:                返回的图数量
            balance_labels:   若 True，强制平衡正负例

        Returns:
            图信息字典列表，附带检索得分
        """
        if not self.source_embeddings:
            return []

        n = len(self.source_embeddings)

        # Pathway A: GNN Embedding — FAISS IndexFlatIP
        embedding_scores = np.zeros(n, dtype=np.float32)

        if self.use_faiss and self.faiss_index is not None:
            qvec = target_embedding.reshape(1, -1).astype(np.float32)
            inner_products, indices = self.faiss_index.search(qvec, n)
            for ip, idx in zip(inner_products[0], indices[0]):
                embedding_scores[idx] = float(ip)
        else:
            qvec = target_embedding.reshape(1, -1)
            src = np.array(self.source_embeddings)
            embedding_scores = cosine_similarity(qvec, src)[0].astype(np.float32)

        # Pathway B: Graph Statistics Similarity
        structural_scores = self._compute_structural_similarity(target_info)

        # 加权融合
        combined = (self.WEIGHT_EMBEDDING * embedding_scores +
                    self.WEIGHT_STRUCTURAL * structural_scores)

        # 平衡正负例（可选）
        if balance_labels:
            half = max(1, k // 2)
            candidate_indices = np.argsort(combined)[::-1][:k * 2]

            pos_pool = [i for i in candidate_indices if self.source_labels[i] in (1, '1', True)]
            neg_pool = [i for i in candidate_indices if self.source_labels[i] in (0, '0', False)]

            selected = pos_pool[:half] + neg_pool[:half]
            if len(selected) < k:
                remaining = [i for i in candidate_indices if i not in selected]
                selected += remaining[:k - len(selected)]
            top_indices = selected[:k]
        else:
            top_indices = np.argsort(combined)[::-1][:k]

        # 构建返回结果
        retrieved = []
        for idx in top_indices:
            graph_data = self.source_graphs[idx].copy()
            graph_data['retrieval_score'] = float(combined[idx])
            graph_data['embedding_sim'] = float(embedding_scores[idx])
            graph_data['structural_sim'] = float(structural_scores[idx])
            graph_data['label'] = self.source_labels[idx]
            retrieved.append(graph_data)

        return retrieved

    # ------------------------------------------------------------------
    # 兼容旧接口名称
    # ------------------------------------------------------------------

    def add_source_molecules(self, embeddings, molecules, labels):
        """兼容旧接口。"""
        return self.add_source_graphs(embeddings, molecules, labels)

    def retrieve_similar_molecules(self, target_embedding, target_smiles="",
                                   k=5, balance_labels=True, **kwargs):
        """兼容旧接口。"""
        target_info = kwargs.get('target_info', {})
        return self.retrieve_similar_graphs(target_embedding, target_info, k, balance_labels)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_retriever(self, filepath: str):
        """持久化检索器状态。"""
        state = {
            'source_embeddings': self.source_embeddings,
            'source_graphs': self.source_graphs,
            'source_labels': self.source_labels,
            'source_stats': self.source_stats,
            'embedding_dim': self.embedding_dim,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    def load_retriever(self, filepath: str):
        """从磁盘恢复检索器状态并重建 FAISS 索引。"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        self.source_embeddings = state['source_embeddings']
        self.source_graphs = state['source_graphs']
        self.source_labels = state['source_labels']
        self.source_stats = state.get('source_stats', [])
        self.embedding_dim = state['embedding_dim']
        if self.use_faiss and self.source_embeddings:
            self._build_faiss_index()


# 向后兼容的类名别名
MolecularRetriever = GraphRetriever


# ---------------------------------------------------------------------------
# 冒烟测试
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    src_graphs = [
        {'graph_id': 'P_0', 'num_nodes': 15, 'num_edges': 40, 'source_domain': 'proteins'},
        {'graph_id': 'P_1', 'num_nodes': 20, 'num_edges': 60, 'source_domain': 'proteins'},
        {'graph_id': 'P_2', 'num_nodes': 8,  'num_edges': 14, 'source_domain': 'proteins'},
        {'graph_id': 'P_3', 'num_nodes': 25, 'num_edges': 80, 'source_domain': 'proteins'},
        {'graph_id': 'P_4', 'num_nodes': 12, 'num_edges': 30, 'source_domain': 'proteins'},
    ]
    src_embs = [rng.standard_normal(64).astype(np.float32) for _ in src_graphs]
    src_embs = [e / np.linalg.norm(e) for e in src_embs]
    src_labels = [1, 1, 0, 0, 1]

    retriever = GraphRetriever(embedding_dim=64)
    retriever.add_source_graphs(src_embs, src_graphs, src_labels)

    tgt_emb = rng.standard_normal(64).astype(np.float32)
    tgt_emb /= np.linalg.norm(tgt_emb)
    tgt_info = {'graph_id': 'DD_42', 'num_nodes': 18, 'num_edges': 50}

    results = retriever.retrieve_similar_graphs(tgt_emb, tgt_info, k=3)
    print(f"Retrieved {len(results)} graphs:")
    for m in results:
        print(f"  {m['graph_id']:10s}  combined={m['retrieval_score']:.3f}  "
              f"embed={m['embedding_sim']:.3f}  struct={m['structural_sim']:.3f}  "
              f"label={m['label']}  domain={m['source_domain']}")