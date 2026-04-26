"""
Domain-Aware Retrieval-Augmented Generation (RAG) for Molecular Graphs
Implements retrieval of source-domain molecules for cross-domain adaptation

通路设计:
  A (Embedding):  GNN 全局嵌入内积相似度（FAISS IndexFlatIP，L2归一化向量等价于余弦）
  B (Structural): MACCS Keys Tanimoto 相似度，聚焦关键药效团/毒性基团的 166 位指纹

修复记录:
  - [BUG] FAISS IndexFlatIP 返回内积，原代码错误地用 1-distance，已修正为直接使用内积值
  - [BUG] 通路B原为占位符 sim=0.5，已替换为真实 MACCS Tanimoto 计算
  - [NEW] retrieve_similar_molecules 新增 target_smiles 参数供通路B使用
  - [NEW] 返回结果中附带 source_domain 元数据，供 Prompt 中标签语义解析
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Optional
import pickle
import faiss

from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys


class MolecularRetriever:
    """
    Domain-aware retrieval system for molecular graphs.

    Retrieves molecules from the source domain that are similar to a target
    molecule using two complementary similarity pathways:

      Pathway A — GNN Embedding (IndexFlatIP via FAISS):
          Captures global structural topology and electronic context.
          Weight: 0.5

      Pathway B — MACCS Keys Tanimoto Similarity:
          Captures presence/absence of specific pharmacophore/toxicophore
          substructures (166-bit binary fingerprint). Compensates for the
          known blind spots of GNN embeddings on small reactive groups
          such as aldehydes, heavy metals, or strained rings.
          Weight: 0.5
    """

    WEIGHT_EMBEDDING = 0.5
    WEIGHT_MACCS     = 0.5

    def __init__(self, embedding_dim: int = 64, use_faiss: bool = True):
        """
        Args:
            embedding_dim: GNN embedding 的维度
            use_faiss:     是否使用 FAISS 加速（建议保持 True）
        """
        self.embedding_dim = embedding_dim
        self.use_faiss = use_faiss

        self.source_embeddings: List[np.ndarray] = []
        self.source_molecules: List[Dict] = []   # 每个 dict 必须含 'smiles' 和 'source_domain'
        self.source_labels: List[Any] = []

        self.faiss_index = None

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def add_source_molecules(self, embeddings: List[np.ndarray],
                             molecules: List[Dict], labels: List[Any]):
        """
        将源域分子加入检索库。

        Args:
            embeddings: L2 归一化后的 GNN 嵌入向量列表
            molecules:  分子元数据字典列表（必须含 'smiles'、'source_domain'）
            labels:     每个分子的原始标签（保留源域语义，不做重命名）
        """
        self.source_embeddings.extend(embeddings)
        self.source_molecules.extend(molecules)
        self.source_labels.extend(labels)

        if self.use_faiss and self.source_embeddings:
            self._build_faiss_index()

    def _build_faiss_index(self):
        """构建 FAISS IndexFlatIP 内积索引（对 L2 归一化向量等价于余弦相似度）。"""
        if not self.source_embeddings:
            return
        arr = np.array(self.source_embeddings, dtype=np.float32)
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        self.faiss_index.add(arr)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve_similar_molecules(self,
                                   target_embedding: np.ndarray,
                                   target_smiles: str,
                                   k: int = 5,
                                   balance_labels: bool = True) -> List[Dict]:
        """
        检索 top-k 相似源域分子，融合两条通路的得分。

        Args:
            target_embedding: 目标分子的 L2 归一化 GNN 嵌入
            target_smiles:    目标分子的 SMILES（用于计算 MACCS 指纹）
            k:                返回的分子数量
            balance_labels:   若 True，从 top-2k 候选中强制平衡正负例数量。
                              解决源域正类不平衡导致 LLM 参考全为正例的问题。

        Returns:
            分子字典列表，每个字典额外包含：
              retrieval_score  — 加权融合得分
              embedding_sim    — 通路 A 得分
              maccs_tanimoto   — 通路 B 得分
              label            — 源域标签（保留原始语义）
        """
        if not self.source_embeddings:
            return []

        n = len(self.source_embeddings)

        # ----------------------------------------------------------
        # Pathway A: GNN Embedding — FAISS IndexFlatIP
        # 注意：IndexFlatIP 直接返回内积值（越大越相似），无需 1-distance
        # ----------------------------------------------------------
        embedding_scores = np.zeros(n, dtype=np.float32)

        if self.use_faiss and self.faiss_index is not None:
            qvec = target_embedding.reshape(1, -1).astype(np.float32)
            inner_products, indices = self.faiss_index.search(qvec, n)
            for ip, idx in zip(inner_products[0], indices[0]):
                embedding_scores[idx] = float(ip)
        else:
            qvec = target_embedding.reshape(1, -1)
            src  = np.array(self.source_embeddings)
            embedding_scores = cosine_similarity(qvec, src)[0].astype(np.float32)

        # ----------------------------------------------------------
        # Pathway B: MACCS Keys Tanimoto
        # 预先计算目标指纹，再批量与源域比较
        # ----------------------------------------------------------
        maccs_scores = np.zeros(n, dtype=np.float32)
        target_mol = Chem.MolFromSmiles(target_smiles)
        if target_mol is not None:
            target_fp = MACCSkeys.GenMACCSKeys(target_mol)
            for i, mol_data in enumerate(self.source_molecules):
                src_mol = Chem.MolFromSmiles(mol_data.get('smiles', ''))
                if src_mol is not None:
                    src_fp = MACCSkeys.GenMACCSKeys(src_mol)
                    maccs_scores[i] = DataStructs.TanimotoSimilarity(target_fp, src_fp)
        # 若目标 SMILES 无效，通路 B 得分保持 0，只用通路 A 结果

        # ----------------------------------------------------------
        # 加权融合 → 按得分排序 → 平衡正负例（可选）
        # ----------------------------------------------------------
        combined = (self.WEIGHT_EMBEDDING * embedding_scores +
                    self.WEIGHT_MACCS     * maccs_scores)

        if balance_labels:
            # 从 top-2k 候选中各取最多 k//2 个正例和负例，
            # 保证 LLM 能同时看到两类参考，避免全正例导致的预测偏差
            half = max(1, k // 2)
            candidate_indices = np.argsort(combined)[::-1][:k * 2]

            pos_pool = [i for i in candidate_indices if self.source_labels[i] in (1, '1', True)]
            neg_pool = [i for i in candidate_indices if self.source_labels[i] in (0, '0', False)]

            selected = pos_pool[:half] + neg_pool[:half]
            # 若一侧不足，用另一侧补齐到 k
            if len(selected) < k:
                remaining = [i for i in candidate_indices if i not in selected]
                selected += remaining[:k - len(selected)]
            top_indices = selected[:k]
        else:
            top_indices = np.argsort(combined)[::-1][:k]

        retrieved = []
        for idx in top_indices:
            mol_data = self.source_molecules[idx].copy()
            mol_data['retrieval_score'] = float(combined[idx])
            mol_data['embedding_sim']   = float(embedding_scores[idx])
            mol_data['maccs_tanimoto']  = float(maccs_scores[idx])
            # 将源域标签附在 'label' 键下，供 Prompt 使用（语义由 DOMAIN_SEMANTICS 解释）
            mol_data['label'] = self.source_labels[idx]
            retrieved.append(mol_data)

        return retrieved

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_retriever(self, filepath: str):
        """持久化检索器状态。"""
        state = {
            'source_embeddings': self.source_embeddings,
            'source_molecules':  self.source_molecules,
            'source_labels':     self.source_labels,
            'embedding_dim':     self.embedding_dim,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    def load_retriever(self, filepath: str):
        """从磁盘恢复检索器状态并重建 FAISS 索引。"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        self.source_embeddings = state['source_embeddings']
        self.source_molecules  = state['source_molecules']
        self.source_labels     = state['source_labels']
        self.embedding_dim     = state['embedding_dim']
        if self.use_faiss and self.source_embeddings:
            self._build_faiss_index()


# ---------------------------------------------------------------------------
# 冒烟测试
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    src_mols = [
        {'smiles': 'CCO',       'name': 'Ethanol',          'source_domain': 'bbbp'},
        {'smiles': 'CCCO',      'name': 'Propanol',         'source_domain': 'bbbp'},
        {'smiles': 'C=O',       'name': 'Formaldehyde',     'source_domain': 'bbbp'},
        {'smiles': 'ClCCl',     'name': 'Dichloromethane',  'source_domain': 'bbbp'},
        {'smiles': 'c1ccccc1',  'name': 'Benzene',          'source_domain': 'bbbp'},
    ]
    src_embs = [rng.standard_normal(64).astype(np.float32) for _ in src_mols]
    src_embs = [e / np.linalg.norm(e) for e in src_embs]
    src_labels = [1, 1, 0, 0, 1]

    retriever = MolecularRetriever(embedding_dim=64)
    retriever.add_source_molecules(src_embs, src_mols, src_labels)

    tgt_emb = rng.standard_normal(64).astype(np.float32)
    tgt_emb /= np.linalg.norm(tgt_emb)

    results = retriever.retrieve_similar_molecules(tgt_emb, 'CC(=O)O', k=3)
    print(f"Retrieved {len(results)} molecules:")
    for m in results:
        print(f"  {m['name']:20s}  combined={m['retrieval_score']:.3f}  "
              f"embed={m['embedding_sim']:.3f}  maccs={m['maccs_tanimoto']:.3f}  "
              f"label={m['label']}  domain={m['source_domain']}")