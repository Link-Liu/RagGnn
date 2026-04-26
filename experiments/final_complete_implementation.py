#!/usr/bin/env python3
"""
Complete REAL implementation for Domain-Aware RAG-MDA.
NO simulation, NO synthetic fallback, NO fake embeddings.

Pipeline:
  1. Load real CSV datasets (Tox21, SIDER, BBBP, etc.)
  2. Convert SMILES to molecular graphs via RDKit
  3. Train GNN (GIN) on source domain (BBBP) with supervised binary classification
  4. Use trained GNN encoder to produce chemically meaningful embeddings
  5. Build a RAG retrieval index from source-domain embeddings
  6. Retrieve similar molecules for each target molecule (GNN embed + MACCS Tanimoto)
  7. Construct Cross-Domain CoT prompts and call the real ARK LLM API
  8. Evaluate predictions against ground truth
"""
import logging
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import sys
import io
import os

# 将项目根目录加入 sys.path（脚本在 experiments/ 子目录下，需要找到 models/ 等兄弟包）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from torch_geometric.loader import DataLoader
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import http.client
import warnings
warnings.filterwarnings('ignore')

# Local modules
from models.gnn_encoder import GNNEncoder
from retrieval.domain_aware_retriever import MolecularRetriever
from prompting.prompt_template import create_detailed_prompt, create_no_rag_prompt
from dataset.mol_graph_utils import (
    smiles_to_graph,
    get_num_atom_features,
    get_num_bond_features,
    batch_smiles_to_graphs,
)


# ---------------------------------------------------------------------------
# ChatAnywhere LLM Interface  (http.client, no third-party dependencies)
# ---------------------------------------------------------------------------

class ARKLLMInterface:
    """
    Interface to ChatAnywhere GPT API via http.client.
    Makes real HTTPS calls - raises RuntimeError on failure.
    """

    _HOST = "api.chatanywhere.tech"
    _PATH = "/v1/chat/completions"

    def __init__(self,
                 api_key: str  = "sk-KXMs07t8WWNkJfzALZvhBB8PED0AtfjdYGUPTKNBvtJ9P8pP",
                 base_url: str = "",   # 保留签名兼容性，此后端不使用
                 model: str    = "gpt-4.1",
                 timeout: int  = 120):
        self.call_count = 0
        self.api_key = api_key
        self.model   = model
        self.timeout = timeout

    def _post(self, payload: dict, timeout: int) -> dict:
        """内部：发送 HTTPS POST，返回解析后的 JSON dict。"""
        body = json.dumps(payload)
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        conn = http.client.HTTPSConnection(self._HOST, timeout=timeout)
        try:
            conn.request("POST", self._PATH, body, headers)
            res  = conn.getresponse()
            data = res.read().decode("utf-8")
            if res.status != 200:
                raise RuntimeError(
                    f"HTTP {res.status} {res.reason}: {data[:300]}"
                )
            return json.loads(data)
        finally:
            conn.close()

    def predict(self, prompt: str) -> Dict:
        """Make a real API call and return parsed prediction."""
        self.call_count += 1
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",   "content": prompt},
            ],
            "max_tokens": 1000,   # 限制输出长度，减少超时风险
        }
        try:
            result  = self._post(payload, self.timeout)
            content = result['choices'][0]['message']['content'].strip()

            # -------------------------------------------------------
            # 精确解析：只识别 Prompt 约定的固定格式 "Answer: 0/1"
            #
            # Prompt 要求 LLM 最后一行写 "Answer: 0" 或 "Answer: 1"，
            # 解析端只读这个锚点，完全不猜测自然语言内容。
            # 找不到时抛出 ValueError，由上层 except 计入 failed，
            # 而不引入任何静默偏差（不存在"保守预测 0"之类的兜底）。
            # -------------------------------------------------------
            import re
            match = re.search(r'Answer\s*:\s*([01])', content, re.IGNORECASE)
            if match is None:
                raise ValueError(
                    f"LLM response did not contain 'Answer: 0/1'.\n"
                    f"Response was: {content[:300]}"
                )
            prediction = int(match.group(1))

            return {
                'prediction': prediction,
                'confidence': 1.0,
                'response': content,
                'tokens_used': result.get('usage', {}).get('total_tokens', 0),
                'api_call_id': f'call_{self.call_count}',
            }
        except Exception as e:
            raise RuntimeError(f"ChatAnywhere API call failed: {e}")

    def check_status(self) -> Dict:
        """Quick connectivity check."""
        try:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user",   "content": "ping"},
                ],
            }
            result = self._post(payload, timeout=15)
            return {'available': True, 'calls': self.call_count}
        except Exception:
            return {'available': False, 'calls': self.call_count}


# ---------------------------------------------------------------------------
# GNN Embedding Engine  (real graph neural network)
# ---------------------------------------------------------------------------

class GNNEmbeddingEngine:
    """
    Wraps GNNEncoder to convert SMILES -> GNN embedding in one call.

    训练流程:
      train_on_source() — 在源域（BBBP）上用二分类任务有监督地训练 GNNEncoder。
                          训练完成后丢弃分类头，encoder 权重被保留，用于后续嵌入提取。
    使用经过训练的 encoder 产生的嵌入具有真实化学语义，
    而非随机初始化的无意义向量。
    """

    def __init__(self, hidden_dim: int = 64, num_layers: int = 3, device: str = 'cpu'):
        self.device = torch.device(device)
        self.hidden_dim = hidden_dim
        num_node_feat = get_num_atom_features()
        num_edge_feat = get_num_bond_features()

        self.encoder = GNNEncoder(
            num_node_features=num_node_feat,
            num_edge_features=num_edge_feat,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        ).to(self.device)
        self.encoder.eval()
        print(f"[GNN] Initialized GIN encoder: node_feat={num_node_feat}, "
              f"edge_feat={num_edge_feat}, hidden={hidden_dim}, layers={num_layers}")

    # ------------------------------------------------------------------
    # 有监督预训练
    # ------------------------------------------------------------------

    def train_on_source(self,
                        smiles_list: List[str],
                        labels: List[int],
                        epochs: int = 30,
                        batch_size: int = 32,
                        lr: float = 1e-3,
                        checkpoint_path: Optional[str] = None) -> None:
        """
        在源域上有监督地训练 GNN encoder（二分类任务）。

        训练步骤:
          1. 将所有 SMILES 转为 PyG Data 图对象
          2. 在 encoder 顶部接一个线性分类头
          3. 用 BCEWithLogitsLoss 训练整体
          4. 训练完成后移除分类头，只保留 encoder 权重
          5. 若指定 checkpoint_path，则保存 encoder 权重

        Args:
            smiles_list:      训练分子的 SMILES 列表
            labels:           对应的二值标签列表（0 / 1）
            epochs:           训练轮数
            batch_size:       mini-batch 大小
            lr:               Adam 学习率
            checkpoint_path:  权重保存路径（.pt 文件），None 表示不保存
        """
        print(f"[GNN-Train] Converting {len(smiles_list)} SMILES to graphs ...")
        graphs = batch_smiles_to_graphs(smiles_list, labels)
        if len(graphs) == 0:
            raise RuntimeError("[GNN-Train] No valid graphs constructed. Check SMILES.")
        print(f"[GNN-Train] Valid graphs: {len(graphs)} / {len(smiles_list)}")

        # ---- 分类头（训练期使用，事后丢弃）----
        classifier = nn.Linear(self.hidden_dim, 1).to(self.device)
        optimizer  = torch.optim.Adam(
            list(self.encoder.parameters()) + list(classifier.parameters()),
            lr=lr, weight_decay=1e-5
        )
        criterion = nn.BCEWithLogitsLoss()
        loader    = DataLoader(graphs, batch_size=batch_size, shuffle=True)

        self.encoder.train()
        best_loss = float('inf')
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for batch in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                emb  = self.encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                logit = classifier(emb).squeeze(-1)
                loss  = criterion(logit, batch.y.float())
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch.num_graphs

            avg_loss = total_loss / len(graphs)
            if avg_loss < best_loss:
                best_loss = avg_loss
            if epoch % 5 == 0 or epoch == 1:
                print(f"  [GNN-Train] Epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}  "
                      f"best={best_loss:.4f}")

        # 训练完成：切回推断模式，丢弃分类头
        self.encoder.eval()
        del classifier
        print(f"[GNN-Train] Training complete. Best loss={best_loss:.4f}")

        # 可选：保存 encoder 权重
        if checkpoint_path:
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.encoder.state_dict(), checkpoint_path)
            print(f"[GNN-Train] Encoder weights saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """
        尝试从磁盘加载预训练权重，成功返回 True，文件不存在返回 False。
        """
        p = Path(checkpoint_path)
        if not p.exists():
            return False
        self.encoder.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device)
        )
        self.encoder.eval()
        print(f"[GNN] Loaded pre-trained weights from {checkpoint_path}")
        return True

    # ------------------------------------------------------------------
    # 推断（嵌入提取）
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_smiles(self, smiles: str) -> Optional[np.ndarray]:
        """Convert a single SMILES to an L2-normalised embedding vector (numpy)."""
        graph = smiles_to_graph(smiles)
        if graph is None:
            return None
        graph = graph.to(self.device)
        emb = self.encoder(graph.x, graph.edge_index, graph.edge_attr)
        emb = emb.cpu().numpy().flatten()
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

    @torch.no_grad()
    def encode_batch(self, smiles_list: List[str]) -> Tuple[List[np.ndarray], List[int]]:
        """
        Encode a batch of SMILES. Returns (embeddings, valid_indices).
        Invalid SMILES are silently skipped.
        """
        embeddings = []
        valid_idx  = []
        for i, smi in enumerate(smiles_list):
            emb = self.encode_smiles(smi)
            if emb is not None:
                embeddings.append(emb)
                valid_idx.append(i)
        return embeddings, valid_idx


# ---------------------------------------------------------------------------
# Main Experiment Class  (real data only)
# ---------------------------------------------------------------------------

class RealExperiment:
    """
    End-to-end experiment using ONLY real datasets and real API calls.
    Raises errors if data files are missing - NO synthetic fallback.
    """

    def __init__(self, data_dir: str = "data", hidden_dim: int = 64,
                 gnn_epochs: int = 30, gnn_batch_size: int = 32,
                 gnn_checkpoint: str = "checkpoints/gnn_encoder.pt"):
        self.data_dir = Path(data_dir)
        self.device   = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[Init] device={self.device}")

        # 1. GNN encoder
        self.gnn = GNNEmbeddingEngine(hidden_dim=hidden_dim, device=self.device)

        # 2. 在源域（BBBP）上训练 GNN（若已有 checkpoint 则跳过训练）
        self._train_gnn(gnn_epochs, gnn_batch_size, gnn_checkpoint)

        # 3. LLM interface  (ChatAnywhere / gpt-4.1)
        self.llm = ARKLLMInterface(
            api_key=os.getenv('CHATANYWHERE_API_KEY',
                              'sk-KXMs07t8WWNkJfzALZvhBB8PED0AtfjdYGUPTKNBvtJ9P8pP'),
            model=os.getenv('CHATANYWHERE_MODEL', 'gpt-4.1'),
        )

        # 4. RAG retriever (will be populated with source-domain embeddings)
        self.retriever = MolecularRetriever(embedding_dim=hidden_dim)

        # 5. Build source domain index（使用训练后的 GNN）
        self._build_source_index()

    # ------------------------------------------------------------------
    # GNN 预训练
    # ------------------------------------------------------------------

    def _train_gnn(self, epochs: int, batch_size: int, checkpoint: str) -> None:
        """
        在 BBBP 数据集上有监督地预训练 GNN encoder。

        逻辑:
          - 若 checkpoint 文件已存在，直接加载，跳过训练（节省时间）
          - 否则从 bbbp.csv 中读取全部数据，调用 gnn.train_on_source() 训练
          - 训练完成后将权重写入 checkpoint
        """
        # 优先加载已有权重
        if self.gnn.load_checkpoint(checkpoint):
            return

        bbbp_path = self.data_dir / 'bbbp.csv'
        if not bbbp_path.exists():
            raise FileNotFoundError(
                f"[GNN-Train] Source training file not found: {bbbp_path}\n"
                "Please ensure bbbp.csv is present in the data directory."
            )

        df = pd.read_csv(bbbp_path)
        # 自动识别 SMILES 列和标签列
        smiles_col = next((c for c in df.columns if 'smiles' in c.lower()), None)
        label_col  = next((c for c in df.columns
                           if c.lower() in ('label', 'p_np', 'y', 'target')), None)
        if smiles_col is None or label_col is None:
            raise ValueError(
                f"[GNN-Train] Cannot find smiles/label columns in bbbp.csv. "
                f"Found: {list(df.columns)}"
            )

        df = df[[smiles_col, label_col]].dropna()
        smiles_list = df[smiles_col].tolist()
        labels      = df[label_col].astype(float).apply(lambda v: int(v >= 0.5)).tolist()

        print(f"[GNN-Train] Starting supervised training on BBBP "
              f"({len(smiles_list)} molecules, {epochs} epochs) ...")
        self.gnn.train_on_source(
            smiles_list=smiles_list,
            labels=labels,
            epochs=epochs,
            batch_size=batch_size,
            checkpoint_path=checkpoint,
        )

    # ------------------------------------------------------------------
    # Source domain index
    # ------------------------------------------------------------------

    def _build_source_index(self):
        """Load source_molecules.csv and build the retrieval index."""
        source_path = self.data_dir / "source_molecules.csv"
        if not source_path.exists():
            raise FileNotFoundError(
                f"Source domain file not found: {source_path}\n"
                "Run `python -m molgnn.datasets.download_datasets` first."
            )

        df = pd.read_csv(source_path)
        print(f"[Source] Loading {len(df)} source molecules ...")
        smiles_list = df['smiles'].tolist()
        labels = df['label'].astype(int).tolist()

        # Encode with real GNN
        embeddings, valid_idx = self.gnn.encode_batch(smiles_list)
        molecules = []
        valid_labels = []
        for i in valid_idx:
            row = df.iloc[i]
            molecules.append({
                'smiles': str(row['smiles']),
                'name': str(row.get('name', f'source_{i}')),
                # source_domain 决定了 Prompt 中标签的语义（如 bbbp 的 1 = 穿透血脑屏障）
                # 若 CSV 有 source_domain 列则取之，否则默认 bbbp（当前配置的源域）
                'source_domain': str(row.get('source_domain', 'bbbp')),
                'description': str(row.get('description', '')),
            })
            valid_labels.append(labels[i])

        self.retriever.add_source_molecules(embeddings, molecules, valid_labels)
        print(f"[Source] Indexed {len(embeddings)} molecules (skipped "
              f"{len(smiles_list) - len(embeddings)} invalid SMILES)")

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def _load_dataset(self, name: str, sample_size: Optional[int] = None
                      ) -> Tuple[pd.DataFrame, str]:
        """
        Load a CSV dataset. Raises FileNotFoundError if missing.
        Returns (dataframe, csv_path).
        """
        csv_path = self.data_dir / f"{name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {csv_path}\n"
                "Run `python -m molgnn.datasets.download_datasets` first."
            )

        df = pd.read_csv(csv_path)

        # Identify columns
        smiles_col = next((c for c in df.columns if 'smiles' in c.lower()), None)
        label_col = next((c for c in df.columns
                          if c.lower() in ('label', 'toxicity', 'target', 'y')), None)
        if smiles_col is None or label_col is None:
            raise ValueError(f"CSV must contain 'smiles' and 'label' columns. "
                             f"Found: {list(df.columns)}")

        df = df[[smiles_col, label_col]].dropna().rename(
            columns={smiles_col: 'smiles', label_col: 'label'})
        df['label'] = df['label'].astype(float).apply(lambda v: int(v >= 0.5))

        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

        print(f"[Data] {name}: {len(df)} molecules loaded from {csv_path}")
        return df, str(csv_path)

    # ------------------------------------------------------------------
    # Single dataset experiment
    # ------------------------------------------------------------------

    def run_dataset(self, dataset_name: str, sample_size: int = 50) -> Dict:
        """
        Run the full RAG-MDA pipeline on one dataset.
        Returns a results dict with predictions and metrics.
        """
        print(f"\n{'='*60}")
        print(f"  Experiment: {dataset_name}  (sample_size={sample_size})")
        print(f"{'='*60}")

        df, csv_path = self._load_dataset(dataset_name, sample_size)

        # Encode target molecules with real GNN
        smiles_list = df['smiles'].tolist()
        labels = df['label'].tolist()
        embeddings, valid_idx = self.gnn.encode_batch(smiles_list)
        print(f"[GNN] Encoded {len(embeddings)}/{len(smiles_list)} molecules")

        # Run prediction pipeline
        predictions = []
        true_labels = []
        details = []
        failed = 0

        for pos, idx in enumerate(valid_idx):
            smi = smiles_list[idx]
            true_label = labels[idx]
            emb = embeddings[pos]
            mol_name = f"{dataset_name}_mol_{idx}"

            print(f"  [{pos+1}/{len(valid_idx)}] {smi[:50]}...", end="")

            try:
                # RAG retrieval (通路A: GNN embedding + 通路B: MACCS Tanimoto)
                retrieved = self.retriever.retrieve_similar_molecules(
                    emb, target_smiles=smi, k=5)

                # Build prompt —— 传入 target_dataset 以正确解析标签语义空间
                target_mol = {
                    'smiles': smi,
                    'name': mol_name,
                    'dataset': dataset_name,
                }
                # 按数据集查 property_description，避免将 SIDER 的副作用误称为 "toxicity"
                _PROP_DESC = {
                    'tox21':   'AhR activation (NR-AhR assay)',
                    'sider':   'hepatobiliary side effects',
                    'clintox': 'clinical toxicity',
                    'bbbp':    'blood-brain barrier penetration',
                }
                prop_desc = _PROP_DESC.get(dataset_name.lower(), dataset_name)

                prompt = create_detailed_prompt(
                    target_molecule=target_mol,
                    retrieved_examples=retrieved,
                    property_description=prop_desc,
                    target_dataset=dataset_name,
                )

                # Call real LLM
                llm_result = self.llm.predict(prompt)
                pred = llm_result['prediction']
                predictions.append(pred)
                true_labels.append(true_label)

                details.append({
                    'smiles': smi,
                    'true_label': true_label,
                    'prediction': pred,
                    'confidence': llm_result['confidence'],
                    'retrieved_count': len(retrieved),
                    'llm_response': llm_result['response'][:200],
                })

                marker = 'OK' if pred == true_label else 'WRONG'
                # 打印 LLM 原始响应摘要，便于诊断解析是否正确
                resp_preview = llm_result['response'][:80].replace('\n', ' ')
                print(f"  pred={pred} truth={true_label} [{marker}]  llm='{resp_preview}'")

            except Exception as e:
                failed += 1
                print(f"  ERROR: {e}")
                continue

        # Compute metrics
        metrics = {}
        if len(predictions) > 0:
            metrics['accuracy'] = float(accuracy_score(true_labels, predictions))
            metrics['f1'] = float(f1_score(true_labels, predictions, zero_division=0))
            metrics['precision'] = float(precision_score(true_labels, predictions, zero_division=0))
            metrics['recall'] = float(recall_score(true_labels, predictions, zero_division=0))
            if len(set(true_labels)) > 1:
                metrics['auc'] = float(roc_auc_score(true_labels, predictions))
            else:
                metrics['auc'] = float('nan')

            correct = sum(1 for p, t in zip(predictions, true_labels) if p == t)
            print(f"\n[Results] {dataset_name}:")
            print(f"  Accuracy : {metrics['accuracy']:.4f}  ({correct}/{len(predictions)})")
            print(f"  F1       : {metrics['f1']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall   : {metrics['recall']:.4f}")
            print(f"  AUC      : {metrics['auc']:.4f}")
            print(f"  Failed   : {failed}")

        return {
            'dataset': dataset_name,
            'total_loaded': len(df),
            'total_encoded': len(embeddings),
            'total_predicted': len(predictions),
            'failed': failed,
            'metrics': metrics,
            'details': details,
            'llm_calls': self.llm.call_count,
        }

    # ------------------------------------------------------------------
    # Multi-dataset experiment suite
    # ------------------------------------------------------------------

    def run_experiment_suite(self, datasets: List[str] = None,
                             sample_size: int = 50) -> Dict:
        """Run experiments on multiple datasets."""
        if datasets is None:
            # Note: BBBP is used as source domain, so it is excluded from test datasets
            datasets = ['tox21', 'sider']

        all_results = {}
        for ds in datasets:
            try:
                result = self.run_dataset(ds, sample_size)
                all_results[ds] = result
            except Exception as e:
                print(f"[ERROR] {ds}: {e}")
                all_results[ds] = {'dataset': ds, 'error': str(e), 'status': 'FAILED'}

        return all_results

    # ------------------------------------------------------------------
    # 消融实验：无 RAG 对照组
    # ------------------------------------------------------------------

    def run_dataset_no_rag(self, dataset_name: str, sample_size: int = 50) -> Dict:
        """
        消融实验：用与完整版结构对齐的 Prompt，但不提供 RAG 检索例子。
        其他流程（GNN 编码、数据加载、指标计算）与 run_dataset() 完全相同，
        以确保两组实验的唯一变量是"有无检索参考"。
        """
        print(f"\n{'='*60}")
        print(f"  [ABLATION / No-RAG] {dataset_name}  (sample_size={sample_size})")
        print(f"{'='*60}")

        df, csv_path = self._load_dataset(dataset_name, sample_size)

        smiles_list = df['smiles'].tolist()
        labels      = df['label'].tolist()
        # GNN 编码仍保留（保证与完整版用相同分子子集），但不做 RAG 检索
        embeddings, valid_idx = self.gnn.encode_batch(smiles_list)
        print(f"[GNN] Encoded {len(embeddings)}/{len(smiles_list)} molecules  "
              f"(embeddings computed but not used for retrieval)")

        _PROP_DESC = {
            'tox21':   'AhR activation (NR-AhR assay)',
            'sider':   'hepatobiliary side effects',
            'clintox': 'clinical toxicity',
            'bbbp':    'blood-brain barrier penetration',
        }
        prop_desc = _PROP_DESC.get(dataset_name.lower(), dataset_name)

        predictions = []
        true_labels = []
        details     = []
        failed      = 0

        for pos, idx in enumerate(valid_idx):
            smi        = smiles_list[idx]
            true_label = labels[idx]
            mol_name   = f"{dataset_name}_mol_{idx}"

            print(f"  [{pos+1}/{len(valid_idx)}] {smi[:50]}...", end="")

            try:
                target_mol = {'smiles': smi, 'name': mol_name, 'dataset': dataset_name}
                prompt = create_no_rag_prompt(
                    target_molecule=target_mol,
                    property_description=prop_desc,
                    target_dataset=dataset_name,
                )

                llm_result = self.llm.predict(prompt)
                pred       = llm_result['prediction']
                predictions.append(pred)
                true_labels.append(true_label)

                details.append({
                    'smiles':        smi,
                    'true_label':    true_label,
                    'prediction':    pred,
                    'confidence':    llm_result['confidence'],
                    'retrieved_count': 0,           # 无检索
                    'llm_response':  llm_result['response'][:200],
                })

                marker = 'OK' if pred == true_label else 'WRONG'
                resp_preview = llm_result['response'][:80].replace('\n', ' ')
                print(f"  pred={pred} truth={true_label} [{marker}]  llm='{resp_preview}'")

            except Exception as e:
                failed += 1
                print(f"  ERROR: {e}")
                continue

        metrics = {}
        if len(predictions) > 0:
            metrics['accuracy']  = float(accuracy_score(true_labels, predictions))
            metrics['f1']        = float(f1_score(true_labels, predictions, zero_division=0))
            metrics['precision'] = float(precision_score(true_labels, predictions, zero_division=0))
            metrics['recall']    = float(recall_score(true_labels, predictions, zero_division=0))
            if len(set(true_labels)) > 1:
                metrics['auc'] = float(roc_auc_score(true_labels, predictions))
            else:
                metrics['auc'] = float('nan')

            correct = sum(1 for p, t in zip(predictions, true_labels) if p == t)
            print(f"\n[No-RAG Results] {dataset_name}:")
            print(f"  Accuracy : {metrics['accuracy']:.4f}  ({correct}/{len(predictions)})")
            print(f"  F1       : {metrics['f1']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall   : {metrics['recall']:.4f}")
            print(f"  AUC      : {metrics['auc']:.4f}")
            print(f"  Failed   : {failed}")

        return {
            'dataset':          dataset_name,
            'condition':        'no_rag',
            'total_loaded':     len(df),
            'total_encoded':    len(embeddings),
            'total_predicted':  len(predictions),
            'failed':           failed,
            'metrics':          metrics,
            'details':          details,
            'llm_calls':        self.llm.call_count,
        }

    def run_ablation_suite(self, datasets: List[str] = None,
                           sample_size: int = 50) -> Dict:
        """
        运行消融实验套件：对每个数据集同时运行 Full-RAG 和 No-RAG 两组。
        返回 {'tox21': {'full_rag': {...}, 'no_rag': {...}}, ...}
        """
        if datasets is None:
            datasets = ['tox21', 'sider']

        ablation_results = {}
        for ds in datasets:
            ablation_results[ds] = {}
            # --- Full RAG ---
            try:
                ablation_results[ds]['full_rag'] = self.run_dataset(ds, sample_size)
            except Exception as e:
                print(f"[ERROR] Full-RAG {ds}: {e}")
                ablation_results[ds]['full_rag'] = {'error': str(e), 'status': 'FAILED'}
            # --- No RAG ---
            try:
                ablation_results[ds]['no_rag'] = self.run_dataset_no_rag(ds, sample_size)
            except Exception as e:
                print(f"[ERROR] No-RAG  {ds}: {e}")
                ablation_results[ds]['no_rag'] = {'error': str(e), 'status': 'FAILED'}

        return ablation_results

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------

    def save_results(self, results: Dict, output_dir: str = "real_experiment_results"):
        """Save experiment results to JSON files."""
        Path(output_dir).mkdir(exist_ok=True)

        # Detailed results
        with open(f"{output_dir}/detailed_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str, ensure_ascii=False)

        # Summary
        summary = {
            'datasets': [],
            'llm_total_calls': self.llm.call_count,
        }
        for ds_name, res in results.items():
            if 'metrics' in res:
                summary['datasets'].append({
                    'name': ds_name,
                    **res['metrics'],
                    'samples': res['total_predicted'],
                })
        with open(f"{output_dir}/summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n[Save] Results saved to {output_dir}/")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def print_report(self, results: Dict):
        """Print a formatted summary report."""
        print("\n" + "=" * 60)
        print("  EXPERIMENT REPORT  (Real Data + Real API)")
        print("=" * 60)
        for ds_name, res in results.items():
            if 'metrics' in res:
                m = res['metrics']
                print(f"\n  {ds_name}:")
                print(f"    Accuracy  = {m['accuracy']:.4f}")
                print(f"    F1 Score  = {m['f1']:.4f}")
                print(f"    AUC       = {m['auc']:.4f}")
                print(f"    Samples   = {res['total_predicted']}")
            else:
                print(f"\n  {ds_name}: FAILED - {res.get('error', 'unknown')}")
        print(f"\n  Total LLM API calls: {self.llm.call_count}")
        print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def print_ablation_comparison(ablation_results: Dict):
    """打印 Full-RAG vs No-RAG 对比表。"""
    print("\n" + "=" * 70)
    print("  ABLATION STUDY: Full-RAG  vs  No-RAG  (Zero-shot)")
    print("=" * 70)
    header = f"{'Dataset':<12} {'Condition':<12} {'Acc':>6} {'F1':>6} {'AUC':>6} {'#Pred':>6}"
    print(header)
    print("-" * 70)
    for ds, conditions in ablation_results.items():
        for cond_key, cond_label in [('full_rag', 'Full-RAG'), ('no_rag', 'No-RAG')]:
            res = conditions.get(cond_key, {})
            m   = res.get('metrics', {})
            if m:
                print(f"{ds:<12} {cond_label:<12} "
                      f"{m.get('accuracy', float('nan')):>6.4f} "
                      f"{m.get('f1', float('nan')):>6.4f} "
                      f"{m.get('auc', float('nan')):>6.4f} "
                      f"{res.get('total_predicted', 0):>6}")
            else:
                print(f"{ds:<12} {cond_label:<12}  FAILED — {res.get('error', '?')[:40]}")
        # Delta 行
        rag_m    = conditions.get('full_rag', {}).get('metrics', {})
        no_rag_m = conditions.get('no_rag',   {}).get('metrics', {})
        if rag_m and no_rag_m:
            d_acc = rag_m.get('accuracy', 0) - no_rag_m.get('accuracy', 0)
            d_f1  = rag_m.get('f1', 0)       - no_rag_m.get('f1', 0)
            d_auc = rag_m.get('auc', float('nan')) - no_rag_m.get('auc', float('nan'))
            print(f"{'':12} {'Δ (RAG gain)':<12} "
                  f"{d_acc:>+6.4f} {d_f1:>+6.4f} {d_auc:>+6.4f}")
        print("-" * 70)
    print("=" * 70)


def main():
    print("=" * 60)
    print("  Domain-Aware RAG-MDA  -  REAL DATA EXPERIMENT + ABLATION")
    print("=" * 60)

    experiment = RealExperiment(
        data_dir="data",
        hidden_dim=64,
        gnn_epochs=30,          # 首次运行会训练；之后从 checkpoint 加载，此参数被忽略
        gnn_batch_size=32,
        gnn_checkpoint="checkpoints/gnn_encoder.pt",
    )

    # Check LLM connectivity
    status = experiment.llm.check_status()
    print(f"[LLM] API status: {status}")
    if not status['available']:
        print("[WARNING] LLM API not reachable. Predictions will fail.")

    # 消融实验套件：Full-RAG + No-RAG 同时运行，共用相同的数据子集和 GNN
    ablation_results = experiment.run_ablation_suite(
        datasets=['tox21', 'sider'],
        sample_size=50,
    )

    # 保存完整结果（分 full_rag / no_rag 两个子目录）
    Path("real_experiment_results").mkdir(exist_ok=True)
    for ds, conditions in ablation_results.items():
        for cond_key, res in conditions.items():
            out_path = f"real_experiment_results/{ds}_{cond_key}.json"
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(res, f, indent=2, default=str, ensure_ascii=False)
    print("\n[Save] Ablation results saved to real_experiment_results/")

    # 打印对比表
    print_ablation_comparison(ablation_results)

    print("\nExperiment completed.")


if __name__ == "__main__":
    main()