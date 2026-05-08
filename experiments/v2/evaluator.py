"""
V2 评估器：
  - Logits AUC: 比较 Enzyme vs Non-enzyme 首 token logits
  - Exact Match: LLM generate → 正则匹配
  - 完整评估报告
"""
import torch, numpy as np, json
from typing import List, Dict
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score
)


class Evaluator:
    """在目标域上评估模型。"""

    def __init__(self, model, device: str = "cuda"):
        self.model = model
        self.device = device

    def evaluate(
        self,
        tgt_list: List,
        target_name: str,
        batch_size: int = 4,
        method: str = "logits",
    ) -> Dict:
        """
        评估目标域。
        method: 'logits' 或 'generate'
        """
        from torch_geometric.data import Batch as PyGBatch
        from experiments.v2.prompts import (
            SYSTEM_PROMPT, get_user_prompt, format_chat_prompt, extract_graph_stats
        )

        self.model.gnn.eval()
        self.model.connector.eval()
        self.model.projector.eval()

        n = len(tgt_list)
        predictions, true_labels, prob_scores = [], [], []
        failed = 0

        print(f"[Eval] {target_name}: {n} graphs, method={method}")

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_data = tgt_list[start:end]

            try:
                pyg_batch = PyGBatch.from_data_list(batch_data).to(self.device)
                prompts = []
                for data in batch_data:
                    stats = extract_graph_stats(data)
                    user_p = get_user_prompt(stats)
                    prompt = format_chat_prompt(
                        self.model.tokenizer, SYSTEM_PROMPT, user_p
                    )
                    prompts.append(prompt)

                results = self.model.predict_logits(pyg_batch, prompts)

                for j, res in enumerate(results):
                    true_label = int(batch_data[j].y.item())
                    predictions.append(res['prediction'])
                    true_labels.append(true_label)
                    prob_scores.append(res['prob_1'])

            except Exception as e:
                failed += len(batch_data)
                print(f"  [ERROR] batch {start}-{end}: {e}")

        # 计算指标
        metrics = {}
        if predictions:
            metrics['accuracy'] = accuracy_score(true_labels, predictions)
            metrics['f1'] = f1_score(true_labels, predictions, zero_division=0)
            metrics['precision'] = precision_score(true_labels, predictions, zero_division=0)
            metrics['recall'] = recall_score(true_labels, predictions, zero_division=0)
            if len(set(true_labels)) > 1:
                metrics['auc'] = roc_auc_score(true_labels, prob_scores)
            else:
                metrics['auc'] = float('nan')

        self._print_metrics(target_name, metrics, len(predictions), failed)

        return {
            'target': target_name,
            'method': method,
            'total': n,
            'predicted': len(predictions),
            'failed': failed,
            'metrics': metrics,
        }

    @staticmethod
    def _print_metrics(name: str, metrics: Dict, n: int, failed: int):
        print(f"\n  [Results] {name}:")
        for k, v in metrics.items():
            print(f"    {k:12s}: {v:.4f}")
        print(f"    {'predicted':12s}: {n}")
        if failed > 0:
            print(f"    {'failed':12s}: {failed}")
