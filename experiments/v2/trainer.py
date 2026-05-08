"""
V2 联合训练器：
  - L_cls:    LLM next-token prediction loss（源域有标签）
  - L_domain: GRL 域判别 loss（源域+目标域无标签）
  - AUC-based early stopping
"""
import os, sys, torch, torch.nn.functional as F, numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from sklearn.metrics import roc_auc_score

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.v2.prompts import (
    SYSTEM_PROMPT, get_user_prompt, format_chat_prompt, extract_graph_stats
)


class JointTrainer:
    """联合训练 GNN + Connector + Projector（+ 可选 Domain Disc）。"""

    LABEL_MAP = {0: "Non-enzyme", 1: "Enzyme"}

    def __init__(self, model, device: str = "cuda"):
        self.model = model
        self.device = device

    def _make_prompts(self, batch_data, tokenizer) -> List[str]:
        """为一个 batch 生成格式化的 prompt。"""
        prompts = []
        for data in batch_data:
            stats = extract_graph_stats(data)
            user_prompt = get_user_prompt(stats)
            prompt = format_chat_prompt(tokenizer, SYSTEM_PROMPT, user_prompt)
            prompts.append(prompt)
        return prompts

    def train(
        self,
        src_list: List,
        tgt_list: List,
        source_name: str,
        target_name: str,
        ckpt_path: str = "checkpoints/v2_proj.pt",
        epochs: int = 50,
        batch_size: int = 8,
        lr_gnn: float = 5e-5,
        lr_proj: float = 2e-4,
        grad_accum: int = 2,
        lambda_domain: float = 0.1,
        use_domain_adversarial: bool = True,
    ):
        """联合训练。"""
        import random
        from torch_geometric.data import Batch as PyGBatch

        print(f"\n{'='*60}")
        print(f"  [V2 Joint Train] {source_name} → {target_name}")
        print(f"  epochs={epochs}, batch={batch_size}, domain_adv={use_domain_adversarial}")
        print(f"{'='*60}")

        # ---- 数据划分 ----
        indices = list(range(len(src_list)))
        random.seed(42)
        random.shuffle(indices)
        n_train = int(0.8 * len(indices))
        train_idx, val_idx = indices[:n_train], indices[n_train:]
        print(f"  Source: {len(src_list)} total, train={len(train_idx)}, val={len(val_idx)}")
        if use_domain_adversarial:
            print(f"  Target: {len(tgt_list)} (unlabeled, for domain adversarial)")

        # ---- DataLoader ----
        class IdxDataset(torch.utils.data.Dataset):
            def __init__(self, idxs):
                self.idxs = idxs
            def __len__(self): return len(self.idxs)
            def __getitem__(self, i): return src_list[self.idxs[i]]

        def collate(batch): return PyGBatch.from_data_list(batch)

        # 平衡采样
        train_ds = IdxDataset(train_idx)
        labels = [int(src_list[i].y.item()) for i in train_idx]
        class_counts = np.array([labels.count(c) for c in [0, 1]])
        weights = 1.0 / class_counts[labels]
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            collate_fn=collate, num_workers=0,
        )
        val_loader = torch.utils.data.DataLoader(
            IdxDataset(val_idx), batch_size=batch_size, shuffle=False,
            collate_fn=collate, num_workers=0,
        )

        # 目标域 DataLoader
        tgt_loader = None
        if use_domain_adversarial and tgt_list:
            class TgtDataset(torch.utils.data.Dataset):
                def __init__(self, data): self.data = data
                def __len__(self): return len(self.data)
                def __getitem__(self, i): return self.data[i]
            tgt_loader = torch.utils.data.DataLoader(
                TgtDataset(tgt_list), batch_size=batch_size, shuffle=True,
                collate_fn=collate, num_workers=0, drop_last=True,
            )

        # ---- 优化器 ----
        param_groups = [
            {"params": self.model.gnn.parameters(), "lr": lr_gnn},
            {"params": self.model.connector.parameters(), "lr": lr_proj},
            {"params": self.model.projector.parameters(), "lr": lr_proj},
        ]
        if use_domain_adversarial:
            param_groups.append(
                {"params": self.model.domain_disc.parameters(), "lr": lr_proj}
            )
        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-5)

        # Scheduler
        steps_per_epoch = max(len(train_loader) // grad_accum, 1)
        total_steps = epochs * steps_per_epoch
        warmup = max(total_steps // 5, 1)
        def lr_lambda(step):
            if step < warmup:
                return float(step) / float(max(warmup, 1))
            prog = float(step - warmup) / float(max(total_steps - warmup, 1))
            return max(0.1, 0.5 * (1.0 + np.cos(np.pi * prog)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # ---- 训练循环 ----
        best_auc = 0.0
        best_state = None
        patience, patience_limit = 0, 10
        use_amp = self.device == "cuda"
        tgt_iter = iter(tgt_loader) if tgt_loader else None

        for epoch in range(1, epochs + 1):
            self.model.gnn.train()
            self.model.connector.train()
            self.model.projector.train()
            optimizer.zero_grad()
            epoch_loss, n_steps = 0.0, 0
            train_correct, train_total = 0, 0

            # DANN alpha
            p = (epoch - 1) / max(epochs - 1, 1)
            dann_alpha = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0

            for batch in train_loader:
                batch = batch.to(self.device)
                B = batch.num_graphs
                prompts = self._make_prompts(
                    [batch[i] for i in range(B)], self.model.tokenizer
                )
                labels_text = [self.LABEL_MAP[int(y.item())] for y in batch.y]

                with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
                    cls_loss, acc = self.model.compute_cls_loss(batch, prompts, labels_text)
                    loss = cls_loss / grad_accum

                loss.backward()

                # 域对抗
                if use_domain_adversarial and tgt_iter is not None:
                    try:
                        tgt_batch = next(tgt_iter)
                    except StopIteration:
                        tgt_iter = iter(tgt_loader)
                        tgt_batch = next(tgt_iter)
                    tgt_batch = tgt_batch.to(self.device)

                    domain_loss = self.model.compute_domain_loss(
                        batch, tgt_batch, alpha=dann_alpha
                    )
                    (lambda_domain * domain_loss / grad_accum).backward()

                n_steps += 1
                epoch_loss += cls_loss.item()
                train_correct += int(acc * B)
                train_total += B

                if n_steps % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.trainable_parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            if n_steps % grad_accum != 0:
                optimizer.step(); scheduler.step(); optimizer.zero_grad()

            avg_loss = epoch_loss / max(n_steps, 1)
            train_acc = train_correct / max(train_total, 1)

            # ---- 验证 AUC ----
            val_auc = self._compute_val_auc(val_loader, src_list)

            print(f"  Epoch {epoch}/{epochs}  "
                  f"loss={avg_loss:.4f} acc={train_acc:.4f}  "
                  f"val_auc={val_auc:.4f}", end="")

            if not np.isnan(val_auc) and val_auc > best_auc:
                best_auc = val_auc
                patience = 0
                best_state = {
                    "gnn": {k: v.cpu().clone() for k, v in self.model.gnn.state_dict().items()},
                    "connector": {k: v.cpu().clone() for k, v in self.model.connector.state_dict().items()},
                    "projector": {k: v.cpu().clone() for k, v in self.model.projector.state_dict().items()},
                }
                print(f"  ← best ✓")
            else:
                patience += 1
                print(f"  (patience {patience}/{patience_limit})")
                if patience >= patience_limit:
                    print(f"  [Early Stop] {patience_limit} epochs w/o improvement")
                    break

        # 恢复最佳
        if best_state:
            self.model.gnn.load_state_dict(best_state["gnn"])
            self.model.connector.load_state_dict(best_state["connector"])
            self.model.projector.load_state_dict(best_state["projector"])
            self.model.gnn.to(self.device)
            self.model.connector.to(self.device)
            self.model.projector.to(self.device)
            Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, ckpt_path)
            print(f"  [Saved] {ckpt_path} (val_auc={best_auc:.4f})")

        self.model.gnn.eval()
        self.model.connector.eval()
        self.model.projector.eval()

    @torch.no_grad()
    def _compute_val_auc(self, val_loader, src_list) -> float:
        """在验证集上用 logits 计算 AUC。"""
        self.model.gnn.eval()
        self.model.connector.eval()
        self.model.projector.eval()
        all_probs, all_trues = [], []
        for batch in val_loader:
            batch = batch.to(self.device)
            B = batch.num_graphs
            prompts = self._make_prompts(
                [batch[i] for i in range(B)], self.model.tokenizer
            )
            results = self.model.predict_logits(batch, prompts)
            for i, res in enumerate(results):
                all_probs.append(res['prob_1'])
                all_trues.append(int(batch[i].y.item()))

        if len(set(all_trues)) > 1:
            return roc_auc_score(all_trues, all_probs)
        return float('nan')
