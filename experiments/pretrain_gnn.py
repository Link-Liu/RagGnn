"""
experiments/pretrain_gnn.py

在源域数据集上有监督预训练 GNN Encoder。
被 final_complete_implementation.py 调用。
"""

import random
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from pathlib import Path
from sklearn.metrics import accuracy_score


def pretrain_gnn_standalone(gnn, dataset, device, epochs=100,
                             batch_size=64, lr=1e-3, ckpt=None,
                             train_ratio=0.8, patience=15):
    """
    对 GINEncoder 进行有监督图分类预训练（二分类）。

    Args:
        gnn:         GINEncoder 实例（来自 LocalLLMInterface.gnn）
        dataset:     PyG Data 列表（源域全部数据）
        device:      torch.device
        epochs:      最大训练轮数
        batch_size:  batch 大小
        lr:          学习率
        ckpt:        checkpoint 保存路径（可选）
        train_ratio: 训练集比例（剩余为验证集）
        patience:    early stopping 的耐心值
    """
    if Path(ckpt).exists() if ckpt else False:
        gnn.load_state_dict(torch.load(ckpt, map_location=device))
        gnn.eval()
        print(f"[GNN-Pretrain] Loaded from {ckpt}")
        return

    # ---- 划分训练集/验证集 ----
    data_list = list(dataset)
    indices = list(range(len(data_list)))
    random.seed(42)
    random.shuffle(indices)
    n_train = int(train_ratio * len(indices))
    train_data = [data_list[i] for i in indices[:n_train]]
    val_data = [data_list[i] for i in indices[n_train:]]

    print(f"[GNN-Pretrain] {len(data_list)} graphs: "
          f"train={len(train_data)}, val={len(val_data)}, "
          f"epochs={epochs}, patience={patience}")

    gnn = gnn.to(device)
    classifier = nn.Linear(gnn.hidden_dim, 1).to(device)
    optimizer = torch.optim.Adam(
        list(gnn.parameters()) + list(classifier.parameters()),
        lr=lr, weight_decay=1e-5,
    )
    criterion = nn.BCEWithLogitsLoss()

    use_cuda = (device == 'cuda' or (isinstance(device, torch.device) and device.type == 'cuda'))
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=use_cuda
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=use_cuda
    )

    best_val_loss = float('inf')
    best_gnn_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        gnn.train()
        classifier.train()
        train_total, train_cnt = 0.0, 0
        train_preds, train_labels = [], []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            emb = gnn(batch.x.float(), batch.edge_index, batch.batch)
            logit = classifier(emb).squeeze(-1)
            loss = criterion(logit, batch.y.float())
            loss.backward()
            optimizer.step()
            train_total += loss.item() * batch.num_graphs
            train_cnt += batch.num_graphs
            train_preds.extend((logit > 0).long().cpu().tolist())
            train_labels.extend(batch.y.long().cpu().tolist())
        train_loss = train_total / max(train_cnt, 1)
        train_acc = accuracy_score(train_labels, train_preds)

        # ---- Validate ----
        gnn.eval()
        classifier.eval()
        val_total, val_cnt = 0.0, 0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                emb = gnn(batch.x.float(), batch.edge_index, batch.batch)
                logit = classifier(emb).squeeze(-1)
                loss = criterion(logit, batch.y.float())
                val_total += loss.item() * batch.num_graphs
                val_cnt += batch.num_graphs
                val_preds.extend((logit > 0).long().cpu().tolist())
                val_labels.extend(batch.y.long().cpu().tolist())
        val_loss = val_total / max(val_cnt, 1)
        val_acc = accuracy_score(val_labels, val_preds)

        # ---- Early stopping ----
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            patience_counter = 0
            best_gnn_state = {k: v.cpu().clone() for k, v in gnn.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1 or improved:
            marker = "  ← best ✓" if improved else f"  (patience {patience_counter}/{patience})"
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}{marker}")

        if not improved:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [Early Stopping] No improvement for {patience} epochs.")
                break

    # ---- 恢复最佳权重 ----
    if best_gnn_state:
        gnn.load_state_dict(best_gnn_state)
        gnn.to(device)
    gnn.eval()
    del classifier
    print(f"[GNN-Pretrain] Done. best_val_loss={best_val_loss:.4f}")

    if ckpt:
        Path(ckpt).parent.mkdir(parents=True, exist_ok=True)
        torch.save(gnn.state_dict(), ckpt)
        print(f"[GNN-Pretrain] Saved -> {ckpt}")
