"""
experiments/pretrain_gnn.py

在源域数据集上有监督预训练 GNN Encoder。
被 final_complete_implementation.py 调用。
"""

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from pathlib import Path


def pretrain_gnn_standalone(gnn, dataset, device, epochs=30,
                             batch_size=32, lr=1e-3, ckpt=None):
    """
    对 GINEncoder 进行有监督图分类预训练（二分类）。
    LLM 天然冻结，GNN 是唯一可训练组件。

    Args:
        gnn:        GINEncoder 实例（来自 LocalLLMInterface.gnn）
        dataset:    PyG Data 列表（训练集）
        device:     torch.device
        epochs:     训练轮数
        batch_size: batch 大小
        lr:         学习率
        ckpt:       checkpoint 保存路径（可选）
    """
    if Path(ckpt).exists() if ckpt else False:
        gnn.load_state_dict(torch.load(ckpt, map_location=device))
        gnn.eval()
        print(f"[GNN-Pretrain] Loaded from {ckpt}")
        return

    print(f"[GNN-Pretrain] Training on {len(dataset)} graphs, {epochs} epochs ...")
    gnn = gnn.to(device)
    classifier = nn.Linear(gnn.hidden_dim, 1).to(device)
    optimizer = torch.optim.Adam(
        list(gnn.parameters()) + list(classifier.parameters()),
        lr=lr, weight_decay=1e-5,
    )
    criterion = nn.BCEWithLogitsLoss()
    
    # A5: DataLoader 参数调优
    use_cuda = device.type == 'cuda'
    loader = DataLoader(
        list(dataset), 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0, # Windows 环境下保持 0 避免多进程错误
        pin_memory=use_cuda
    )

    gnn.train()
    best_loss = float('inf')
    for epoch in range(1, epochs + 1):
        total, cnt = 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            emb = gnn(batch.x.float(), batch.edge_index, batch.batch)
            logit = classifier(emb).squeeze(-1)
            loss = criterion(logit, batch.y.float())
            loss.backward()
            optimizer.step()
            total += loss.item() * batch.num_graphs
            cnt += batch.num_graphs
        avg = total / max(cnt, 1)
        if avg < best_loss:
            best_loss = avg
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  loss={avg:.4f}  best={best_loss:.4f}")

    gnn.eval()
    del classifier
    print(f"[GNN-Pretrain] Done. best_loss={best_loss:.4f}")

    if ckpt:
        Path(ckpt).parent.mkdir(parents=True, exist_ok=True)
        torch.save(gnn.state_dict(), ckpt)
        print(f"[GNN-Pretrain] Saved -> {ckpt}")
