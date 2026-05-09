#!/usr/bin/env python3
"""
V2 实验主入口 — 默认运行所有迁移对 × 所有消融

迁移对:
  PROTEINS → DD, DD → PROTEINS

消融配置:
  1. Text-Only:  纯文字统计信息，无 graph tokens
  2. w/o GRL:    GNN + Connector + LLM，无域对抗
  3. Full:       GNN + Connector + LLM + GRL 域对抗

用法:
    python run_experiment.py                            # 跑所有迁移对 × 所有消融
    python run_experiment.py --pairs PROTEINS-DD        # 只跑一个方向
    python run_experiment.py --pairs PROTEINS-DD DD-PROTEINS --no_ablation  # 跑指定方向，只跑 Full
"""
import os, sys, json, argparse, torch, numpy as np, gc
from pathlib import Path
from datetime import datetime
from typing import List, Dict

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.v2.data_utils import load_and_prepare
from experiments.v2.models import GraphLLMv2
from experiments.v2.trainer import JointTrainer
from experiments.v2.evaluator import Evaluator
from experiments.v2.prompts import (
    SYSTEM_PROMPT, get_text_only_prompt, format_chat_prompt, extract_graph_stats
)

# 默认迁移对
DEFAULT_PAIRS = [
    ("PROTEINS", "DD"),
    ("DD", "PROTEINS"),
]


# ============================================================
# GNN 预训练（有监督图分类）
# ============================================================
def _pretrain_gnn_v2(gnn, src_list, device, ckpt=None,
                     epochs=100, batch_size=64, lr=1e-3, patience=15):
    """对 GINEncoderNodeLevel 做有监督预训练。"""
    import random
    from torch_geometric.data import Batch as PyGBatch
    from torch_geometric.nn import global_mean_pool
    from sklearn.metrics import accuracy_score

    indices = list(range(len(src_list)))
    random.seed(42)
    random.shuffle(indices)
    n_train = int(0.8 * len(indices))
    train_data = [src_list[i] for i in indices[:n_train]]
    val_data = [src_list[i] for i in indices[n_train:]]
    print(f"\n  [GNN-Pretrain] {len(src_list)} graphs: train={len(train_data)}, val={len(val_data)}")

    gnn = gnn.to(device)
    classifier = torch.nn.Linear(gnn.hidden_dim, 1).to(device)
    optimizer = torch.optim.Adam(
        list(gnn.parameters()) + list(classifier.parameters()),
        lr=lr, weight_decay=1e-5,
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    def collate(batch): return PyGBatch.from_data_list(batch)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, collate_fn=collate
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, shuffle=False, collate_fn=collate
    )

    best_val_loss, best_state, pat = float('inf'), None, 0
    for epoch in range(1, epochs + 1):
        gnn.train(); classifier.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            node_feats, batch_idx = gnn(batch.x.float(), batch.edge_index, batch.batch)
            graph_emb = global_mean_pool(node_feats, batch_idx)  # [B, hidden]
            logit = classifier(graph_emb).squeeze(-1)
            loss = criterion(logit, batch.y.float())
            loss.backward()
            optimizer.step()

        # 验证
        gnn.eval(); classifier.eval()
        val_loss, val_cnt = 0.0, 0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                node_feats, batch_idx = gnn(batch.x.float(), batch.edge_index, batch.batch)
                graph_emb = global_mean_pool(node_feats, batch_idx)
                logit = classifier(graph_emb).squeeze(-1)
                loss = criterion(logit, batch.y.float())
                val_loss += loss.item() * batch.num_graphs
                val_cnt += batch.num_graphs
                val_preds.extend((logit > 0).long().cpu().tolist())
                val_labels.extend(batch.y.long().cpu().tolist())

        vl = val_loss / max(val_cnt, 1)
        va = accuracy_score(val_labels, val_preds)
        improved = vl < best_val_loss
        if improved:
            best_val_loss = vl; pat = 0
            best_state = {k: v.cpu().clone() for k, v in gnn.state_dict().items()}
        else:
            pat += 1

        if epoch % 5 == 0 or epoch == 1 or improved:
            tag = "← best ✓" if improved else f"(patience {pat}/{patience})"
            print(f"    Epoch {epoch:3d}/{epochs}  val_loss={vl:.4f} val_acc={va:.4f}  {tag}")

        if pat >= patience:
            print(f"    [Early Stop] {patience} epochs w/o improvement")
            break

    if best_state:
        gnn.load_state_dict(best_state)
        gnn.to(device)
    gnn.eval()
    del classifier
    print(f"  [GNN-Pretrain] Done. best_val_loss={best_val_loss:.4f}")

    if ckpt:
        Path(ckpt).parent.mkdir(parents=True, exist_ok=True)
        torch.save(gnn.state_dict(), ckpt)
        print(f"  [GNN-Pretrain] Saved → {ckpt}")


# ============================================================
# Text-Only Baseline
# ============================================================
def run_text_only_baseline(model, tgt_list, target_name, max_samples=200):
    """消融：不用 graph tokens，只给 LLM 图统计文字。"""
    from torch_geometric.data import Batch as PyGBatch
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

    import random as _rnd
    _all = list(tgt_list)  # 完整复制
    _rnd.seed(42)
    _rnd.shuffle(_all)
    samples = _all[:max_samples]
    # 检查类别分布
    class_counts = {}
    for d in samples:
        c = int(d.y.item())
        class_counts[c] = class_counts.get(c, 0) + 1
    print(f"\n  [Text-Only] Evaluating {len(samples)} graphs (class dist: {class_counts})")

    preds, trues, probs = [], [], []
    for data in samples:
        stats = extract_graph_stats(data)
        user_p = get_text_only_prompt(stats)
        prompt = format_chat_prompt(model.tokenizer, SYSTEM_PROMPT, user_p)

        input_ids = model.tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.llm(input_ids=input_ids, return_dict=True)
            last_logits = outputs.logits[0, -1, :]

        logit_1 = last_logits[model._token_1].item()
        logit_0 = last_logits[model._token_0].item()
        exp_1, exp_0 = np.exp(logit_1), np.exp(logit_0)
        p1 = exp_1 / (exp_1 + exp_0)

        preds.append(1 if p1 > 0.5 else 0)
        trues.append(int(data.y.item()))
        probs.append(p1)

    acc = accuracy_score(trues, preds)
    auc = roc_auc_score(trues, probs) if len(set(trues)) > 1 else float('nan')
    f1 = f1_score(trues, preds, zero_division=0)
    print(f"  [Text-Only] Acc={acc:.4f}  AUC={auc:.4f}  F1={f1:.4f}")
    return {"accuracy": acc, "auc": auc, "f1": f1, "method": "text_only", "n_samples": len(samples)}


# ============================================================
# 单次实验
# ============================================================
def run_single(
    source_name: str, target_name: str,
    data_dir: str = "data",
    use_domain_adversarial: bool = False,
    epochs: int = 50, batch_size: int = 8,
    run_text_only: bool = False,
):
    """运行一个迁移对的一种配置。"""
    tag = "full" if use_domain_adversarial else "no_grl"
    print(f"\n{'='*60}")
    print(f"  {source_name} → {target_name}  [{tag.upper()}]")
    print(f"  domain_adv={use_domain_adversarial}, epochs={epochs}, batch={batch_size}")
    print(f"{'='*60}")

    # 1. 加载数据
    src_list, tgt_list, unified_dim = load_and_prepare(data_dir, source_name, target_name)

    # 2. 创建模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GraphLLMv2(
        num_node_features=unified_dim,
        gnn_hidden_dim=128, gnn_num_layers=3,
        num_query_tokens=16, connector_layers=2,
        device=device,
    )

    # 3. Text-Only（可选）
    text_only_result = None
    if run_text_only:
        text_only_result = run_text_only_baseline(model, tgt_list, target_name)

    # 4. GNN 预训练（有监督图分类，初始化 GNN 权重）
    gnn_ckpt = f"checkpoints/v2_gnn_{source_name.lower()}.pt"
    if not Path(gnn_ckpt).exists():
        _pretrain_gnn_v2(model.gnn, src_list, device, ckpt=gnn_ckpt)
    else:
        model.gnn.load_state_dict(torch.load(gnn_ckpt, map_location='cpu', weights_only=False))
        model.gnn.to(device)
        print(f"  [GNN] Loaded pretrained weights: {gnn_ckpt}")

    # 5. 联合训练
    ckpt = f"checkpoints/v2_{source_name.lower()}_{target_name.lower()}_{tag}.pt"
    if not Path(ckpt).exists():
        trainer = JointTrainer(model, device)
        trainer.train(
            src_list, tgt_list, source_name, target_name,
            ckpt_path=ckpt, epochs=epochs, batch_size=batch_size,
            use_domain_adversarial=use_domain_adversarial,
        )
    else:
        model.load_checkpoint(ckpt)
        print(f"  [Skip] Using existing checkpoint: {ckpt}")

    # 6. 评估目标域
    evaluator = Evaluator(model, device)
    result = evaluator.evaluate(tgt_list, target_name, batch_size=4)

    # 7. 释放显存
    del model, evaluator
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "pair": f"{source_name}→{target_name}",
        "tag": tag,
        "metrics": result.get("metrics", {}),
        "text_only": text_only_result,
    }


# ============================================================
# 完整实验：所有迁移对 × 所有消融
# ============================================================
def run_all(
    pairs: List[tuple] = None,
    data_dir: str = "data",
    epochs: int = 50,
    batch_size: int = 8,
    run_ablation: bool = True,
):
    """运行所有迁移对和消融实验。"""
    if pairs is None:
        pairs = DEFAULT_PAIRS

    all_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for src, tgt in pairs:
        pair_key = f"{src}→{tgt}"
        all_results[pair_key] = {}

        if run_ablation:
            # 消融: Text-Only + Ours
            r = run_single(src, tgt, data_dir,
                           use_domain_adversarial=False, epochs=epochs,
                           batch_size=batch_size, run_text_only=True)
            all_results[pair_key]["text_only"] = r["text_only"]
            all_results[pair_key]["ours"] = r["metrics"]
        else:
            # 只跑 Ours
            r = run_single(src, tgt, data_dir,
                           use_domain_adversarial=False, epochs=epochs,
                           batch_size=batch_size, run_text_only=False)
            all_results[pair_key]["ours"] = r["metrics"]

    # ---- 打印汇总表 ----
    _print_summary(all_results, run_ablation)

    # ---- 保存结果 ----
    results_dir = Path("results/v2")
    results_dir.mkdir(parents=True, exist_ok=True)
    result_file = results_dir / f"results_{timestamp}.json"
    with open(result_file, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[Saved] {result_file}")
    return all_results


def _print_summary(results: Dict, ablation: bool):
    """打印汇总对比表。"""
    print(f"\n{'#'*70}")
    print(f"  EXPERIMENT SUMMARY")
    print(f"{'#'*70}")

    if ablation:
        print(f"\n  {'Transfer Pair':<22s} {'Config':<16s} {'AUC':>8s} {'Acc':>8s} {'F1':>8s}")
        print(f"  {'─'*62}")
        for pair, data in results.items():
            if data.get("text_only"):
                tb = data["text_only"]
                print(f"  {pair:<22s} {'Text-Only':<16s} "
                      f"{tb.get('auc',0):>8.4f} {tb.get('accuracy',0):>8.4f} {tb.get('f1',0):>8.4f}")
            if data.get("ours"):
                m = data["ours"]
                print(f"  {'':<22s} {'Ours':<16s} "
                      f"{m.get('auc',0):>8.4f} {m.get('accuracy',0):>8.4f} {m.get('f1',0):>8.4f}")
            print(f"  {'─'*62}")
    else:
        print(f"\n  {'Transfer Pair':<22s} {'AUC':>8s} {'Acc':>8s} {'F1':>8s} {'Precision':>10s} {'Recall':>8s}")
        print(f"  {'─'*66}")
        for pair, data in results.items():
            m = data.get("ours", {})
            print(f"  {pair:<22s} {m.get('auc',0):>8.4f} {m.get('accuracy',0):>8.4f} "
                  f"{m.get('f1',0):>8.4f} {m.get('precision',0):>10.4f} {m.get('recall',0):>8.4f}")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V2 Graph-LLM Transfer Learning")
    parser.add_argument("--pairs", nargs="*", default=None,
                        help="迁移对，格式: PROTEINS-DD DD-PROTEINS（默认跑所有）")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--no_ablation", action="store_true",
                        help="跳过消融，只跑 Full 版本")
    parser.add_argument("--data_dir", default="data")
    args = parser.parse_args()

    # 解析迁移对
    pairs = None
    if args.pairs:
        pairs = []
        for p in args.pairs:
            parts = p.split("-")
            if len(parts) == 2:
                pairs.append((parts[0], parts[1]))
            else:
                print(f"[WARN] Invalid pair format: {p}, expected SRC-TGT")

    run_all(
        pairs=pairs,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        run_ablation=not args.no_ablation,
    )
