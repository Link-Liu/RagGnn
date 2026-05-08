#!/usr/bin/env python3
"""
V2 实验主入口：三种消融配置
  1. Text-Only Baseline: 纯文字统计信息，无 graph tokens
  2. Ours w/o GRL: GNN + Connector + LLM，无域对抗
  3. Ours (Full): 完整版 GNN + Connector + LLM + GRL 域对抗

用法:
    python run_experiment.py                  # 运行完整版
    python run_experiment.py --ablation all   # 运行所有消融
"""
import os, sys, json, argparse, torch, numpy as np
from pathlib import Path
from datetime import datetime

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


# ============================================================
# Text-Only Baseline
# ============================================================
def run_text_only_baseline(model, tgt_list, target_name):
    """消融：不用 graph tokens，只给 LLM 图统计文字。"""
    from torch_geometric.data import Batch as PyGBatch
    from sklearn.metrics import roc_auc_score, accuracy_score

    print(f"\n{'='*60}")
    print(f"  [Text-Only Baseline] {target_name}")
    print(f"{'='*60}")

    preds, trues, probs = [], [], []
    for data in tgt_list:
        stats = extract_graph_stats(data)
        user_p = get_text_only_prompt(stats)
        prompt = format_chat_prompt(model.tokenizer, SYSTEM_PROMPT, user_p)

        # 直接用 LLM（无 graph tokens）
        input_ids = model.tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.llm(input_ids=input_ids, return_dict=True)
            last_logits = outputs.logits[0, -1, :]

        enz_logit = last_logits[model._enzyme_first_token].item()
        non_logit = last_logits[model._non_enzyme_first_token].item()

        # Softmax
        exp_e, exp_n = np.exp(enz_logit), np.exp(non_logit)
        p_enz = exp_e / (exp_e + exp_n)

        pred = 1 if p_enz > 0.5 else 0
        preds.append(pred)
        trues.append(int(data.y.item()))
        probs.append(p_enz)

    acc = accuracy_score(trues, preds)
    auc = roc_auc_score(trues, probs) if len(set(trues)) > 1 else float('nan')
    print(f"  Accuracy: {acc:.4f}")
    print(f"  AUC:      {auc:.4f}")
    return {"accuracy": acc, "auc": auc, "method": "text_only"}


# ============================================================
# 主实验流程
# ============================================================
def run_experiment(
    source_name: str = "PROTEINS",
    target_name: str = "DD",
    data_dir: str = "data",
    use_domain_adversarial: bool = True,
    epochs: int = 50,
    batch_size: int = 8,
    run_text_only: bool = False,
):
    """运行一个迁移对的完整实验。"""

    print(f"\n{'#'*60}")
    print(f"  V2 Experiment: {source_name} → {target_name}")
    print(f"  Domain Adversarial: {use_domain_adversarial}")
    print(f"{'#'*60}")

    # 1. 加载数据
    src_list, tgt_list, unified_dim = load_and_prepare(
        data_dir, source_name, target_name
    )

    # 2. 创建模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GraphLLMv2(
        num_node_features=unified_dim,
        gnn_hidden_dim=128,
        gnn_num_layers=3,
        num_query_tokens=8,
        connector_layers=2,
        device=device,
    )

    # 3. Text-Only Baseline（可选）
    text_only_result = None
    if run_text_only:
        text_only_result = run_text_only_baseline(model, tgt_list[:200], target_name)

    # 4. 训练
    tag = "full" if use_domain_adversarial else "no_grl"
    ckpt = f"checkpoints/v2_{source_name.lower()}_{target_name.lower()}_{tag}.pt"

    if not Path(ckpt).exists():
        trainer = JointTrainer(model, device)
        trainer.train(
            src_list, tgt_list, source_name, target_name,
            ckpt_path=ckpt,
            epochs=epochs,
            batch_size=batch_size,
            use_domain_adversarial=use_domain_adversarial,
        )
    else:
        model.load_checkpoint(ckpt)
        print(f"[Skip] Using existing checkpoint: {ckpt}")

    # 5. 评估
    evaluator = Evaluator(model, device)
    result = evaluator.evaluate(tgt_list, target_name, batch_size=4)

    # 6. 保存结果
    results_dir = Path("results/v2")
    results_dir.mkdir(parents=True, exist_ok=True)
    all_results = {
        "experiment": f"{source_name}→{target_name}",
        "tag": tag,
        "timestamp": datetime.now().isoformat(),
        "target_metrics": result['metrics'],
        "text_only_baseline": text_only_result,
    }
    result_file = results_dir / f"{source_name}_{target_name}_{tag}.json"
    with open(result_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[Results] Saved to {result_file}")

    return all_results


# ============================================================
# 消融实验
# ============================================================
def run_ablation(source_name="PROTEINS", target_name="DD"):
    """运行完整消融：Text-Only + w/o GRL + Full。"""
    results = {}

    # 1. Ours w/o GRL
    print("\n" + "="*60)
    print("  ABLATION: Ours w/o GRL")
    print("="*60)
    results["no_grl"] = run_experiment(
        source_name, target_name,
        use_domain_adversarial=False,
        run_text_only=True,
    )

    # 释放显存
    torch.cuda.empty_cache()

    # 2. Ours (Full)
    print("\n" + "="*60)
    print("  ABLATION: Ours (Full)")
    print("="*60)
    results["full"] = run_experiment(
        source_name, target_name,
        use_domain_adversarial=True,
        run_text_only=False,
    )

    # 打印对比表
    print(f"\n{'='*60}")
    print(f"  ABLATION SUMMARY: {source_name} → {target_name}")
    print(f"{'='*60}")
    print(f"  {'Config':<20s} {'AUC':>8s} {'Acc':>8s} {'F1':>8s}")
    print(f"  {'-'*44}")

    # Text-Only
    if results["no_grl"].get("text_only_baseline"):
        tb = results["no_grl"]["text_only_baseline"]
        print(f"  {'Text-Only':<20s} {tb.get('auc',0):.4f}   {tb.get('accuracy',0):.4f}   {'N/A':>8s}")

    for tag, label in [("no_grl", "Ours w/o GRL"), ("full", "Ours (Full)")]:
        m = results[tag].get("target_metrics", {})
        print(f"  {label:<20s} {m.get('auc',0):.4f}   {m.get('accuracy',0):.4f}   {m.get('f1',0):.4f}")

    return results


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V2 Graph-LLM Transfer Learning")
    parser.add_argument("--source", default="PROTEINS", help="Source dataset")
    parser.add_argument("--target", default="DD", help="Target dataset")
    parser.add_argument("--ablation", default=None, choices=["all"],
                        help="Run ablation study")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--no_grl", action="store_true", help="Disable domain adversarial")
    args = parser.parse_args()

    if args.ablation == "all":
        run_ablation(args.source, args.target)
    else:
        run_experiment(
            source_name=args.source,
            target_name=args.target,
            use_domain_adversarial=not args.no_grl,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
