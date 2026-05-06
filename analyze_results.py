import json

with open('detailed_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

out = open('analysis_output.txt', 'w', encoding='utf-8')

def p(s=""):
    out.write(s + "\n")

p("=" * 90)
p("                        实验结果详细分析报告")
p("=" * 90)

all_results = []

for task_key in data.keys():
    task_data = data[task_key]
    for method_key in task_data.keys():
        method_data = task_data[method_key]
        if isinstance(method_data, dict) and 'metrics' in method_data:
            m = method_data['metrics']
            details = method_data.get('details', [])
            
            true_0 = sum(1 for d in details if d['true_label'] == 0)
            true_1 = sum(1 for d in details if d['true_label'] == 1)
            pred_0 = sum(1 for d in details if d['prediction'] == 0)
            pred_1 = sum(1 for d in details if d['prediction'] == 1)
            tp = sum(1 for d in details if d['true_label'] == 1 and d['prediction'] == 1)
            tn = sum(1 for d in details if d['true_label'] == 0 and d['prediction'] == 0)
            fp = sum(1 for d in details if d['true_label'] == 0 and d['prediction'] == 1)
            fn = sum(1 for d in details if d['true_label'] == 1 and d['prediction'] == 0)
            
            result = {
                'task': task_key, 'method': method_key,
                'total': method_data.get('total_target', 0),
                'failed': method_data.get('failed', 0),
                'accuracy': m.get('accuracy', 0), 'f1': m.get('f1', 0),
                'precision': m.get('precision', 0), 'recall': m.get('recall', 0),
                'auc': m.get('auc', 0),
                'true_0': true_0, 'true_1': true_1,
                'pred_0': pred_0, 'pred_1': pred_1,
                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            }
            all_results.append(result)

# 1. Metrics table
p("\n1. 各任务指标汇总表")
p("-" * 90)
p(f"{'任务':<20} {'方法':<10} {'样本':>5} {'Acc':>8} {'F1':>8} {'Prec':>8} {'Recall':>8} {'AUC':>8}")
p("-" * 90)
for r in all_results:
    p(f"{r['task']:<20} {r['method']:<10} {r['total']:>5} {r['accuracy']:>8.4f} {r['f1']:>8.4f} {r['precision']:>8.4f} {r['recall']:>8.4f} {r['auc']:>8.4f}")

# 2. Confusion matrix
p("\n\n2. 各任务混淆矩阵分析")
p("-" * 90)
p(f"{'任务':<20} {'方法':<10} {'真0':>5} {'真1':>5} {'预0':>5} {'预1':>5} {'TP':>5} {'TN':>5} {'FP':>5} {'FN':>5}")
p("-" * 90)
for r in all_results:
    p(f"{r['task']:<20} {r['method']:<10} {r['true_0']:>5} {r['true_1']:>5} {r['pred_0']:>5} {r['pred_1']:>5} {r['tp']:>5} {r['tn']:>5} {r['fp']:>5} {r['fn']:>5}")

# 3. Comparison
p("\n\n3. full_rag vs no_rag 对比分析")
p("-" * 90)

tasks = sorted(set(r['task'] for r in all_results))

for task in tasks:
    rag = [r for r in all_results if r['task'] == task and r['method'] == 'full_rag']
    no_rag = [r for r in all_results if r['task'] == task and r['method'] == 'no_rag']
    
    if rag and no_rag:
        rag = rag[0]; no_rag = no_rag[0]
        p(f"\n  任务: {task}")
        for metric in ['accuracy', 'f1', 'precision', 'recall', 'auc']:
            diff = rag[metric] - no_rag[metric]
            sign = "+" if diff > 0 else ""
            winner = "RAG胜" if diff > 0 else ("平局" if diff == 0 else "NoRAG胜")
            p(f"    {metric:>10}: RAG={rag[metric]:.4f}  NoRAG={no_rag[metric]:.4f}  差值={sign}{diff:.4f}  [{winner}]")

# 4. Overall
p("\n\n4. 总体统计")
p("-" * 90)
rag_results = [r for r in all_results if r['method'] == 'full_rag']
no_rag_results = [r for r in all_results if r['method'] == 'no_rag']

for metric in ['accuracy', 'f1', 'precision', 'recall', 'auc']:
    avg_rag = sum(r[metric] for r in rag_results) / len(rag_results)
    avg_no_rag = sum(r[metric] for r in no_rag_results) / len(no_rag_results)
    diff = avg_rag - avg_no_rag
    sign = "+" if diff > 0 else ""
    p(f"  平均 {metric:>10}: RAG={avg_rag:.4f}  NoRAG={avg_no_rag:.4f}  差值={sign}{diff:.4f}")

# 5. Prediction bias
p("\n\n5. 预测偏置分析（模型倾向性）")
p("-" * 90)

for r in all_results:
    total = r['total']
    pred_0_pct = r['pred_0'] / total * 100
    pred_1_pct = r['pred_1'] / total * 100
    true_0_pct = r['true_0'] / total * 100
    true_1_pct = r['true_1'] / total * 100
    
    p(f"  {r['task']:<20} [{r['method']:<8}]")
    p(f"    真实分布: 类0={r['true_0']:>4}({true_0_pct:5.1f}%)  类1={r['true_1']:>4}({true_1_pct:5.1f}%)")
    p(f"    预测分布: 类0={r['pred_0']:>4}({pred_0_pct:5.1f}%)  类1={r['pred_1']:>4}({pred_1_pct:5.1f}%)")
    
    if pred_0_pct > 85:
        p(f"    [!] 严重偏向预测为类0！")
    elif pred_1_pct > 85:
        p(f"    [!] 严重偏向预测为类1！")
    p()

# 6. RAG wins
p("\n6. RAG方法胜出统计")
p("-" * 90)

rag_wins = {m: 0 for m in ['accuracy', 'f1', 'precision', 'recall', 'auc']}
total_tasks = 0

for task in tasks:
    rag = [r for r in all_results if r['task'] == task and r['method'] == 'full_rag']
    no_rag = [r for r in all_results if r['task'] == task and r['method'] == 'no_rag']
    if rag and no_rag:
        total_tasks += 1
        for metric in rag_wins:
            if rag[0][metric] > no_rag[0][metric]:
                rag_wins[metric] += 1

for metric, wins in rag_wins.items():
    p(f"  {metric:>10}: RAG胜出 {wins}/{total_tasks} 个任务")

p("\n" + "=" * 90)
out.close()
print("Analysis complete. Output written to analysis_output.txt")
