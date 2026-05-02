"""
Prompt Engineering for Frozen LLM in Domain-Aware RAG-MDA (Graph Classification Version)

核心设计原则：标签语义空间解耦 (Label Space Decoupling)
------------------------------------------------------
RAG 检索到的源域图不能将其标签直接映射到目标任务。
例如 PROTEINS 的 label=1 表示"是酶"，不能直接复制到 DD 上。

正确做法：
  1. 明确告知 LLM 源域任务是什么
  2. 展示源域标签时使用其真实语义
  3. 引入显式的跨域推理链（Cross-Domain Chain-of-Thought）

数据集对：
  - PROTEINS <-> DD        (蛋白质结构图 — 酶分类)
  - COX2     <-> COX2_MD   (环氧合酶-2抑制剂)
  - BZR      <-> BZR_MD    (苯二氮卓受体配体)
"""

from typing import List, Dict, Any, Optional


# ---------------------------------------------------------------------------
# 域语义注册表 (Domain Semantics Registry)
# ---------------------------------------------------------------------------

DOMAIN_SEMANTICS: Dict[str, Dict] = {
    # ===== 蛋白质结构图数据集 =====
    'proteins': {
        'task': 'Protein Structure Classification (PROTEINS)',
        'label_pos': 'Enzyme — protein is an enzyme with catalytic function',
        'label_neg': 'Non-enzyme — protein does NOT have enzymatic activity',
        'key_features': (
            'secondary structure element (helix/sheet/coil) composition, '
            'contact map topology, amino acid connectivity patterns, '
            'active site subgraph motifs, graph density and clustering coefficient'
        ),
        'assay_context': (
            "IMPORTANT — This label comes from the PROTEINS dataset (TUDataset).\n"
            "\n"
            "Each graph represents a protein's secondary structure:\n"
            "  - Nodes are secondary structure elements (SSEs: helices, sheets, coils)\n"
            "  - Edges connect SSEs that are neighbors in the amino acid sequence\n"
            "    or in 3D space (within 6 Angstroms)\n"
            "\n"
            "  label=1  The protein IS an enzyme (has catalytic function)\n"
            "  label=0  The protein is NOT an enzyme\n"
            "\n"
            "Key structural features for enzyme classification:\n"
            "  - Presence of catalytic triad/dyad subgraph motifs\n"
            "  - Specific arrangements of helices around the active site\n"
            "  - Higher local clustering coefficient in the active region\n"
            "  - Characteristic helix-sheet-helix patterns in enzyme folds\n"
        ),
    },
    'dd': {
        'task': 'Protein Structure Classification (DD)',
        'label_pos': 'Enzyme — protein is an enzyme with catalytic function',
        'label_neg': 'Non-enzyme — protein does NOT have enzymatic activity',
        'key_features': (
            'amino acid contact network topology, residue-level spatial contacts, '
            'graph diameter, degree distribution, community structure'
        ),
        'assay_context': (
            "IMPORTANT — This label comes from the DD dataset (TUDataset).\n"
            "\n"
            "Each graph represents a protein's amino acid contact network:\n"
            "  - Nodes are amino acid residues\n"
            "  - Edges connect residues that are spatially close (< 6 Angstroms)\n"
            "\n"
            "  label=1  The protein IS an enzyme\n"
            "  label=0  The protein is NOT an enzyme\n"
            "\n"
            "Key features:\n"
            "  - Contact network topology reflects 3D fold\n"
            "  - Enzyme-specific contact patterns in active sites\n"
            "  - Different degree distributions for enzyme vs non-enzyme folds\n"
        ),
    },

    # ===== COX-2 抑制剂数据集 =====
    'cox2': {
        'task': 'COX-2 Inhibitor Activity Classification (COX2)',
        'label_pos': 'Active — molecule is a COX-2 inhibitor',
        'label_neg': 'Inactive — molecule does NOT inhibit COX-2',
        'key_features': (
            'molecular graph topology, pharmacophore patterns, '
            'sulfonamide/sulfone groups, diaryl heterocycle scaffolds, '
            'selectivity features for COX-2 over COX-1'
        ),
        'assay_context': (
            "IMPORTANT — This label comes from the COX2 dataset.\n"
            "\n"
            "COX-2 (Cyclooxygenase-2) is an enzyme involved in inflammation.\n"
            "  - Nodes represent atoms in the molecular graph\n"
            "  - Edges represent chemical bonds\n"
            "\n"
            "  label=1  The molecule IS a COX-2 inhibitor (active)\n"
            "  label=0  The molecule is NOT a COX-2 inhibitor (inactive)\n"
            "\n"
            "Known COX-2 inhibitor structural features:\n"
            "  - Diaryl heterocycle core (e.g., pyrazole, isoxazole)\n"
            "  - Sulfonamide or methylsulfone group for COX-2 selectivity\n"
            "  - Specific substitution patterns on aromatic rings\n"
        ),
    },
    'cox2_md': {
        'task': 'COX-2 Inhibitor Activity Classification (COX2_MD)',
        'label_pos': 'Active — molecule is a COX-2 inhibitor (MD descriptors)',
        'label_neg': 'Inactive — molecule does NOT inhibit COX-2 (MD descriptors)',
        'key_features': (
            'molecular dynamics descriptors, binding free energy components, '
            'protein-ligand interaction fingerprints, conformational features'
        ),
        'assay_context': (
            "IMPORTANT — This label comes from the COX2_MD dataset.\n"
            "\n"
            "Same molecules as COX2 but node features are derived from\n"
            "molecular dynamics simulations rather than static descriptors.\n"
            "\n"
            "  label=1  The molecule IS a COX-2 inhibitor (active)\n"
            "  label=0  The molecule is NOT a COX-2 inhibitor (inactive)\n"
        ),
    },

    # ===== BZR 受体配体数据集 =====
    'bzr': {
        'task': 'Benzodiazepine Receptor Ligand Activity (BZR)',
        'label_pos': 'Active — molecule is a BZR ligand (binds to receptor)',
        'label_neg': 'Inactive — molecule does NOT bind to BZR',
        'key_features': (
            'benzodiazepine core scaffold, heterocyclic ring systems, '
            'electrostatic surface features, H-bond donor/acceptor patterns, '
            'lipophilicity indicators'
        ),
        'assay_context': (
            "IMPORTANT — This label comes from the BZR dataset.\n"
            "\n"
            "BZR (Benzodiazepine Receptor) is a target for anxiolytic/sedative drugs.\n"
            "  - Nodes represent atoms\n"
            "  - Edges represent bonds\n"
            "\n"
            "  label=1  The molecule binds to the benzodiazepine receptor (active)\n"
            "  label=0  The molecule does NOT bind (inactive)\n"
        ),
    },
    'bzr_md': {
        'task': 'Benzodiazepine Receptor Ligand Activity (BZR_MD)',
        'label_pos': 'Active — BZR ligand (MD descriptors)',
        'label_neg': 'Inactive — not a BZR ligand (MD descriptors)',
        'key_features': (
            'molecular dynamics descriptors, receptor-ligand interaction features, '
            'binding pose conformational descriptors'
        ),
        'assay_context': (
            "IMPORTANT — This label comes from the BZR_MD dataset.\n"
            "\n"
            "Same molecules as BZR but node features are from MD simulations.\n"
            "\n"
            "  label=1  Active BZR ligand\n"
            "  label=0  Inactive\n"
        ),
    },
}

_UNKNOWN_DOMAIN = {
    'task': 'Unknown graph classification task',
    'label_pos': 'Label = 1',
    'label_neg': 'Label = 0',
    'key_features': 'graph structural features',
}


def _get_domain_info(dataset_name: str) -> Dict:
    """查询域语义，大小写不敏感，未知域返回通用占位。"""
    return DOMAIN_SEMANTICS.get(dataset_name.lower(), _UNKNOWN_DOMAIN)


# ---------------------------------------------------------------------------
# 主要接口：跨域推理链 Prompt（供 RealExperiment 调用）
# ---------------------------------------------------------------------------

def create_detailed_prompt(target_graph_info: Dict,
                           retrieved_examples: List[Dict],
                           property_description: str = 'graph classification',
                           target_dataset: str = 'proteins',
                           source_dataset: Optional[str] = None,
                           target_label: Any = None,
                           include_target_label: bool = False,
                           graph_tokens_text: Optional[str] = None) -> str:
    """
    构造跨域 RAG Prompt（图分类版本）。

    设计原则：严格防止信息泄露
    -------------------------------------------------------
    不提供（会让 LLM 绕过 GNN 直接推断）：
      - graph_id / num_nodes / num_edges / avg_degree / density
      - feature_summary / top_feat_indices
      - key_features（专家提示，相当于答案提示）
      - assay_context 中的结构描述（过强领域先验）
    只提供：
      - 标签语义定义（0/1 的含义）
      - GNN graph tokens（唯一的图结构信号）
      - RAG 参考图的 graph tokens + 标签

    Args:
        target_graph_info:    保留兼容性，不从中读取统计量
        retrieved_examples:   检索到的源域参考图列表
        property_description: 目标任务属性名
        target_dataset:       目标数据集名称
        source_dataset:       源域数据集名称
        target_label:         真实标签（仅调试用）
        include_target_label: 是否将真实标签写入 Prompt（仅调试）
        graph_tokens_text:    GNN 嵌入序列化文本，唯一的图结构信息来源

    Returns:
        格式化后的 Prompt 字符串
    """
    target_sem = _get_domain_info(target_dataset)

    if source_dataset is None and retrieved_examples:
        source_dataset = retrieved_examples[0].get('source_domain', '')
    source_sem = _get_domain_info(source_dataset or 'proteins')

    parts = []

    # Section 1: 角色定义
    parts.append(
        "You are a graph classification expert. "
        "Your task is to predict a binary label for a target graph "
        "using its GNN embedding representation and structurally similar "
        "reference graphs retrieved from a related source domain.\n"
        "IMPORTANT: Base your prediction ONLY on the GNN graph tokens provided. "
        "Do NOT rely on dataset names or domain-specific prior knowledge."
    )

    # Section 2: 标签语义定义（只告知 0/1 含义）
    parts.append(f"\n{'='*60}")
    parts.append("CLASSIFICATION TASK")
    parts.append(f"{'='*60}")
    parts.append(f"  Property : {property_description}")
    parts.append(f"  Label 0  : {target_sem['label_neg']}")
    parts.append(f"  Label 1  : {target_sem['label_pos']}")

    # Section 3: 目标图 GNN Token（唯一图结构信息来源）
    parts.append(f"\n{'='*60}")
    parts.append("TARGET GRAPH — GNN EMBEDDING TOKENS")
    parts.append(f"{'='*60}")
    if graph_tokens_text:
        parts.append(
            "  The graph has been encoded by a pre-trained GNN encoder.\n"
            "  The most significant embedding dimensions (L2-normalized) are:\n"
            f"  {graph_tokens_text}\n"
            "  Format: [dim_index:value, ...] — values reflect learned structural patterns."
        )
    else:
        parts.append("  (GNN embedding not available for this graph.)")

    # Section 4: RAG 参考图（只给 graph token + 标签，不给统计量）
    parts.append(f"\n{'='*60}")
    parts.append("REFERENCE GRAPHS FROM SOURCE DOMAIN")
    parts.append(f"{'='*60}")
    parts.append(
        "  These graphs come from a RELATED but DIFFERENT dataset.\n"
        "  Use their GNN tokens to gauge structural similarity.\n"
        "  Do NOT directly copy their labels — reason from token similarity.\n"
        f"  Source label semantics: "
        f"0 = {source_sem['label_neg']} | 1 = {source_sem['label_pos']}"
    )
    parts.append("")

    if retrieved_examples:
        for i, ex in enumerate(retrieved_examples[:5]):
            label = ex.get('label', '?')
            score = ex.get('retrieval_score', 0.0)
            ref_tokens = ex.get('graph_tokens_text', None)

            if label in (0, '0'):
                label_str = f"0 ({source_sem['label_neg']})"
            elif label in (1, '1'):
                label_str = f"1 ({source_sem['label_pos']})"
            else:
                label_str = str(label)

            line = f"  [{i+1}] similarity={score:.3f}  label={label_str}"
            if ref_tokens:
                line += f"\n       GNN tokens: {ref_tokens}"
            parts.append(line)
    else:
        parts.append("  (No reference graphs retrieved.)")

    # Section 5: 推理指令
    parts.append(f"\n{'='*60}")
    parts.append("REASONING INSTRUCTION")
    parts.append(f"{'='*60}")
    parts.append(
        "  Step 1 — Compare GNN tokens with reference tokens:\n"
        "    Which references have the most similar embedding patterns?\n"
        "    What do the shared active dimensions suggest about graph structure?"
    )
    parts.append(
        f"  Step 2 — Bridge the domain gap:\n"
        f"    References are from [{source_sem['task']}], "
        f"but you predict [{target_sem['task']}].\n"
        f"    Focus on structural similarity signals, not domain-specific rules."
    )
    parts.append(
        "  Step 3 — Make your prediction:\n"
        "    Based on GNN token similarity and reference labels, decide the label."
    )

    # Section 6: 输出格式
    parts.append(f"\n{'='*60}")
    parts.append("OUTPUT FORMAT  (strictly follow)")
    parts.append(f"{'='*60}")
    parts.append(
        "  You may reason freely above. But your FINAL LINE must be exactly:\n"
        "\n"
        "      Answer: 0\n"
        "  or\n"
        "      Answer: 1\n"
        "\n"
        f"  where  0 = {target_sem['label_neg']}\n"
        f"         1 = {target_sem['label_pos']}\n"
        "\n"
        "  The parser only reads 'Answer: <digit>'. Any other format causes an error."
    )

    if include_target_label and target_label is not None:
        parts.append(f"\n  [DEBUG] Ground truth = {target_label}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 消融实验：无 RAG 对照 Prompt
# ---------------------------------------------------------------------------

def create_no_rag_prompt(target_graph_info: Dict,
                          property_description: str = 'graph classification',
                          target_dataset: str = 'proteins',
                          graph_tokens_text: Optional[str] = None) -> str:
    """
    消融实验用 Prompt：不提供 RAG 检索结果。
    graph_tokens_text 应传入 <graph_token> 占位符，供 soft token 注入使用。
    """
    target_sem = _get_domain_info(target_dataset)

    parts = []

    parts.append(
        "You are an expert in graph-based machine learning and graph "
        "classification. Your task is to predict a label for a target graph "
        "based on its GNN embedding and structural properties."
    )

    # GNN Embedding Token Section（与 Full-RAG prompt 一致的 soft token 注入点）
    parts.append(f"\n{'='*60}")
    parts.append("TARGET GRAPH — GNN EMBEDDING TOKENS")
    parts.append(f"{'='*60}")
    if graph_tokens_text:
        parts.append(
            "  The graph has been encoded by a pre-trained GNN encoder.\n"
            "  The most significant embedding dimensions (L2-normalized) are:\n"
            f"  {graph_tokens_text}\n"
            "  Format: [dim_index:value, ...] — values reflect learned structural patterns."
        )
    else:
        parts.append("  (GNN embedding not available for this graph.)")

    parts.append(f"\n{'='*60}")
    parts.append(f"TARGET GRAPH  (Task: {target_sem['task']})")
    parts.append(f"{'='*60}")
    parts.append(f"  Graph ID   : {target_graph_info.get('graph_id', 'N/A')}")
    parts.append(f"  Num Nodes  : {target_graph_info.get('num_nodes', 'N/A')}")
    parts.append(f"  Num Edges  : {target_graph_info.get('num_edges', 'N/A')}")
    parts.append(f"  Avg Degree : {target_graph_info.get('avg_degree', 'N/A')}")
    if 'density' in target_graph_info:
        parts.append(f"  Density    : {target_graph_info['density']:.4f}")
    if 'feature_summary' in target_graph_info:
        parts.append(f"  Features   : {target_graph_info['feature_summary']}")
    parts.append(f"  Predict    : {property_description}  "
                 f"(0 = {target_sem['label_neg']} | 1 = {target_sem['label_pos']})")
    parts.append(f"  Key signals: {target_sem['key_features']}")

    assay_context = target_sem.get('assay_context')
    if assay_context:
        parts.append(f"\n{'='*60}")
        parts.append("TASK DEFINITION CLARIFICATION")
        parts.append(f"{'='*60}")
        parts.append(assay_context)

    parts.append(f"\n{'='*60}")
    parts.append("REFERENCE GRAPHS  [ABLATION: No RAG retrieval]")
    parts.append(f"{'='*60}")
    parts.append(
        "  No reference graphs are provided in this ablation condition.\n"
        "  Please rely on the GNN embedding tokens and the structural\n"
        "  properties of the target graph for your prediction."
    )

    parts.append(f"\n{'='*60}")
    parts.append("REASONING INSTRUCTION  [Zero-shot]")
    parts.append(f"{'='*60}")
    parts.append(
        "  Step 1 — Analyse the GNN embedding tokens:\n"
        "    Examine the activated dimensions and their values.\n"
        "    What structural patterns do they suggest?"
    )
    parts.append(
        f"  Step 2 — Apply domain knowledge:\n"
        f"    Based on known structure–property relationships for\n"
        f"    [{target_sem['task']}], reason whether the graph features\n"
        f"    indicate {property_description}."
    )
    parts.append(
        f"  Step 3 — Make your prediction:\n"
        f"    Decide the most likely label."
    )

    parts.append(f"\n{'='*60}")
    parts.append("OUTPUT FORMAT  (strictly follow)")
    parts.append(f"{'='*60}")
    parts.append(
        "  You may reason freely above. But your FINAL LINE must be exactly:\n"
        "\n"
        "      Answer: 0\n"
        "  or\n"
        "      Answer: 1\n"
        "\n"
        f"  where  0 = {target_sem['label_neg']}\n"
        f"         1 = {target_sem['label_pos']}\n"
        "\n"
        "  The parser only reads 'Answer: <digit>'. Any other format causes an error."
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 兼容旧接口
# ---------------------------------------------------------------------------

def create_prompt(target_graph_info: Dict, retrieved_examples: List[Dict],
                  target_label: Any = None, include_target_label: bool = False) -> str:
    """轻量版 Prompt，向后兼容。"""
    parts = [
        "You are an expert in graph classification. "
        "Predict the label of the target graph using the reference examples."
    ]
    parts.append(f"\nTarget: graph_id={target_graph_info.get('graph_id', 'N/A')}  "
                 f"nodes={target_graph_info.get('num_nodes', '?')}  "
                 f"edges={target_graph_info.get('num_edges', '?')}")
    parts.append("\nReferences:")
    for i, ex in enumerate(retrieved_examples[:5]):
        parts.append(f"  {i+1}. graph={ex.get('graph_id','N/A')}  label={ex.get('label','?')}")
    parts.append("\nRespond with 0 or 1 only.")
    if include_target_label and target_label is not None:
        parts.append(f"[DEBUG] Ground truth = {target_label}")
    return "\n".join(parts)


def create_few_shot_prompt(target_graph_info: Dict, retrieved_examples: List[Dict],
                           target_label: Any = None, include_target_label: bool = False,
                           num_examples: int = 3) -> str:
    """Few-shot Prompt（保留兼容接口）。"""
    return create_detailed_prompt(
        target_graph_info=target_graph_info,
        retrieved_examples=retrieved_examples[:num_examples],
        target_label=target_label,
        include_target_label=include_target_label,
    )


def format_graph_info(graph_info: Dict) -> str:
    """格式化图信息为单行字符串。"""
    parts = []
    if 'graph_id' in graph_info:
        parts.append(f"ID: {graph_info['graph_id']}")
    if 'num_nodes' in graph_info:
        parts.append(f"Nodes: {graph_info['num_nodes']}")
    if 'num_edges' in graph_info:
        parts.append(f"Edges: {graph_info['num_edges']}")
    return "; ".join(parts) if parts else "Unknown graph"


# ---------------------------------------------------------------------------
# 快速验证
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    target = {
        'graph_id': 'PROTEINS_42',
        'num_nodes': 15,
        'num_edges': 34,
        'avg_degree': 4.53,
        'density': 0.3238,
    }
    refs = [
        {'graph_id': 'DD_100', 'num_nodes': 20, 'num_edges': 56, 'label': 1,
         'source_domain': 'dd', 'retrieval_score': 0.82},
        {'graph_id': 'DD_205', 'num_nodes': 12, 'num_edges': 28, 'label': 0,
         'source_domain': 'dd', 'retrieval_score': 0.71},
    ]

    print("=== Full RAG Prompt ===")
    print(create_detailed_prompt(
        target_graph_info=target,
        retrieved_examples=refs,
        property_description='enzyme classification',
        target_dataset='proteins',
        source_dataset='dd',
    ))

    print("\n\n=== No-RAG Prompt ===")
    print(create_no_rag_prompt(
        target_graph_info=target,
        property_description='enzyme classification',
        target_dataset='proteins',
    ))