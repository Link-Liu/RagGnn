"""
Prompt Engineering for Frozen LLM in Domain-Aware RAG-MDA

核心设计原则：标签语义空间解耦 (Label Space Decoupling)
------------------------------------------------------
RAG 检索到的源域分子不能将其标签直接映射到目标任务。
例如 BBBP 的 label=1 表示"能穿透血脑屏障"，不能直接说成"有毒性=1"。

正确做法：
  1. 明确告知 LLM 源域任务是什么（BBBP / Tox21 / SIDER 各有语义）
  2. 展示源域标签时使用其真实语义（如 "BBB Penetration: Yes"）
  3. 引入显式的跨域推理链（Cross-Domain Chain-of-Thought），让 LLM
     自主推断结构特征如何跨任务关联，而不是照搬标签数字
"""

from typing import List, Dict, Any, Optional


# ---------------------------------------------------------------------------
# 域语义注册表 (Domain Semantics Registry)
# 记录每个数据集的任务描述、正例/负例的语义含义
# 新增数据集时在此处注册，无需修改 Prompt 函数
# ---------------------------------------------------------------------------

DOMAIN_SEMANTICS: Dict[str, Dict] = {
    # 源域
    'bbbp': {
        'task': 'Blood-Brain Barrier Penetration (BBBP)',
        'label_pos': 'Penetrates BBB (lipophilic / CNS-active scaffold)',
        'label_neg': 'Does NOT penetrate BBB (hydrophilic or efflux-susceptible)',
        'key_features': 'lipophilicity, molecular weight, H-bond donor/acceptor count, P-gp efflux',
    },
    # 目标域
    'tox21': {
        # 注意：实验实际使用 task_index=2 即 NR-AhR 单一检测，不是所有 12 个检测
        'task': 'Tox21 NR-AhR (Aryl Hydrocarbon Receptor Activation Assay)',
        'label_pos': 'AhR Agonist — activates the Aryl Hydrocarbon Receptor in vitro',
        'label_neg': 'AhR Inactive — does not activate AhR in this specific assay',
        'key_features': 'planar aromatic/polycyclic structure, halogenated aromatics (PCBs/dioxin-like), '
                        'molecular planarity enabling intercalation into AhR ligand-binding pocket',
        # assay_context: 精确描述本实验使用的单一检测任务，避免 LLM 混淆"有毒"和"AhR激活"
        'assay_context': (
            "IMPORTANT — This label comes from a SINGLE specific biochemical assay: NR-AhR.\n"
            "\n"
            "The AhR (Aryl Hydrocarbon Receptor) is a ligand-activated transcription factor.\n"
            "  label=1  The molecule ACTIVATES AhR in a cell-based reporter assay.\n"
            "           This is NOT equivalent to 'toxic' — it means the molecule binds "
            "and activates this specific nuclear receptor pathway.\n"
            "  label=0  The molecule does NOT activate AhR in this assay.\n"
            "           This does NOT mean it is safe — it simply lacks AhR agonist activity.\n"
            "\n"
            "Known AhR agonist structural features to look for:\n"
            "  - Planar polycyclic aromatic hydrocarbons (PAHs): naphthalene, anthracene, pyrene cores\n"
            "  - Halogenated aromatics: PCB-like structures, dioxins (chlorinated/brominated biphenyls)\n"
            "  - Indoles, carbazoles, and other N-containing heteroaromatics\n"
            "  - Flavonoids and phytochemicals with planar conjugated systems\n"
            "  - High molecular planarity allowing fit into the AhR-PAS-B hydrophobic cavity\n"
            "\n"
            "Do NOT predict based on general toxicity intuition. "
            "Ask specifically: 'Does this molecule look like a planar aromatic AhR ligand?'"
        ),
    },
    'sider': {
        # 注意：实验实际使用 task_index=0 即 Hepatobiliary disorders（肝胆疾病副作用）
        'task': 'SIDER Hepatobiliary Disorders (Drug Side-Effect Database)',
        'label_pos': 'Associated with hepatobiliary side effects (liver/gallbladder/bile duct disorders)',
        'label_neg': 'No recorded hepatobiliary side effects in clinical data',
        'key_features': 'hepatotoxic substructures, reactive metabolites (quinones, epoxides), '
                        'mitochondrial liability, bile salt export pump (BSEP) inhibition motifs',
        # assay_context: 告知 LLM 这是基于上市药物临床不良反应数据库的标签，不是实验检测
        'assay_context': (
            "IMPORTANT — This label comes from the SIDER clinical drug side-effect database, "
            "specifically for the category: Hepatobiliary Disorders (MedDRA SOC).\n"
            "\n"
            "  label=1  This approved drug is reported to cause liver, gallbladder, or bile duct "
            "disorders in clinical use (e.g. drug-induced liver injury, cholestasis, jaundice).\n"
            "  label=0  No hepatobiliary side effects recorded for this drug in clinical data.\n"
            "\n"
            "Key structural features associated with hepatobiliary side effects:\n"
            "  - Reactive metabolite formation: compounds that form quinones, epoxides, or "
            "acyl glucuronides via CYP450 metabolism\n"
            "  - BSEP inhibition: bulky lipophilic compounds (clogP > 3) with H-bond acceptors\n"
            "  - Mitochondrial toxicity: uncouplers, electron transport chain inhibitors\n"
            "  - Idiosyncratic hepatotoxins: often contain nitrogen heterocycles, sulfonamides, "
            "or aromatic amines that form reactive nitroso/hydroxylamine intermediates\n"
            "\n"
            "Do NOT predict based on general toxicity. Ask: "
            "'Does this drug's structure suggest reactive metabolite formation or bile transport inhibition?'"
        ),
    },
    'clintox': {
        'task': 'Clinical Toxicity (ClinTox)',
        'label_pos': 'Failed clinical trials due to toxicity',
        'label_neg': 'Approved / passed clinical trials',
        'key_features': 'hERG inhibition, hepatotoxicity, genotoxicity',
    },
}

_UNKNOWN_DOMAIN = {
    'task': 'Unknown molecular property task',
    'label_pos': 'Label = 1',
    'label_neg': 'Label = 0',
    'key_features': 'structural features',
}


def _get_domain_info(dataset_name: str) -> Dict:
    """查询域语义，大小写不敏感，未知域返回通用占位。"""
    return DOMAIN_SEMANTICS.get(dataset_name.lower(), _UNKNOWN_DOMAIN)


# ---------------------------------------------------------------------------
# 主要接口：跨域推理链 Prompt（供 RealExperiment 调用）
# ---------------------------------------------------------------------------

def create_detailed_prompt(target_molecule: Dict,
                           retrieved_examples: List[Dict],
                           property_description: str = 'toxicity',
                           target_dataset: str = 'tox21',
                           source_dataset: Optional[str] = None,
                           target_label: Any = None,
                           include_target_label: bool = False) -> str:
    """
    构造跨域 RAG Prompt，核心逻辑：特征迁移与标签预测解耦。

    Args:
        target_molecule:      目标分子信息字典（需含 'smiles'、'name'）
        retrieved_examples:   检索到的源域参考分子列表
        property_description: 目标任务的属性名（如 'toxicity'）
        target_dataset:       目标数据集名称（用于查询语义注册表）
        source_dataset:       源域数据集名称（若为 None 则从例子的 source_domain 字段推断）
        target_label:         真实标签（仅用于调试，默认不显示）
        include_target_label: 是否将真实标签写入 Prompt（仅调试用）

    Returns:
        格式化后的 Prompt 字符串
    """
    target_sem = _get_domain_info(target_dataset)

    # 推断源域数据集名称（取第一个检索结果的 source_domain 字段）
    if source_dataset is None and retrieved_examples:
        source_dataset = retrieved_examples[0].get('source_domain', 'bbbp')
    source_sem = _get_domain_info(source_dataset or 'bbbp')

    parts = []

    # ------------------------------------------------------------------
    # Section 1: 角色定义
    # ------------------------------------------------------------------
    parts.append(
        "You are an expert cheminformatician specialising in cross-domain "
        "molecular property prediction. Your task is to predict a property "
        "for a target molecule by reasoning from structurally similar molecules "
        "retrieved from a *different* dataset with a *different* labelling task."
    )

    # ------------------------------------------------------------------
    # Section 2: 目标分子
    # ------------------------------------------------------------------
    parts.append(f"\n{'='*60}")
    parts.append(f"TARGET MOLECULE  (Task: {target_sem['task']})")
    parts.append(f"{'='*60}")
    parts.append(f"  SMILES : {target_molecule.get('smiles', 'N/A')}")
    parts.append(f"  Name   : {target_molecule.get('name', 'N/A')}")
    parts.append(f"  Predict: {property_description}  "
                 f"(0 = {target_sem['label_neg']} | 1 = {target_sem['label_pos']})")
    parts.append(f"  Key chemical signals to watch: {target_sem['key_features']}")

    # 若目标域有 assay_context，插入任务定义澄清块
    # 这是领域知识，不是数据泄露：帮助 LLM 正确理解标签的真实含义
    assay_context = target_sem.get('assay_context')
    if assay_context:
        parts.append(f"\n{'='*60}")
        parts.append("TASK DEFINITION CLARIFICATION")
        parts.append(f"{'='*60}")
        parts.append(assay_context)

    # ------------------------------------------------------------------
    # Section 3: 源域参考分子（标签保留原始语义，不强行改名）
    # ------------------------------------------------------------------
    parts.append(f"\n{'='*60}")
    parts.append(f"SOURCE DOMAIN REFERENCES  (Task: {source_sem['task']})")
    parts.append(f"{'='*60}")
    parts.append(
        "  These molecules are labelled for a DIFFERENT task than the one you "
        "need to predict. Use their structural features as chemical evidence, "
        "NOT their labels as direct answers."
    )
    parts.append(
        f"  Source label semantics: "
        f"0 = {source_sem['label_neg']} | 1 = {source_sem['label_pos']}"
    )
    parts.append("")

    if retrieved_examples:
        for i, ex in enumerate(retrieved_examples[:5]):
            smiles = ex.get('smiles', 'N/A')
            name   = ex.get('name', f'ref_{i+1}')
            label  = ex.get('label', '?')
            score  = ex.get('retrieval_score', 0.0)
            maccs  = ex.get('maccs_tanimoto', None)

            # 标签用源域语义翻译，不混用目标任务名
            if label in (0, '0'):
                label_str = f"0 ({source_sem['label_neg']})"
            elif label in (1, '1'):
                label_str = f"1 ({source_sem['label_pos']})"
            else:
                label_str = str(label)

            line = (f"  [{i+1}] SMILES={smiles}  name={name}\n"
                    f"       Source label ({source_sem['task']}): {label_str}\n"
                    f"       Retrieval score={score:.3f}")
            if maccs is not None:
                line += f"  MACCS Tanimoto={maccs:.3f}"
            parts.append(line)
    else:
        parts.append("  (No similar molecules found in source domain.)")

    # ------------------------------------------------------------------
    # Section 4: 跨域推理链指令（Cross-Domain Chain-of-Thought）
    # ------------------------------------------------------------------
    parts.append(f"\n{'='*60}")
    parts.append("CROSS-DOMAIN REASONING INSTRUCTION")
    parts.append(f"{'='*60}")
    parts.append(
        "  Step 1 — Identify shared structural features:\n"
        "    What substructures does the target molecule share with the "
        "references? (e.g., aromatic rings, halogens, reactive carbonyls, "
        "heavy atoms, H-bond donors/acceptors)"
    )
    parts.append(
        f"  Step 2 — Bridge the domain gap:\n"
        f"    The references were labelled for [{source_sem['task']}], "
        f"but you need to predict [{target_sem['task']}].\n"
        f"    Reason: how do the physicochemical features that drive "
        f"[{source_sem['task']}] mechanistically relate to "
        f"[{target_sem['task']}]?\n"
        f"    (They may correlate, anticorrelate, or be independent — "
        f"use your chemical knowledge to judge.)"
    )
    parts.append(
        "  Step 3 — Make your prediction:\n"
        "    Based on Steps 1–2, decide whether the target molecule is "
        f"likely to exhibit {property_description}."
    )

    # ------------------------------------------------------------------
    # Section 5: 输出格式（严格结构化）
    # ------------------------------------------------------------------
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

    # 调试用：真实标签
    if include_target_label and target_label is not None:
        parts.append(f"\n  [DEBUG] Ground truth = {target_label}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 兼容旧接口（简单版，无跨域解耦）
# ---------------------------------------------------------------------------

def create_prompt(target_molecule: Dict, retrieved_examples: List[Dict],
                  target_label: Any = None, include_target_label: bool = False) -> str:
    """轻量版 Prompt，向后兼容，不含跨域推理链。"""
    parts = [
        "You are an expert in molecular property prediction. "
        "Predict the property of the target molecule using the reference examples."
    ]
    parts.append(f"\nTarget: SMILES={target_molecule.get('smiles', 'N/A')}  "
                 f"name={target_molecule.get('name', 'N/A')}")
    parts.append("\nReferences:")
    for i, ex in enumerate(retrieved_examples[:5]):
        parts.append(f"  {i+1}. SMILES={ex.get('smiles','N/A')}  label={ex.get('label','?')}")
    parts.append("\nRespond with 0 or 1 only.")
    if include_target_label and target_label is not None:
        parts.append(f"[DEBUG] Ground truth = {target_label}")
    return "\n".join(parts)


def create_few_shot_prompt(target_molecule: Dict, retrieved_examples: List[Dict],
                           target_label: Any = None, include_target_label: bool = False,
                           num_examples: int = 3) -> str:
    """Few-shot Prompt（保留兼容接口）。"""
    return create_detailed_prompt(
        target_molecule=target_molecule,
        retrieved_examples=retrieved_examples[:num_examples],
        target_label=target_label,
        include_target_label=include_target_label,
    )


def format_molecular_info(molecule: Dict) -> str:
    """格式化分子信息为单行字符串。"""
    parts = []
    if 'smiles' in molecule:
        parts.append(f"SMILES: {molecule['smiles']}")
    if 'name' in molecule:
        parts.append(f"Name: {molecule['name']}")
    if 'formula' in molecule:
        parts.append(f"Formula: {molecule['formula']}")
    if 'mw' in molecule:
        parts.append(f"MW: {molecule['mw']}")
    return "; ".join(parts) if parts else "Unknown molecule"


# ---------------------------------------------------------------------------
# 消融实验：无 RAG 对照 Prompt（与 create_detailed_prompt 结构对齐）
# ---------------------------------------------------------------------------

def create_no_rag_prompt(target_molecule: Dict,
                          property_description: str = 'toxicity',
                          target_dataset: str = 'tox21') -> str:
    """
    消融实验用 Prompt：结构与 create_detailed_prompt 完全对齐，
    但 Section 3（源域参考分子）留空，Section 4 去掉跨域 Step 1。

    目的：衡量 RAG 检索带来的增益（Full - No-RAG = RAG contribution）。

    Args:
        target_molecule:      目标分子信息字典（需含 'smiles'、'name'）
        property_description: 目标任务的属性名（如 'toxicity'）
        target_dataset:       目标数据集名称（用于查询语义注册表）

    Returns:
        格式化后的无-RAG Prompt 字符串
    """
    target_sem = _get_domain_info(target_dataset)

    parts = []

    # Section 1: 角色定义（与完整版相同）
    parts.append(
        "You are an expert cheminformatician specialising in molecular property "
        "prediction. Your task is to predict a property for a target molecule "
        "based solely on your chemical knowledge and the molecule's structure."
    )

    # Section 2: 目标分子（与完整版相同）
    parts.append(f"\n{'='*60}")
    parts.append(f"TARGET MOLECULE  (Task: {target_sem['task']})")
    parts.append(f"{'='*60}")
    parts.append(f"  SMILES : {target_molecule.get('smiles', 'N/A')}")
    parts.append(f"  Name   : {target_molecule.get('name', 'N/A')}")
    parts.append(f"  Predict: {property_description}  "
                 f"(0 = {target_sem['label_neg']} | 1 = {target_sem['label_pos']})")
    parts.append(f"  Key chemical signals to watch: {target_sem['key_features']}")

    # 若目标域有 assay_context，插入任务定义澄清块（与完整版相同）
    assay_context = target_sem.get('assay_context')
    if assay_context:
        parts.append(f"\n{'='*60}")
        parts.append("TASK DEFINITION CLARIFICATION")
        parts.append(f"{'='*60}")
        parts.append(assay_context)

    # Section 3: 无检索例子（消融关键）
    parts.append(f"\n{'='*60}")
    parts.append("REFERENCE MOLECULES  [ABLATION: No RAG retrieval]")
    parts.append(f"{'='*60}")
    parts.append(
        "  No reference molecules are provided in this ablation condition.\n"
        "  Please rely exclusively on your chemical knowledge and the\n"
        "  structural features of the target molecule."
    )

    # Section 4: 推理指令（去掉依赖检索结果的 Step 1，直接基于结构推理）
    parts.append(f"\n{'='*60}")
    parts.append("REASONING INSTRUCTION  [Zero-shot]")
    parts.append(f"{'='*60}")
    parts.append(
        "  Step 1 — Analyse the target structure:\n"
        "    Identify key functional groups, ring systems, heteroatoms,\n"
        "    halogens, reactive moieties, and physicochemical indicators\n"
        "    (logP, H-bond donors/acceptors, molecular weight, planarity)."
    )
    parts.append(
        f"  Step 2 — Apply domain knowledge:\n"
        f"    Based on known structure–activity relationships for\n"
        f"    [{target_sem['task']}], reason whether the identified features\n"
        f"    are associated with {property_description}."
    )
    parts.append(
        f"  Step 3 — Make your prediction:\n"
        f"    Decide whether the target molecule is likely to exhibit\n"
        f"    {property_description}."
    )

    # Section 5: 输出格式（与完整版完全相同）
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
# 快速验证
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')   # 修复 Windows 控制台 Unicode 乱码
    target = {'smiles': 'c1ccc(Cl)cc1', 'name': 'Chlorobenzene', 'dataset': 'tox21'}
    refs = [
        {'smiles': 'c1ccccc1',   'name': 'Benzene',      'label': 1,
         'source_domain': 'bbbp', 'retrieval_score': 0.82, 'maccs_tanimoto': 0.74},
        {'smiles': 'ClCCl',      'name': 'DCM',           'label': 0,
         'source_domain': 'bbbp', 'retrieval_score': 0.71, 'maccs_tanimoto': 0.61},
    ]
    print(create_detailed_prompt(
        target_molecule=target,
        retrieved_examples=refs,
        property_description='toxicity',
        target_dataset='tox21',
        source_dataset='bbbp',
    ))