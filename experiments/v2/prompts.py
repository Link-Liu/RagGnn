"""
V2 Prompt 模板：为 LLM 设计的 System + User prompt。
标签使用 0/1（单 token，无语义偏差）。
"""
from typing import Dict, Optional


SYSTEM_PROMPT = (
    "You are a binary classifier for protein graphs. "
    "Given the protein graph information, output 1 if it is an enzyme, or 0 if it is not. "
    "Output only the number, nothing else."
)


def get_user_prompt(graph_stats: Optional[Dict] = None) -> str:
    """生成 User prompt。"""
    parts = [
        "Classify the following protein graph. "
        "Output strictly 1 (enzyme) or 0 (non-enzyme)."
    ]
    if graph_stats:
        stats_str = (
            f"\nGraph statistics: "
            f"nodes={graph_stats.get('num_nodes', '?')}, "
            f"edges={graph_stats.get('num_edges', '?')}, "
            f"density={graph_stats.get('density', 0):.4f}, "
            f"avg_degree={graph_stats.get('avg_degree', 0):.2f}"
        )
        parts.append(stats_str)
    parts.append("\nProtein Graph: <graph_tokens>")
    return "\n".join(parts)


def get_text_only_prompt(graph_stats: Dict) -> str:
    """消融实验：纯文字版本（不用 graph tokens）。"""
    return (
        f"Classify the following protein graph. "
        f"Output strictly 1 (enzyme) or 0 (non-enzyme).\n"
        f"Graph statistics: "
        f"nodes={graph_stats.get('num_nodes', '?')}, "
        f"edges={graph_stats.get('num_edges', '?')}, "
        f"density={graph_stats.get('density', 0):.4f}, "
        f"avg_degree={graph_stats.get('avg_degree', 0):.2f}, "
        f"clustering_coeff={graph_stats.get('avg_clustering', 0):.4f}"
    )


def format_chat_prompt(tokenizer, system: str, user: str) -> str:
    """用 tokenizer 的 chat template 格式化。"""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback: 手动拼接
        return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


def extract_graph_stats(data) -> Dict:
    """从 PyG Data 提取图统计信息。"""
    N = data.num_nodes
    E = data.edge_index.shape[1] if data.edge_index.numel() > 0 else 0
    density = E / (N * (N - 1) + 1e-6)
    avg_deg = E / max(N, 1)
    return {
        'num_nodes': N,
        'num_edges': E,
        'density': density,
        'avg_degree': avg_deg,
    }
