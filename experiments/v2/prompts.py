"""
V2 Prompt 模板：为 LLM 设计的 System + User prompt。
支持 LLaMA-3 chat template 格式。
"""
from typing import Dict, Optional


SYSTEM_PROMPT = (
    "You are an expert biological assistant. "
    "Your task is to classify whether the given protein graph is an enzyme or a non-enzyme. "
    "You will receive the protein graph information as embedding tokens. "
    "Based on these embeddings, make your classification decision."
)


def get_user_prompt(graph_stats: Optional[Dict] = None) -> str:
    """生成 User prompt。graph_stats 可选地嵌入图统计信息。"""
    parts = [
        "Determine the class of the given protein molecule. "
        "Output the final decision strictly as either 'Enzyme' or 'Non-enzyme'."
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
        f"Determine the class of the given protein molecule. "
        f"Output the final decision strictly as either 'Enzyme' or 'Non-enzyme'.\n"
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
