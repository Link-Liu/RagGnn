"""
V2 模型组件：
  - GINEncoderNodeLevel: 3层 GIN，输出节点级特征
  - DynamicQueryConnector: Q-Former 风格的交叉注意力连接器
  - LLMProjector: 2层 MLP 投影到 LLM 词嵌入空间
  - DomainDiscriminator: GRL + MLP 域判别器（作用在 soft tokens 上）
  - GraphLLMv2: 完整模型（GNN + Connector + Projector + LLM）

LLM: Qwen2.5-3B-Instruct (hidden_size=2560)
"""
import os, torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool
from transformers import AutoTokenizer
from modelscope import snapshot_download

try:
    from transformers import AutoModelForCausalLM as HFAutoModel
except Exception:
    HFAutoModel = None


# ============================================================
# 1. GNN Encoder（节点级输出，不做图级池化）
# ============================================================
class GINEncoderNodeLevel(nn.Module):
    """3 层 GIN，输出节点级特征用于下游 Dynamic Query Connector。"""
    def __init__(self, num_node_features: int, hidden_dim: int = 128,
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(num_node_features, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim),
        )
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = dropout

    def forward(self, x, edge_index, batch=None):
        """返回节点级特征 [total_N, hidden_dim] 和 batch 索引。"""
        h = self.input_proj(x.float())
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h, batch  # [total_N, hidden_dim], [total_N]

    def graph_pool(self, h, batch):
        """辅助方法：节点特征 → 图级特征（用于 RAG 编码等）"""
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        return torch.cat([h_mean, h_max], dim=-1)  # [B, 2*hidden]


# ============================================================
# 2. Dynamic Query Connector（Q-Former 风格）
# ============================================================
class DynamicQueryConnector(nn.Module):
    """
    使用可学习的 query tokens 通过交叉注意力从节点特征中提取定长图表示。

    流程：
      query_tokens [Q, d]  ×  node_features [N, d]
            ↓ cross-attention (Q=queries, K=V=nodes)
      output [Q, d]  → 定长图特征矩阵

    类似 BLIP-2 的 Q-Former，每个 query token 学习关注图的不同部分。
    """
    def __init__(self, hidden_dim: int = 128, num_query_tokens: int = 8,
                 num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_query_tokens = num_query_tokens
        self.query_tokens = nn.Parameter(torch.randn(num_query_tokens, hidden_dim) * 0.02)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_dim, nhead=num_heads,
                dim_feedforward=hidden_dim * 2, dropout=dropout,
                batch_first=True, norm_first=True,
            ) for _ in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_feats: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_feats: [total_N, hidden_dim]
            batch: [total_N] 每个节点属于哪个图
        Returns:
            [B, num_query_tokens, hidden_dim]
        """
        B = batch.max().item() + 1
        device = node_feats.device

        # 构建 padded KV（每个图的节点数不同）
        counts = torch.bincount(batch, minlength=B)
        max_nodes = counts.max().item()

        kv = torch.zeros(B, max_nodes, node_feats.shape[1], device=device)
        kv_mask = torch.ones(B, max_nodes, dtype=torch.bool, device=device)  # True=忽略

        offset = 0
        for b in range(B):
            n = counts[b].item()
            kv[b, :n] = node_feats[offset:offset+n]
            kv_mask[b, :n] = False
            offset += n

        # 扩展 query tokens
        queries = self.query_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, Q, d]

        # 多层交叉注意力
        out = queries
        for layer in self.layers:
            out = layer(out, kv, memory_key_padding_mask=kv_mask)

        return self.out_norm(out)  # [B, Q, hidden_dim]


# ============================================================
# 3. LLM Projector（2 层 MLP）
# ============================================================
class LLMProjector(nn.Module):
    """将 connector 输出投影到 LLM 词嵌入空间。"""
    def __init__(self, hidden_dim: int = 128, llm_dim: int = 2560):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, llm_dim),
            nn.LayerNorm(llm_dim),
        )

    def forward(self, x):
        """x: [B, num_tokens, hidden_dim] → [B, num_tokens, llm_dim]"""
        return self.proj(x)


# ============================================================
# 4. GRL + Domain Discriminator（作用在 soft tokens 上）
# ============================================================
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class DomainDiscriminator(nn.Module):
    """域判别器：判断 soft tokens 来自源域还是目标域。"""
    def __init__(self, llm_dim: int = 2560):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(llm_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, soft_tokens: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """
        soft_tokens: [B, num_tokens, llm_dim]
        Returns: [B, 1] 域判别 logits
        """
        # 池化 soft tokens → 单向量
        pooled = soft_tokens.mean(dim=1)  # [B, llm_dim]
        x_rev = GradientReversal.apply(pooled, alpha)
        return self.net(x_rev)


# ============================================================
# 5. GraphLLMv2：完整模型
# ============================================================

# ModelScope 配置
MODEL_ID = os.environ.get("MODELSCOPE_MODEL_ID", "LLM-Research/Meta-Llama-3.1-8B-Instruct")
MODELSCOPE_CACHE_DIR = os.environ.get("MODELSCOPE_CACHE_DIR", None)
MODELSCOPE_REVISION = os.environ.get("MODELSCOPE_REVISION", "master")


class GraphLLMv2(nn.Module):
    """
    完整的 Graph-LLM 模型：
      GNN → Dynamic Query Connector → LLM Projector → 冻结 LLM
    """
    def __init__(
        self,
        num_node_features: int,
        gnn_hidden_dim: int = 128,
        gnn_num_layers: int = 3,
        num_query_tokens: int = 16,
        connector_layers: int = 2,
        model_name_or_path: str = MODEL_ID,
        cache_dir: str = MODELSCOPE_CACHE_DIR,
        modelscope_revision: str = MODELSCOPE_REVISION,
        load_in_8bit: bool = False,
        device: str = "auto",
    ):
        super().__init__()
        self.call_count = 0
        self.num_query_tokens = num_query_tokens

        # ---- 设备 ----
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # ---- 加载 LLM（ModelScope 下载 + transformers 加载）----
        model_path = Path(model_name_or_path)
        if model_path.exists():
            model_dir = str(model_path)
            print(f"[LLM] Local model: {model_dir}")
        else:
            if cache_dir is None:
                project_root = Path(__file__).resolve().parents[2]
                cache_dir = str(project_root / "checkpoints" / "modelscope")
            os.makedirs(cache_dir, exist_ok=True)
            print(f"[LLM] Downloading from ModelScope: {model_name_or_path}")
            model_dir = snapshot_download(
                model_name_or_path,
                cache_dir=cache_dir,
                revision=modelscope_revision,
                ignore_file_pattern=['original/*'],
            )

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # LLM（用 transformers 加载本地模型）
        print(f"[LLM] Loading {model_dir} ...")
        from transformers import AutoModelForCausalLM
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
        }
        if load_in_8bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = None

        self.llm = AutoModelForCausalLM.from_pretrained(model_dir, **load_kwargs)
        if not load_in_8bit:
            self.llm = self.llm.to(self.device)

        # 冻结 LLM
        for param in self.llm.parameters():
            param.requires_grad = False
        self.llm.gradient_checkpointing_enable()
        print("[LLM] All LLM parameters FROZEN + gradient checkpointing ON")

        llm_dim = self.llm.config.hidden_size
        print(f"[LLM] hidden_size = {llm_dim}")

        # ---- 可训练组件 ----
        self.gnn = GINEncoderNodeLevel(
            num_node_features=num_node_features,
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_num_layers,
        ).to(self.device)

        self.connector = DynamicQueryConnector(
            hidden_dim=gnn_hidden_dim,
            num_query_tokens=num_query_tokens,
            num_heads=4,
            num_layers=connector_layers,
        ).to(self.device)

        self.projector = LLMProjector(
            hidden_dim=gnn_hidden_dim,
            llm_dim=llm_dim,
        ).to(self.device)

        self.domain_disc = DomainDiscriminator(llm_dim=llm_dim).to(self.device)

        # ---- 缓存 label token IDs ----
        self._enzyme_ids = self.tokenizer.encode("Enzyme", add_special_tokens=False)
        self._non_enzyme_ids = self.tokenizer.encode("Non-enzyme", add_special_tokens=False)
        # 用于快速 logits 对比的首 token
        self._enzyme_first_token = self._enzyme_ids[0]
        self._non_enzyme_first_token = self._non_enzyme_ids[0]

        trainable = sum(p.numel() for p in self.trainable_parameters())
        total_llm = sum(p.numel() for p in self.llm.parameters())
        print(f"[Params] Trainable: {trainable:,} | Frozen (LLM): {total_llm:,} | "
              f"Ratio: {100*trainable/(total_llm+trainable):.3f}%")

    def trainable_parameters(self):
        return (list(self.gnn.parameters())
                + list(self.connector.parameters())
                + list(self.projector.parameters())
                + list(self.domain_disc.parameters()))

    # ----------------------------------------------------------
    # 前向：GNN → Connector → Projector → soft tokens
    # ----------------------------------------------------------
    def encode_graph(self, pyg_batch) -> torch.Tensor:
        """PyG batch → soft tokens [B, num_tokens, llm_dim]"""
        node_feats, batch_idx = self.gnn(
            pyg_batch.x.to(self.device),
            pyg_batch.edge_index.to(self.device),
            pyg_batch.batch.to(self.device),
        )
        graph_tokens = self.connector(node_feats, batch_idx)  # [B, Q, hidden]
        soft_tokens = self.projector(graph_tokens)             # [B, Q, llm_dim]
        return soft_tokens

    # ----------------------------------------------------------
    # 训练 Loss
    # ----------------------------------------------------------
    def compute_cls_loss(self, pyg_batch, prompts: List[str],
                         labels: List[str]) -> Tuple[torch.Tensor, float]:
        """
        分类 loss：LLM next-token prediction。
        labels: ["Enzyme", "Non-enzyme", ...]
        """
        B = len(prompts)
        soft_tokens = self.encode_graph(pyg_batch)  # [B, Q, llm_dim]

        embed_fn = self.llm.get_input_embeddings()
        max_txt_len = 512

        # 构建每个样本的 input_embeds + label
        batch_inputs_embeds = []
        batch_labels = []
        for i in range(B):
            # Prompt embedding
            prompt_ids = self.tokenizer.encode(prompts[i], add_special_tokens=False)[:max_txt_len]
            prompt_embeds = embed_fn(torch.tensor(prompt_ids, device=self.device))

            # Label embedding
            label_ids = self.tokenizer.encode(labels[i], add_special_tokens=False)
            eos_id = self.tokenizer.eos_token_id or self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
            label_ids_full = label_ids + [eos_id]
            label_embeds = embed_fn(torch.tensor(label_ids_full, device=self.device))

            # 拼接：[soft_tokens, prompt, label]
            seq_embeds = torch.cat([
                soft_tokens[i],                # [Q, llm_dim]
                prompt_embeds,                 # [P, llm_dim]
                label_embeds,                  # [L, llm_dim]
            ], dim=0)
            batch_inputs_embeds.append(seq_embeds)

            # Labels：只在 label 部分计算 loss
            n_prefix = soft_tokens.shape[1] + len(prompt_ids)
            seq_labels = [-100] * n_prefix + label_ids_full
            batch_labels.append(torch.tensor(seq_labels, device=self.device))

        # Padding
        max_len = max(e.shape[0] for e in batch_inputs_embeds)
        pad_embed = embed_fn(torch.tensor([self.tokenizer.pad_token_id], device=self.device))

        padded_embeds, padded_labels, attn_masks = [], [], []
        for i in range(B):
            pad_len = max_len - batch_inputs_embeds[i].shape[0]
            if pad_len > 0:
                padded_embeds.append(torch.cat([pad_embed.repeat(pad_len, 1),
                                                 batch_inputs_embeds[i]]))
                padded_labels.append(torch.cat([torch.full((pad_len,), -100,
                                                            device=self.device),
                                                 batch_labels[i]]))
                attn_masks.append([0]*pad_len + [1]*batch_inputs_embeds[i].shape[0])
            else:
                padded_embeds.append(batch_inputs_embeds[i])
                padded_labels.append(batch_labels[i])
                attn_masks.append([1]*batch_inputs_embeds[i].shape[0])

        inputs_embeds = torch.stack(padded_embeds)
        label_ids = torch.stack(padded_labels)
        attention_mask = torch.tensor(attn_masks, device=self.device)

        outputs = self.llm(
            inputs_embeds=inputs_embeds.to(self.llm.dtype),
            attention_mask=attention_mask,
            labels=label_ids,
            return_dict=True,
            use_cache=False,
        )

        # 计算准确率
        accuracy = 0.0
        with torch.no_grad():
            for i in range(B):
                label_positions = (label_ids[i] != -100).nonzero(as_tuple=True)[0]
                if len(label_positions) > 0:
                    pred_pos = label_positions[0] - 1
                    pred_token = outputs.logits[i, pred_pos].argmax().item()
                    true_token = label_ids[i][label_positions[0]].item()
                    if pred_token == true_token:
                        accuracy += 1
        accuracy /= max(B, 1)

        return outputs.loss, accuracy

    # ----------------------------------------------------------
    # 域对抗 Loss
    # ----------------------------------------------------------
    def compute_domain_loss(self, src_batch, tgt_batch,
                            alpha: float = 1.0) -> torch.Tensor:
        """在 soft tokens 层面做域对抗。"""
        src_tokens = self.encode_graph(src_batch)  # [B_s, Q, llm_dim]
        tgt_tokens = self.encode_graph(tgt_batch)  # [B_t, Q, llm_dim]

        src_pred = self.domain_disc(src_tokens.float(), alpha)
        tgt_pred = self.domain_disc(tgt_tokens.float(), alpha)

        loss = (
            F.binary_cross_entropy_with_logits(src_pred, torch.zeros_like(src_pred)) +
            F.binary_cross_entropy_with_logits(tgt_pred, torch.ones_like(tgt_pred))
        ) / 2.0
        return loss

    # ----------------------------------------------------------
    # 推理：Logits 方式
    # ----------------------------------------------------------
    @torch.no_grad()
    def predict_logits(self, pyg_batch, prompts: List[str]) -> List[Dict]:
        """用 LLM logits 做分类，比较 Enzyme vs Non-enzyme 首 token 概率。"""
        self.call_count += len(prompts)
        B = len(prompts)
        soft_tokens = self.encode_graph(pyg_batch)

        embed_fn = self.llm.get_input_embeddings()
        max_txt_len = 512

        batch_inputs_embeds, batch_attn = [], []
        for i in range(B):
            prompt_ids = self.tokenizer.encode(prompts[i], add_special_tokens=False)[:max_txt_len]
            prompt_embeds = embed_fn(torch.tensor(prompt_ids, device=self.device))
            seq = torch.cat([soft_tokens[i], prompt_embeds], dim=0)
            batch_inputs_embeds.append(seq)
            batch_attn.append([1] * seq.shape[0])

        # Padding
        max_len = max(e.shape[0] for e in batch_inputs_embeds)
        pad_embed = embed_fn(torch.tensor([self.tokenizer.pad_token_id], device=self.device))
        for i in range(B):
            pad_len = max_len - batch_inputs_embeds[i].shape[0]
            if pad_len > 0:
                batch_inputs_embeds[i] = torch.cat([pad_embed.repeat(pad_len, 1),
                                                     batch_inputs_embeds[i]])
                batch_attn[i] = [0]*pad_len + batch_attn[i]

        inputs_embeds = torch.stack(batch_inputs_embeds)
        attention_mask = torch.tensor(batch_attn, device=self.device)

        outputs = self.llm(
            inputs_embeds=inputs_embeds.to(self.llm.dtype),
            attention_mask=attention_mask,
            return_dict=True, use_cache=False,
        )

        last_logits = outputs.logits[:, -1, :]  # [B, vocab]
        enz_logit = last_logits[:, self._enzyme_first_token]
        non_logit = last_logits[:, self._non_enzyme_first_token]
        binary_logits = torch.stack([non_logit, enz_logit], dim=-1)  # [B, 2]: [Non, Enz]
        probs = F.softmax(binary_logits.float(), dim=-1)

        results = []
        for i in range(B):
            p_enz = probs[i, 1].item()
            p_non = probs[i, 0].item()
            pred = 1 if p_enz > p_non else 0
            results.append({
                'prediction': pred,
                'prob_enzyme': p_enz,
                'prob_non_enzyme': p_non,
                'prob_1': p_enz,  # 兼容 V1 评估
                'confidence': max(p_enz, p_non),
                'method': 'logits_v2',
            })
        return results

    # ----------------------------------------------------------
    # 资源释放
    # ----------------------------------------------------------
    def release(self):
        """显式释放 GPU 显存。"""
        if hasattr(self, 'llm') and self.llm is not None:
            del self.llm
        torch.cuda.empty_cache()
        print("[LLM] GPU memory released.")

    def load_checkpoint(self, path: str) -> bool:
        if not Path(path).exists():
            return False
        state = torch.load(path, map_location='cpu')
        self.gnn.load_state_dict(state['gnn'])
        self.connector.load_state_dict(state['connector'])
        self.projector.load_state_dict(state['projector'])
        self.gnn.to(self.device)
        self.connector.to(self.device)
        self.projector.to(self.device)
        print(f"[Checkpoint] Loaded from {path}")
        return True
