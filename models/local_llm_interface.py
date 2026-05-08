"""
models/local_llm_interface.py

基于 ModelScope 加载本地 LLM（如 LLM-Research/Meta-Llama-3.1-8B）。

核心设计：
  - LLM 参数完全冻结（requires_grad = False）
  - GNN Encoder + Graph Projector 参数可训练
  - 提示词中的 <graph_token> 占位符在前向时被 soft embedding 替换
  - 支持训练（compute_loss）和推理（predict / generate）两种模式

用法示例：
    interface = LocalLLMInterface(
        model_name_or_path="LLM-Research/Meta-Llama-3.1-8B",
        num_node_features=89,
        gnn_hidden_dim=128,
        num_graph_tokens=8,
    )
    interface.train_step(pyg_batch, prompts, labels_text)
    result = interface.predict(pyg_batch, prompt)
"""

import re
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Dict, Tuple
from pathlib import Path

from modelscope import snapshot_download, AutoModel, AutoTokenizer

try:
    from modelscope import AutoModelForCausalLM as ModelScopeAutoModelForCausalLM
except Exception:
    ModelScopeAutoModelForCausalLM = AutoModel
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool


# ============================================================
# 特殊 token 定义
# ============================================================
GRAPH_TOKEN = "<graph_token>"


# ============================================================
# GNN Encoder（GIN，与 gnn_encoder.py 独立，避免循环依赖）
# ============================================================
class GINEncoder(nn.Module):
    def __init__(self, num_node_features: int, hidden_dim: int = 128,
                 num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # ---- 可学习的输入特征投影（替代零填充直接进入 GINConv）----
        # 将任意维度的节点特征先映射到 hidden_dim，
        # 这样即使输入被零填充（如 4→89 维），网络也能学到只关注有意义的维度
        self.input_proj = nn.Sequential(
            nn.Linear(num_node_features, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            # 所有层都是 hidden_dim → hidden_dim（输入已被投影）
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = dropout
        # 多通道池化投影：concat(mean, max) → hidden_dim
        self.pool_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x, edge_index, batch=None):
        # 先投影特征到 hidden_dim，再做消息传递
        h = self.input_proj(x.float())
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        if batch is not None:
            h_mean = global_mean_pool(h, batch)   # [B, hidden_dim]
            h_max = global_max_pool(h, batch)     # [B, hidden_dim]
        else:
            h_mean = h.mean(dim=0, keepdim=True)
            h_max = h.max(dim=0, keepdim=True).values
        # 多通道池化 → 投影回 hidden_dim（保持下游接口不变）
        h_cat = torch.cat([h_mean, h_max], dim=-1)  # [B, 2*hidden_dim]
        return self.pool_proj(h_cat)  # [B, hidden_dim]


# ============================================================
# Gradient Reversal Layer（域对抗训练核心）
# ============================================================
class GradientReversal(torch.autograd.Function):
    """梯度反转层：前向不变，反向时反转梯度方向。"""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class DomainDiscriminator(nn.Module):
    """域判别器：判断 GNN embedding 来自源域还是目标域。"""
    def __init__(self, input_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x, alpha: float = 1.0):
        x_rev = GradientReversal.apply(x, alpha)
        return self.net(x_rev)


# ============================================================
# Graph Projector：GNN 嵌入 → LLM 词嵌入空间
# ============================================================
class GraphProjector(nn.Module):
    """
    残差式 Graph Projector — 解决冻结 LLM 无法理解 soft tokens 的问题。

    核心思路：
      soft_token = base_tokens + scale * delta(graph_emb)

      - base_tokens: 可学习参数，被 gen_loss 直接优化（类似 prompt tuning）
        → 会快速收敛到 LLM embedding 空间中合适的位置
      - delta: 由 GNN embedding 产生的小扰动
        → 提供图级别的区分能力

    这样 LLM 主要看到"它认识的" base_tokens（在其 embedding 空间内），
    delta 提供微小但关键的图特定偏移来影响分类结果。
    """
    def __init__(self, gnn_dim: int, llm_dim: int, num_tokens: int = 8,
                 delta_scale: float = 0.5):
        super().__init__()
        self.num_tokens = num_tokens
        self.llm_dim = llm_dim
        self.delta_scale = delta_scale

        # ---- 可学习的 base tokens（类似 prompt tuning）----
        # 这些参数被 gen_loss 直接优化，不需要穿过 Projector MLP
        self.base_tokens = nn.Parameter(
            torch.randn(num_tokens, llm_dim) * 0.02
        )

        # ---- 图特定的扰动网络（低秩分解 rank-4）----
        # rank-1 问题：32 个 token 共享一个方向，只有 1 维图信息
        # rank-4：4 个独立方向 + 逐 token 加权混合 → 多维图信息
        # 参数量：128→256→4*4096 + 128→32*4 ≈ 4.3M（原 33.5M 的 13%）
        self.delta_rank = 4
        self.delta_shared = nn.Sequential(
            nn.Linear(gnn_dim, 256),
            nn.GELU(),
            nn.Linear(256, llm_dim * self.delta_rank),  # 4 个独立方向 [4*llm_dim]
        )
        # 每个 token 对 4 个方向的混合权重
        self.delta_token_gate = nn.Sequential(
            nn.Linear(gnn_dim, num_tokens * self.delta_rank),
            nn.Tanh(),  # [-1, 1]
        )

        # ---- 直接分类头：绕过冻结 LLM 的梯度瓶颈 ----
        # delta 编码了图特定信息，直接用它做分类可以给 GNN 强梯度
        self.delta_classifier = nn.Sequential(
            nn.Linear(llm_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, graph_emb: torch.Tensor) -> torch.Tensor:
        """
        graph_emb: [B, gnn_dim]
        return:    [B, num_tokens, llm_dim]
        """
        B = graph_emb.size(0)

        # base: 所有图共享的基础 tokens，被 gen_loss 直接优化
        base = self.base_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, num_tok, llm_dim]

        # delta: rank-4 分解 — 4 个方向 + 逐 token 混合
        delta_dirs = self.delta_shared(graph_emb)  # [B, 4*llm_dim]
        delta_dirs = delta_dirs.view(B, self.delta_rank, self.llm_dim)  # [B, 4, llm_dim]
        token_gates = self.delta_token_gate(graph_emb)  # [B, num_tokens*4]
        token_gates = token_gates.view(B, self.num_tokens, self.delta_rank)  # [B, num_tok, 4]
        # [B, num_tok, 4] @ [B, 4, llm_dim] → [B, num_tok, llm_dim]
        delta = torch.bmm(token_gates, delta_dirs)

        # 残差组合：base（LLM 能理解）+ 小扰动（图特定信息）
        return base + self.delta_scale * delta




# ============================================================
# 主接口类
# ============================================================
class LocalLLMInterface:
    """
    基于 ModelScope 的本地 LLM 接口。

    LLM 参数完全冻结，只训练 GNN Encoder + Graph Projector + Classifier Head。
    Graph tokens 通过 concat 注入到序列头部（GraphPrompter 风格）。
    """

    def __init__(
        self,
        model_name_or_path: str,       # ModelScope 模型 ID 或本地路径
        num_node_features: int,        # GNN 输入节点特征维度
        gnn_hidden_dim: int = 128,     # GNN 隐层维度
        gnn_num_layers: int = 4,       # GNN 层数
        num_graph_tokens: int = 8,     # 每图注入的 soft token 数
        max_new_tokens: int = 16,      # 不需要思维链，只需 "Answer: 0/1"，16 token 足够
        device: str = "auto",
        load_in_8bit: bool = False,    # 是否用 8-bit 量化节省显存
        modelscope_cache_dir: Optional[str] = None,
        modelscope_revision: str = "master",
    ):
        self.num_graph_tokens = num_graph_tokens
        self.max_new_tokens = max_new_tokens
        self.max_txt_len = 512  # 参考 GraphPrompter 的 max_txt_len
        self.call_count = 0
        
        # 标签 token 缓存（只缓存 "0" 和 "1" 两个固定标签）
        self._label_cache = {}

        # ---- 设备 ----
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # ---- 解析模型目录：优先使用本地路径，否则从 ModelScope 下载 ----
        model_path = Path(model_name_or_path)
        if model_path.exists():
            self.model_dir = str(model_path)
            print(f"[LLM] Using local model dir: {self.model_dir}")
        else:
            if modelscope_cache_dir is None:
                project_root = Path(__file__).resolve().parents[1]
                modelscope_cache_dir = str(project_root / "checkpoints" / "modelscope")
            os.makedirs(modelscope_cache_dir, exist_ok=True)
            print(
                f"[LLM] Downloading from ModelScope: {model_name_or_path} "
                f"(cache_dir={modelscope_cache_dir}, revision={modelscope_revision})"
            )
            self.model_dir = snapshot_download(
                model_name_or_path,
                cache_dir=modelscope_cache_dir,
                revision=modelscope_revision,
                ignore_file_pattern=['original/*'],
            )

        # ---- Tokenizer ----
        print(f"[LLM] Loading tokenizer from: {self.model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # 注：<graph_token> 注册已废弃（改用 concat 注入），保留以兼容旧代码引用
        # self.tokenizer.add_tokens([GRAPH_TOKEN], special_tokens=True)

        # ---- LLM ----
        print(f"[LLM] Loading model from: {self.model_dir} to {self.device}")
        load_kwargs = dict(
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        if load_in_8bit:
            # 新版 transformers 废弃了 load_in_8bit，改用 BitsAndBytesConfig
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["device_map"] = {"": self.device.index if self.device.index is not None else 0}
            print("[LLM] 8-bit quantization enabled via BitsAndBytesConfig")
        else:
            # 显式指定显卡，防止 accelerate 误判到 CPU
            load_kwargs["device_map"] = {"": self.device.index if self.device.index is not None else 0}

        self.llm = ModelScopeAutoModelForCausalLM.from_pretrained(
            self.model_dir, **load_kwargs
        )
        
        # 确认位置
        actual_device = next(self.llm.parameters()).device
        print(f"[LLM] Model actually loaded on: {actual_device}")
        if actual_device.type == 'cpu' and self.device.type == 'cuda':
            print("[LLM] WARNING: Model is on CPU but CUDA is requested! Attempting force move...")
            self.llm = self.llm.to(self.device)


        # ---- 冻结 LLM 全部参数 ----
        for param in self.llm.parameters():
            param.requires_grad = False
        print("[LLM] All LLM parameters FROZEN.")

        # ---- 开启 Gradient Checkpointing，用时间换显存 ----
        self.llm.gradient_checkpointing_enable()
        print("[LLM] Gradient checkpointing ENABLED (saves ~60-70% activation memory).")

        llm_dim = self.llm.config.hidden_size

        # ---- GNN Encoder（可训练）----
        self.gnn = GINEncoder(
            num_node_features=num_node_features,
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_num_layers,
        ).to(self.device)

        # ---- Graph Projector（可训练）----
        self.projector = GraphProjector(
            gnn_dim=gnn_hidden_dim,
            llm_dim=llm_dim,
            num_tokens=num_graph_tokens,
        ).to(self.device)

        # ---- Domain Discriminator（域对抗训练）----
        self.domain_disc = DomainDiscriminator(
            input_dim=gnn_hidden_dim,
        ).to(self.device)

        embed_fn = self.llm.get_input_embeddings()

        # ---- 用有意义的文本 embedding 初始化 base_tokens ----
        # 让 base_tokens 从一个 LLM 已知的位置出发，而非随机噪声
        init_text = "The graph classification result based on structural analysis is"
        init_ids = self.tokenizer.encode(init_text, add_special_tokens=False)
        # 截取或重复以匹配 num_graph_tokens
        while len(init_ids) < num_graph_tokens:
            init_ids = init_ids + init_ids
        init_ids = init_ids[:num_graph_tokens]
        with torch.no_grad():
            init_embeds = embed_fn(torch.tensor(init_ids, device=self.device))
            self.projector.base_tokens.data.copy_(init_embeds.float())
        print(f"[Proj] base_tokens initialized from text: '{init_text}'")

        print(f"[GNN] node_feat={num_node_features}, hidden={gnn_hidden_dim}, "
              f"layers={gnn_num_layers}, graph_tokens={num_graph_tokens}")

        trainable = sum(p.numel() for p in self.trainable_parameters())
        total_llm = sum(p.numel() for p in self.llm.parameters())
        print(f"[Params] Trainable (GNN+Proj): {trainable:,} | "
              f"Frozen (LLM): {total_llm:,} | "
              f"Ratio: {100*trainable/(total_llm+trainable):.3f}%")

    # ----------------------------------------------------------
    # 可训练参数
    # ----------------------------------------------------------
    def trainable_parameters(self):
        return (list(self.gnn.parameters()) 
                + list(self.projector.parameters())
                + list(self.domain_disc.parameters()))

    # ----------------------------------------------------------
    # 图编码
    # ----------------------------------------------------------
    def _encode_graph(self, pyg_batch) -> torch.Tensor:
        """PyG batch → [B, gnn_hidden_dim]"""
        x = pyg_batch.x.to(self.device)
        edge_index = pyg_batch.edge_index.to(self.device)
        batch_vec = pyg_batch.batch.to(self.device)
        return self.gnn(x, edge_index, batch_vec)

    def _get_soft_tokens(self, graph_emb: torch.Tensor) -> torch.Tensor:
        """graph_emb [B, hidden] → [B, num_tokens, llm_dim]"""
        return self.projector(graph_emb)

    # ----------------------------------------------------------
    # Chat template 包装（Instruct 模型需要）
    # ----------------------------------------------------------
    def _wrap_chat_template(self, prompts: List[str]) -> List[str]:
        """用 Instruct 模型的 chat template 包装 prompt 文本。"""
        if not hasattr(self.tokenizer, 'apply_chat_template'):
            return prompts
        wrapped = []
        for p in prompts:
            try:
                text = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False, add_generation_prompt=True
                )
                wrapped.append(text)
            except Exception:
                wrapped.append(p)
        return wrapped

    # ----------------------------------------------------------
    # 前向（训练用）：计算 cross-entropy loss
    # GraphPrompter 风格: 手动拼接 [bos] + [graph] + [text] + [label+eos]
    # + Instruct 模型: 用 chat template 包装 prompt
    # ----------------------------------------------------------
    def compute_loss(
        self,
        pyg_batch,
        prompts: List[str],
        labels_text: List[str],

        **kwargs,
    ) -> Tuple[torch.Tensor, float]:
        """
        生成式 loss + embedding 对齐 loss：训练 GNN + Projector（残差式）。
        
        梯度流向：
          1. gen_loss → LLM(frozen) → soft_tokens(base+delta) → Projector → GNN
          2. align_loss → soft_tokens → Projector → GNN
        
        base_tokens 被 gen_loss 直接优化（类似 prompt tuning），
        delta_proj 提供图特定的扰动。
        
        Returns:
            (loss, accuracy): loss 用于反向传播，accuracy 用于监控训练效果
        """
        B = pyg_batch.num_graphs

        # 1. GNN 编码 → soft tokens
        graph_emb = self._encode_graph(pyg_batch)          # [B, hidden]
        soft_tokens = self._get_soft_tokens(graph_emb)     # [B, num_tok, llm_dim]

        # 2. 真实标签（用于 accuracy 计算）
        true_labels = [int(l) for l in labels_text]

        # 3. Embedding 对齐 loss（让 soft tokens 留在 LLM 能理解的空间内）
        #    核心思想：soft tokens 应靠近 LLM 词表中的真实 token embedding,
        #    这样冻结的 LLM 才能正确处理它们（而非当作噪声忽略）
        embed_fn = self.llm.get_input_embeddings()
        with torch.no_grad():
            # 随机采样 256 个真实 token embedding 作为锚点
            vocab_size = embed_fn.weight.shape[0]
            sample_ids = torch.randint(0, vocab_size, (256,), device=self.device)
            anchor_embeds = embed_fn(sample_ids)                       # [256, llm_dim]
            anchor_norm = F.normalize(anchor_embeds.float(), dim=-1)   # [256, llm_dim]

        # 计算每个 soft token 与最近真实 token 的余弦相似度
        soft_flat = soft_tokens.view(-1, soft_tokens.size(-1)).float()  # [B*num_tok, llm_dim]
        soft_norm = F.normalize(soft_flat, dim=-1)
        cos_sim = torch.mm(soft_norm, anchor_norm.t())    # [B*num_tok, 256]
        max_sim = cos_sim.max(dim=-1).values              # [B*num_tok]
        align_loss = (1.0 - max_sim).mean()               # 越接近真实 token 越好

        # 4. 用 chat template 包装后 tokenize
        wrapped = self._wrap_chat_template(prompts)
        text_enc = self.tokenizer(wrapped, add_special_tokens=False)
        label_enc = self.tokenizer(labels_text, add_special_tokens=False)

        # 5. 获取特殊 token 的 embedding
        #    注意：不再手动添加 BOS，因为 apply_chat_template 已在文本开头
        #    包含 <|begin_of_text|>，tokenize 后第一个 token 就是 BOS
        pad_embed = embed_fn(torch.tensor([self.tokenizer.pad_token_id], device=self.device))
        soft_tokens = soft_tokens.to(pad_embed.dtype)

        # 6. 逐样本手动拼接 embeddings: [soft_tokens] + [text(含BOS)]
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_ids = []

        for i in range(B):
            label_ids_i = label_enc.input_ids[i] + [self.tokenizer.eos_token_id]
            text_ids_i = text_enc.input_ids[i][:self.max_txt_len] + label_ids_i
            text_embeds = embed_fn(torch.tensor(text_ids_i, device=self.device))
            seq_embeds = torch.cat([soft_tokens[i], text_embeds], dim=0)

            batch_inputs_embeds.append(seq_embeds)
            batch_attention_mask.append([1] * seq_embeds.shape[0])

            n_ignore = seq_embeds.shape[0] - len(label_ids_i)
            label_for_loss = [-100] * n_ignore + label_ids_i
            batch_label_ids.append(label_for_loss)

        # 7. 左侧 padding 对齐
        max_length = max(x.shape[0] for x in batch_inputs_embeds)
        for i in range(B):
            pad_len = max_length - batch_inputs_embeds[i].shape[0]
            if pad_len > 0:
                batch_inputs_embeds[i] = torch.cat([pad_embed.repeat(pad_len, 1), batch_inputs_embeds[i]])
                batch_attention_mask[i] = [0] * pad_len + batch_attention_mask[i]
                batch_label_ids[i] = [-100] * pad_len + batch_label_ids[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0)
        attention_mask = torch.tensor(batch_attention_mask, device=self.device)
        label_ids = torch.tensor(batch_label_ids, device=self.device)

        # 8. LLM 前向
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=label_ids,
            return_dict=True,
            use_cache=False,
        )

        # 9. 从 LLM logits 提取 accuracy（零额外计算开销）
        token_0_id = self.tokenizer.encode("0", add_special_tokens=False)[0]
        token_1_id = self.tokenizer.encode("1", add_special_tokens=False)[0]
        correct = 0
        for i in range(B):
            label_positions = (label_ids[i] != -100).nonzero(as_tuple=True)[0]
            if len(label_positions) > 0:
                pred_pos = label_positions[0] - 1  # LLM 在此位置预测下一个 token
                binary_logit = outputs.logits[i, pred_pos, [token_0_id, token_1_id]]
                pred = binary_logit.argmax().item()
                if pred == true_labels[i]:
                    correct += 1
        accuracy = correct / max(B, 1)

        # 10. 直接分类损失：绕过冻结 LLM，给 GNN+delta 直接的分类梯度
        #     这是最强的训练信号，不经过 LLM 的任何层
        delta_only = soft_tokens - self.projector.base_tokens.unsqueeze(0).to(soft_tokens.dtype)
        delta_pooled = delta_only.mean(dim=1)  # [B, llm_dim]
        delta_logits = self.projector.delta_classifier(delta_pooled.float())  # [B, 1]
        true_labels_tensor = torch.tensor(true_labels, dtype=torch.float32, device=self.device)
        cls_loss = F.binary_cross_entropy_with_logits(
            delta_logits.squeeze(-1), true_labels_tensor
        )

        # 11. delta 对比损失（margin-based，不依赖大 batch）
        delta_norm = F.normalize(delta_pooled.float(), dim=-1)  # [B, llm_dim]
        sim_matrix = torch.mm(delta_norm, delta_norm.t())  # [B, B]
        label_eq = (true_labels_tensor.unsqueeze(0) == true_labels_tensor.unsqueeze(1)).float()
        mask = 1.0 - torch.eye(B, device=self.device)

        margin_pos, margin_neg = 0.5, -0.1
        pos_loss = ((margin_pos - sim_matrix) * label_eq * mask).clamp(min=0)
        neg_loss = ((sim_matrix - margin_neg) * (1 - label_eq) * mask).clamp(min=0)
        n_pairs = mask.sum().clamp(min=1)
        contrastive_loss = (pos_loss.sum() + neg_loss.sum()) / n_pairs

        # 12. 四重 loss：
        #   gen_loss         — LLM 生成 loss（主训练信号，穿过冻结 LLM）
        #   align_loss       — embedding 对齐（保持 soft tokens 在 LLM 空间内）
        #   cls_loss         — 直接分类（绕过 LLM，最强梯度信号）
        #   contrastive_loss — delta 区分度
        gen_loss = outputs.loss
        total_loss = (gen_loss
                      + 0.5 * align_loss
                      + 0.1 * cls_loss
                      + 0.1 * contrastive_loss)

        return total_loss, accuracy

    # ----------------------------------------------------------
    # 推理（generate）
    # 参考 GraphPrompter: [bos] + [graph] + [text] → generate → 完整 decode
    # ----------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        pyg_batch,
        prompts: List[str],
        max_new_tokens: Optional[int] = None,
    ) -> List[str]:
        """
        GraphPrompter 风格的推理。
        手动拼接 [BOS] + [graph_tokens] + [text_tokens]，
        直接用 inputs_embeds 调用 generate()，
        decode 完整输出并返回（包含 prompt 部分，在 predict_batch 中 regex 提取）。
        """
        max_new_tokens = max_new_tokens or self.max_new_tokens

        graph_emb = self._encode_graph(pyg_batch)
        soft_tokens = self._get_soft_tokens(graph_emb)

        # 用 chat template 包装后 tokenize
        wrapped = self._wrap_chat_template(prompts)
        text_enc = self.tokenizer(wrapped, add_special_tokens=False)

        embed_fn = self.llm.get_input_embeddings()
        pad_embed = embed_fn(torch.tensor([self.tokenizer.pad_token_id], device=self.device))
        soft_tokens = soft_tokens.to(pad_embed.dtype)

        B = len(prompts)
        batch_inputs_embeds = []
        batch_attention_mask = []

        for i in range(B):
            text_ids_i = text_enc.input_ids[i][:self.max_txt_len]
            text_embeds = embed_fn(torch.tensor(text_ids_i, device=self.device))
            # [graph_tokens] + [text_tokens(含BOS)]
            seq_embeds = torch.cat([soft_tokens[i], text_embeds], dim=0)
            batch_inputs_embeds.append(seq_embeds)
            batch_attention_mask.append([1] * seq_embeds.shape[0])

        # 左侧 padding
        max_length = max(x.shape[0] for x in batch_inputs_embeds)
        for i in range(B):
            pad_len = max_length - batch_inputs_embeds[i].shape[0]
            if pad_len > 0:
                batch_inputs_embeds[i] = torch.cat([pad_embed.repeat(pad_len, 1), batch_inputs_embeds[i]])
                batch_attention_mask[i] = [0] * pad_len + batch_attention_mask[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0)
        attention_mask = torch.tensor(batch_attention_mask, device=self.device)

        # generate（直接用 inputs_embeds，参考 GraphPrompter）
        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            attention_mask=attention_mask,
            do_sample=False,
            use_cache=True,
        )

        # decode 完整输出（包含 prompt 部分），在 predict_batch 中用 regex 提取
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return pred

    # ----------------------------------------------------------
    # predict（支持 batch）
    # ----------------------------------------------------------
    @torch.no_grad()
    def predict_batch(self, pyg_batch, prompts: List[str]) -> List[Dict]:
        """
        批量推理，返回结果字典列表。
        """
        self.call_count += len(prompts)
        results = self.generate(pyg_batch, prompts)
        
        batch_results = []
        for content in results:
            content = content.strip()
            
            # inputs_embeds 模式下，generate() 只输出新生成的 token（不含 prompt）
            # 所以输出通常就是 "0" 或 "1"（可能带空格或换行）
            
            # 策略 1：直接匹配输出开头的 0 或 1（最高置信度）
            head_match = re.match(r'\s*([01])\b', content)
            if head_match:
                prediction = int(head_match.group(1))
                confidence = 1.0
            # 策略 2：如果输出包含 "Answer: X"（兼容完整 decode 的情况）
            elif re.search(r'Answer\s*:\s*([01])', content, re.IGNORECASE):
                matches = re.findall(r'Answer\s*:\s*([01])', content, re.IGNORECASE)
                prediction = int(matches[-1])
                confidence = 1.0
            else:
                # 策略 3：在整个输出中找任意 0/1
                all_digits = re.findall(r'[01]', content)
                if all_digits:
                    prediction = int(all_digits[0])  # 取第一个
                    confidence = 0.5
                    print(f"  [WARN] Unexpected output format, fallback -> {prediction}  (output: {content[:50]!r})")
                else:
                    prediction = np.random.randint(0, 2)
                    confidence = 0.1
                    print(f"  [WARN] No 0/1 found, random -> {prediction}  (output: {content[:50]!r})")
            
            batch_results.append({
                'prediction': prediction,
                'confidence': confidence,
                'response': content,
                'tokens_used': 0,
            })
        return batch_results



    # ----------------------------------------------------------
    # 方案3: LLM logits 分类（替代 generate）
    # ----------------------------------------------------------
    @torch.no_grad()
    def _get_llm_classification_logits(self, pyg_batch, prompts: List[str]) -> torch.Tensor:
        """
        用 LLM 的 next-token logits 做分类，不需要 generate。
        
        直接取 LLM 在序列最后一个位置对 "0" 和 "1" token 的 logits，
        比 generate + regex 更快、更精确。
        
        Returns:
            binary_logits: [B, 2]  对 "0" 和 "1" 两个 token 的 logits
        """
        graph_emb = self._encode_graph(pyg_batch)
        soft_tokens = self._get_soft_tokens(graph_emb)

        wrapped = self._wrap_chat_template(prompts)
        text_enc = self.tokenizer(wrapped, add_special_tokens=False)

        embed_fn = self.llm.get_input_embeddings()
        pad_embed = embed_fn(torch.tensor([self.tokenizer.pad_token_id], device=self.device))
        soft_tokens = soft_tokens.to(pad_embed.dtype)

        B = len(prompts)
        batch_inputs_embeds = []
        batch_attention_mask = []

        for i in range(B):
            text_ids_i = text_enc.input_ids[i][:self.max_txt_len]
            text_embeds = embed_fn(torch.tensor(text_ids_i, device=self.device))
            seq_embeds = torch.cat([soft_tokens[i], text_embeds], dim=0)
            batch_inputs_embeds.append(seq_embeds)
            batch_attention_mask.append([1] * seq_embeds.shape[0])

        max_length = max(x.shape[0] for x in batch_inputs_embeds)
        for i in range(B):
            pad_len = max_length - batch_inputs_embeds[i].shape[0]
            if pad_len > 0:
                batch_inputs_embeds[i] = torch.cat([pad_embed.repeat(pad_len, 1), batch_inputs_embeds[i]])
                batch_attention_mask[i] = [0] * pad_len + batch_attention_mask[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0)
        attention_mask = torch.tensor(batch_attention_mask, device=self.device)

        # LLM 前向（不 generate，只取 logits）
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=False,
        )

        # 取每个序列最后一个非 padding 位置的 logits
        last_logits = outputs.logits[:, -1, :]  # [B, vocab_size]

        # 获取 "0" 和 "1" 对应的 token ID
        token_0_id = self.tokenizer.encode("0", add_special_tokens=False)[0]
        token_1_id = self.tokenizer.encode("1", add_special_tokens=False)[0]

        binary_logits = last_logits[:, [token_0_id, token_1_id]]  # [B, 2]
        return binary_logits

    @torch.no_grad()
    def predict_with_llm_logits(self, pyg_batch, prompts: List[str]) -> List[Dict]:
        """
        用 LLM logits 做分类。
        delta_classifier 仅在训练时提供辅助梯度，推理不使用。
        """
        self.call_count += len(prompts)
        binary_logits = self._get_llm_classification_logits(pyg_batch, prompts)
        probs = F.softmax(binary_logits, dim=-1)  # [B, 2]
        preds = binary_logits.argmax(dim=-1).cpu().tolist()

        batch_results = []
        for i in range(len(prompts)):
            prob_0 = probs[i, 0].item()
            prob_1 = probs[i, 1].item()
            batch_results.append({
                'prediction': preds[i],
                'confidence': max(prob_0, prob_1),
                'response': str(preds[i]),
                'prob_0': prob_0,
                'prob_1': prob_1,
                'tokens_used': 0,
                'method': 'llm_logits',
            })
        return batch_results



    @torch.no_grad()
    def predict(self, pyg_batch, prompt: str) -> Dict:
        """单图推理（封装 predict_batch）"""
        return self.predict_batch(pyg_batch, [prompt])[0]

    # ----------------------------------------------------------
    # 资源释放（防止 CUDA OOM）
    # ----------------------------------------------------------
    def release(self):
        """显式释放 GPU 显存。在重新创建 LLM 前调用。
        注意：不使用 model.cpu()（8B 模型移到 CPU 可能导致系统卡死），
        直接 del 并清空 CUDA 缓存。
        """
        import gc
        if hasattr(self, 'llm') and self.llm is not None:
            del self.llm
            self.llm = None
        if hasattr(self, 'gnn') and self.gnn is not None:
            del self.gnn
            self.gnn = None
        if hasattr(self, 'projector') and self.projector is not None:
            del self.projector
            self.projector = None

        self._label_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[LLM] GPU memory released.")

    # ----------------------------------------------------------
    # check_status（兼容旧接口）
    # ----------------------------------------------------------
    def check_status(self) -> Dict:
        return {'available': True, 'calls': self.call_count, 'backend': 'modelscope'}

    # ----------------------------------------------------------
    # 保存 / 加载 GNN+Projector checkpoint
    # ----------------------------------------------------------
    def save_checkpoint(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'gnn': self.gnn.state_dict(),
            'projector': self.projector.state_dict(),
        }, path)
        print(f"[Checkpoint] Saved -> {path}")

    def load_checkpoint(self, path: str) -> bool:
        p = Path(path)
        if not p.exists():
            return False
        ckpt = torch.load(path, map_location=self.device)
        try:
            self.gnn.load_state_dict(ckpt['gnn'])
            self.projector.load_state_dict(ckpt['projector'])
            print(f"[Checkpoint] Loaded <- {path}")
            return True
        except RuntimeError as e:
            # 架构变更导致 state_dict 不匹配，删除旧 checkpoint 重新训练
            print(f"[Checkpoint] Architecture mismatch, deleting old checkpoint: {path}")
            print(f"  Detail: {e}")
            p.unlink()
            return False
