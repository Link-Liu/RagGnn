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
from typing import List, Optional, Dict
from pathlib import Path

from modelscope import snapshot_download, AutoModel, AutoTokenizer

try:
    from modelscope import AutoModelForCausalLM as ModelScopeAutoModelForCausalLM
except Exception:
    ModelScopeAutoModelForCausalLM = AutoModel
from torch_geometric.nn import GINConv, global_mean_pool


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
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            in_dim = num_node_features if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = dropout

    def forward(self, x, edge_index, batch=None):
        h = x.float()
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        if batch is not None:
            h = global_mean_pool(h, batch)
        else:
            h = h.mean(dim=0, keepdim=True)
        return h  # [B, hidden_dim]


# ============================================================
# Graph Projector：GNN 嵌入 → LLM 词嵌入空间
# ============================================================
class GraphProjector(nn.Module):
    """
    改进版 Projector，参考 LLaVA 的 MLP Projector。
    
    相比原版（Linear→Sigmoid→Linear）的改进：
      - GELU 替代 Sigmoid：避免梯度饱和，梯度信号更强
      - 加入 LayerNorm：稳定训练过程
      - 增加网络深度：更强的映射能力
    """
    def __init__(self, gnn_dim: int, llm_dim: int, num_tokens: int = 8):
        super().__init__()
        self.num_tokens = num_tokens
        self.llm_dim = llm_dim
        mid_dim = min(gnn_dim * 4, llm_dim)
        self.proj = nn.Sequential(
            nn.Linear(gnn_dim, mid_dim),
            nn.GELU(),
            nn.LayerNorm(mid_dim),
            nn.Linear(mid_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, llm_dim * num_tokens),
        )

    def forward(self, graph_emb: torch.Tensor) -> torch.Tensor:
        """
        graph_emb: [B, gnn_dim]
        return:    [B, num_tokens, llm_dim]
        """
        B = graph_emb.size(0)
        out = self.proj(graph_emb)
        return out.view(B, self.num_tokens, self.llm_dim)




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

        # ---- 分类头（辅助 loss，直接从 GNN embedding 做二分类）----
        # 解决“生成式 loss 信号穿不透冻结 LLM”的问题
        self.classifier_head = nn.Sequential(
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(gnn_hidden_dim, 2),
        ).to(self.device)

        print(f"[GNN] node_feat={num_node_features}, hidden={gnn_hidden_dim}, "
              f"layers={gnn_num_layers}, graph_tokens={num_graph_tokens}")

        trainable = sum(p.numel() for p in self.trainable_parameters())
        total_llm = sum(p.numel() for p in self.llm.parameters())
        print(f"[Params] Trainable (GNN+Proj+ClsHead): {trainable:,} | "
              f"Frozen (LLM): {total_llm:,} | "
              f"Ratio: {100*trainable/(total_llm+trainable):.3f}%")

    # ----------------------------------------------------------
    # 可训练参数
    # ----------------------------------------------------------
    def trainable_parameters(self):
        return (list(self.gnn.parameters()) 
                + list(self.projector.parameters())
                + list(self.classifier_head.parameters()))

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
        cls_weight: float = 0.3,
        gen_weight: float = 0.7,
        epoch: int = 0,
        total_epochs: int = 1,
    ) -> torch.Tensor:
        """
        混合 loss：分类头辅助 loss + 生成式 loss。
        
        核心设计：
          - 生成式 loss 是主信号（训练 Projector，让 LLM 理解 soft token）
          - 分类头 loss 是辅助信号（帮助 GNN 快速学到好的图表示）
          - 渐进退火：前期分类头权重较高帮 GNN 热身，后期生成式 loss 主导
        
        梯度流向：
          cls_loss → classifier_head → graph_emb → GNN  (不经过 Projector)
          gen_loss → LLM(frozen) → soft_tokens → Projector → graph_emb → GNN
        
        Args:
            cls_weight: 分类头 loss 的基础权重（会被退火调低）
            gen_weight: 生成式 loss 的基础权重（会被退火调高）
            epoch:        当前训练轮次（用于退火调度）
            total_epochs: 总训练轮次
        """
        B = pyg_batch.num_graphs

        # 1. GNN 编码 → soft tokens
        graph_emb = self._encode_graph(pyg_batch)          # [B, hidden]
        soft_tokens = self._get_soft_tokens(graph_emb)     # [B, num_tok, llm_dim]

        # ========== 分类头 loss（辅助信号，训练 GNN）==========
        cls_logits = self.classifier_head(graph_emb)       # [B, 2]
        cls_labels = torch.tensor(
            [int(l) for l in labels_text], device=self.device, dtype=torch.long
        )
        cls_loss = nn.functional.cross_entropy(cls_logits, cls_labels)

        # ========== 生成式 loss（主信号，训练 Projector + GNN）==========
        # 2. 用 chat template 包装后 tokenize
        wrapped = self._wrap_chat_template(prompts)
        text_enc = self.tokenizer(wrapped, add_special_tokens=False)
        label_enc = self.tokenizer(labels_text, add_special_tokens=False)

        # 3. 获取特殊 token 的 embedding
        embed_fn = self.llm.get_input_embeddings()
        bos_embed = embed_fn(torch.tensor([self.tokenizer.bos_token_id], device=self.device))
        pad_embed = embed_fn(torch.tensor([self.tokenizer.pad_token_id], device=self.device))
        soft_tokens = soft_tokens.to(bos_embed.dtype)

        # 4. 逐样本手动拼接 embeddings
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_ids = []

        for i in range(B):
            label_ids_i = label_enc.input_ids[i] + [self.tokenizer.eos_token_id]
            text_ids_i = text_enc.input_ids[i][:self.max_txt_len] + label_ids_i
            text_embeds = embed_fn(torch.tensor(text_ids_i, device=self.device))
            seq_embeds = torch.cat([bos_embed, soft_tokens[i], text_embeds], dim=0)

            batch_inputs_embeds.append(seq_embeds)
            batch_attention_mask.append([1] * seq_embeds.shape[0])

            n_ignore = seq_embeds.shape[0] - len(label_ids_i)
            label_for_loss = [-100] * n_ignore + label_ids_i
            batch_label_ids.append(label_for_loss)

        # 5. 左侧 padding 对齐
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

        # 6. LLM 前向
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=label_ids,
            return_dict=True,
            use_cache=False,
        )
        gen_loss = outputs.loss

        # ========== 渐进退火：分类头权重逐渐降低，生成式 loss 逐渐升高 ==========
        # 前期：cls 帮助 GNN 快速学到好的图表示
        # 后期：gen_loss 主导，让 Projector 充分学习 LLM 空间映射
        progress = epoch / max(total_epochs, 1)
        effective_cls_w = cls_weight * max(0.1, 1.0 - progress)   # 逐渐降到 cls_weight * 0.1
        effective_gen_w = gen_weight + cls_weight * min(0.9, progress)  # 逐渐吸收 cls 的权重

        total_loss = effective_cls_w * cls_loss + effective_gen_w * gen_loss
        return total_loss

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
        bos_embed = embed_fn(torch.tensor([self.tokenizer.bos_token_id], device=self.device))
        pad_embed = embed_fn(torch.tensor([self.tokenizer.pad_token_id], device=self.device))
        # 统一 dtype
        soft_tokens = soft_tokens.to(bos_embed.dtype)

        B = len(prompts)
        batch_inputs_embeds = []
        batch_attention_mask = []

        for i in range(B):
            text_ids_i = text_enc.input_ids[i][:self.max_txt_len]
            text_embeds = embed_fn(torch.tensor(text_ids_i, device=self.device))
            # [BOS] + [graph_tokens] + [text_tokens]
            seq_embeds = torch.cat([bos_embed, soft_tokens[i], text_embeds], dim=0)
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
    # 方案1: 分类头直接推理（训练-推理一致性修复）
    # ----------------------------------------------------------
    @torch.no_grad()
    def predict_with_classifier(self, pyg_batch) -> List[Dict]:
        """
        使用 classifier_head 直接预测，不经过 LLM generate。
        
        训练时 classifier_head 占 70% 的 loss 权重，
        因此它应该是最可靠的预测信号。
        """
        self.call_count += pyg_batch.num_graphs
        graph_emb = self._encode_graph(pyg_batch)              # [B, hidden]
        cls_logits = self.classifier_head(graph_emb)            # [B, 2]
        cls_probs = F.softmax(cls_logits, dim=-1)               # [B, 2]
        cls_preds = cls_logits.argmax(dim=-1).cpu().tolist()     # [B]
        
        batch_results = []
        for i in range(pyg_batch.num_graphs):
            prob_0 = cls_probs[i, 0].item()
            prob_1 = cls_probs[i, 1].item()
            pred = cls_preds[i]
            batch_results.append({
                'prediction': pred,
                'confidence': max(prob_0, prob_1),
                'response': str(pred),
                'prob_0': prob_0,
                'prob_1': prob_1,
                'tokens_used': 0,
                'method': 'classifier_head',
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
        bos_embed = embed_fn(torch.tensor([self.tokenizer.bos_token_id], device=self.device))
        pad_embed = embed_fn(torch.tensor([self.tokenizer.pad_token_id], device=self.device))
        soft_tokens = soft_tokens.to(bos_embed.dtype)

        B = len(prompts)
        batch_inputs_embeds = []
        batch_attention_mask = []

        for i in range(B):
            text_ids_i = text_enc.input_ids[i][:self.max_txt_len]
            text_embeds = embed_fn(torch.tensor(text_ids_i, device=self.device))
            seq_embeds = torch.cat([bos_embed, soft_tokens[i], text_embeds], dim=0)
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
        用 LLM logits 做分类（替代 generate + regex）。
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

    # ----------------------------------------------------------
    # 方案2: 集成推理（分类头 + LLM logits）
    # ----------------------------------------------------------
    @torch.no_grad()
    def predict_batch_ensemble(
        self, pyg_batch, prompts: List[str],
        cls_weight: float = 0.6, llm_weight: float = 0.4,
    ) -> List[Dict]:
        """
        集成推理：融合 classifier_head 和 LLM logits 的概率。
        
        Args:
            cls_weight: 分类头概率的权重
            llm_weight: LLM logits 概率的权重
        """
        self.call_count += len(prompts)

        # 路径 A: 分类头
        graph_emb = self._encode_graph(pyg_batch)
        cls_logits = self.classifier_head(graph_emb)       # [B, 2]
        cls_probs = F.softmax(cls_logits, dim=-1)           # [B, 2]

        # 路径 B: LLM logits
        llm_logits = self._get_llm_classification_logits(pyg_batch, prompts)  # [B, 2]
        llm_probs = F.softmax(llm_logits, dim=-1)           # [B, 2]

        # 加权融合
        final_probs = cls_weight * cls_probs + llm_weight * llm_probs  # [B, 2]
        preds = final_probs.argmax(dim=-1).cpu().tolist()

        batch_results = []
        for i in range(len(prompts)):
            batch_results.append({
                'prediction': preds[i],
                'confidence': final_probs[i].max().item(),
                'response': str(preds[i]),
                'prob_0': final_probs[i, 0].item(),
                'prob_1': final_probs[i, 1].item(),
                'cls_prob_1': cls_probs[i, 1].item(),
                'llm_prob_1': llm_probs[i, 1].item(),
                'tokens_used': 0,
                'method': 'ensemble',
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
        """显式释放 GPU 显存。在重新创建 LLM 前调用。"""
        import gc
        if hasattr(self, 'llm') and self.llm is not None:
            self.llm.cpu()
            del self.llm
            self.llm = None
        if hasattr(self, 'gnn') and self.gnn is not None:
            self.gnn.cpu()
            del self.gnn
            self.gnn = None
        if hasattr(self, 'projector') and self.projector is not None:
            self.projector.cpu()
            del self.projector
            self.projector = None
        if hasattr(self, 'classifier_head') and self.classifier_head is not None:
            self.classifier_head.cpu()
            del self.classifier_head
            self.classifier_head = None
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
            'classifier_head': self.classifier_head.state_dict(),
        }, path)
        print(f"[Checkpoint] Saved -> {path}")

    def load_checkpoint(self, path: str) -> bool:
        p = Path(path)
        if not p.exists():
            return False
        ckpt = torch.load(path, map_location=self.device)
        self.gnn.load_state_dict(ckpt['gnn'])
        self.projector.load_state_dict(ckpt['projector'])
        # 兼容旧 checkpoint（没有 classifier_head）
        if 'classifier_head' in ckpt:
            self.classifier_head.load_state_dict(ckpt['classifier_head'])
        print(f"[Checkpoint] Loaded <- {path}")
        return True
