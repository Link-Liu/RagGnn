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
    """参考 GraphPrompter 的 Projector 架构：Linear → Sigmoid → Linear"""
    def __init__(self, gnn_dim: int, llm_dim: int, num_tokens: int = 1):
        super().__init__()
        self.num_tokens = num_tokens
        self.llm_dim = llm_dim
        # GraphPrompter 风格: gnn_dim → mid → llm_dim * num_tokens
        mid_dim = min(gnn_dim * 4, llm_dim)  # 512*4=2048, cap at llm_dim
        self.proj = nn.Sequential(
            nn.Linear(gnn_dim, mid_dim),
            nn.Sigmoid(),
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
# Soft Token 注入
# ============================================================
def inject_graph_tokens(
    input_embeds: torch.Tensor,      # [B, seq_len, llm_dim]
    soft_tokens: torch.Tensor,       # [B, num_tokens, llm_dim]
    input_ids: torch.Tensor,         # [B, seq_len]
    graph_token_id: int,
    num_graph_tokens: int,
) -> torch.Tensor:
    """
    将 input_embeds 中前 num_graph_tokens 个 graph_token_id 位置
    替换为 soft_tokens。
    """
    B, L, D = input_embeds.shape
    
    # 找到所有 graph_token_id 的位置
    mask = (input_ids == graph_token_id)
    num_found = mask.sum().item()
    
    # 如果根本没找到占位符（例如在消融实验 No-RAG 模式下），直接返回原 embedding
    if num_found == 0:
        return input_embeds

    # 获取索引
    batch_idx, pos_idx = torch.where(mask)
    
    # 安全检查：如果找到的数量不是预期的 B * num_graph_tokens，
    # 说明某些 prompt 占位符不对，回退到循环处理以保证健壮性
    if num_found != B * num_graph_tokens:
        out = input_embeds.clone()
        # 确保类型一致
        soft_tokens = soft_tokens.to(out.dtype)
        for b in range(B):
            b_mask = (input_ids[b] == graph_token_id)
            b_pos = b_mask.nonzero(as_tuple=True)[0]
            # 取前 num_graph_tokens 个进行替换
            for k, pos in enumerate(b_pos[:num_graph_tokens]):
                out[b, pos, :] = soft_tokens[b, k, :]
        return out

    # 理想情况：使用向量化加速
    batch_idx = batch_idx.view(B, num_graph_tokens)
    pos_idx = pos_idx.view(B, num_graph_tokens)

    out = input_embeds.clone()
    # 核心修复：确保赋值时类型一致 (BFloat16 vs Float32)
    out[batch_idx, pos_idx] = soft_tokens.to(out.dtype)
    return out


# ============================================================
# 主接口类
# ============================================================
class LocalLLMInterface:
    """
    基于 ModelScope 的本地 LLM 接口。

    LLM 参数完全冻结，只训练 GNN Encoder + Graph Projector。
    提示词中的 <graph_token> 在前向时被 GNN soft embedding 替换。
    """

    def __init__(
        self,
        model_name_or_path: str,       # ModelScope 模型 ID 或本地路径
        num_node_features: int,        # GNN 输入节点特征维度
        gnn_hidden_dim: int = 128,     # GNN 隐层维度
        gnn_num_layers: int = 4,       # GNN 层数
        num_graph_tokens: int = 1,     # 参考 GraphPrompter，只需 1 个 soft token
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

        # 注册 <graph_token> 特殊 token
        self.tokenizer.add_tokens([GRAPH_TOKEN], special_tokens=True)
        self.graph_token_id = self.tokenizer.convert_tokens_to_ids(GRAPH_TOKEN)

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

        # 扩展 embedding 层（因为新增了 <graph_token>）
        if not hasattr(self.llm, "resize_token_embeddings"):
            raise RuntimeError("Loaded ModelScope model does not support token embedding resize.")
        self.llm.resize_token_embeddings(len(self.tokenizer))

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
        return list(self.gnn.parameters()) + list(self.projector.parameters())

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
    ) -> torch.Tensor:
        """
        融合 GraphPrompter 的 concat 注入 + Instruct 模型的 chat template。
        [BOS] + [graph_tokens] + [chat_template(prompt)] + [label + EOS]
        只对 label 位置计算 cross-entropy loss。
        """
        B = pyg_batch.num_graphs

        # 1. GNN 编码 → soft tokens
        graph_emb = self._encode_graph(pyg_batch)          # [B, hidden]
        soft_tokens = self._get_soft_tokens(graph_emb)     # [B, num_tok, llm_dim]

        # 2. 用 chat template 包装后 tokenize（不加 special tokens，BOS 手动加）
        wrapped = self._wrap_chat_template(prompts)
        text_enc = self.tokenizer(wrapped, add_special_tokens=False)
        label_enc = self.tokenizer(labels_text, add_special_tokens=False)

        # 3. 获取特殊 token 的 embedding
        embed_fn = self.llm.get_input_embeddings()
        bos_embed = embed_fn(torch.tensor([self.tokenizer.bos_token_id], device=self.device))  # [1, D]
        pad_embed = embed_fn(torch.tensor([self.tokenizer.pad_token_id], device=self.device))  # [1, D]

        # 4. 逐样本手动拼接 embeddings
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_ids = []

        for i in range(B):
            # label: label_tokens + EOS
            label_ids_i = label_enc.input_ids[i] + [self.tokenizer.eos_token_id]
            # text: prompt tokens + label tokens
            text_ids_i = text_enc.input_ids[i][:self.max_txt_len] + label_ids_i

            text_embeds = embed_fn(torch.tensor(text_ids_i, device=self.device))
            # 拼接: [BOS] + [graph_tokens] + [text + label]
            seq_embeds = torch.cat([bos_embed, soft_tokens[i], text_embeds], dim=0)

            batch_inputs_embeds.append(seq_embeds)
            batch_attention_mask.append([1] * seq_embeds.shape[0])

            # Label: prompt 部分用 -100 忽略，label 部分参与 loss
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
        return outputs.loss

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
            # 从完整输出中提取 "Answer: 0/1"（取最后一个匹配，因为完整输出包含 prompt）
            matches = re.findall(r'Answer\s*:\s*([01])', content, re.IGNORECASE)
            if matches:
                prediction = int(matches[-1])  # 取最后一个匹配（即模型生成的）
                confidence = 1.0
            else:
                # 取输出末尾的数字作为 fallback
                # 完整输出中末尾就是模型新生成的部分
                tail = content[-30:] if len(content) > 30 else content
                tail_digits = re.findall(r'[01]', tail)
                if tail_digits:
                    prediction = int(tail_digits[-1])
                    confidence = 0.5
                    print(f"  [WARN] No 'Answer: X' found, tail fallback -> {prediction}  (tail: {tail!r})")
                else:
                    prediction = np.random.randint(0, 2)
                    confidence = 0.1
                    print(f"  [WARN] Cannot determine label, random fallback -> {prediction}  (tail: {tail!r})")
            
            batch_results.append({
                'prediction': prediction,
                'confidence': confidence,
                'response': content,
                'tokens_used': 0,
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
        self.gnn.load_state_dict(ckpt['gnn'])
        self.projector.load_state_dict(ckpt['projector'])
        print(f"[Checkpoint] Loaded <- {path}")
        return True
