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
    def __init__(self, gnn_dim: int, llm_dim: int, num_tokens: int = 8):
        super().__init__()
        self.num_tokens = num_tokens
        self.llm_dim = llm_dim
        self.proj = nn.Sequential(
            nn.Linear(gnn_dim, llm_dim * 2),
            nn.GELU(),
            nn.Linear(llm_dim * 2, llm_dim * num_tokens),
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
        num_graph_tokens: int = 8,     # 每图注入的 soft token 数
        max_new_tokens: int = 32,      # 生成时最大新 token 数
        device: str = "auto",
        load_in_8bit: bool = False,    # 是否用 8-bit 量化节省显存
        modelscope_cache_dir: Optional[str] = None,
        modelscope_revision: str = "master",
    ):
        self.num_graph_tokens = num_graph_tokens
        self.max_new_tokens = max_new_tokens
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
            load_kwargs["load_in_8bit"] = True
            load_kwargs["device_map"] = {"": self.device.index if self.device.index is not None else 0}
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
    # 前向（训练用）：计算 cross-entropy loss
    # ----------------------------------------------------------
    def compute_loss(
        self,
        pyg_batch,
        prompts: List[str],
        labels_text: List[str],
    ) -> torch.Tensor:
        """
        Args:
            pyg_batch:    PyG Batch 对象（含 x, edge_index, batch, y）
            prompts:      含 <graph_token> 占位符的提示词列表
            labels_text:  标签文本列表（如 ["0", "1", ...]）
        Returns:
            loss scalar（只对标签位置计算）
        """
        B = pyg_batch.num_graphs

        # 1. GNN 编码 → soft tokens
        graph_emb = self._encode_graph(pyg_batch)          # [B, hidden]
        soft_tokens = self._get_soft_tokens(graph_emb)     # [B, num_tok, llm_dim]

        # 2. Tokenize 提示词（每次重新 tokenize，不缓存 — 每个 batch 的 prompt 不同，
        #    缓存只会导致 GPU 显存无限增长而永远不命中）
        enc = self.tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        # 3. 获取词嵌入并注入 soft tokens
        embed_fn = self.llm.get_input_embeddings()
        input_embeds = embed_fn(input_ids)                 # [B, seq, llm_dim]
        input_embeds = inject_graph_tokens(
            input_embeds, soft_tokens,
            input_ids, self.graph_token_id, self.num_graph_tokens,
        )

        # 4. Tokenize 标签 (A3: 缓存优化)
        # 预先 Tokenize 核心标签 "0" 和 "1"
        if "0" not in self._label_cache:
            for l_str in ["0", "1"]:
                l_enc = self.tokenizer(
                    l_str, return_tensors="pt", add_special_tokens=False
                )
                self._label_cache[l_str] = l_enc["input_ids"].to(self.device)
                # 记录 mask（全 1，因为标签通常是一个 token）
                self._label_cache[l_str + "_mask"] = l_enc["attention_mask"].to(self.device)

        # 构造 batch 的 label_ids 和 label_mask
        label_ids = torch.cat([self._label_cache[l] for l in labels_text], dim=0)
        label_mask = torch.cat([self._label_cache[l + "_mask"] for l in labels_text], dim=0)
        label_embeds = embed_fn(label_ids)

        # 5. 拼接 prompt + label
        full_embeds = torch.cat([input_embeds, label_embeds], dim=1)
        full_mask = torch.cat([attention_mask, label_mask], dim=1)

        # 6. 构造 labels（prompt 位置忽略，label 位置计算 loss）
        ignore = torch.full(
            (B, input_ids.size(1)), -100,
            dtype=torch.long, device=self.device,
        )
        labels = torch.cat([ignore, label_ids], dim=1)

        outputs = self.llm(
            inputs_embeds=full_embeds,
            attention_mask=full_mask,
            labels=labels,
        )
        return outputs.loss

    # ----------------------------------------------------------
    # 推理（generate）
    # ----------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        pyg_batch,
        prompts: List[str],
        max_new_tokens: Optional[int] = None,
    ) -> List[str]:
        """返回每个 prompt 的生成文本列表。"""
        max_new_tokens = max_new_tokens or self.max_new_tokens

        graph_emb = self._encode_graph(pyg_batch)
        soft_tokens = self._get_soft_tokens(graph_emb)

        enc = self.tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        embed_fn = self.llm.get_input_embeddings()
        input_embeds = embed_fn(input_ids)
        input_embeds = inject_graph_tokens(
            input_embeds, soft_tokens,
            input_ids, self.graph_token_id, self.num_graph_tokens,
        )

        out_ids = self.llm.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)

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
            # 解析 "Answer: 0/1"
            match = re.search(r'Answer\s*:\s*([01])', content, re.IGNORECASE)
            if match is not None:
                prediction = int(match.group(1))
                confidence = 1.0
            else:
                # 回退策略：搜索最后出现的 0 或 1
                digits = re.findall(r'[01]', content[-50:])
                if digits:
                    prediction = int(digits[-1])
                    confidence = 0.5
                    print(f"  [WARN] No 'Answer: X' found, fallback to last digit -> {prediction}  (response: {content[-60:]!r})")
                else:
                    # 最终回退：搜索整个文本中 '1' 和 '0' 出现次数
                    cnt_0 = content.lower().count('label 0') + content.lower().count('class 0')
                    cnt_1 = content.lower().count('label 1') + content.lower().count('class 1')
                    if cnt_1 > cnt_0:
                        prediction = 1
                        print(f"  [WARN] No digit found, inferred label=1 from keyword counts (label_1={cnt_1}, label_0={cnt_0})")
                    elif cnt_0 > cnt_1:
                        prediction = 0
                        print(f"  [WARN] No digit found, inferred label=0 from keyword counts (label_0={cnt_0}, label_1={cnt_1})")
                    else:
                        # 真的无法判断时，用随机预测避免系统性偏向 majority class
                        prediction = np.random.randint(0, 2)
                        print(f"  [WARN] Cannot determine label, random fallback -> {prediction}  (response: {content[:80]!r})")
                    confidence = 0.1
            
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
