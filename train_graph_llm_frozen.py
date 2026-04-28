"""
train_graph_llm_frozen.py

核心改动（相比 train_graph_llm.py）：
  1. LLM 完全冻结，不使用 LoRA
  2. 提示词中插入 <graph_token> 占位符
  3. GNN Encoder + Graph Projector 将图嵌入投影到 LLM 词嵌入空间
  4. 前向时用图嵌入替换 <graph_token> 对应位置的词嵌入

训练参数：
  - 只有 GNN Encoder 和 Graph Projector 参与梯度更新
  - 优化器只包含这两部分参数
"""

import json
import pathlib
import pickle
import torch
import torch.nn as nn
import os
import copy
import gc
import warnings
from tqdm import tqdm
from pathlib import Path
from typing import List, Optional

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    default_data_collator,
)
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader as GeoDataLoader

warnings.filterwarnings("ignore")

# ============================================================
# 路径配置（按需修改）
# ============================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# ============================================================
# 超参数（可用 argparse 替换）
# ============================================================
LLM_NAME_OR_PATH = "meta-llama/Llama-2-7b-hf"   # 或本地路径
GNN_HIDDEN_DIM   = 256        # GNN 隐层维度
GNN_NUM_LAYERS   = 4          # GNN 层数
GNN_EPOCHS       = 30         # GNN 预训练轮数
GNN_BATCH_SIZE   = 32
GRAPH_TOKEN_NUM  = 8          # 每张图注入的 token 数（soft tokens 数量）
PROJ_LR          = 3e-4       # Projector 学习率
GNN_LR           = 1e-4       # GNN Encoder 学习率
WEIGHT_DECAY     = 1e-5
TRAIN_EPOCHS     = 10
BATCH_SIZE       = 4
GRAD_ACCUM_STEPS = 4
MAX_NEW_TOKENS   = 32
SEED             = 42

GRAPH_TOKEN      = "<graph_token>"   # 提示词中的占位符


# ============================================================
# GNN Encoder（GIN）
# ============================================================
from torch_geometric.nn import GINConv, global_mean_pool
import torch.nn.functional as F


class GINEncoder(nn.Module):
    def __init__(self, num_node_features: int, hidden_dim: int = 256,
                 num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
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
        self.hidden_dim = hidden_dim

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
        return h   # [B, hidden_dim]


# ============================================================
# Graph Projector：将 GNN 嵌入 → LLM 词嵌入空间
# 输出 shape: [B, graph_token_num, llm_embed_dim]
# ============================================================
class GraphProjector(nn.Module):
    def __init__(self, gnn_dim: int, llm_dim: int, num_tokens: int = 8):
        super().__init__()
        self.num_tokens = num_tokens
        self.proj = nn.Sequential(
            nn.Linear(gnn_dim, llm_dim * 2),
            nn.GELU(),
            nn.Linear(llm_dim * 2, llm_dim * num_tokens),
        )
        self.llm_dim = llm_dim

    def forward(self, graph_emb):
        """
        graph_emb: [B, gnn_dim]
        return:    [B, num_tokens, llm_dim]
        """
        B = graph_emb.size(0)
        out = self.proj(graph_emb)                   # [B, llm_dim * num_tokens]
        out = out.view(B, self.num_tokens, self.llm_dim)
        return out


# ============================================================
# 构建含 <graph_token> 的提示词
# ============================================================
def build_prompt(graph_info: dict, graph_token: str = GRAPH_TOKEN,
                 num_tokens: int = GRAPH_TOKEN_NUM,
                 task_desc: str = "graph classification") -> str:
    """
    在提示词开头插入 num_tokens 个 <graph_token> 作为图表示的 soft prefix。
    """
    tokens_str = " ".join([graph_token] * num_tokens)
    prompt = (
        f"Graph representation: {tokens_str}\n"
        f"Task: {task_desc}\n"
        f"Graph statistics — "
        f"nodes={graph_info.get('num_nodes', '?')}, "
        f"edges={graph_info.get('num_edges', '?')}, "
        f"avg_degree={graph_info.get('avg_degree', '?'):.2f}.\n"
        f"Based on the graph representation above, predict the label (0 or 1).\n"
        f"Answer:"
    )
    return prompt


# ============================================================
# 数据准备工具
# ============================================================
def load_tu_dataset(name: str, root: str = "data") -> TUDataset:
    from dataset.mol_graph_utils import ensure_node_features
    ds = TUDataset(root=root, name=name, use_node_attr=True)
    ensure_node_features(ds, name)
    return ds


def graph_info_from_data(data) -> dict:
    n = data.num_nodes
    e = data.num_edges
    avg_deg = (2 * e / n) if n > 0 else 0
    return {"num_nodes": n, "num_edges": e, "avg_degree": avg_deg}


# ============================================================
# 图 Token 注入：替换 input_embeds 中的 <graph_token> 位置
# ============================================================
def inject_graph_tokens(
    input_embeds: torch.Tensor,          # [B, seq_len, llm_dim]
    graph_soft_tokens: torch.Tensor,     # [B, num_tokens, llm_dim]
    token_ids: torch.Tensor,             # [B, seq_len]  input_ids
    graph_token_id: int,
    num_graph_tokens: int,
) -> torch.Tensor:
    """
    将 input_embeds 中连续 num_graph_tokens 个 graph_token_id 位置
    替换为 graph_soft_tokens。
    """
    B, seq_len, dim = input_embeds.shape
    out = input_embeds.clone()
    for b in range(B):
        positions = (token_ids[b] == graph_token_id).nonzero(as_tuple=True)[0]
        if len(positions) >= num_graph_tokens:
            positions = positions[:num_graph_tokens]
        for k, pos in enumerate(positions):
            if k < num_graph_tokens:
                out[b, pos, :] = graph_soft_tokens[b, k, :]
    return out


# ============================================================
# 主体：GraphFrozenLLMTrainer
# ============================================================
class GraphFrozenLLMTrainer:
    """
    冻结 LLM，只训练 GNN Encoder + Graph Projector。
    提示词中包含 <graph_token>，前向时用图嵌入替换对应位置。
    """

    def __init__(
        self,
        llm_name_or_path: str = LLM_NAME_OR_PATH,
        gnn_hidden_dim:   int  = GNN_HIDDEN_DIM,
        gnn_num_layers:   int  = GNN_NUM_LAYERS,
        num_graph_tokens: int  = GRAPH_TOKEN_NUM,
        device: str = "auto",
    ):
        self.num_graph_tokens = num_graph_tokens
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
                      if device == "auto" else torch.device(device)

        # ---- LLM & Tokenizer ----
        print(f"[LLM] Loading {llm_name_or_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name_or_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # 注册 <graph_token> 特殊 token
        self.tokenizer.add_tokens([GRAPH_TOKEN], special_tokens=True)
        self.graph_token_id = self.tokenizer.convert_tokens_to_ids(GRAPH_TOKEN)

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name_or_path,
            torch_dtype=torch.bfloat16,
        )
        # 扩展 embedding 层（因为加了新 token）
        self.llm.resize_token_embeddings(len(self.tokenizer))

        # ---- 冻结 LLM 全部参数 ----
        for param in self.llm.parameters():
            param.requires_grad = False
        print("[LLM] All LLM parameters FROZEN.")

        # ---- GNN Encoder（延迟初始化，需知道 num_node_features）----
        self.gnn_hidden_dim = gnn_hidden_dim
        self.gnn_num_layers = gnn_num_layers
        self.gnn: Optional[GINEncoder] = None

        # ---- Graph Projector ----
        llm_embed_dim = self.llm.config.hidden_size
        self.projector = GraphProjector(
            gnn_dim=gnn_hidden_dim,
            llm_dim=llm_embed_dim,
            num_tokens=num_graph_tokens,
        ).to(self.device)

        self.llm_embed_dim = llm_embed_dim

    def _init_gnn(self, num_node_features: int):
        self.gnn = GINEncoder(
            num_node_features=num_node_features,
            hidden_dim=self.gnn_hidden_dim,
            num_layers=self.gnn_num_layers,
        ).to(self.device)
        print(f"[GNN] Initialized: node_feat={num_node_features}, "
              f"hidden={self.gnn_hidden_dim}, layers={self.gnn_num_layers}")

    def trainable_parameters(self):
        """只返回 GNN + Projector 的参数（LLM 已冻结）。"""
        params = list(self.projector.parameters())
        if self.gnn is not None:
            params += list(self.gnn.parameters())
        return params

    def _encode_graph_batch(self, batch_pyg) -> torch.Tensor:
        """
        将一个 PyG batch 编码为图嵌入。
        Returns: [B, gnn_hidden_dim]
        """
        x = batch_pyg.x.to(self.device)
        edge_index = batch_pyg.edge_index.to(self.device)
        batch_vec  = batch_pyg.batch.to(self.device)
        return self.gnn(x, edge_index, batch_vec)   # [B, hidden_dim]

    def _get_soft_tokens(self, graph_emb: torch.Tensor) -> torch.Tensor:
        """graph_emb [B, hidden] → soft_tokens [B, num_tokens, llm_dim]"""
        return self.projector(graph_emb.to(self.device))

    def forward_with_graph_tokens(
        self,
        batch_pyg,
        prompts: List[str],
        labels_text: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        前向：编码图 → 注入 soft token → LLM cross-entropy loss

        Args:
            batch_pyg:    PyG Batch 对象
            prompts:      文本提示词列表（含 <graph_token>）
            labels_text:  标签文本（如 "0" / "1"），用于计算 loss
        Returns:
            loss scalar
        """
        B = batch_pyg.num_graphs

        # 1. 编码图
        graph_emb   = self._encode_graph_batch(batch_pyg)        # [B, hidden]
        soft_tokens = self._get_soft_tokens(graph_emb)            # [B, num_tok, llm_dim]

        # 2. Tokenize 提示词
        enc = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=256,
        )
        input_ids      = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        # 3. 获取 LLM 词嵌入
        embed_layer = self.llm.get_input_embeddings()
        input_embeds = embed_layer(input_ids)                     # [B, seq, dim]

        # 4. 注入 graph soft tokens
        input_embeds = inject_graph_tokens(
            input_embeds, soft_tokens,
            input_ids, self.graph_token_id,
            self.num_graph_tokens,
        )

        if labels_text is not None:
            # 拼接标签 token 计算 loss
            label_enc = self.tokenizer(
                labels_text, return_tensors="pt", padding=True,
                add_special_tokens=False,
            )
            label_ids  = label_enc["input_ids"].to(self.device)
            label_mask = label_enc["attention_mask"].to(self.device)

            label_embeds = embed_layer(label_ids)
            full_embeds  = torch.cat([input_embeds, label_embeds], dim=1)
            full_mask    = torch.cat([attention_mask, label_mask], dim=1)

            # labels：prompt 位置 = -100（忽略），label 位置 = 真实 token id
            ignore_len = input_ids.size(1)
            labels = torch.full(
                (B, full_embeds.size(1)), -100,
                dtype=torch.long, device=self.device,
            )
            labels[:, ignore_len:] = label_ids

            outputs = self.llm(
                inputs_embeds=full_embeds,
                attention_mask=full_mask,
                labels=labels,
            )
            return outputs.loss

        else:
            # 推理模式
            outputs = self.llm(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
            )
            return outputs

    @torch.no_grad()
    def generate(self, batch_pyg, prompts: List[str],
                 max_new_tokens: int = MAX_NEW_TOKENS) -> List[str]:
        """推理：生成预测结果。"""
        graph_emb   = self._encode_graph_batch(batch_pyg)
        soft_tokens = self._get_soft_tokens(graph_emb)

        enc = self.tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=256,
        )
        input_ids      = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        embed_layer  = self.llm.get_input_embeddings()
        input_embeds = embed_layer(input_ids)
        input_embeds = inject_graph_tokens(
            input_embeds, soft_tokens,
            input_ids, self.graph_token_id,
            self.num_graph_tokens,
        )

        out_ids = self.llm.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        return self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)


# ============================================================
# GNN 预训练（有监督图分类，为 Projector 提供好的初始特征）
# ============================================================
def pretrain_gnn(gnn: GINEncoder, dataset, epochs: int = GNN_EPOCHS,
                 batch_size: int = GNN_BATCH_SIZE, device=None, ckpt: str = None):
    """在源域数据集上有监督地预训练 GNN。"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classifier = nn.Linear(gnn.hidden_dim, 1).to(device)
    optimizer  = torch.optim.Adam(
        list(gnn.parameters()) + list(classifier.parameters()),
        lr=1e-3, weight_decay=1e-5,
    )
    criterion = nn.BCEWithLogitsLoss()
    loader    = GeoDataLoader(list(dataset), batch_size=batch_size, shuffle=True)

    gnn.train()
    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        total, cnt = 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            emb   = gnn(batch.x.float(), batch.edge_index, batch.batch)
            logit = classifier(emb).squeeze(-1)
            loss  = criterion(logit, batch.y.float())
            loss.backward()
            optimizer.step()
            total += loss.item() * batch.num_graphs
            cnt   += batch.num_graphs
        avg = total / max(cnt, 1)
        if avg < best_loss:
            best_loss = avg
        if epoch % 5 == 0:
            print(f"  [GNN pretrain] Epoch {epoch}/{epochs}  loss={avg:.4f}")

    gnn.eval()
    del classifier
    print(f"[GNN pretrain] Done. best_loss={best_loss:.4f}")
    if ckpt:
        Path(ckpt).parent.mkdir(parents=True, exist_ok=True)
        torch.save(gnn.state_dict(), ckpt)
        print(f"[GNN pretrain] Saved -> {ckpt}")


# ============================================================
# 训练主循环
# ============================================================
def train(
    source_dataset_name: str = "PROTEINS",
    target_dataset_name: str = "DD",
    llm_name_or_path:    str = LLM_NAME_OR_PATH,
    gnn_pretrain:        bool = True,
):
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 加载数据 ----
    from dataset.mol_graph_utils import (
        load_dataset, ensure_node_features, dataset_to_list, unify_feature_dim_lists,
    )
    src_ds = load_dataset(str(DATA_DIR), source_dataset_name)
    tgt_ds = load_dataset(str(DATA_DIR), target_dataset_name)
    ensure_node_features(src_ds, source_dataset_name)
    ensure_node_features(tgt_ds, target_dataset_name)

    src_list = dataset_to_list(src_ds)
    tgt_list = dataset_to_list(tgt_ds)
    unified_dim = unify_feature_dim_lists(src_list, tgt_list,
                                          source_dataset_name, target_dataset_name)

    # ---- 初始化 Trainer ----
    trainer = GraphFrozenLLMTrainer(
        llm_name_or_path=llm_name_or_path,
        gnn_hidden_dim=GNN_HIDDEN_DIM,
        gnn_num_layers=GNN_NUM_LAYERS,
        num_graph_tokens=GRAPH_TOKEN_NUM,
    )
    trainer._init_gnn(num_node_features=unified_dim)

    # ---- GNN 预训练（可选加载 checkpoint）----
    gnn_ckpt = str(CHECKPOINT_DIR / f"gnn_{source_dataset_name.lower()}.pt")
    if gnn_pretrain:
        if Path(gnn_ckpt).exists():
            trainer.gnn.load_state_dict(torch.load(gnn_ckpt, map_location=device))
            trainer.gnn.eval()
            print(f"[GNN] Loaded from {gnn_ckpt}")
        else:
            pretrain_gnn(trainer.gnn, src_list, device=device, ckpt=gnn_ckpt)

    # ---- 只优化 GNN + Projector ----
    optimizer = torch.optim.AdamW(
        [
            {"params": trainer.gnn.parameters(),       "lr": GNN_LR},
            {"params": trainer.projector.parameters(), "lr": PROJ_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    # 可训练参数统计
    trainable = sum(p.numel() for p in trainer.trainable_parameters() if p.requires_grad)
    total     = sum(p.numel() for p in trainer.llm.parameters()) + trainable
    print(f"[Params] Trainable: {trainable:,} / Total: {total:,} "
          f"({100*trainable/total:.2f}%)")

    # ---- 简单划分训练/验证集（按 8:2）----
    import random
    random.seed(SEED)
    indices = list(range(len(tgt_list)))
    random.shuffle(indices)
    n_train = int(0.8 * len(indices))
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:]
    train_data = [tgt_list[i] for i in train_idx]
    val_data   = [tgt_list[i] for i in val_idx]

    train_loader = GeoDataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = GeoDataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False)

    best_val_acc = -1.0
    best_state   = None

    # ---- 训练循环 ----
    for epoch in range(1, TRAIN_EPOCHS + 1):
        trainer.gnn.train()
        trainer.projector.train()

        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            # 构建提示词
            prompts = []
            labels_text = []
            for i in range(batch.num_graphs):
                info = graph_info_from_data(
                    type("D", (), {
                        "num_nodes": batch.ptr[i+1].item() - batch.ptr[i].item(),
                        "num_edges": (batch.batch == i).sum().item() * 2,
                        "avg_degree": 0,
                    })()
                )
                prompts.append(build_prompt(info))
                labels_text.append(str(int(batch.y[i].item())))

            loss = trainer.forward_with_graph_tokens(batch, prompts, labels_text)
            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainer.trainable_parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * GRAD_ACCUM_STEPS

        avg_loss = epoch_loss / len(train_loader)
        print(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f}")

        # ---- 验证 ----
        trainer.gnn.eval()
        trainer.projector.eval()
        correct, total_val = 0, 0

        for batch in val_loader:
            prompts = []
            for i in range(batch.num_graphs):
                info = {"num_nodes": 0, "num_edges": 0, "avg_degree": 0.0}
                prompts.append(build_prompt(info))

            preds = trainer.generate(batch, prompts)
            for pred_text, true_label in zip(preds, batch.y.tolist()):
                pred_digit = 1 if "1" in pred_text.split("Answer:")[-1][:5] else 0
                correct   += int(pred_digit == int(true_label))
                total_val += 1

        val_acc = correct / max(total_val, 1)
        print(f"[Epoch {epoch}] Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                "gnn": copy.deepcopy(trainer.gnn.state_dict()),
                "projector": copy.deepcopy(trainer.projector.state_dict()),
                "epoch": epoch,
            }
            print(f"  [Best] saved at epoch {epoch}, val_acc={val_acc:.4f}")

    # ---- 保存最佳 checkpoint ----
    if best_state:
        ckpt_path = CHECKPOINT_DIR / f"graph_frozen_llm_{source_dataset_name}_{target_dataset_name}.pt"
        torch.save(best_state, ckpt_path)
        print(f"[Done] Best model saved -> {ckpt_path}  (val_acc={best_val_acc:.4f})")

    return trainer, best_state


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    train(
        source_dataset_name="PROTEINS",
        target_dataset_name="DD",
        llm_name_or_path=LLM_NAME_OR_PATH,
        gnn_pretrain=True,
    )
