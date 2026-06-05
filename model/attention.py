"""
GQA 分组查询注意力（Grouped-Query Attention）

核心思想：Q 有 num_heads 个头，但 K 和 V 只有 num_kv_heads 个头。
每个 KV 头被多个 Q 头共享，减少 KV Cache 的显存占用。

与标准 MHA 的区别：
  MHA:  Q heads=8, KV heads=8  → 每个头独立的 K,V
  GQA:  Q heads=8, KV heads=4  → 每 2 个 Q 头共享 1 个 KV 头
  MQA:  Q heads=8, KV heads=1  → 所有 Q 头共享 1 个 KV 头

KV Cache：
  推理时，缓存已经计算过的 K 和 V，避免对历史 token 重复计算。
  新 token 只需要计算自己的 K,V，拼接到缓存后面。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .rope import RotaryEmbedding, apply_rotary_pos_emb


class KVCache:
    """KV Cache：缓存历史 token 的 K 和 V

    推理时使用，训练时不需要。
    """

    def __init__(self):
        self.key_cache = []
        self.value_cache = []

    def update(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor):
        """将当前 token 的 K,V 追加到缓存

        参数：
            layer_idx: 层索引
            key: (batch, 1, num_kv_heads, head_dim) 当前 token 的 K
            value: (batch, 1, num_kv_heads, head_dim) 当前 token 的 V

        返回：
            full_key, full_value: 拼接后的完整 K,V（包括历史）

        逻辑：
        - 如果该层第一次访问，直接 append
        - 否则用 torch.cat 沿 seq 维度拼接
        """
        if layer_idx >= len(self.key_cache):
            self.key_cache.append(key)
            self.value_cache.append(value)
        else:
            self.key_cache[layer_idx] = torch.cat((self.key_cache[layer_idx], key), dim=1)
            self.value_cache[layer_idx] = torch.cat((self.value_cache[layer_idx], value), dim=1)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """获取缓存的序列长度"""
        if layer_idx >= len(self.key_cache):
            return 0
        return self.key_cache[layer_idx].shape[1]


class CausalSelfAttention(nn.Module):
    """GQA 因果自注意力

    关键点：
    - Q 有 num_heads 个头，K,V 只有 num_kv_heads 个头
    - 使用 RoPE 编码位置
    - 支持 KV Cache（推理加速）
    - 因果 mask：每个位置只能看到自己和前面的 token
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim  # = hidden_size // num_heads
        self.num_groups = config.num_heads // config.num_kv_heads  # 每组 Q 共享一个 KV

        self.q_proj = nn.Linear(config.hidden_size, config.num_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_heads * config.head_dim, config.hidden_size, bias=False)

        # RoPE 位置编码
        self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len, config.rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        kv_cache: KVCache = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """前向传播

        参数：
            x: (batch, seq_len, hidden_size)
            freqs: (seq_len, head_dim) RoPE 频率
            kv_cache: KVCache 实例（推理时传入，训练时为 None）
            layer_idx: 当前层索引

        完整流程（11 步）：
        1. 线性投影: x → Q, K, V
        2. reshape 成多头: (batch, seq, num_heads, head_dim)
        3. 对 Q, K 应用 RoPE
        4. KV Cache 更新（推理时）
        5. GQA: 把 KV 头 repeat 到和 Q 头数一样
        6. 转置: (batch, num_heads, seq, head_dim)
        7. 计算注意力分数: Q @ K^T / sqrt(head_dim)
        8. 因果 mask（上三角设为 -inf）
        9. softmax + 加权求和
        10. 转置回 (batch, seq, num_heads, head_dim)，合并头
        11. 输出投影
        """
        batch_size, seq_len, _ = x.shape

        # 步骤 1: 线性投影
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 步骤 2: reshape 成多头形式
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # 步骤 3: RoPE 位置编码
        q, k = apply_rotary_pos_emb(q, k, freqs)

        # 步骤 4: KV Cache
        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx, k, v)
        # 步骤 5: GQA 广播 — 把 KV 头复制到和 Q 头数一样
        if self.num_groups > 1:
            # unsqueeze(3) 在第 3 维插入一个维度
            k = k.unsqueeze(3)  # (1, 10, 4, 1, 64)
            # expand 复制 2 次
            k = k.expand(-1, -1, -1, self.num_groups, -1)  # (1, 10, 4, 2, 64)
            # reshape 把 4×2 合并成 8
            k = k.reshape(batch_size, -1, self.num_heads, self.head_dim)  # (1, 10, 8, 64)

            # v 同理
            v = v.unsqueeze(3).expand(-1, -1, -1, self.num_groups, -1)
            v = v.reshape(batch_size, -1, self.num_heads, self.head_dim)
        # 步骤 6: 转置 (batch, seq, heads, dim) → (batch, heads, seq, dim)
        # 为了方便做 Q @ K^T 矩阵乘法，heads 维要放到 seq 前面
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 步骤 7: 注意力分数 = Q @ K^T / √head_dim
        # 每个 Q 和每个 K 做点积，除以 √64 防止数值过大
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # 步骤 8: 因果 mask — 不能偷看未来的 token
        if seq_len > 1:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
            scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # 步骤 9: softmax 归一化 → 加权求和
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # 步骤 10-11: 转置回来 + 合并头 + 输出投影
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        return self.o_proj(attn_output)
