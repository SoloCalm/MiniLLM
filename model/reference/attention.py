"""参考代码 - attention.py
GQA (Grouped Query Attention) + KV Cache 实现

GQA 与 MHA 的区别：
- MHA: 每个 Q head 有自己的 K/V head（8Q + 8K + 8V）
- GQA: 多个 Q head 共享一组 K/V（8Q + 4K + 4V），节省 KV Cache 显存

KV Cache 的作用：
- 自回归生成时，每个新 token 只需要和之前的 K/V 做注意力
- 缓存之前的 K/V，避免重复计算（将 O(n²) 降到 O(n) 每步）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import ModelConfig
from .rope import RotaryEmbedding, apply_rotary_pos_emb


# ================================================================
# TODO 1-2: KVCache
# ================================================================
class KVCache:
    """KV 缓存：存储每层历史 token 的 K 和 V

    用两个 list（每层一个 entry）实现：
    - key_cache[layer_idx]: shape (batch, past_seq_len, kv_heads, head_dim)
    - value_cache[layer_idx]: shape 同上

    首次调用 update 时 append，之后 concat 新的 K/V。
    """

    def __init__(self):
        self.key_cache = []
        self.value_cache = []

    def update(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor):
        """更新指定层的 KV 缓存

        首次（prefill 阶段）：直接存储
        后续（decode 阶段）：沿 seq_len 维度拼接新 token 的 K/V

        Args:
            layer_idx: Transformer 层索引
            key: 新 token 的 K，shape (batch, new_seq_len, kv_heads, head_dim)
            value: 新 token 的 V，shape 同上

        Returns:
            完整的 (key, value)：包含历史 + 新 token
        """
        if layer_idx >= len(self.key_cache):
            # 首次：直接缓存（prefill，输入完整序列）
            self.key_cache.append(key)
            self.value_cache.append(value)
        else:
            # 后续：拼接（decode，每次只输入 1 个新 token）
            # dim=1 是 seq_len 维度
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key], dim=1)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value], dim=1)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """获取已缓存的序列长度（用于 RoPE 频率偏移）"""
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[1]


# ================================================================
# TODO 3-11: CausalSelfAttention
# ================================================================
class CausalSelfAttention(nn.Module):
    """带因果掩码的多头自注意力（GQA 版本）

    GQA 核心思想：
    - Q 有 num_heads 个投影（8 个）
    - K/V 只有 num_kv_heads 个投影（4 个）
    - 每组 Q head (2个) 共享 1 组 K/V head
    - 推理时 KV Cache 只存 kv_heads 份，显存减半

    forward 流程 11 步：
    1. 线性投影 (Q/K/V/O)
    2. Reshape 为多头
    3. 应用 RoPE 位置编码
    4. 更新 KV Cache
    5. GQA 广播（将 K/V 复制给共享的 Q 组）
    6. 转置为 (batch, heads, seq, dim)
    7. 计算注意力分数
    8. 应用因果掩码（上三角填 -inf）
    9. Softmax + 加权求和
    10. 合并多头
    11. 输出投影
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_heads      # Q 头数 (8)
        self.num_kv_heads = config.num_kv_heads # K/V 头数 (4)
        self.head_dim = config.head_dim        # 每头维度 (64)
        self.num_groups = config.num_heads // config.num_kv_heads  # Q/KV 组数 (2)

        # 四个线性投影，无 bias（现代 LLM 标准做法，减少参数）
        # Q 投影最大: hidden → heads * head_dim
        # K/V 投影较小: hidden → kv_heads * head_dim（GQA 的显存节省来源）
        self.q_proj = nn.Linear(config.hidden_size, config.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len, config.rope_theta)

    def forward(self, x, freqs, kv_cache=None, layer_idx=0):
        """
        Args:
            x: 输入, shape (batch, seq_len, hidden_size)
            freqs: RoPE 频率, shape (seq_len, head_dim//2)
            kv_cache: KV 缓存实例（prefill 阶段可为 None）
            layer_idx: 当前层索引（用于 KV Cache 定位）
        """
        batch_size, seq_len, _ = x.shape

        # 步骤 1: 线性投影
        #   (batch, seq, hidden) → (batch, seq, heads*head_dim)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 步骤 2: reshape 成多头
        #   (batch, seq, heads*head_dim) → (batch, seq, heads, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # 步骤 3: 对 Q 和 K 应用 RoPE（V 不旋转）
        q, k = apply_rotary_pos_emb(q, k, freqs)

        # 步骤 4: 更新 KV Cache
        #   prefill 阶段: 缓存整个序列的 K/V
        #   decode 阶段: 只缓存新 1 个 token 的 K/V，拼接到历史后面
        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx, k, v)

        # 步骤 5: GQA 广播 — 最关键的一步！
        #   如果 num_groups=2, kv_heads=4, heads=8:
        #   k 的 shape 从 (batch, seq, 4, dim) 扩展为 (batch, seq, 8, dim)
        #   具体做法：将第 3 维复制 num_groups 次，再 reshape
        if self.num_groups > 1:
            # unsqueeze(3) → (batch, seq, kv_heads, 1, dim)
            # expand → (batch, seq, kv_heads, num_groups, dim)
            # reshape → (batch, seq, kv_heads*num_groups, dim) = (batch, seq, heads, dim)
            k = k.unsqueeze(3).expand(-1, -1, -1, self.num_groups, -1)
            k = k.reshape(batch_size, -1, self.num_heads, self.head_dim)
            v = v.unsqueeze(3).expand(-1, -1, -1, self.num_groups, -1)
            v = v.reshape(batch_size, -1, self.num_heads, self.head_dim)

        # 步骤 6: 转置为 (batch, heads, seq, dim) — 方便做矩阵乘法
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 步骤 7: 计算注意力分数
        #   (batch, heads, seq_q, dim) @ (batch, heads, dim, seq_k)
        #   → (batch, heads, seq_q, seq_k) 每个 token 对其他 token 的注意力权重
        #   除以 sqrt(dim) 防止点积过大导致 softmax 饱和
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # 步骤 8: 因果掩码 — 保证每个 token 只能看到自己和之前的 token
        #   triu(diagonal=1) 生成上三角矩阵，True 的位置填 -inf
        #   只在 seq_len > 1 时应用（decode 阶段 seq_len=1，不需要 mask）
        if seq_len > 1:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
            scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # 步骤 9: Softmax 归一化 → 加权求和
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # 步骤 10-11: 合并多头 + 输出投影
        #   (batch, heads, seq, dim) → (batch, seq, heads*dim) → (batch, seq, hidden)
        attn_output = attn_output.transpose(1, 2).contiguous()  # contiguous 因为 transpose 后内存不连续
        attn_output = attn_output.view(batch_size, seq_len, -1)
        return self.o_proj(attn_output)
