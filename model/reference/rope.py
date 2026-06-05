"""参考代码 - rope.py
RoPE (Rotary Position Embedding) — 用旋转矩阵编码位置信息

相比绝对位置编码的优势：
1. 能泛化到训练时没见过的更长序列
2. 注意力分数天然包含"相对位置"信息（旋转角度差 = 位置差）
3. 与 KV Cache 兼容（只需对新 token 做旋转，不需要重新计算旧 token）
"""

import torch
import torch.nn as nn


# ================================================================
# TODO 1: precompute_rope_frequencies — 预计算旋转频率
# ================================================================
def precompute_rope_frequencies(dim: int, max_seq_len: int, theta: float = 10000.0):
    """预计算每个位置的旋转频率（只算一次，缓存复用）

    核心公式: freq[i] = 1 / theta^(2i/dim)
    每个位置 m 的旋转角度 = m * freq[i]

    直觉: 低维（i 小）频率高 → 编码局部位置差异
          高维（i 大）频率低 → 编码全局位置差异
    类似傅里叶变换的多尺度位置编码。

    Args:
        dim: 每个 head 的维度（必须是偶数，因为要两两配对做旋转）
        max_seq_len: 支持的最大序列长度
        theta: 基础频率，越大能编码越长的序列（LLaMA 用 10000）

    Returns:
        freqs: shape (max_seq_len, dim//2)，每个位置的旋转频率
    """
    # 步骤 1: freq[i] = 1 / (theta ^ (2i / dim))
    #   arange(0, dim, 2) → [0, 2, 4, ..., dim-2]，只取偶数索引
    freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

    # 步骤 2: freqs[m, i] = m * freq[i]，用外积
    #   每行是一个位置，每列是一个频率分量
    m = torch.arange(max_seq_len).float()
    freqs = torch.outer(m, freq)

    return freqs


# ================================================================
# TODO 2: apply_rotary_pos_emb — 对 Q,K 应用旋转
# ================================================================
def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, freqs: torch.Tensor):
    """对 Q 和 K 向量应用旋转位置编码

    旋转操作将 dim 维向量视为 dim/2 个二维平面向量，每个平面旋转不同角度：
    [x1, x2] → [x1*cos(θ) - x2*sin(θ),  x1*sin(θ) + x2*cos(θ)]

    为什么只旋转 Q 和 K，不旋转 V？
    → 注意力分数 = Q·K^T，旋转后 Q_m·K_n 包含了 cos(m-n) 的信息，
      天然编码了相对位置。V 不需要位置信息。

    Args:
        q: Query 张量, shape (..., seq_len, head_dim)
        k: Key 张量, shape (..., seq_len, head_dim)
        freqs: 预计算的频率, shape (seq_len, head_dim//2)

    Returns:
        旋转后的 q 和 k（与输入同 dtype）
    """
    # 步骤 1: 转 float32（旋转需要精确计算）
    # 步骤 2: 拆成 (..., dim//2, 2) 对 — 每两个元素组成一个旋转平面
    q_pairs = q.float().reshape(*q.shape[:-1], -1, 2)
    k_pairs = k.float().reshape(*k.shape[:-1], -1, 2)

    # 步骤 3: 计算 cos, sin 并 unsqueeze 到 (1, seq_len, 1, dim//2)
    #   这样可以 broadcast 到 q_pairs 的 shape
    cos = freqs.cos().unsqueeze(0).unsqueeze(2)
    sin = freqs.sin().unsqueeze(0).unsqueeze(2)

    # 步骤 4: 二维旋转矩阵:
    #   x1' = x1*cos(θ) - x2*sin(θ)
    #   x2' = x1*sin(θ) + x2*cos(θ)
    q_rot = torch.zeros_like(q_pairs)
    q_rot[..., 0] = q_pairs[..., 0] * cos - q_pairs[..., 1] * sin
    q_rot[..., 1] = q_pairs[..., 0] * sin + q_pairs[..., 1] * cos

    k_rot = torch.zeros_like(k_pairs)
    k_rot[..., 0] = k_pairs[..., 0] * cos - k_pairs[..., 1] * sin
    k_rot[..., 1] = k_pairs[..., 0] * sin + k_pairs[..., 1] * cos

    # 步骤 5: 展平回原始 shape，并转回原始 dtype（如 bfloat16）
    return q_rot.flatten(-2).type_as(q), k_rot.flatten(-2).type_as(k)


# ================================================================
# TODO 3: RotaryEmbedding 类
# ================================================================
class RotaryEmbedding(nn.Module):
    """RoPE 频率的 Module 封装

    使用 register_buffer 确保频率随模型移动（to(device)）但不参与梯度计算。
    forward 只做切片，返回 [0:seq_len] 的频率，支持 KV Cache 场景下的偏移。
    """

    def __init__(self, dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        # register_buffer: 不是参数，但会随模型保存/加载，且跟随 .to(device) 移动
        self.register_buffer("freqs", precompute_rope_frequencies(dim, max_seq_len, theta))

    def forward(self, seq_len: int):
        """返回 [0:seq_len] 的频率子集

        在 KV Cache 场景中，传入 cache_len + new_seq_len 获取完整频率，
        然后在外部切片 freqs[cache_len:] 只取新 token 的频率。
        """
        return self.freqs[:seq_len]
