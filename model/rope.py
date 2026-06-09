"""
RoPE（Rotary Position Embedding）旋转位置编码

核心思想：
  不像绝对位置编码加到输入上，RoPE 通过"旋转"Q 和 K 向量来编码位置信息。
  好处：天然支持相对位置，外推性好（训练 1024 长度，推理时可以更长）。

工作原理：
  1. 把 head_dim 维的向量分成 dim/2 个 2D 平面
  2. 每个平面上，位置 m 的向量旋转 m * θ_i 角度
  3. θ_i = 1 / (10000^(2i/dim))，不同维度用不同频率旋转

  两个向量的点积 = cos((m-n) * θ_i)，自然包含了相对位置 (m-n) 的信息。

★ 为什么多尺度频率？
  低维（i 小）频率高 → 捕捉相邻 token 的位置差异
  高维（i 大）频率低 → 捕捉远距离的位置差异
  类似傅里叶变换，用不同频率编码不同粒度的位置信息。

★ 为什么只旋转 Q 和 K，不旋转 V？
  注意力分数 = Q·K^T，旋转后 Q_m·K_n 包含了 cos(m-n) 的信息，
  天然编码了相对位置。V 不参与点积计算，不需要位置信息。

★ 为什么用 register_buffer？
  频率不参与训练（不需要梯度），但需要跟随模型保存/加载和 .to(device) 移动。
  register_buffer 正好满足这三个需求：不占参数、可序列化、可迁移设备。
"""

import torch
import torch.nn as nn


def precompute_rope_frequencies(dim: int, max_seq_len: int, theta: float = 10000.0):
    """预计算 RoPE 的旋转频率

    返回形状: (max_seq_len, dim // 2)

    步骤：
    1. 计算频率向量 freq: shape=(dim//2,)
       公式: freq[i] = 1 / (theta ^ (2i / dim))
       提示: 用 torch.arange(0, dim, 2) 生成 [0, 2, 4, ..., dim-2]
             然后计算 theta^(2i/dim) 的倒数

    2. 计算频率矩阵 freqs: shape=(max_seq_len, dim//2)
       公式: freqs[m, i] = m * freq[i]
       提示: 用 torch.outer(m, freq) 计算外积

    返回 freqs（后续会用 cos(freqs) 和 sin(freqs) 得到旋转矩阵）

    ★ 举例：dim=64, theta=10000
       arange(0, 64, 2) → [0, 2, 4, ..., 62]（共 32 个偶数索引）
       freq[i] = 1 / 10000^(2i/64)
       → i=0: freq=1.0（最高频，每步都转）
       → i=31: freq≈0.00158（最低频，很久才转一圈）
       freqs[m, i] = m * freq[i]：第 m 个位置在第 i 个频率上的旋转角度
    """
    # 步骤 1: freq[i] = 1 / (theta ^ (2i / dim))
    #   arange(0, dim, 2) → [0, 2, 4, ..., dim-2]，只取偶数索引
    freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

    # 步骤 2: freqs[m, i] = m * freq[i]，用外积
    #   每行是一个位置，每列是一个频率分量
    m = torch.arange(max_seq_len).float()
    freqs = torch.outer(m, freq)

    return freqs


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, freqs: torch.Tensor):
    """对 Q 和 K 应用旋转位置编码

    参数:
        q: (batch, seq_len, num_heads, head_dim)
        k: (batch, seq_len, num_kv_heads, head_dim)
        freqs: (seq_len, head_dim // 2) — 来自 precompute_rope_frequencies

    步骤：
    1. 把 q, k 转成 float32 避免 bf16 精度问题
       提示: q.float()

    2. 把最后一维拆成 (dim//2, 2) 的形式
       提示: q.reshape(*q.shape[:-1], -1, 2)

    3. 从 freqs 计算 cos 和 sin: shape=(seq_len, dim//2)
       提示: freqs.cos(), freqs.sin()
       然后 unsqueeze 到 (1, seq_len, 1, dim//2) 以便广播

    4. 旋转变换：
       q_rot[..., 0] = q_pairs[..., 0] * cos - q_pairs[..., 1] * sin
       q_rot[..., 1] = q_pairs[..., 0] * sin + q_pairs[..., 1] * cos
       k 同理

    5. 展平回原始形状: q_rot.flatten(-2)，转回原始 dtype

    返回旋转后的 q 和 k（形状不变）

    ★ 二维旋转矩阵的直觉：
       把 64 维向量看成 32 个二维平面向量，每个平面旋转不同角度：
       [x1, x2] → [x1*cos(θ) - x2*sin(θ),  x1*sin(θ) + x2*cos(θ)]
       32 个角度各不相同（来自不同的 freq[i]），实现多尺度位置编码。

    ★ 为什么要转 float32？
       bf16 只有 7 位有效数字，旋转涉及 cos/sin 乘法，精度不够会导致
       旋转后向量长度改变（理论上旋转不改变长度），累积误差影响训练稳定性。
    """
    # 步骤 1: 转 float32，拆成 (..., dim//2, 2) 对
    q_pairs = q.float().reshape(*q.shape[:-1], -1, 2)
    k_pairs = k.float().reshape(*k.shape[:-1], -1, 2)

    # 步骤 2: cos/sin 并 unsqueeze 到能 broadcast
    cos = freqs.cos().unsqueeze(0).unsqueeze(2)
    sin = freqs.sin().unsqueeze(0).unsqueeze(2)

    # 步骤 3: 二维旋转
    q_rot = torch.zeros_like(q_pairs)
    q_rot[..., 0] = q_pairs[..., 0] * cos - q_pairs[..., 1] * sin
    q_rot[..., 1] = q_pairs[..., 0] * sin + q_pairs[..., 1] * cos

    k_rot = torch.zeros_like(k_pairs)
    k_rot[..., 0] = k_pairs[..., 0] * cos - k_pairs[..., 1] * sin
    k_rot[..., 1] = k_pairs[..., 0] * sin + k_pairs[..., 1] * cos

    # 步骤 4: 展平回原始 shape，转回原始 dtype
    return q_rot.flatten(-2).type_as(q), k_rot.flatten(-2).type_as(k)


class RotaryEmbedding(nn.Module):
    """RoPE 模块：预计算频率并提供 apply 方法

    用法:
        rope = RotaryEmbedding(head_dim, max_seq_len)
        freqs = rope(seq_len)
        q_rotated, k_rotated = apply_rotary_pos_emb(q, k, freqs)

    ★ 为什么封装成 nn.Module？
       方便和其他组件统一管理：
       - self.rope 会跟随 model.to(device) 移动
       - 保存 checkpoint 时自动包含
       - forward 只做切片，支持 KV Cache 场景下的偏移
    """

    def __init__(self, dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        # register_buffer: 不是参数，但会随模型保存/加载，且跟随 .to(device) 移动
        self.register_buffer("freqs", precompute_rope_frequencies(dim, max_seq_len, theta))

    def forward(self, seq_len: int):
        """返回前 seq_len 个位置的频率

        ★ KV Cache 场景：
           prefill: rope(seq_len) 返回完整频率
           decode: rope(1) 只返回当前 token 的频率
           注意：decode 时需要考虑已缓存的长度偏移，
           这部分逻辑在 modeling_llm.py 的 forward 中处理。
        """
        return self.freqs[:seq_len]
