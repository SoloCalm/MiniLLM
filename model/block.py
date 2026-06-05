"""
Transformer Block — 将 Attention + FFN + RMSNorm 组装成一个完整的 Transformer 层

block.py 的作用：
  把之前写的三个独立模块（RMSNorm、CausalSelfAttention、SwiGLU）组装成一个可复用的层。
  模型有 12 层，每层都是一个 TransformerBlock，堆叠起来就是完整的 Transformer。

Pre-norm 结构（LLaMA2 风格）：
  x = x + Attention(RMSNorm(x))    # 先 norm，再 attention，再残差
  x = x + FFN(RMSNorm(x))          # 先 norm，再 FFN，再残差

为什么用 Pre-norm 而不是 Post-norm：
  - Pre-norm 训练更稳定，不容易梯度爆炸/消失
  - 不需要专门的学习率 warmup
  - 大多数现代 LLM 都用 Pre-norm
"""

import torch
import torch.nn as nn

from .config import ModelConfig
from .attention import CausalSelfAttention, KVCache
from .ffn import SwiGLU


class RMSNorm(nn.Module):
    """RMSNorm（Root Mean Square Layer Normalization）

    和 LayerNorm 的区别：
    - LayerNorm: (x - 均值) / 标准差 — 同时标准化位置和幅度
    - RMSNorm: x / sqrt(mean(x²)) — 只标准化幅度，不减均值

    公式：RMSNorm(x) = x / RMS(x) * weight
    其中 RMS(x) = sqrt(mean(x²) + eps)

    为什么更好：
    - 计算更快（省掉均值计算）
    - 效果和 LayerNorm 相当
    - LLaMA、Qwen、Gemma 都用 RMSNorm

    ★ 为什么需要归一化？
       Transformer 层很深（12层），不归一化的话每层输出的数值会越来越大或越来越小，
       导致训练不稳定。归一化让每层输出的幅度保持在合理范围。
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps  # 防止除零的小常数
        # 可学习的缩放参数，初始值全 1（让模型自己学是否需要缩放）
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        参数：x: (..., dim)
        返回：(..., dim)，形状不变

        ★ 为什么要转 float32？
           bf16 只有 7 位有效数字，计算 mean(x²) 时精度不够，
           累积误差会导致归一化结果不准确，影响训练稳定性。
        """
        # 步骤 1-2: 转 float32，计算 RMS = sqrt(mean(x²) + eps)
        rms = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        # 步骤 3-4: 归一化并缩放，转回原始 dtype
        return (x.float() / rms * self.weight).type_as(x)


class TransformerBlock(nn.Module):
    """一个完整的 Transformer 层

    结构：
      x ─→ norm1 ─→ Attention ─→ + x（残差）
       └─→ norm2 ─→ FFN       ─→ + x（残差）

    ★ 残差连接的作用：
       x = x + SubLayer(x)
       梯度回传时有一条"高速公路"直接流过（∂x/∂x = 1），
       即使 SubLayer 的梯度很小，梯度也不会消失。
       这是深层 Transformer（12层、24层甚至更多）能训练的关键。

    ★ layer_idx 的作用：
       每层有独立的 KV Cache 缓存，layer_idx 用来区分是第几层的缓存。
    """

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx  # 层索引，用于 KV Cache 定位
        # Attention 分支：先 norm 再 attention
        self.norm1 = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = CausalSelfAttention(config)
        # FFN 分支：先 norm 再 ffn
        self.norm2 = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        kv_cache: KVCache = None,
    ) -> torch.Tensor:
        """前向传播

        Pre-norm 残差连接（核心就两行）：
          x = x + Attention(norm1(x))
          x = x + FFN(norm2(x))

        参数：
          x: 输入, shape (batch, seq_len, hidden_size)
          freqs: RoPE 频率
          kv_cache: KV 缓存（可选，推理时使用）
        """
        # Attention 分支：归一化 → 注意力计算 → 残差连接
        x = x + self.attn(self.norm1(x), freqs, kv_cache, self.layer_idx)
        # FFN 分支：归一化 → 前馈网络 → 残差连接
        x = x + self.ffn(self.norm2(x))
        return x
