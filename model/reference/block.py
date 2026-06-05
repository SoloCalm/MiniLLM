"""参考代码 - block.py
RMSNorm + Transformer Block 实现

为什么用 RMSNorm 而不是 LayerNorm？
- LayerNorm: (x - mean) / sqrt(var + eps)，先去均值再归一化
- RMSNorm: x / sqrt(mean(x²) + eps)，只做缩放，不去均值
- 优势：计算量更小（省了均值计算），实践中效果相当或更好
- LLaMA、Gemma、Mistral 等主流模型都用 RMSNorm

为什么用 Pre-norm 而不是 Post-norm？
- Pre-norm: x + SubLayer(LayerNorm(x))  — 先归一化再计算
- Post-norm: LayerNorm(x + SubLayer(x)) — 先计算再归一化
- Pre-norm 的优势：梯度流更稳定，训练更不容易梯度爆炸/消失
- 现代 LLM 几乎全部使用 Pre-norm
"""

import torch
import torch.nn as nn
from .config import ModelConfig
from .attention import CausalSelfAttention, KVCache
from .ffn import SwiGLU


# ================================================================
# TODO 1: RMSNorm
# ================================================================
class RMSNorm(nn.Module):
    """RMS (Root Mean Square) 归一化

    公式: RMSNorm(x) = (x / RMS(x)) * weight
    其中 RMS(x) = sqrt(mean(x²) + eps)

    相比 LayerNorm 少了一步"减去均值"，只做缩放。
    weight 是可学习参数，初始为 1，让网络自己决定是否需要缩放。

    Args:
        dim: 归一化的维度（通常 = hidden_size）
        eps: 防止除零的小常数
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习缩放参数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 步骤 1: 转 float32（归一化需要精确计算，避免 bf16 精度不足）
        # 步骤 2: RMS = sqrt(mean(x^2) + eps) — 沿最后一维计算
        rms = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        # 步骤 3: 归一化并乘以可学习权重，再转回原始 dtype
        return (x.float() / rms * self.weight).type_as(x)


# ================================================================
# TODO 2: TransformerBlock
# ================================================================
class TransformerBlock(nn.Module):
    """单个 Transformer 解码器 Block

    结构（Pre-norm 残差）：
    ┌─────────────┐
    │  x ─────────│────────────────────→ + ──→ + ──→ out
    │  ↓          │                      ↑      ↑
    │  RMSNorm    │                      │      │
    │  ↓          │                      │      │
    │  Attention ─│───→ + ───────────────┘      │
    │             │      ↑                      │
    │             │  RMSNorm                    │
    │             │      ↑                      │
    │             └──────┤                      │
    │                    x                      │
    │                                           │
    │  x ──────────────────────────────────→ + ─┘
    │  ↓                                     ↑
    │  RMSNorm                               │
    │  ↓                                     │
    │  SwiGLU FFN ───────────────────────────┘
    └─────────────┘

    残差连接的作用：让梯度直接流过，避免深层网络的梯度消失。
    """

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx  # 用于 KV Cache 索引（每层独立缓存）
        # Attention 分支
        self.norm1 = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = CausalSelfAttention(config)
        # FFN 分支
        self.norm2 = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn = SwiGLU(config)

    # TODO 3: forward — Pre-norm 残差连接
    def forward(self, x, freqs, kv_cache=None):
        """
        Args:
            x: 输入, shape (batch, seq_len, hidden_size)
            freqs: RoPE 频率
            kv_cache: KV 缓存（可选）
        """
        # Attention 分支: x + Attention(LayerNorm(x))
        #   norm1 → attn（内部处理 RoPE、KV Cache、GQA、因果 mask）
        x = x + self.attn(self.norm1(x), freqs, kv_cache, self.layer_idx)
        # FFN 分支: x + FFN(LayerNorm(x))
        #   norm2 → SwiGLU
        x = x + self.ffn(self.norm2(x))
        return x
