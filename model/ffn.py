"""
SwiGLU 前馈网络（Feed-Forward Network）

标准 FFN:    FFN(x) = W2 · ReLU(W1 · x)
SwiGLU FFN:  FFN(x) = W2 · (SiLU(W_gate · x) ⊙ W_up · x)

其中 SiLU(x) = x · σ(x)，σ 是 sigmoid 函数
⊙ 是逐元素乘法

为什么用 SwiGLU：
  - 比 ReLU 效果好（实验验证）
  - gate 机制让网络学会"选择性放行"信息
  - LLaMA、Qwen、Mistral 等主流模型都用 SwiGLU

三个线性层的参数量 = 3 * hidden_size * intermediate_size
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class SwiGLU(nn.Module):
    """SwiGLU 前馈网络

    公式：FFN(x) = down_proj(SiLU(gate_proj(x)) * up_proj(x))

    三个线性层：
    - gate_proj: hidden → intermediate（控制信息流的"门"）
    - up_proj: hidden → intermediate（信息变换）
    - down_proj: intermediate → hidden（投影回去）
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        参数：x: (batch, seq_len, hidden_size)
        返回：(batch, seq_len, hidden_size)

        公式：down_proj( SiLU(gate_proj(x)) * up_proj(x) )

        步骤：
        1. gate = gate_proj(x)
        2. up = up_proj(x)
        3. return down_proj(F.silu(gate) * up)
        """
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
