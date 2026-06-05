"""参考代码 - ffn.py
SwiGLU 前馈网络 — 现代 LLM 标准 FFN 方案

对比传统 FFN (ReLU + 两层)：
- SwiGLU: 用 SiLU 激活替代 ReLU，加了 gate 投影实现"门控"
- 效果：同等参数量下效果更好（论文 "GLU Variants Improve Transformer"）
- 代价：比标准 FFN 多一个线性层，参数量 = 3 * hidden * intermediate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import ModelConfig


class SwiGLU(nn.Module):
    """SwiGLU 前馈网络

    公式: output = down_proj( SiLU(gate_proj(x)) * up_proj(x) )

    三个投影的作用：
    - gate_proj: 门控，决定哪些信息通过（经 SiLU 激活，接近 0 的被抑制）
    - up_proj: 值投影，提供被门控筛选的值
    - down_proj: 将 intermediate_size 压缩回 hidden_size

    为什么 intermediate_size = 1376？
    - 标准 LLaMA 用 2/3 * 4 * hidden 作为中间维度
    - 这里 1376 ≈ 8/3 * 512，保证展开比约为 2.7x
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        # TODO 1: 创建三个线性层
        # gate 和 up: hidden → intermediate（展开维度）
        # down: intermediate → hidden（压缩回原始维度）
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO 2: 实现 SwiGLU 前向传播
        # SiLU(x) = x * sigmoid(x)，比 ReLU 更平滑，负值有小的通过量
        # gate * up = 元素级相乘，gate 充当"阀门"控制每个神经元的输出
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
