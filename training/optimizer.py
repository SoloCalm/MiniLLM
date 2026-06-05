"""
优化器和学习率调度器

实现 AdamW + Cosine Decay with Warmup。
这是预训练最常用的组合。
"""

import math
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def create_optimizer(model, lr: float, weight_decay: float, betas=(0.9, 0.95)):
    """创建 AdamW 优化器

    关键点：
    - 只对需要梯度的参数做 weight_decay
    - bias 和 LayerNorm/RMSNorm 的 weight 不做 weight_decay
    - 这是 LLM 训练的标准做法

    参数：
        model: 模型
        lr: 学习率
        weight_decay: 权重衰减
        betas: Adam 的 beta 参数

    返回：AdamW 优化器
    """
    # 1. 把参数分成两组：decay 和 no_decay
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bias' in name or 'norm' in name or 'tok_emb' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    # 2. 创建参数组，分别设置 weight_decay
    optimizer_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return AdamW(optimizer_groups, lr=lr, betas=betas)


def create_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """创建 Cosine Decay 学习率调度器（带 Warmup）

    学习率变化：
    - 0 ~ warmup_steps：线性从 0 增长到 lr
    - warmup_steps ~ total_steps：cosine 衰减到 0

    为什么需要 warmup：
    - 训练初期模型参数随机，梯度不稳定
    - 大学习率会导致梯度爆炸
    - warmup 让模型先用小学习率"热身"

    为什么用 cosine decay：
    - 比线性衰减更平滑
    - 后期学习率很小，让模型精细收敛
    - 是 LLM 预训练的标准选择
    """
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps  # 线性增长
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))  # cosine 衰减

    return LambdaLR(optimizer, lr_lambda)
