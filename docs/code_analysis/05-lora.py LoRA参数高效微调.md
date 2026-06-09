# 05-lora.py LoRA 参数高效微调

## 逐段源码与解析

### 1. LoRA 线性层 LoRALinear (L1-78)

```python
class LoRALinear(nn.Module):
    """LoRA 线性层

    作为 nn.Linear 的 drop-in 替换：
    原始：y = Wx + b
    LoRA：y = Wx + b + (alpha/r) * B(A(x))

    - A: d×r 矩阵（随机初始化）
    - B: r×d 矩阵（初始化为 0）
    - r: rank（越小参数越少）
    - alpha: 缩放因子
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features

        # 复制原始权重（冻结）
        self.weight = nn.Parameter(original_linear.weight.data.clone(), requires_grad=False)
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data.clone(), requires_grad=False)
        else:
            self.bias = None

        # LoRA 参数（可训练）
        # A 矩阵：随机初始化
        self.lora_A = nn.Parameter(torch.randn(self.in_features, r) * 0.01)
        # B 矩阵：初始化为 0（训练开始时 ΔW=0，不影响原始模型）
        self.lora_B = nn.Parameter(torch.zeros(r, self.out_features))

        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        公式：y = Wx + b + (alpha/r) * dropout(x) @ A @ B
        """
        # 原始线性变换
        base_output = F.linear(x, self.weight, self.bias)
        # LoRA 部分：dropout(x) @ A @ B * scaling
        lora_output = self.lora_dropout(x) @ self.lora_A @ self.lora_B * self.scaling
        return base_output + lora_output
```

**LoRA 数据流：**
```
输入: x (batch, seq_len, 512)
    ↓
原始线性变换: Wx + b  # 冻结，不更新
    ↓
LoRA 部分:
    dropout(x) → x' (batch, seq_len, 512)
    x' @ A → (batch, seq_len, r=8)
    @ B → (batch, seq_len, 512)
    * scaling (alpha/r = 16/8 = 2)
    ↓
输出: Wx + b + 2 * B(A(x'))
```

---

### 2. 应用 LoRA Apply LoRA (L92-146)

```python
def apply_lora_to_model(
    model,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: list = None,
):
    """对模型的指定层应用 LoRA

    步骤：
    1. 冻结模型所有参数
    2. 遍历模型的所有模块
    3. 对 target_modules 中的层，替换为 LoRA 版本
    4. 只有 LoRA 的 A, B 参数需要训练
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]

    # 1. 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

    # 2. 遍历模型，找到目标层，替换为 LoRA 版本
    lora_count = 0
    for name, module in model.named_modules():
        # 检查是否是目标层（比如 "layers.0.attn.q_proj"）
        if isinstance(module, nn.Linear) and any(target in name for target in target_modules):
            # 创建 LoRA 层（drop-in 替换）
            lora_layer = LoRALinear(
                module,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )

            # 替换原模块
            _replace_module(model, name, lora_layer)
            lora_count += 1

    print(f"已对 {lora_count} 个层应用 LoRA（rank={r}, alpha={lora_alpha}）")

    # 3. 打印可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")
```

**LoRA 应用示例：**
```
原始模型层：
    layers.0.attn.q_proj: Linear(512, 512)
    layers.0.attn.k_proj: Linear(512, 256)
    layers.0.ffn.gate_proj: Linear(512, 1376)
    ...

应用 LoRA 后：
    layers.0.attn.q_proj: LoRALinear(512, 512, r=8)
    layers.0.attn.k_proj: LoRALinear(512, 256, r=8)
    layers.0.ffn.gate_proj: LoRALinear(512, 1376, r=8)
    ...
```

---

### 3. 合并 LoRA 权重 Merge LoRA (L148-168)

```python
def merge_lora_weights(model):
    """合并 LoRA 权重到原始权重

    推理时使用：W' = W + (alpha/r) * BA
    合并后推理速度不变（不需要额外计算 LoRA 部分）

    步骤：
    1. 遍历所有 LoRALinear 层
    2. 计算 ΔW = (α/r) * B @ A
    3. 加到原始权重上
    4. 删除 LoRA 参数（可选）
    """
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            # A: (in_features, r), B: (r, out_features)
            # delta_W = (A @ B)^T: (in_features, out_features) -> T -> (out_features, in_features)
            lora_delta = (module.lora_A @ module.lora_B).T  # (out_features, in_features)
            module.weight.data += module.scaling * lora_delta

    print("LoRA 权重已合并到原始权重")
```

---

## 关键设计决策

### 1. 为什么 B 初始化为 0？
- 训练开始时 ΔW = B @ A = 0
- 模型输出和原始模型一样
- 避免随机初始化破坏预训练知识

### 2. 为什么用 scaling = alpha/r？
- alpha 是缩放因子，r 是 rank
- scaling 控制 LoRA 更新的幅度
- alpha/r 越大，LoRA 更新越明显

### 3. 为什么冻结原始权重？
- 原始权重已经学到了很好的表征
- 只训练 LoRA 参数，避免破坏原始知识
- 减少可训练参数，节省显存

### 4. LoRA vs 全参数微调
| 方法 | 可训练参数 | 显存 | 效果 |
|------|------------|------|------|
| 全参微调 | 100% | 高 | 基准 |
| LoRA r=8 | 2.28% | 低 | 接近全参 |

LoRA 用 2.28% 参数达到可比效果，显存省 17%。

### 5. LoRA 应用到哪些层？
```python
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
```
- 注意力层：Q, K, V, O 投影
- FFN 层：gate, up, down 投影
- 不包括 Embedding 和 LM Head（参数太少，效果不明显）
