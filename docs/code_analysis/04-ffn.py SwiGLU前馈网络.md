# 04-ffn.py SwiGLU 前馈网络

## 逐段源码与解析

### 1. SwiGLU 模块 (L1-56)

```python
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

        公式：down_proj( SiLU(gate_proj(x)) * up_proj(x) )

        步骤：
        1. gate = gate_proj(x)
        2. up = up_proj(x)
        3. return down_proj(F.silu(gate) * up)
        """
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

**SwiGLU 数据流：**
```
输入: x (batch, seq_len, 512)
    ↓
gate_proj: 512 → 1376
up_proj: 512 → 1376
    ↓
gate: (batch, seq_len, 1376)
up: (batch, seq_len, 1376)
    ↓
SiLU(gate) = gate * sigmoid(gate)
    ↓
SiLU(gate) * up  # 逐元素乘法（门控）
    ↓
down_proj: 1376 → 512
    ↓
输出: (batch, seq_len, 512)
```

---

## 标准 FFN vs SwiGLU

### 标准 FFN
```python
FFN(x) = W2 · ReLU(W1 · x)
```
- 两个线性层：W1 (512→1376), W2 (1376→512)
- ReLU 激活函数

### SwiGLU FFN
```python
FFN(x) = W2 · (SiLU(W_gate · x) ⊙ W_up · x)
```
- 三个线性层：W_gate, W_up (512→1376), W2 (1376→512)
- SiLU 激活函数
- 门控机制：gate 控制信息流

**参数量对比：**
```
标准 FFN: 2 × 512 × 1376 = 1,409,024
SwiGLU: 3 × 512 × 1376 = 2,113,536
```

SwiGLU 多了 50% 参数，但效果更好。

---

## 为什么用 SwiGLU？

### 1. 门控机制
- gate 决定哪些信息可以通过
- 类似 LSTM 的遗忘门
- 让网络学会"选择性放行"信息

### 2. SiLU 激活函数
- SiLU(x) = x * sigmoid(x)
- 比 ReLU 更平滑
- 避免 ReLU 的"死神经元"问题

### 3. 实验验证
- LLaMA、Qwen、Mistral 等主流模型都用 SwiGLU
- 实验证明 SwiGLU 效果比 ReLU 好

---

## 中间层维度为什么是 1376？

标准 FFN 的中间层维度通常是 hidden_size × 4 = 2048。
SwiGLU 有 3 个线性层（比标准 FFN 多 1 个），所以需要缩小中间层维度以保持总参数量相近。

```
标准 FFN: 2 × 512 × 2048 = 2,097,152
SwiGLU: 3 × 512 × 1376 = 2,113,536  # 参数量相近
```

1376 是通过实验调优得到的最佳值。
