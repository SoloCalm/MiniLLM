# 02-rope.py 旋转位置编码

## 逐段源码与解析

### 1. 预计算频率 Precompute Frequencies (L1-67)

```python
def precompute_rope_frequencies(dim: int, max_seq_len: int, theta: float = 10000.0):
    """预计算 RoPE 的旋转频率

    返回形状: (max_seq_len, dim // 2)

    步骤：
    1. 计算频率向量 freq: shape=(dim//2,)
       公式: freq[i] = 1 / (theta ^ (2i / dim))

    2. 计算频率矩阵 freqs: shape=(max_seq_len, dim//2)
       公式: freqs[m, i] = m * freq[i]
    """
    # 步骤 1: freq[i] = 1 / (theta ^ (2i / dim))
    freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

    # 步骤 2: freqs[m, i] = m * freq[i]，用外积
    m = torch.arange(max_seq_len).float()
    freqs = torch.outer(m, freq)

    return freqs
```

**频率计算示例（dim=64, theta=10000）：**
```
arange(0, 64, 2) → [0, 2, 4, ..., 62]（共 32 个偶数索引）
freq[i] = 1 / 10000^(2i/64)
→ i=0: freq=1.0（最高频，每步都转）
→ i=31: freq≈0.00158（最低频，很久才转一圈）

freqs[m, i] = m * freq[i]：第 m 个位置在第 i 个频率上的旋转角度
```

---

### 2. 应用旋转位置编码 Apply Rotary Position Embedding (L69-125)

```python
def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, freqs: torch.Tensor):
    """对 Q 和 K 应用旋转位置编码

    参数:
        q: (batch, seq_len, num_heads, head_dim)
        k: (batch, seq_len, num_kv_heads, head_dim)
        freqs: (seq_len, head_dim // 2)

    步骤：
    1. 把 q, k 转成 float32 避免 bf16 精度问题
    2. 把最后一维拆成 (dim//2, 2) 的形式
    3. 从 freqs 计算 cos 和 sin
    4. 旋转变换：
       q_rot[..., 0] = q_pairs[..., 0] * cos - q_pairs[..., 1] * sin
       q_rot[..., 1] = q_pairs[..., 0] * sin + q_pairs[..., 1] * cos
    5. 展平回原始形状
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
```

**旋转矩阵直觉：**
```
把 64 维向量看成 32 个二维平面向量，每个平面旋转不同角度：
[x1, x2] → [x1*cos(θ) - x2*sin(θ), x1*sin(θ) + x2*cos(θ)]
32 个角度各不相同（来自不同的 freq[i]），实现多尺度位置编码。
```

---

### 3. RoPE 模块 RotaryEmbedding (L127-157)

```python
class RotaryEmbedding(nn.Module):
    """RoPE 模块：预计算频率并提供 apply 方法

    用法:
        rope = RotaryEmbedding(head_dim, max_seq_len)
        freqs = rope(seq_len)
        q_rotated, k_rotated = apply_rotary_pos_emb(q, k, freqs)
    """

    def __init__(self, dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        # register_buffer: 不是参数，但会随模型保存/加载，且跟随 .to(device) 移动
        self.register_buffer("freqs", precompute_rope_frequencies(dim, max_seq_len, theta))

    def forward(self, seq_len: int):
        """返回前 seq_len 个位置的频率

        KV Cache 场景：
        - prefill: rope(seq_len) 返回完整频率
        - decode: rope(1) 只返回当前 token 的频率
        """
        return self.freqs[:seq_len]
```

---

## 为什么用 RoPE？

### 1. 相对位置编码
- 两个向量的点积 = cos((m-n) * θ_i)
- 天然包含了相对位置 (m-n) 的信息
- 不需要额外学习位置嵌入

### 2. 外推性好
- 训练 1024 长度，推理时可以更长
- 因为频率是连续的，可以外推到更长序列

### 3. 多尺度频率
- 低维（i 小）频率高 → 捕捉相邻 token 的位置差异
- 高维（i 大）频率低 → 捕捉远距离的位置差异
- 类似傅里叶变换，用不同频率编码不同粒度的位置信息

### 4. 为什么只旋转 Q 和 K，不旋转 V？
- 注意力分数 = Q·K^T
- 旋转后 Q_m·K_n 包含了 cos(m-n) 的信息
- V 不参与点积计算，不需要位置信息

### 5. 为什么用 register_buffer？
- 频率不参与训练（不需要梯度）
- 但需要跟随模型保存/加载和 .to(device) 移动
- register_buffer 正好满足这三个需求

---

## 为什么转 float32？

bf16 只有 7 位有效数字，旋转涉及 cos/sin 乘法，精度不够会导致：
- 旋转后向量长度改变（理论上旋转不改变长度）
- 累积误差影响训练稳定性

转成 float32 可以避免这些问题。
