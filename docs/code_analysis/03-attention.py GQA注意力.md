# 03-attention.py GQA 注意力

## 逐段源码与解析

### 1. KV Cache (L1-64)

```python
class KVCache:
    """KV Cache：缓存历史 token 的 K 和 V

    推理时使用，训练时不需要。
    """

    def __init__(self):
        self.key_cache = []
        self.value_cache = []

    def update(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor):
        """将当前 token 的 K,V 追加到缓存

        参数：
            layer_idx: 层索引
            key: (batch, 1, num_kv_heads, head_dim) 当前 token 的 K
            value: (batch, 1, num_kv_heads, head_dim) 当前 token 的 V

        返回：
            full_key, full_value: 拼接后的完整 K,V（包括历史）
        """
        if layer_idx >= len(self.key_cache):
            self.key_cache.append(key)
            self.value_cache.append(value)
        else:
            self.key_cache[layer_idx] = torch.cat((self.key_cache[layer_idx], key), dim=1)
            self.value_cache[layer_idx] = torch.cat((self.value_cache[layer_idx], value), dim=1)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """获取缓存的序列长度"""
        if layer_idx >= len(self.key_cache):
            return 0
        return self.key_cache[layer_idx].shape[1]
```

**KV Cache 工作原理：**
```
初始状态：cache = []
第 1 个 token：cache = [k1, v1]
第 2 个 token：cache = [k1+k2, v1+v2]  # 拼接
第 3 个 token：cache = [k1+k2+k3, v1+v2+v3]
...
```

---

### 2. GQA 注意力 CausalSelfAttention (L66-172)

```python
class CausalSelfAttention(nn.Module):
    """GQA 因果自注意力

    关键点：
    - Q 有 num_heads 个头，K,V 只有 num_kv_heads 个头
    - 使用 RoPE 编码位置
    - 支持 KV Cache（推理加速）
    - 因果 mask：每个位置只能看到自己和前面的 token
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim  # = hidden_size // num_heads
        self.num_groups = config.num_heads // config.num_kv_heads  # 每组 Q 共享一个 KV

        self.q_proj = nn.Linear(config.hidden_size, config.num_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_heads * config.head_dim, config.hidden_size, bias=False)

        # RoPE 位置编码
        self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len, config.rope_theta)
```

**GQA 参数量对比：**
```
标准 MHA：Q=8头, K=8头, V=8头
GQA：Q=8头, K=4头, V=4头（每2个Q头共享1个KV头）
MQA：Q=8头, K=1头, V=1头（所有Q头共享1个KV头）

GQA 优势：比 MHA 省显存，比 MQA 效果好
```

---

### 3. 前向传播 Forward (L91-172)

```python
def forward(
    self,
    x: torch.Tensor,
    freqs: torch.Tensor,
    kv_cache: KVCache = None,
    layer_idx: int = 0,
) -> torch.Tensor:
    """前向传播（11 步）"""
    batch_size, seq_len, _ = x.shape

    # 步骤 1: 线性投影
    q = self.q_proj(x)  # (batch, seq, num_heads * head_dim)
    k = self.k_proj(x)  # (batch, seq, num_kv_heads * head_dim)
    v = self.v_proj(x)  # (batch, seq, num_kv_heads * head_dim)

    # 步骤 2: reshape 成多头形式
    q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
    k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
    v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

    # 步骤 3: RoPE 位置编码
    q, k = apply_rotary_pos_emb(q, k, freqs)

    # 步骤 4: KV Cache（推理时）
    if kv_cache is not None:
        k, v = kv_cache.update(layer_idx, k, v)

    # 步骤 5: GQA 广播 — 把 KV 头复制到和 Q 头数一样
    if self.num_groups > 1:
        # unsqueeze(3) 在第 3 维插入一个维度
        k = k.unsqueeze(3)  # (1, 10, 4, 1, 64)
        # expand 复制 2 次
        k = k.expand(-1, -1, -1, self.num_groups, -1)  # (1, 10, 4, 2, 64)
        # reshape 把 4×2 合并成 8
        k = k.reshape(batch_size, -1, self.num_heads, self.head_dim)  # (1, 10, 8, 64)

        # v 同理
        v = v.unsqueeze(3).expand(-1, -1, -1, self.num_groups, -1)
        v = v.reshape(batch_size, -1, self.num_heads, self.head_dim)

    # 步骤 6: 转置 (batch, seq, heads, dim) → (batch, heads, seq, dim)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # 步骤 7: 注意力分数 = Q @ K^T / √head_dim
    scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

    # 步骤 8: 因果 mask — 不能偷看未来的 token
    if seq_len > 1:
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    # 步骤 9: softmax 归一化 → 加权求和
    attn_weights = F.softmax(scores, dim=-1)
    attn_output = torch.matmul(attn_weights, v)

    # 步骤 10-11: 转置回来 + 合并头 + 输出投影
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch_size, seq_len, -1)
    return self.o_proj(attn_output)
```

**注意力计算流程：**
```
输入: x (batch, seq_len, hidden_size=512)
    ↓ 线性投影
Q: (batch, seq_len, 512) → (batch, seq_len, 8, 64)
K: (batch, seq_len, 256) → (batch, seq_len, 4, 64)
V: (batch, seq_len, 256) → (batch, seq_len, 4, 64)
    ↓ RoPE
Q, K 旋转位置编码
    ↓ GQA 广播
K: (batch, seq_len, 4, 64) → (batch, seq_len, 8, 64)
V: (batch, seq_len, 4, 64) → (batch, seq_len, 8, 64)
    ↓ 注意力计算
scores = Q @ K^T / √64  # (batch, 8, seq_len, seq_len)
    ↓ 因果 mask
scores = scores.masked_fill(mask, -inf)
    ↓ softmax
attn_weights = softmax(scores)
    ↓ 加权求和
attn_output = attn_weights @ V  # (batch, 8, seq_len, 64)
    ↓ 合并头
attn_output: (batch, seq_len, 512)
    ↓ 输出投影
output: (batch, seq_len, 512)
```

---

## 关键设计决策

### 1. 为什么用 GQA？
- **标准 MHA**：Q=8头, K=8头, V=8头 → KV Cache 大
- **GQA**：Q=8头, K=4头, V=4头 → KV Cache 减半，效果接近 MHA
- **MQA**：Q=8头, K=1头, V=1头 → KV Cache 最小，但效果差

GQA 是 MHA 和 MQA 的平衡点，主流模型（LLaMA、Qwen）都用 GQA。

### 2. 为什么用因果 Mask？
- 自回归生成时，每个位置只能看到自己和前面的 token
- 不能偷看未来的 token（否则就是作弊）
- 训练时用 mask 实现，推理时用 KV Cache 实现

### 3. 为什么除以 √head_dim？
- Q @ K^T 的值可能很大（尤其是 head_dim 大时）
- 大值经过 softmax 会变成 one-hot，梯度消失
- 除以 √64 缩放，让 softmax 输出更平滑

### 4. 为什么用线性投影而非 nn.Linear？
- 代码更透明，方便理解
- 后续 LoRA 注入时需要直接操作 weight 张量
- 不需要绕过 nn.Linear 的内部封装
