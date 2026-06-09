# 01-transformer.py 模型主干

## 逐段源码与解析

### 1. 模型配置 ModelConfig (L1-50)

```python
class ModelConfig:
    """LLM 模型配置（38M 参数，适配 6GB GPU）"""

    # 模型架构参数
    vocab_size: int = 6400           # 词表大小（BPE分词器，自训练6400词表）
    hidden_size: int = 512           # 隐藏层维度（所有 Transformer 层的宽度）
    num_layers: int = 12             # Transformer Block 层数
    num_heads: int = 8               # Query 注意力头数
    num_kv_heads: int = 4            # Key/Value 注意力头数（GQA：KV 头数 < Q 头数，节省显存）
    intermediate_size: int = 1376    # FFN 中间层维度（SwiGLU 的 gate/up 投影宽度）
    max_seq_len: int = 1024          # 最大序列长度
    rope_theta: float = 10000.0      # RoPE 基础频率（越大支持越长序列）
    rms_norm_eps: float = 1e-6       # RMSNorm 防除零的小常数

    # 特殊 token
    bos_token_id: int = 1            # <bos> 开始 token
    eos_token_id: int = 2            # <eos> 结束 token
    pad_token_id: int = 0            # <pad> 填充 token
```

**为什么这样设计配置类：**
- 使用类属性作为默认值，清晰可见
- `__init__` 支持 kwargs 覆盖，方便实验
- 参数量估算函数，方便验证设计
- 所有模块都从这里读取配置，修改一处即可全局生效

---

### 2. 完整模型 MiniLLM (L51-120)

```python
class MiniLLM(nn.Module):
    """Decoder-only Transformer（LLaMA2 架构）"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # 1. Token Embedding 层，将 token ids 转换为 hidden_size 维的向量表示
        self.tok_emb = nn.Embedding(config.vocab_size, config.hidden_size)

        # 2. TransformerBlock 层，每个层包含一个 Self-Attention 层和一个 FeedForward 层
        self.layers = nn.ModuleList([TransformerBlock(config, i) for i in range(config.num_layers)])

        # 3. RMSNorm 层，对 TransformerBlock 输出进行归一化
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        # 4. LM Head 层，将归一化后的 hidden_size 维向量转换为 vocab_size 维的概率分布
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 5. Rotary Embedding 层，用于计算 RoPE 频率
        self.rope = RotaryEmbedding(config.head_dim, config.max_seq_len, config.rope_theta)

        # 权重绑定：lm_head 和 tok_emb 共享权重，减少 ~3.3M 参数
        self.lm_head.weight = self.tok_emb.weight
```

**架构图：**
```
Token Embeddings
      ↓
[TransformerBlock × 12]
      ↓
RMSNorm
      ↓
LM Head (权重绑定)
```

---

### 3. 前向传播 Forward (L121-170)

```python
def forward(
    self,
    input_ids: torch.Tensor,
    kv_cache: KVCache = None,
) -> torch.Tensor:
    """前向传播

    参数：
        input_ids: (batch, seq_len) token ids
        kv_cache: KV Cache（推理时传入，训练时为 None）

    返回：
        logits: (batch, seq_len, vocab_size)
    """
    batch_size, seq_len = input_ids.shape

    # 1. Token Embedding：将 token ids 转换为 hidden_size 维的向量表示
    x = self.tok_emb(input_ids)

    # 2. 计算 RoPE 频率（需要考虑 KV Cache 中已有的长度作为偏移）
    if kv_cache is not None:
        cache_len = kv_cache.get_seq_length(0)
    else:
        cache_len = 0
    freqs = self.rope(cache_len + seq_len)
    freqs = freqs[cache_len:]  # 只取当前 token 对应的频率

    # 3. 逐层经过 TransformerBlock
    for layer in self.layers:
        x = layer(x, freqs, kv_cache)

    # 4. 最终 RMSNorm + LM Head
    x = self.norm(x)
    logits = self.lm_head(x)
    return logits
```

**数据流：**
```
input_ids: (batch, seq_len)
    ↓ Token Embedding
x: (batch, seq_len, 512)
    ↓ RoPE 频率计算
freqs: (seq_len, 64)
    ↓ 12 层 TransformerBlock
x: (batch, seq_len, 512)
    ↓ RMSNorm
x: (batch, seq_len, 512)
    ↓ LM Head
logits: (batch, seq_len, 6400)
```

---

### 4. 生成 Generate (L171-230)

```python
@torch.no_grad()
def generate(
    self,
    input_ids: torch.Tensor,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
) -> torch.Tensor:
    """自回归生成

    参数：
        input_ids: (batch, seq_len) 输入 token ids
        max_new_tokens: 最多生成多少个 token
        temperature: 温度（0=贪心，>1=更随机）
        top_p: Nucleus sampling 阈值
        top_k: Top-K 过滤

    返回：
        output: (batch, seq_len + new_tokens) 完整序列
    """
    for _ in range(max_new_tokens):
        # 1. 前向传播，得到 logits
        logits = self(input_ids)

        # 2. 取最后一个位置的 logits
        next_token_logits = logits[:, -1, :]  # (batch, vocab_size)

        # 3. 温度缩放
        if temperature > 0:
            next_token_logits = next_token_logits / temperature

        # 4. Top-K 过滤
        if top_k > 0:
            top_k_values, _ = torch.topk(next_token_logits, top_k)
            next_token_logits[next_token_logits < top_k_values[:, -1:]] = float('-inf')

        # 5. Top-P (Nucleus) 过滤
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = float('-inf')

        # 6. 采样
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)

        # 7. 拼接
        input_ids = torch.cat([input_ids, next_token], dim=1)

        # 8. 停止条件（遇到 EOS）
        if (next_token == self.config.eos_token_id).all():
            break

    return input_ids
```

**生成流程：**
```
输入: [BOS] user_text [EOS]
    ↓
循环 256 次（或遇到 EOS）：
    ↓
    1. 前向传播 → logits
    2. 取最后一个位置 → next_token_logits
    3. 温度缩放
    4. Top-K 过滤
    5. Top-P 过滤
    6. 采样 → next_token
    7. 拼接到输入
    ↓
输出: [BOS] user_text [EOS] assistant_text [EOS]
```

---

## 关键设计决策

### 1. 为什么用权重绑定？
- lm_head 和 tok_emb 共享权重
- 减少 ~3.3M 参数（6400 × 512）
- 提高训练效率，减少过拟合风险

### 2. 为什么用 Pre-norm？
- 训练更稳定，不容易梯度爆炸/消失
- 不需要专门的学习率 warmup
- 大多数现代 LLM 都用 Pre-norm（LLaMA、Qwen、Gemma）

### 3. 为什么用 GQA？
- Q 有 8 个头，K/V 只有 4 个头
- 每 2 个 Q 头共享 1 个 KV 头
- 减少 KV Cache 显存占用，加速推理

### 4. 为什么支持 KV Cache？
- 推理时缓存历史 token 的 K/V
- 新 token 只需要计算自己的 K/V
- 避免对历史 token 重复计算，加速自回归生成
