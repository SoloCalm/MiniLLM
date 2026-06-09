# 10-数据 pipeline 六步处理

## 数据处理流程概览

```
Step 1: 原始数据 → 清洗
Step 2: 清洗数据 → Tokenize
Step 3: Tokenize → 切分
Step 4: 切分数据 → 训练格式
Step 5: 训练格式 → 模型输入
Step 6: 模型输入 → 训练
```

---

## 详细步骤

### Step 1: 数据清洗 (data_utils/clean_pretrain.py)

```python
# 清洗预训练语料
# - 去除 HTML 标签
# - 去除特殊字符
# - 去重
# - 过滤过短/过长的文本
```

**清洗前后对比：**
```
清洗前：
    "<p>如何才能摆脱拖延症？</p>\n\n\n治愈拖延症并不容易..."

清洗后：
    "如何才能摆脱拖延症？治愈拖延症并不容易..."
```

---

### Step 2: Tokenize (scripts/tokenize_to_disk.py)

```python
def main():
    # 加载 tokenizer
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(args.tokenizer_path)

    # 读取 JSONL
    items = load_jsonl(args.data_path)
    texts = [item["text"] for item in items]

    # 统计总 token 数
    total_tokens = 0
    for text in texts:
        tokens = tokenizer.encode(text)
        total_tokens += len(tokens)

    # 写入 .npy 文件（内存映射）
    array = np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=np.uint16,
        shape=(total_tokens,),
    )

    offset = 0
    for text in texts:
        tokens = tokenizer.encode(text)
        array[offset:offset + len(tokens)] = tokens
        offset += len(tokens)

    array.flush()
```

**Tokenize 流程：**
```
原始文本：
    "如何才能摆脱拖延症？"

Tokenize 后：
    [101, 102, 103, 104, 105]

存储为 numpy uint16：
    [101, 102, 103, 104, 105]  # 每个 token 占 2 字节
```

---

### Step 3: 切分数据 (training/data_loader.py)

```python
class PretrainDatasetMmap(Dataset):
    """预训练数据集（numpy 版，推荐使用）"""

    def __init__(self, token_ids_path: Path, max_length: int = 1024):
        self.max_length = max_length

        # numpy 加载：uint16 只占 2 字节/token
        self.token_ids = np.load(token_ids_path).astype(np.int64)
        self.total_tokens = len(self.token_ids)

    def __len__(self):
        # 可用的训练样本数
        return (self.total_tokens - 1) // self.max_length

    def __getitem__(self, idx):
        """按索引取第 idx 个固定长度的片段"""
        start = idx * self.max_length
        chunk = self.token_ids[start:start + self.max_length]

        input_ids = torch.from_numpy(chunk)
        return {
            "input_ids": input_ids,
            "labels": input_ids,
        }
```

**切分流程：**
```
原始 token 序列：
    [101, 102, 103, ..., 5000, 101, 102, 103, ..., 6000]

按 max_length=1024 切分：
    chunk1: [101, 102, ..., 1024]
    chunk2: [1025, 1026, ..., 2048]
    ...

训练样本数：
    total_tokens // max_length = 395M // 1024 ≈ 386K 个样本
```

---

### Step 4: 训练格式 (training/sft.py)

```python
def preprocess(self, item: dict) -> dict:
    """将一条原始数据转成 input_ids 和 labels"""
    # 构造 user 部分和 assistant 部分的文本
    if "conversations" in item:
        user_text = ""
        assistant_text = ""
        for turn in item["conversations"]:
            if turn["role"] == "user":
                user_text = turn["content"]
            elif turn["role"] == "assistant":
                assistant_text = turn["content"]

    # tokenize 两部分
    user_ids = self.tokenizer.encode(user_text)
    assistant_ids = self.tokenizer.encode(assistant_text)

    # 拼接：[BOS] user tokens [EOS] assistant tokens [EOS]
    input_ids = [bos_id] + user_ids + [eos_id] + assistant_ids + [eos_id]

    # 构造 labels：prompt 部分设为 -100
    prompt_len = 1 + len(user_ids) + 1
    labels = [-100] * prompt_len + assistant_ids + [eos_id]

    return {
        "input_ids": input_ids,
        "labels": labels,
    }
```

**训练格式示例：**
```
原始数据：
    {"conversations": [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"}
    ]}

训练格式：
    input_ids:  [1, 101, 102, 2, 201, 202, 2]
    labels:     [-100, -100, -100, -100, 201, 202, 2]
                └── prompt 部分 ──┘  └─ assistant ─┘
```

---

### Step 5: 模型输入 (training/data_loader.py)

```python
def create_dataloader(dataset, batch_size: int, shuffle: bool = True):
    """创建 DataLoader

    使用自定义的 collate_fn 处理动态 padding
    """
    def collate_fn(batch):
        """动态 padding：pad 到 batch 内最长序列"""
        keys = batch[0].keys()
        padded = {}
        for key in keys:
            sequences = [item[key] for item in batch]
            # 找 batch 内最大长度
            max_len = max(seq.shape[0] for seq in sequences)
            # pad 到最大长度
            padded_seqs = []
            for seq in sequences:
                if seq.shape[0] < max_len:
                    pad_len = max_len - seq.shape[0]
                    pad = torch.full((pad_len,), 0, dtype=seq.dtype)
                    padded_seq = torch.cat([seq, pad])
                else:
                    padded_seq = seq
                padded_seqs.append(padded_seq)
            padded[key] = torch.stack(padded_seqs)
        return padded

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )
```

**动态 Padding 示例：**
```
batch 中的样本长度：
    sample1: [1, 101, 102, 2]  # 长度 4
    sample2: [1, 201, 202, 301, 302, 2]  # 长度 6
    sample3: [1, 401, 402, 2]  # 长度 4

动态 Padding 后（pad 到最长 6）：
    sample1: [1, 101, 102, 2, 0, 0]  # 补 2 个 0
    sample2: [1, 201, 202, 301, 302, 2]  # 不变
    sample3: [1, 401, 402, 2, 0, 0]  # 补 2 个 0
```

---

### Step 6: 训练 (training/pretrain.py)

```python
def train_one_step(model, batch, optimizer, scheduler, config):
    """训练一步"""
    model.train()
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    # 1. 前向传播
    logits = model(input_ids)

    # 2. 计算 loss（右移 + 交叉熵）
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, config.vocab_size),
        shift_labels.view(-1),
        ignore_index=config.pad_token_id,
    )

    # 3. 反向传播
    loss.backward()

    # 4. 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

    # 5. 更新参数
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    return loss.item()
```

**训练流程：**
```
输入: batch["input_ids"] (batch, seq_len)
    ↓
模型前向传播
logits: (batch, seq_len, vocab_size)
    ↓
右移 + 交叉熵
loss: scalar
    ↓
反向传播
    ↓
梯度裁剪
    ↓
更新参数
    ↓
返回 loss
```

---

## 数据流总结

```
原始数据 (JSONL)
    ↓ Step 1: 清洗
清洗数据
    ↓ Step 2: Tokenize
Token 序列 (numpy uint16)
    ↓ Step 3: 切分
训练样本 (固定长度)
    ↓ Step 4: 训练格式
input_ids + labels
    ↓ Step 5: 模型输入
batch (动态 padding)
    ↓ Step 6: 训练
模型更新
```

---

## 关键优化点

### 1. 预 Tokenize + 内存映射
- 解决 OOM 问题
- 395M tokens × 2 bytes = ~791 MB（vs 11 GB）
- 训练时按需加载，节省内存

### 2. 动态 Padding
- 不同样本长度不同时，pad 到 batch 内最长
- 避免固定长度 padding 的浪费
- 提高训练效率

### 3. 权重绑定
- lm_head 和 tok_emb 共享权重
- 减少 ~3.3M 参数
- 提高训练效率

### 4. 梯度累积
- 有效 batch = batch_size × grad_accum
- 小显存也能用大 batch
- 提高训练稳定性
