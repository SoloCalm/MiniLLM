# 07-data_loader.py 数据加载与 Loss

## 逐段源码与解析

### 1. 预训练数据集 PretrainDataset (L1-90)

```python
class PretrainDataset(Dataset):
    """预训练数据集

    数据格式：每行 {"text": "一段文本..."}
    处理方式：把所有文本拼接，按 max_length 切成固定长度的块

    为什么不一条一条训练：
    - 预训练数据每条长度不一，直接训练效率低
    - 拼接后切块，每个样本都是 max_length，batch 训练更高效
    - 这是 LLM 预训练的标准做法
    """

    def __init__(self, data_path: Path, tokenizer, max_length: int = 1024, max_lines: int = None):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 1. 读取文本（可限制行数，避免 OOM）
        items = load_jsonl(data_path)
        if max_lines:
            items = items[:max_lines]
        texts = [item["text"] for item in items]
        print(f"加载 {len(texts)} 条文本")

        # 2. 拼接所有文本，一起 tokenize
        all_tokens = []
        for text in texts:
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)
        print(f"总 token 数: {len(all_tokens)}")

        # 3. 按 max_length 切块
        self.samples = []
        for i in range(0, len(all_tokens), max_length):
            chunk = all_tokens[i:i + max_length]
            if len(chunk) > 1:  # 至少 2 个 token 才有意义
                self.samples.append(chunk)
        print(f"训练样本数: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """返回一个训练样本"""
        chunk = self.samples[idx]
        input_ids = torch.tensor(chunk, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "labels": input_ids,   # 和 input_ids 一样，loss 内部右移
        }
```

**预训练数据处理流程：**
```
原始文本：
    {"text": "如何才能摆脱拖延症？治愈拖延症并不容易..."}
    {"text": "清晨的阳光透过窗帘洒进房间..."}
    ...

拼接后 tokenize：
    [101, 102, 103, ..., 5000, 101, 102, 103, ..., 6000]

按 max_length=1024 切块：
    chunk1: [101, 102, ..., 1024]
    chunk2: [1025, 1026, ..., 2048]
    ...
```

---

### 2. 内存映射数据集 PretrainDatasetMmap (L96-145)

```python
class PretrainDatasetMmap(Dataset):
    """预训练数据集（numpy 版，推荐使用）

    解决原版 PretrainDataset 的 OOM 问题：

    问题：
      原版在 __init__ 中一次性 tokenize 所有文本到 Python list，
      127 万条文本 → 3.95 亿 token → Python int 占 28 字节 → ~11GB 内存 → OOM

    解决方案（参考 MicroLM）：
      1. 用 scripts/tokenize_to_disk.py 提前 tokenize，存成 .npy 文件
      2. numpy uint16 存储：2 字节/token（vs Python int 28 字节，省 14 倍）
      3. numpy 加载到内存：3.95 亿 token × 2 字节 = ~791 MB（可接受）
    """

    def __init__(self, token_ids_path: Path, max_length: int = 1024):
        self.max_length = max_length

        # numpy 加载：uint16 只占 2 字节/token，比 Python list 省 14 倍
        self.token_ids = np.load(token_ids_path).astype(np.int64)
        self.total_tokens = len(self.token_ids)

        print(f"加载 token ids: {self.total_tokens} tokens ({self.total_tokens * 2 / 1e6:.1f} MB)")

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

**内存映射对比：**
```
Python List:
    395M tokens × 28 bytes/token = ~11 GB  # OOM!

numpy uint16:
    395M tokens × 2 bytes/token = ~791 MB  # 可接受
```

---

### 3. SFT 数据集 SFTDataset (L151-251)

```python
class SFTDataset(Dataset):
    """SFT 数据集

    数据格式：{"instruction": ..., "input": ..., "output": ...}
    或：{"conversations": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}

    关键点：
    - 用 chat_template 把对话转成 token 序列
    - labels 中 prompt 部分设为 -100（只对 assistant 回复计算 loss）
    """

    def preprocess(self, item: dict) -> dict:
        """将一条原始数据转成 input_ids 和 labels"""
        # 构造 user 部分和 assistant 部分的文本
        if "conversations" in item:
            # minimind 格式
            user_text = ""
            assistant_text = ""
            for turn in item["conversations"]:
                if turn["role"] == "user":
                    user_text = turn["content"]
                elif turn["role"] == "assistant":
                    assistant_text = turn["content"]
        elif "instruction" in item:
            # BelleGroup 格式
            user_text = item["instruction"]
            if item.get("input"):
                user_text += "\n" + item["input"]
            assistant_text = item["output"]
        else:
            return None

        # 3. tokenize 两部分
        user_ids = self.tokenizer.encode(user_text)
        assistant_ids = self.tokenizer.encode(assistant_text)

        # 4. 拼接：[BOS] user tokens [EOS] assistant tokens [EOS]
        bos_id = 1  # <s>
        eos_id = 2  # </s>

        input_ids = [bos_id] + user_ids + [eos_id] + assistant_ids + [eos_id]

        # 5. 构造 labels：prompt 部分设为 -100
        prompt_len = 1 + len(user_ids) + 1  # BOS + user + EOS
        labels = [-100] * prompt_len + assistant_ids + [eos_id]

        return {
            "input_ids": input_ids,
            "labels": labels,
        }
```

**SFT 数据处理流程：**
```
原始数据：
    {"conversations": [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"}
    ]}

Tokenize 后：
    input_ids:  [1, 101, 102, 2, 201, 202, 2]
    labels:     [-100, -100, -100, -100, 201, 202, 2]
                └── prompt 部分 ──┘  └─ assistant ─┘
```

---

### 4. DPO 数据集 DPODataset (L257-361)

```python
class DPODataset(Dataset):
    """DPO 偏好数据集

    数据格式：{"prompt": "...", "chosen": "...", "rejected": "..."}

    每条数据包含一个问题的两个回答：
    - chosen: 人类偏好的回答
    - rejected: 人类不偏好的回答
    """

    def __getitem__(self, idx):
        """返回第 idx 条数据的 4 个 tensor"""
        sample = self.samples[idx]
        prompt = sample["prompt"]
        chosen = sample["chosen"]
        rejected = sample["rejected"]

        # 2. Tokenize
        prompt_ids = self.tokenizer.encode(prompt)
        chosen_ids = self.tokenizer.encode(chosen)
        rejected_ids = self.tokenizer.encode(rejected)

        # 4. 拼接 input_ids
        chosen_input_ids = [bos_id] + prompt_ids + [eos_id] + chosen_ids + [eos_id]
        rejected_input_ids = [bos_id] + prompt_ids + [eos_id] + rejected_ids + [eos_id]

        # 5. 构造 labels（prompt 部分设为 -100）
        prompt_len = 1 + len(prompt_ids) + 1
        chosen_labels = [-100] * prompt_len + chosen_ids + [eos_id]
        rejected_labels = [-100] * prompt_len + rejected_ids + [eos_id]

        return {
            "chosen_input_ids": torch.tensor(chosen_input_ids, dtype=torch.long),
            "chosen_labels": torch.tensor(chosen_labels, dtype=torch.long),
            "rejected_input_ids": torch.tensor(rejected_input_ids, dtype=torch.long),
            "rejected_labels": torch.tensor(rejected_labels, dtype=torch.long),
        }
```

**DPO 数据处理流程：**
```
原始数据：
    {"prompt": "如何学习编程？",
     "chosen": "建议从Python开始...",
     "rejected": "不知道..."}

Tokenize 后：
    chosen_input_ids:    [1, prompt_ids, 2, chosen_ids, 2]
    chosen_labels:       [-100..., chosen_ids, 2]
    rejected_input_ids:  [1, prompt_ids, 2, rejected_ids, 2]
    rejected_labels:     [-100..., rejected_ids, 2]
```

---

### 5. 数据加载器工厂 Create DataLoader (L367-400)

```python
def create_dataloader(dataset, batch_size: int, shuffle: bool = True):
    """创建 DataLoader

    使用自定义的 collate_fn 处理动态 padding
    （不同样本长度不同时，pad 到 batch 内最长）
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
                    pad = torch.full((pad_len,), 0, dtype=seq.dtype)  # pad_token_id=0
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

## Loss 计算原理

### 预训练 Loss
```python
# 所有 token 都计算 loss
shift_logits = logits[:, :-1, :]
shift_labels = labels[:, 1:]
loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=pad_token_id)
```

### SFT Loss
```python
# 只有 assistant 部分计算 loss（prompt 部分是 -100）
shift_logits = logits[:, :-1, :]
shift_labels = labels[:, 1:]
loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
```

**为什么用 -100？**
- PyTorch 的 cross_entropy 会自动忽略 label=-100 的位置
- SFT 中 prompt 部分设为 -100，不参与 loss 计算
- 只优化 assistant 部分，让模型学会回答问题
