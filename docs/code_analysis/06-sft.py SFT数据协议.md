# 06-sft.py SFT 数据协议

## 逐段源码与解析

### 1. SFTDataset 数据集 (L1-90)

```python
class SFTDataset(Dataset):
    """SFT 数据集

    数据格式：{"instruction": ..., "input": ..., "output": ...}
    或：{"conversations": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}

    关键点：
    - 用 chat_template 把对话转成 token 序列
    - labels 中 prompt 部分设为 -100（只对 assistant 回复计算 loss）
    """

    def __init__(self, data_path: Path, tokenizer, max_length: int = 512, max_lines: int = None):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 1. 读取数据（可限制行数）
        items = load_jsonl(data_path)
        if max_lines:
            items = items[:max_lines]
        print(f"加载 {len(items)} 条 SFT 数据")

        # 2. 对每条数据预处理
        self.samples = []
        for item in items:
            sample = self.preprocess(item)
            if sample is not None:
                self.samples.append(sample)

        print(f"有效样本数: {len(self.samples)}")

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

        if not user_text or not assistant_text:
            return None

        # 3. tokenize 两部分
        user_ids = self.tokenizer.encode(user_text)
        assistant_ids = self.tokenizer.encode(assistant_text)

        # 4. 拼接：[BOS] user tokens [EOS] assistant tokens [EOS]
        bos_id = 1  # <s>
        eos_id = 2  # </s>

        input_ids = [bos_id] + user_ids + [eos_id] + assistant_ids + [eos_id]

        # 5. 构造 labels：prompt 部分设为 -100
        #    prompt = [BOS] user tokens [EOS]（包括 user 部分和分隔符）
        prompt_len = 1 + len(user_ids) + 1  # BOS + user + EOS
        labels = [-100] * prompt_len + assistant_ids + [eos_id]

        # 6. 截断
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]

        return {
            "input_ids": input_ids,
            "labels": labels,
        }
```

**数据格式示例：**
```json
{
    "conversations": [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！有什么我可以帮助你的吗？"}
    ]
}
```

**Tokenize 后：**
```
input_ids:  [1, 101, 102, 2, 201, 202, 301, 302, 2]
labels:     [-100, -100, -100, -100, 201, 202, 301, 302, 2]
            └── prompt 部分（不计算loss）──┘  └─ assistant 部分 ─┘
```

---

### 2. SFT 训练循环 (L78-187)

```python
def main():
    parser = argparse.ArgumentParser(description="SFT 微调 MiniLLM")
    parser.add_argument("--pretrained-path", type=Path, required=True, help="预训练模型路径")
    parser.add_argument("--data-path", type=Path, default=Path("data/minimind_dataset/lora_identity.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sft"))
    parser.add_argument("--lr", type=float, default=1e-5)            # SFT 学习率比预训练小 30 倍
    parser.add_argument("--epochs", type=int, default=3)              # 训练 3 轮
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)  # warmup 比例（3%）
    parser.add_argument("--tokenizer-path", type=str, default="tokenizer/bpe.model")
    args = parser.parse_args()

    # 步骤 1 - 加载预训练模型
    config = ModelConfig()
    model = MiniLLM(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"加载预训练模型: {args.pretrained_path}")
    ckpt = torch.load(args.pretrained_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)

    # 步骤 2 - 加载 SFT 数据
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(args.tokenizer_path)
    train_dataset = SFTDataset(args.data_path, tokenizer, args.max_length, args.max_lines)
    train_dataloader = create_dataloader(train_dataset, args.batch_size)

    # 步骤 3 - 创建优化器（lr=1e-5，比预训练小 30 倍）
    total_steps = len(train_dataloader) * args.epochs
    optimizer = create_optimizer(model, args.lr, weight_decay=0.01)
    scheduler = create_scheduler(optimizer, int(total_steps * args.warmup_ratio), total_steps)

    # 步骤 4 - 训练循环
    step = 0
    for epoch in range(args.epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = train_one_step(model, batch, optimizer, config)

            step += 1
            if step % args.log_interval == 0:
                print(f"  step {step}/{total_steps} | loss: {loss:.4f}")

            scheduler.step()

            if step % args.save_interval == 0:
                ckpt_path = output_dir / f"ckpt_step{step}.pt"
                torch.save({"model": model.state_dict(), "step": step}, ckpt_path)

    # 步骤 5 - 保存最终模型
    final_path = output_dir / "ckpt_final.pt"
    torch.save({"model": model.state_dict(), "step": step}, final_path)
```

---

### 3. 训练一步 Train One Step (L28-76)

```python
def train_one_step(model, batch, optimizer, config):
    """训练一步

    和 pretrain.py 的 train_one_step 完全一样。
    SFTDataset 的 labels 已经把 prompt 部分设为 -100，
    cross_entropy 会自动忽略 label=-100 的位置。
    """
    # 第一步：前向传播
    logits = model(batch["input_ids"])

    # 第二步：算 loss（右移 + 交叉熵）
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = batch["labels"][:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, config.vocab_size),
        shift_labels.view(-1),
        ignore_index=-100,  # prompt 部分是 -100，自动跳过
    )

    # 第三步：反向传播
    loss.backward()

    # 第四步：梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

    # 第五步：更新参数
    optimizer.step()
    optimizer.zero_grad()

    return loss.item()
```

---

## 预训练 vs SFT 的区别

| | 预训练 | SFT |
|---|---|---|
| **目标** | 学习语言本身 | 学习遵循指令 |
| **数据格式** | 纯文本 | 指令-回答对 |
| **Loss计算** | 所有token | 只有assistant部分 |
| **学习率** | 3e-4（较大） | 1e-5（较小） |
| **训练轮数** | 1轮（数据量大） | 3轮（数据量小） |

**为什么 SFT 学习率小？**
- 预训练模型已经学到了很好的语言知识
- 大学习率会破坏已学知识（灾难性遗忘）
- 小学习率只做微调，保持语言能力

**为什么 SFT 只在 assistant 部分计算 Loss？**
- 我们希望模型学会"回答问题"
- 不希望模型学会"重复问题"
- prompt 部分设为 -100，不参与 loss 计算
