# 09-dpo.py DPO 偏好对齐

## 逐段源码与解析

### 1. 计算 Log Probabilities (L36-86)

```python
def compute_log_probs(model, input_ids, labels):
    """计算模型在 labels 非 -100 位置的 log probability

    参数：
        model: 语言模型
        input_ids: (batch, seq_len) 输入 token ids
        labels: (batch, seq_len) 标签（-100 表示不计算 loss）

    返回：
        seq_log_probs: (batch,) 每个序列的 log probability
    """
    # 1. 前向传播，得到 logits
    logits = model(input_ids)  # (batch, seq_len, vocab_size)

    # 2. 右移（和 SFT 一样，预测下一个 token）
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # 3. 计算 log softmax
    log_probs = F.log_softmax(shift_logits, dim=-1)

    # 4. 收集每个位置对应 label 的 log prob
    mask = (shift_labels != -100).float()
    clamped_labels = shift_labels.clamp(min=0)  # -100 → 0（占位）
    token_log_probs = log_probs.gather(
        dim=-1,
        index=clamped_labels.unsqueeze(-1)
    ).squeeze(-1)

    # 5. mask 掉 label=-100 的位置
    token_log_probs = token_log_probs * mask

    # 6. 求和得到序列的 log probability
    seq_log_probs = token_log_probs.sum(dim=-1)

    return seq_log_probs
```

**Log Probabilities 计算流程：**
```
输入: input_ids (batch, seq_len)
    ↓
模型前向传播
logits: (batch, seq_len, vocab_size)
    ↓
右移
shift_logits: (batch, seq_len-1, vocab_size)
shift_labels: (batch, seq_len-1)
    ↓
Log Softmax
log_probs: (batch, seq_len-1, vocab_size)
    ↓
收集对应 label 的 log prob
token_log_probs: (batch, seq_len-1)
    ↓
Mask 掉 -100 位置
token_log_probs: (batch, seq_len-1)
    ↓
求和
seq_log_probs: (batch,)
```

---

### 2. DPO Loss (L88-162)

```python
def dpo_loss(
    model,
    ref_model,
    batch,
    beta: float = 0.2,
):
    """计算 DPO Loss

    参数：
        model: 当前训练的模型（Policy Model）
        ref_model: 参考模型（冻结的 SFT 模型）
        batch: 包含 chosen 和 rejected 的数据
        beta: DPO 温度参数（越大越保守，越小越激进）

    返回：
        loss: DPO loss
        chosen_rewards: chosen 回答的隐式奖励
        rejected_rewards: rejected 回答的隐式奖励
        reward_margin: chosen - rejected 的奖励差
    """
    # 1. 计算 Policy 模型的 log probs（有梯度，会更新）
    policy_chosen_logps = compute_log_probs(model,
                                            batch["chosen_input_ids"],
                                            batch["chosen_labels"])
    policy_rejected_logps = compute_log_probs(model,
                                              batch["rejected_input_ids"],
                                              batch["rejected_labels"])

    # 2. 计算 Reference 模型的 log probs（不需要梯度，冻结）
    with torch.no_grad():
        ref_chosen_logps = compute_log_probs(ref_model,
                                             batch["chosen_input_ids"],
                                             batch["chosen_labels"])
        ref_rejected_logps = compute_log_probs(ref_model,
                                               batch["rejected_input_ids"],
                                               batch["rejected_labels"])

    # 3. 计算 log ratio: log(π_θ / π_ref)
    chosen_log_ratio = policy_chosen_logps - ref_chosen_logps
    rejected_log_ratio = policy_rejected_logps - ref_rejected_logps

    # 4. DPO Loss
    logits = beta * (chosen_log_ratio - rejected_log_ratio)
    loss = -F.logsigmoid(logits).mean()

    # 5. 计算奖励（用于监控训练过程）
    chosen_rewards = beta * chosen_log_ratio
    rejected_rewards = beta * rejected_log_ratio
    reward_margin = (chosen_rewards - rejected_rewards).mean()

    return loss, chosen_rewards.mean(), rejected_rewards.mean(), reward_margin
```

**DPO Loss 公式：**
```
L_DPO = -E[log σ(β × (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]

其中：
- π_θ: 当前训练的模型
- π_ref: 参考模型（冻结的 SFT 模型）
- y_w: chosen（好的回答）
- y_l: rejected（差的回答）
- β: 温度参数
- σ: sigmoid 函数
```

---

### 3. DPO 训练循环 (L165-315)

```python
def main():
    parser = argparse.ArgumentParser(description="DPO 偏好对齐")
    parser.add_argument("--sft-path", type=Path, required=True, help="SFT 模型路径")
    parser.add_argument("--data-path", type=str, default="data/minimind_dataset/dpo.jsonl")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dpo"))
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    # 步骤 1：加载 SFT 模型（作为 model 和 ref_model）
    config = ModelConfig()
    model = MiniLLM(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载 SFT 权重
    ckpt = torch.load(args.sft_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)

    # 创建 reference model（冻结副本）
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # 步骤 2：加载 DPO 数据集
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(args.tokenizer_path)
    train_dataset = DPODataset(args.data_path, tokenizer, args.max_length)
    train_dataloader = create_dataloader(train_dataset, args.batch_size)

    # 步骤 3：创建优化器（DPO 学习率很小）
    optimizer = create_optimizer(model, args.lr, weight_decay=0.01)
    total_steps = len(train_dataloader) * args.epochs
    scheduler = create_scheduler(optimizer, int(total_steps * 0.03), total_steps)

    # 步骤 4：DPO 训练循环
    step = 0
    for epoch in range(args.epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            # 计算 DPO loss
            loss, chosen_reward, rejected_reward, reward_margin = dpo_loss(
                model, ref_model, batch, beta=args.beta
            )

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            step += 1
            if step % args.log_interval == 0:
                print(f"  step {step}/{total_steps} | "
                      f"loss: {loss:.4f} | "
                      f"chosen: {chosen_reward:.4f} | "
                      f"rejected: {rejected_reward:.4f} | "
                      f"margin: {reward_margin:.4f}")

    # 步骤 5：保存最终模型
    final_path = output_dir / "ckpt_final.pt"
    torch.save({"model": model.state_dict(), "step": step}, final_path)
```

---

## DPO vs PPO

### PPO（Proximal Policy Optimization）
- 需要训练 Reward Model
- 需要多个模型（Policy, Reference, Reward, Value）
- 训练复杂，容易崩溃
- 效果好，但实现难度大

### DPO（Direct Preference Optimization）
- 不需要 Reward Model
- 只需要 2 个模型（Policy, Reference）
- 实现简单，训练稳定
- 效果接近 PPO

**为什么 DPO 更受欢迎：**
1. 不需要训练 Reward Model（省掉一个模型）
2. 实现简单（就是一个特殊的 loss）
3. 训练稳定（不像 PPO 容易崩溃）

---

## β 参数影响

| β | Margin | 特点 |
|---|--------|------|
| 0.1 | 0.1222 | 激进，偏离参考模型远 |
| 0.2 | 0.2236 | 平衡，推荐 |
| 0.5 | 0.3199 | 保守，接近参考模型 |

**β 的作用：**
- β 越大，越保守，越依赖偏好差距
- β 越小，越激进，偏离参考模型越远
- β=0.2 是平衡点，margin 适中，loss 较低
