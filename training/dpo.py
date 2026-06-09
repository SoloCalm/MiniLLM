"""
DPO（Direct Preference Optimization）偏好对齐

DPO 的核心思想：
  不需要训练 Reward Model（PPO 的做法），直接用偏好数据优化策略模型。

DPO Loss 公式：
  L_DPO = -E[log σ(β · (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]

  其中：
  - π: 当前训练的模型
  - π_ref: 参考模型（冻结的 SFT 模型）
  - y_w: chosen（好的回答）
  - y_l: rejected（差的回答）
  - β: 温度参数，控制对偏好差异的敏感度
  - σ: sigmoid 函数

为什么 DPO 比 PPO 更受欢迎：
  - 不需要训练 Reward Model（省掉一个模型）
  - 实现简单（就是一个特殊的 loss）
  - 训练稳定（不像 PPO 容易崩溃）
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from model.config import ModelConfig
from model.modeling_llm import MiniLLM
from training.data_loader import DPODataset, create_dataloader
from training.optimizer import create_optimizer, create_scheduler


def compute_log_probs(model, input_ids, labels):
    """计算模型在 labels 非 -100 位置的 log probability

    参数：
        model: 语言模型
        input_ids: (batch, seq_len) 输入 token ids
        labels: (batch, seq_len) 标签（-100 表示不计算 loss）

    返回：
        seq_log_probs: (batch,) 每个序列的 log probability

    流程：
        input_ids → model → logits → log_softmax → 收集 → mask → 求和
    """
    # 1. 前向传播，得到 logits
    # logits[b, t, v] 表示位置 t 预测词表中第 v 个词的分数
    logits = model(input_ids)  # (batch, seq_len, vocab_size)

    # 2. 右移（和 SFT 一样，预测下一个 token）
    # 为什么要右移？
    # input_ids:  [BOS] [t1] [t2] [t3]
    # labels:     [-100] [t1] [t2] [t3]
    # shift_logits: [BOS]→[t1] 的预测, [t1]→[t2] 的预测, ...
    # shift_labels: [t1], [t2], [t3], ...
    shift_logits = logits[:, :-1, :].contiguous()    # (batch, seq_len-1, vocab_size)
    shift_labels = labels[:, 1:].contiguous()        # (batch, seq_len-1)

    # 3. 计算 log softmax（把 logits 转成概率，再取 log）
    # log_probs[b, t, v] = log(P(token_v | 前面的 token))
    log_probs = F.log_softmax(shift_logits, dim=-1)  # (batch, seq_len-1, vocab_size)

    # 4. 收集每个位置对应 label 的 log prob
    # 例如：label=101，就取出 log_probs[b, t, 101]
    # 注意：shift_labels 中有 -100，gather 无法处理负数索引
    # 解决：先 clamp 到 0，gather 后再 mask 掉
    mask = (shift_labels != -100).float()
    clamped_labels = shift_labels.clamp(min=0)  # -100 → 0（占位）
    token_log_probs = log_probs.gather(
        dim=-1,
        index=clamped_labels.unsqueeze(-1)  # (batch, seq_len-1, 1)
    ).squeeze(-1)  # (batch, seq_len-1)

    # 5. mask 掉 label=-100 的位置（prompt 部分不计算 loss）
    token_log_probs = token_log_probs * mask

    # 6. 求和得到序列的 log probability
    # log P(y|x) = Σ log P(y_t | x, y_{<t})
    seq_log_probs = token_log_probs.sum(dim=-1)  # (batch,)

    return seq_log_probs


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

    DPO Loss 公式：
        L = -log σ(β × (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))

    直觉理解：
        - 如果 model 比 ref_model 更喜欢 chosen，log_ratio > 0
        - 如果 model 比 ref_model 更喜欢 rejected，log_ratio < 0
        - 目标：让 chosen_log_ratio > rejected_log_ratio
    """
    # 1. 计算 Policy 模型的 log probs（有梯度，会更新）
    policy_chosen_logps = compute_log_probs(model,
                                            batch["chosen_input_ids"],
                                            batch["chosen_labels"])
    policy_rejected_logps = compute_log_probs(model,
                                              batch["rejected_input_ids"],
                                              batch["rejected_labels"])

    # 2. 计算 Reference 模型的 log probs（不需要梯度，冻结）
    # 为什么要用 ref_model？
    # - 防止 model 偏离太远（类似正则化）
    # - 提供"参考点"，让 model 知道该往哪个方向优化
    with torch.no_grad():
        ref_chosen_logps = compute_log_probs(ref_model,
                                             batch["chosen_input_ids"],
                                             batch["chosen_labels"])
        ref_rejected_logps = compute_log_probs(ref_model,
                                               batch["rejected_input_ids"],
                                               batch["rejected_labels"])

    # 3. 计算 log ratio: log(π_θ / π_ref)
    # chosen_log_ratio > 0: model 比 ref 更喜欢 chosen（好）
    # chosen_log_ratio < 0: model 比 ref 不喜欢 chosen（不好）
    chosen_log_ratio = policy_chosen_logps - ref_chosen_logps
    rejected_log_ratio = policy_rejected_logps - ref_rejected_logps

    # 4. DPO Loss
    # logits = β × (chosen_log_ratio - rejected_log_ratio)
    # loss = -log σ(logits)
    #
    # 为什么这样设计？
    # - 如果 chosen_log_ratio > rejected_log_ratio，logits > 0
    # - σ(大正数) ≈ 1，-log(1) ≈ 0，loss 小（正确行为）
    # - 如果 chosen_log_ratio < rejected_log_ratio，logits < 0
    # - σ(大负数) ≈ 0，-log(0) ≈ ∞，loss 大（需要修正）
    logits = beta * (chosen_log_ratio - rejected_log_ratio)
    loss = -F.logsigmoid(logits).mean()

    # 5. 计算奖励（用于监控训练过程）
    # chosen_rewards: 模型对 chosen 的偏好程度（越大越好）
    # rejected_rewards: 模型对 rejected 的偏好程度（越小越好）
    # reward_margin: chosen - rejected（越大越好，说明模型更偏好 chosen）
    chosen_rewards = beta * chosen_log_ratio
    rejected_rewards = beta * rejected_log_ratio
    reward_margin = (chosen_rewards - rejected_rewards).mean()

    return loss, chosen_rewards.mean(), rejected_rewards.mean(), reward_margin


def main():
    parser = argparse.ArgumentParser(description="DPO 偏好对齐")
    parser.add_argument("--sft-path", type=Path, required=True, help="SFT 模型路径")
    parser.add_argument("--data-path", type=str, default="data/dpo/train.jsonl")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dpo"))
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--tokenizer-path", type=str, default="tokenizer/bpe.model")  # tokenizer 路径
    args = parser.parse_args()

    import copy
    import sentencepiece as spm

    # ============================================================
    # 步骤 1：加载 SFT 模型（作为 model 和 ref_model）
    # ============================================================
    # 创建模型配置和模型实例
    config = ModelConfig()
    model = MiniLLM(config)

    # 选择设备：有 GPU 用 GPU，没有用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 加载 SFT 权重
    # SFT 模型已经学会了基本的对话能力，DPO 在此基础上学习偏好
    print(f"加载 SFT 模型: {args.sft_path}")
    ckpt = torch.load(args.sft_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    print(f"模型参数量: {model.count_parameters() / 1e6:.1f}M")

    # 创建 reference model（冻结副本）
    # 为什么需要 ref_model？
    # - 防止 model 偏离太远（类似正则化）
    # - 提供"参考点"，让 model 知道该往哪个方向优化
    # - ref_model 是冻结的，不会更新
    ref_model = copy.deepcopy(model)
    ref_model.eval()  # 设为评估模式（关闭 dropout 等）
    for param in ref_model.parameters():
        param.requires_grad = False  # 冻结所有参数
    print("Reference model 已创建（冻结）")

    # ============================================================
    # 步骤 2：加载 DPO 数据集
    # ============================================================
    # 加载 SentencePiece tokenizer
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(args.tokenizer_path)

    # 创建 DPO 数据集
    # 每条数据包含 prompt、chosen、rejected
    train_dataset = DPODataset(args.data_path, tokenizer, args.max_length)

    # 创建 DataLoader（自动 batch 和 padding）
    train_dataloader = create_dataloader(train_dataset, args.batch_size)
    print(f"训练样本数: {len(train_dataset)}")

    # ============================================================
    # 步骤 3：创建优化器（DPO 学习率很小）
    # ============================================================
    # DPO 学习率比 SFT 小 10-100 倍
    # 原因：防止偏离 Reference Model 太远
    optimizer = create_optimizer(model, args.lr, weight_decay=0.01)

    # 计算总步数，创建学习率调度器（warmup + cosine decay）
    total_steps = len(train_dataloader) * args.epochs
    scheduler = create_scheduler(optimizer, int(total_steps * 0.03), total_steps)

    # ============================================================
    # 步骤 4：DPO 训练循环
    # ============================================================
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化统计变量
    step = 0
    total_loss = 0.0
    total_chosen_reward = 0.0
    total_rejected_reward = 0.0

    # 训练循环
    for epoch in range(args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*50}")

        for batch in train_dataloader:
            # 把 batch 搬到 GPU
            batch = {k: v.to(device) for k, v in batch.items()}

            # 计算 DPO loss
            # 返回：loss, chosen_reward, rejected_reward, reward_margin
            loss, chosen_reward, rejected_reward, reward_margin = dpo_loss(
                model, ref_model, batch, beta=args.beta
            )

            # 反向传播（计算梯度）
            loss.backward()

            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # 更新参数
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            # 累计统计
            total_loss += loss.item()
            total_chosen_reward += chosen_reward.item()
            total_rejected_reward += rejected_reward.item()
            step += 1

            # 打印日志
            if step % args.log_interval == 0:
                avg_loss = total_loss / args.log_interval
                avg_chosen = total_chosen_reward / args.log_interval
                avg_rejected = total_rejected_reward / args.log_interval
                margin = avg_chosen - avg_rejected

                print(f"  step {step}/{total_steps} | "
                      f"loss: {avg_loss:.4f} | "
                      f"chosen: {avg_chosen:.4f} | "
                      f"rejected: {avg_rejected:.4f} | "
                      f"margin: {margin:.4f}")

                # 重置统计
                total_loss = 0.0
                total_chosen_reward = 0.0
                total_rejected_reward = 0.0

        # 每个 epoch 保存 checkpoint
        ckpt_path = output_dir / f"ckpt_epoch{epoch+1}.pt"
        torch.save({"model": model.state_dict(), "step": step}, ckpt_path)
        print(f"保存 checkpoint: {ckpt_path}")

    # ============================================================
    # 步骤 5：保存最终模型
    # ============================================================
    final_path = output_dir / "ckpt_final.pt"
    torch.save({"model": model.state_dict(), "step": step}, final_path)
    print(f"\nDPO 完成！最终模型保存到: {final_path}")


if __name__ == "__main__":
    main()
