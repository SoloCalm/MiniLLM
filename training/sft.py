"""
SFT 微调脚本

SFT（Supervised Fine-Tuning）的核心：教模型遵循指令、进行对话。
- 数据：(prompt, answer) 对
- Loss：只在 answer 部分计算（prompt 部分 -100）
- 学习率：比预训练小很多（1e-5 vs 3e-4），避免破坏已学知识

SFT vs 预训练的区别：
- 预训练：所有 token 都计算 loss（学习语言本身）
- SFT：只有 assistant 部分计算 loss（学习遵循指令）
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import sentencepiece as spm
import wandb

from model.config import ModelConfig
from model.modeling_llm import MiniLLM
from training.optimizer import create_optimizer, create_scheduler
from training.data_loader import SFTDataset, create_dataloader


def train_one_step(model, batch, optimizer, config):
    """训练一步

    和 pretrain.py 的 train_one_step 完全一样。
    SFTDataset 的 labels 已经把 prompt 部分设为 -100，
    cross_entropy 会自动忽略 label=-100 的位置。

    步骤：
    1. 前向传播
    2. 计算 loss（ignore_index=-100，只计算 assistant 部分）
    3. 反向传播
    4. 梯度裁剪
    5. 更新参数
    6. 清零梯度
    """
    # 第一步：前向传播（算预测结果）
    logits = model(batch["input_ids"])

    # 第二步：算 loss（右移 + 交叉熵）
    # 为什么要右移？
    # input_ids:  [BOS] [user问题] [EOS] [assistant回答] [EOS]
    # labels:     [-100 ... -100]       [assistant回答] [EOS]
    #
    # 模型在位置 0 看到 BOS，要预测下一个 token
    # 模型在位置 1 看到 BOS+user，要预测下一个 token
    # ...
    # 但最后一个位置没有"下一个 token"可以预测，所以丢弃
    shift_logits = logits[:, :-1, :].contiguous()    # 丢弃最后一个位置
    shift_labels = batch["labels"][:, 1:].contiguous()  # 丢弃第一个位置
    loss = F.cross_entropy(
        shift_logits.view(-1, config.vocab_size),
        shift_labels.view(-1),
        ignore_index=-100,  # prompt 部分是 -100，自动跳过
    )

    # 第三步：反向传播（计算每个参数梯度）
    loss.backward()

    # 第四步：梯度裁剪（防止梯度爆炸）
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

    # 第五步：更新参数（根据梯度更新）
    optimizer.step()

    # 第六步：清零梯度（准备下一次反向传播）
    optimizer.zero_grad()

    return loss.item()


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
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--max-lines", type=int, default=None)       # 限制加载行数
    parser.add_argument("--tokenizer-path", type=str, default="tokenizer/bpe.model")  # tokenizer 路径
    args = parser.parse_args()

    # ============================================================
    # 步骤 1 - 加载预训练模型
    # ============================================================
    config = ModelConfig()  # 创建配置对象
    model = MiniLLM(config)  # 根据配置创建模型

    # 选择设备：有 GPU 用 GPU，没有用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 从预训练 checkpoint 加载权重
    print(f"加载预训练模型: {args.pretrained_path}")
    ckpt = torch.load(args.pretrained_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)  # 把模型搬到 GPU
    print(f"模型参数量: {model.count_parameters() / 1e6:.1f}M")

    # ============================================================
    # 步骤 2 - 加载 SFT 数据
    # ============================================================
    # 加载 tokenizer（把文字转成 token ids 的工具）
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(args.tokenizer_path)

    # 创建 SFT 数据集（处理对话数据，构造 labels mask）
    train_dataset = SFTDataset(args.data_path, tokenizer, args.max_length, args.max_lines)

    # 创建 DataLoader（每次取 batch_size 个样本，pad 到同一长度）
    train_dataloader = create_dataloader(train_dataset, args.batch_size)
    print(f"训练样本数: {len(train_dataset)}")

    # ============================================================
    # 步骤 3 - 创建优化器（lr=1e-5，比预训练小 30 倍）
    # ============================================================
    # 计算总步数：每个 epoch 的 batch 数 × epoch 数
    total_steps = len(train_dataloader) * args.epochs

    # 创建 AdamW 优化器（参数分组：bias/norm 不做 weight_decay）
    optimizer = create_optimizer(model, args.lr, weight_decay=0.01)

    # 创建学习率调度器（warmup + cosine decay）
    scheduler = create_scheduler(optimizer, int(total_steps * args.warmup_ratio), total_steps)

    # ============================================================
    # 步骤 4 - 训练循环（按 epoch 而不是 step）
    # ============================================================
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    step = 0           # 当前步数
    total_loss = 0.0   # 累计 loss

    for epoch in range(args.epochs):           # 外层：遍历整个数据集 3 次
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*50}")

        for batch in train_dataloader:         # 内层：每次取 4 条对话
            # 把 batch 搬到 GPU
            batch = {k: v.to(device) for k, v in batch.items()}

            # 训练一步：前向传播 → 算 loss → 反向传播 → 更新参数
            loss = train_one_step(model, batch, optimizer, config)

            total_loss += loss
            step += 1

            # 每隔 log_interval 步打印日志
            if step % args.log_interval == 0:
                avg_loss = total_loss / args.log_interval
                current_lr = scheduler.get_last_lr()[0]
                print(f"  step {step}/{total_steps} | loss: {avg_loss:.4f} | lr: {current_lr:.2e}")
                total_loss = 0.0

            # 更新学习率
            scheduler.step()

            # 每隔 save_interval 步保存 checkpoint
            if step % args.save_interval == 0:
                ckpt_path = output_dir / f"ckpt_step{step}.pt"
                torch.save({"model": model.state_dict(), "step": step}, ckpt_path)
                print(f"  保存 checkpoint: {ckpt_path}")

    # ============================================================
    # 步骤 5 - 保存最终模型
    # ============================================================
    final_path = output_dir / "ckpt_final.pt"
    torch.save({"model": model.state_dict(), "step": step}, final_path)
    print(f"\nSFT 完成！最终模型保存到: {final_path}")


if __name__ == "__main__":
    main()
