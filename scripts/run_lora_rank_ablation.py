"""LoRA rank 消融实验脚本

在自研 41M 模型上测试不同 LoRA rank 的效果：
- rank = 4（最小，参数最少）
- rank = 8（中等）
- rank = 16（最大，参数最多）

目标：比较不同 rank 下的：
1. 可训练参数量
2. 显存占用
3. 训练 loss 下降
4. 最终效果

用法：
    python scripts/run_lora_rank_ablation.py --data-path data/belle_chat_0.4M.jsonl
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.config import ModelConfig
from model.modeling_llm import MiniLLM
from training.lora import apply_lora_to_model, merge_lora_weights
from training.optimizer import create_optimizer, create_scheduler


class SimpleDataset:
    """简单的内存映射数据集"""

    def __init__(self, token_ids, max_length):
        self.token_ids = token_ids
        self.max_length = max_length

    def __len__(self):
        return (len(self.token_ids) - 1) // self.max_length

    def __getitem__(self, idx):
        start = idx * self.max_length
        chunk = self.token_ids[start:start + self.max_length]
        return {
            "input_ids": torch.from_numpy(chunk.astype(np.int64)),
            "labels": torch.from_numpy(chunk.astype(np.int64)),
        }


def create_dataloader(dataset, batch_size, max_length):
    """创建 DataLoader"""
    from torch.utils.data import DataLoader

    def collate_fn(batch):
        input_ids = torch.stack([item["input_ids"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        return {"input_ids": input_ids, "labels": labels}

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )


def count_parameters(model):
    """统计可训练参数"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def measure_gpu_memory(model):
    """测量显存占用"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**3  # GB
    return 0


def train_one_step(model, batch, optimizer, model_config):
    """训练一步"""
    logits = model(batch["input_ids"])

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = batch["labels"][:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, model_config.vocab_size),
        shift_labels.view(-1),
        ignore_index=-100,
    )

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), model_config.max_grad_norm)
    optimizer.step()
    optimizer.zero_grad()

    return loss.item()


def run_single_experiment(rank, data_path, config):
    """运行单个 LoRA rank 实验"""
    print(f"\n{'='*60}")
    print(f"实验: LoRA rank = {rank}")
    print(f"{'='*60}")

    # 创建模型
    model_config = ModelConfig(
        vocab_size=6400,
        hidden_size=512,
        num_layers=12,
        num_heads=8,
        num_kv_heads=4,
        intermediate_size=1376,
        max_seq_len=1024,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
    )

    model = MiniLLM(model_config)

    # 加载预训练权重
    ckpt_path = Path("outputs/pretrained/ckpt_final.pt")
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        print(f"加载预训练权重: {ckpt_path}")

    # 应用 LoRA
    apply_lora_to_model(
        model,
        r=rank,
        lora_alpha=rank * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    # 统计参数
    trainable, total = count_parameters(model)
    print(f"可训练参数: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    # 重置显存统计
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # 加载数据（使用内存映射避免 OOM）
    if data_path.suffix == ".npy":
        token_ids = np.load(data_path, mmap_mode="r")
        # 只取前 max_steps * batch_size * max_length 个 token
        max_tokens = min(len(token_ids), config["max_steps"] * config["batch_size"] * config["max_length"] * 2)
        token_ids = token_ids[:max_tokens]
        dataset = SimpleDataset(token_ids, config["max_length"])
    else:
        raise ValueError(f"不支持的数据格式: {data_path.suffix}")

    dataloader = create_dataloader(
        dataset,
        batch_size=config["batch_size"],
        max_length=config["max_length"],
    )

    # 训练配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = create_optimizer(
        model,
        lr=config["lr"],
        weight_decay=0.1,
        betas=(0.9, 0.95),
    )

    num_epochs = config["epochs"]
    steps_per_epoch = min(config["max_steps"], len(dataloader))
    total_steps = steps_per_epoch * num_epochs

    scheduler = create_scheduler(
        optimizer,
        warmup_steps=config["warmup_steps"],
        total_steps=total_steps,
    )

    # 训练
    print(f"开始训练: {steps_per_epoch} 步/epoch, {num_epochs} epochs")
    start_time = time.time()

    train_losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        step = 0

        for batch in dataloader:
            if step >= steps_per_epoch:
                break

            batch = {k: v.to(device) for k, v in batch.items()}
            loss = train_one_step(model, batch, optimizer, model_config)

            epoch_loss += loss
            step += 1
            scheduler.step()

            if step % config["log_interval"] == 0:
                avg_loss = epoch_loss / config["log_interval"]
                print(f"  step {step}/{steps_per_epoch} | loss: {avg_loss:.4f}")
                epoch_loss = 0.0

        train_losses.append(epoch_loss / steps_per_epoch if steps_per_epoch > 0 else 0)

    duration = time.time() - start_time

    # 测量最终显存
    final_memory = measure_gpu_memory(model)

    # 合并 LoRA 权重
    merge_lora_weights(model)

    # 保存模型
    output_dir = Path(f"outputs/sft_lora_rank{rank}")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model": model.state_dict(),
        "config": model_config.__dict__,
        "train_losses": train_losses,
    }, output_dir / "ckpt_final.pt")

    # 记录结果
    results = {
        "rank": rank,
        "lora_alpha": rank * 2,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_ratio": trainable / total * 100,
        "train_losses": train_losses,
        "duration_seconds": duration,
        "duration_minutes": duration / 60,
        "gpu_memory_gb": final_memory,
    }

    print(f"\n实验完成:")
    print(f"  Rank: {rank}")
    print(f"  可训练参数: {trainable:,} ({trainable/total*100:.2f}%)")
    print(f"  训练 loss: {train_losses[-1] if train_losses else 'N/A'}")
    print(f"  耗时: {duration/60:.2f} 分钟")
    print(f"  显存: {final_memory:.2f} GB")

    return results


def main():
    parser = argparse.ArgumentParser(description="LoRA rank 消融实验")
    parser.add_argument("--data-path", type=str,
                        default="data/pretrain_tokenized/train_ids.npy",
                        help="训练数据路径")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="每个实验最大训练步数")
    parser.add_argument("--epochs", type=int, default=1,
                        help="训练轮数")
    args = parser.parse_args()

    print(f"=" * 60)
    print(f"LoRA rank 消融实验")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"测试 rank: [4, 8, 16]")
    print(f"=" * 60)

    # 训练配置
    config = {
        "batch_size": args.batch_size,
        "lr": 1e-4,
        "max_length": 512,
        "grad_accum": 4,
        "warmup_steps": 100,
        "max_steps": args.max_steps,
        "epochs": args.epochs,
        "log_interval": 50,
    }

    data_path = Path(args.data_path)

    # 运行实验
    all_results = []
    ranks = [4, 8, 16]

    for rank in ranks:
        results = run_single_experiment(rank, data_path, config)
        all_results.append(results)

    # 汇总结果
    print(f"\n{'='*60}")
    print(f"实验汇总")
    print(f"{'='*60}")
    print(f"\n{'Rank':<8} {'可训练参数':<15} {'占比':<10} {'最终 Loss':<12} {'耗时(分)':<10} {'显存(GB)':<10}")
    print("-" * 65)

    for r in all_results:
        print(f"{r['rank']:<8} {r['trainable_params']:>12,} {r['trainable_ratio']:>8.2f}% "
              f"{r['train_losses'][-1]:>10.4f} {r['duration_minutes']:>8.2f} "
              f"{r['gpu_memory_gb']:>8.2f}")

    # 保存汇总结果
    output_file = Path("results/lora_rank_ablation.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump({
            "config": config,
            "results": all_results,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到: {output_file}")

    # 生成对比表（用于简历）
    print(f"\n--- 对比表（简历用）---")
    print(f"| LoRA Rank | 可训练参数 | 占比 | 最终 Loss | 显存 |")
    print(f"|-----------|------------|------|-----------|------|")
    for r in all_results:
        print(f"| r={r['rank']:<7} | {r['trainable_params']:>10,} | {r['trainable_ratio']:.2f}% | "
              f"{r['train_losses'][-1]:.4f} | {r['gpu_memory_gb']:.2f} GB |")


if __name__ == "__main__":
    main()
