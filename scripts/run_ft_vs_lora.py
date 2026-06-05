"""全参 vs LoRA 对比实验脚本

对比全参 SFT 和 LoRA SFT 的差异：
1. 参数量（全参 100% vs LoRA ~1-2%）
2. 显存占用
3. 训练速度
4. 最终效果（loss）

用法：
    python scripts/run_ft_vs_lora.py
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


def create_dataloader(dataset, batch_size):
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


def measure_gpu_memory():
    """测量当前显存占用"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**3
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


def run_experiment(method, data_path, config):
    """运行单个实验

    method: "full" 或 "lora"
    """
    print(f"\n{'='*60}")
    print(f"实验: {method.upper()} SFT")
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

    # 根据方法应用 LoRA 或全参
    if method == "lora":
        apply_lora_to_model(
            model,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
    else:
        # 全参：所有参数都可训练
        for param in model.parameters():
            param.requires_grad = True

    # 统计参数
    trainable, total = count_parameters(model)
    print(f"可训练参数: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    # 重置显存统计
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # 加载数据（内存映射）
    token_ids = np.load(data_path, mmap_mode="r")
    max_tokens = min(len(token_ids), config["max_steps"] * config["batch_size"] * config["max_length"] * 2)
    token_ids = token_ids[:max_tokens]
    dataset = SimpleDataset(token_ids, config["max_length"])

    dataloader = create_dataloader(
        dataset,
        batch_size=config["batch_size"],
    )

    # 训练配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 优化器：全参用较小 lr，LoRA 用正常 lr
    lr = config["lr_ft"] if method == "full" else config["lr_lora"]
    optimizer = create_optimizer(
        model,
        lr=lr,
        weight_decay=0.1,
        betas=(0.9, 0.95),
    )

    steps_per_epoch = min(config["max_steps"], len(dataloader))
    total_steps = steps_per_epoch * config["epochs"]

    scheduler = create_scheduler(
        optimizer,
        warmup_steps=config["warmup_steps"],
        total_steps=total_steps,
    )

    # 训练
    print(f"开始训练: lr={lr}, {steps_per_epoch} 步/epoch")
    start_time = time.time()
    start_memory = measure_gpu_memory()

    total_loss = 0.0
    for epoch in range(config["epochs"]):
        step = 0

        for batch in dataloader:
            if step >= steps_per_epoch:
                break

            batch = {k: v.to(device) for k, v in batch.items()}
            loss = train_one_step(model, batch, optimizer, model_config)

            total_loss += loss
            step += 1
            scheduler.step()

            if step % config["log_interval"] == 0:
                avg_loss = total_loss / config["log_interval"]
                print(f"  step {step}/{steps_per_epoch} | loss: {avg_loss:.4f}")
                total_loss = 0.0

    final_loss = total_loss / max(1, steps_per_epoch % config["log_interval"])
    duration = time.time() - start_time
    final_memory = measure_gpu_memory()

    # 合并 LoRA（如果是 LoRA 方法）
    if method == "lora":
        merge_lora_weights(model)

    # 保存模型
    output_dir = Path(f"outputs/sft_{method}")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model": model.state_dict(),
        "config": model_config.__dict__,
        "method": method,
        "trainable_params": trainable,
    }, output_dir / "ckpt_final.pt")

    # 记录结果
    results = {
        "method": method,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_ratio": trainable / total * 100,
        "lr": lr,
        "final_loss": final_loss,
        "duration_seconds": duration,
        "duration_minutes": duration / 60,
        "start_memory_gb": start_memory,
        "peak_memory_gb": final_memory,
    }

    print(f"\n实验完成:")
    print(f"  方法: {method.upper()}")
    print(f"  可训练参数: {trainable:,} ({trainable/total*100:.2f}%)")
    print(f"  学习率: {lr}")
    print(f"  耗时: {duration/60:.2f} 分钟")
    print(f"  峰值显存: {final_memory:.2f} GB")

    return results


def main():
    parser = argparse.ArgumentParser(description="全参 vs LoRA 对比实验")
    parser.add_argument("--data-path", type=str,
                        default="data/pretrain_tokenized/train_ids.npy",
                        help="训练数据路径")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Batch size")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="每个实验最大训练步数")
    parser.add_argument("--epochs", type=int, default=1,
                        help="训练轮数")
    args = parser.parse_args()

    print(f"=" * 60)
    print(f"全参 vs LoRA 对比实验")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"=" * 60)

    # 训练配置
    config = {
        "batch_size": args.batch_size,
        "lr_ft": 5e-6,         # 全参微调用小 lr
        "lr_lora": 1e-4,       # LoRA 用正常 lr
        "max_length": 512,
        "warmup_steps": 50,
        "max_steps": args.max_steps,
        "epochs": args.epochs,
        "log_interval": 50,
    }

    data_path = Path(args.data_path)

    # 运行实验
    ft_results = run_experiment("full", data_path, config)
    lora_results = run_experiment("lora", data_path, config)

    # 汇总对比
    print(f"\n{'='*60}")
    print(f"对比汇总")
    print(f"{'='*60}")

    print(f"\n{'指标':<15} {'全参 SFT':<15} {'LoRA SFT':<15} {'差异':<15}")
    print("-" * 60)

    # 参数量对比
    print(f"{'可训练参数':<15} {ft_results['trainable_params']:>12,} "
          f"{lora_results['trainable_params']:>12,} "
          f"{ft_results['trainable_params']/lora_results['trainable_params']:.1f}x")

    # 参数占比
    print(f"{'参数占比':<15} {ft_results['trainable_ratio']:>12.2f}% "
          f"{lora_results['trainable_ratio']:>12.2f}%")

    # 显存对比
    print(f"{'峰值显存(GB)':<15} {ft_results['peak_memory_gb']:>12.2f} "
          f"{lora_results['peak_memory_gb']:>12.2f} "
          f"{ft_results['peak_memory_gb']/lora_results['peak_memory_gb']:.1f}x")

    # 耗时对比
    print(f"{'耗时(分钟)':<15} {ft_results['duration_minutes']:>12.2f} "
          f"{lora_results['duration_minutes']:>12.2f} "
          f"{ft_results['duration_minutes']/lora_results['duration_minutes']:.1f}x")

    # 保存对比结果
    output_file = Path("results/ft_vs_lora_ablation.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump({
            "config": config,
            "full_ft": ft_results,
            "lora": lora_results,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到: {output_file}")

    # 生成对比表（简历用）
    print(f"\n--- 对比表（简历用）---")
    print(f"| 方法 | 可训练参数 | 占比 | 峰值显存 | 耗时 |")
    print(f"|------|------------|------|----------|------|")
    print(f"| 全参 SFT | {ft_results['trainable_params']:,} | {ft_results['trainable_ratio']:.2f}% | "
          f"{ft_results['peak_memory_gb']:.2f} GB | {ft_results['duration_minutes']:.1f} min |")
    print(f"| LoRA SFT | {lora_results['trainable_params']:,} | {lora_results['trainable_ratio']:.2f}% | "
          f"{lora_results['peak_memory_gb']:.2f} GB | {lora_results['duration_minutes']:.1f} min |")


if __name__ == "__main__":
    main()
