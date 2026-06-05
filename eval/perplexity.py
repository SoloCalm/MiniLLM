"""
困惑度（Perplexity, PPL）评估

PPL 是衡量语言模型好坏的标准指标：
  PPL = exp(average_cross_entropy_loss)

  - PPL 越低，模型越好
  - 随机模型的 PPL ≈ vocab_size（6400）
  - 好的中文模型 PPL 通常在 10-50 范围

PPL 的局限：
  - 只衡量"语言建模"能力，不衡量"回答问题"能力
  - 需要结合生成质量评估才有意义

用法：
    python eval/perplexity.py --model-path outputs/dpo/ckpt_final.pt --data-path data/pretrain_tokenized/train_ids.npy
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import math

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset

from model.config import ModelConfig
from model.modeling_llm import MiniLLM


class SimpleDataset(Dataset):
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


@torch.no_grad()
def compute_perplexity(
    model: MiniLLM,
    dataloader: DataLoader,
    device: str = "cuda",
    max_batches: int = 0,
) -> float:
    """计算困惑度

    参数：
        model: LLM 模型
        dataloader: 测试数据的 DataLoader
        device: 设备
        max_batches: 最多评估多少个 batch（0=全部）

    返回：
        ppl: 困惑度值

    步骤：
    1. 遍历 dataloader
    2. 对每个 batch 前向传播，计算 loss
    3. 累加所有 loss
    4. PPL = exp(总 loss / 总 token 数)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    batch_count = 0

    for batch in dataloader:
        if max_batches > 0 and batch_count >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # 前向传播
        logits = model(input_ids)

        # 右移
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # 计算 loss
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )

        # 统计
        mask = (shift_labels != -100).float()
        num_tokens = mask.sum().item()

        total_loss += loss.item()
        total_tokens += num_tokens
        batch_count += 1

        if batch_count % 100 == 0:
            print(f"  已处理 {batch_count} 个 batch，{total_tokens} 个 token")

    # 计算 PPL
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    ppl = math.exp(avg_loss)

    return ppl


def main():
    parser = argparse.ArgumentParser(description="计算 Perplexity")
    parser.add_argument("--model-path", type=str, required=True, help="模型 checkpoint 路径")
    parser.add_argument("--data-path", type=str, required=True, help="测试数据路径（.npy）")
    parser.add_argument("--max-length", type=int, default=512, help="最大序列长度")
    parser.add_argument("--max-batches", type=int, default=100, help="最多评估多少个 batch")
    parser.add_argument("--batch-size", type=int, default=4, help="batch 大小")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    # 加载模型
    print(f"加载模型: {args.model_path}")
    config = ModelConfig()
    model = MiniLLM(config)

    # checkpoint 包含 model state_dict，需要 weights_only=False
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()
    print(f"模型参数量: {model.count_parameters() / 1e6:.1f}M")

    # 加载数据
    print(f"加载数据: {args.data_path}")
    token_ids = np.load(args.data_path, mmap_mode="r")
    # 只取一部分数据用于评估
    max_tokens = min(len(token_ids), args.max_batches * args.batch_size * args.max_length * 2)
    token_ids = token_ids[:max_tokens]

    dataset = SimpleDataset(token_ids, args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    print(f"数据集大小: {len(dataset)} 样本")

    # 计算 PPL
    print(f"\n计算 Perplexity...")
    ppl = compute_perplexity(model, dataloader, device, args.max_batches)

    print(f"\n{'='*50}")
    print(f"Perplexity (PPL): {ppl:.2f}")
    print(f"{'='*50}")

    # 保存结果
    output_file = Path("results/perplexity.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    import json
    with open(output_file, "w") as f:
        json.dump({
            "model": args.model_path,
            "ppl": ppl,
            "max_batches": args.max_batches,
        }, f, indent=2)

    print(f"结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
