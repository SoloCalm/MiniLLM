"""KV Cache 加速对比

对比开/关 KV Cache 的推理速度。

用法：
    python scripts/kv_cache_benchmark.py --checkpoint outputs/dpo/ckpt_final.pt
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import time

import torch
import sentencepiece as spm

from model.config import ModelConfig
from model.modeling_llm import MiniLLM, KVCache


def generate_without_kv_cache(model, input_ids, max_new_tokens=50):
    """不使用 KV Cache 生成"""
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        # 每次都用完整序列前向传播
        logits = model(generated)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

    return generated


def generate_with_kv_cache(model, input_ids, max_new_tokens=50):
    """使用 KV Cache 生成"""
    kv_cache = KVCache()
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        # 只传最后一个 token
        logits = model(generated[:, -1:], kv_cache)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

    return generated


def benchmark(model, tokenizer, prompt, max_new_tokens=50, device="cuda"):
    """对比推理速度"""
    # tokenize
    prompt_ids = tokenizer.encode(prompt)
    bos_id = 1
    eos_id = 2
    input_ids = [bos_id] + prompt_ids + [eos_id]
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    # 预热
    for _ in range(3):
        with torch.no_grad():
            _ = generate_with_kv_cache(model, input_tensor.clone(), max_new_tokens=10)

    # 不使用 KV Cache
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        output_no_cache = generate_without_kv_cache(model, input_tensor.clone(), max_new_tokens)
    torch.cuda.synchronize()
    time_no_cache = time.time() - start

    # 使用 KV Cache
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        output_with_cache = generate_with_kv_cache(model, input_tensor.clone(), max_new_tokens)
    torch.cuda.synchronize()
    time_with_cache = time.time() - start

    return time_no_cache, time_with_cache


def main():
    parser = argparse.ArgumentParser(description="KV Cache 加速对比")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型 checkpoint 路径")
    parser.add_argument("--max-new-tokens", type=int, default=50, help="生成 token 数")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 加载 tokenizer
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load("tokenizer/bpe.model")

    # 加载模型
    print(f"加载模型: {args.checkpoint}")
    config = ModelConfig()
    model = MiniLLM(config)

    # checkpoint 包含 model state_dict，需要 weights_only=False
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()
    print(f"模型参数量: {model.count_parameters() / 1e6:.1f}M")

    # 测试不同 prompt 长度
    test_prompts = [
        "你好",
        "今天天气怎么样",
        "人工智能是什么",
        "如何学习编程",
    ]

    print(f"\n{'='*60}")
    print(f"KV Cache 加速对比（生成 {args.max_new_tokens} 个 token）")
    print(f"{'='*60}")

    total_no_cache = 0
    total_with_cache = 0

    for prompt in test_prompts:
        time_no_cache, time_with_cache = benchmark(
            model, tokenizer, prompt, args.max_new_tokens, device
        )
        speedup = time_no_cache / time_with_cache if time_with_cache > 0 else 0

        print(f"\nPrompt: {prompt}")
        print(f"  无 KV Cache: {time_no_cache:.3f}s")
        print(f"  有 KV Cache: {time_with_cache:.3f}s")
        print(f"  加速比: {speedup:.2f}x")

        total_no_cache += time_no_cache
        total_with_cache += time_with_cache

    avg_speedup = total_no_cache / total_with_cache if total_with_cache > 0 else 0

    print(f"\n{'='*60}")
    print(f"平均加速比: {avg_speedup:.2f}x")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
