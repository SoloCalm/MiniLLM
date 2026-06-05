"""
结构化评测脚本

用固定 prompt 测试不同 checkpoint 的生成质量，输出对比表格。

用法：
    # 评测单个模型
    python eval/benchmark.py --checkpoint outputs/dpo/ckpt_final.pt

    # 对比多个模型
    python eval/benchmark.py \
        --checkpoints outputs/pretrained/ckpt_final.pt outputs/sft/ckpt_final.pt outputs/dpo/ckpt_final.pt \
        --labels Pretrain SFT DPO

    # 输出到文件
    python eval/benchmark.py --checkpoint outputs/dpo/ckpt_final.pt --output results/benchmark.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import sentencepiece as spm

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.config import ModelConfig
from model.modeling_llm import MiniLLM


# 评测 prompt 集（覆盖不同能力维度）
EVAL_PROMPTS = [
    # 基础对话
    {"prompt": "你好", "category": "对话", "expected_keywords": ["你好"]},
    {"prompt": "你叫什么名字", "category": "对话", "expected_keywords": ["名字", "叫"]},

    # 知识问答
    {"prompt": "人工智能是什么", "category": "知识", "expected_keywords": ["人工智能", "智能"]},
    {"prompt": "中国的首都是哪里", "category": "知识", "expected_keywords": ["北京", "中国"]},

    # 创作
    {"prompt": "写一首关于春天的诗", "category": "创作", "expected_keywords": ["春", "花"]},
    {"prompt": "用一句话描述大海", "category": "创作", "expected_keywords": []},

    # 指令遵循
    {"prompt": "把下面的句子翻译成英文：今天天气很好", "category": "指令", "expected_keywords": []},
    {"prompt": "列举三个学习方法", "category": "指令", "expected_keywords": []},
]


def load_model(checkpoint_path: str, device: str = "cuda"):
    """加载模型"""
    config = ModelConfig()
    model = MiniLLM(config)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    return model


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 100,
             temperature: float = 0.7, device: str = "cuda") -> str:
    """生成回答"""
    bos_id, eos_id = 1, 2
    prompt_ids = tokenizer.encode(prompt)
    input_ids = [bos_id] + prompt_ids + [eos_id]
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        output = model.generate(input_tensor, max_new_tokens=max_new_tokens,
                                temperature=temperature, top_p=0.9, top_k=50)

    gen_ids = output[0].cpu().tolist()[len(input_ids):]
    gen_ids = [t for t in gen_ids if t not in [eos_id, 0]]
    return tokenizer.decode(gen_ids) if gen_ids else "(空)"


def score_response(response: str, expected_keywords: list) -> dict:
    """评分：关键词命中 + 长度 + 重复度"""
    if response == "(空)":
        return {"keyword_score": 0, "length_score": 0, "repeat_score": 0, "total": 0}

    # 关键词命中
    keyword_hits = sum(1 for kw in expected_keywords if kw in response)
    keyword_score = keyword_hits / max(len(expected_keywords), 1)

    # 长度合理性（太短或太长都扣分）
    length = len(response)
    if length < 5:
        length_score = 0.2
    elif length < 20:
        length_score = 0.6
    elif length < 200:
        length_score = 1.0
    else:
        length_score = 0.7

    # 重复度（重复越多越低分）
    unique_chars = len(set(response))
    total_chars = len(response)
    repeat_score = min(unique_chars / max(total_chars * 0.3, 1), 1.0)

    total = keyword_score * 0.5 + length_score * 0.3 + repeat_score * 0.2
    return {
        "keyword_score": round(keyword_score, 2),
        "length_score": round(length_score, 2),
        "repeat_score": round(repeat_score, 2),
        "total": round(total, 2),
    }


def benchmark_model(model, tokenizer, label: str, device: str = "cuda") -> dict:
    """评测单个模型"""
    print(f"\n{'='*50}")
    print(f"评测: {label}")
    print(f"{'='*50}")

    results = []
    total_score = 0

    for i, item in enumerate(EVAL_PROMPTS):
        response = generate(model, tokenizer, item["prompt"], device=device)
        score = score_response(response, item["expected_keywords"])
        total_score += score["total"]

        results.append({
            "prompt": item["prompt"],
            "category": item["category"],
            "response": response[:200],
            "score": score,
        })

        status = "✅" if score["total"] >= 0.5 else "⚠️" if score["total"] >= 0.2 else "❌"
        print(f"  {status} [{item['category']}] {item['prompt']}")
        print(f"     回复: {response[:80]}...")
        print(f"     得分: {score['total']:.2f}")

    avg_score = total_score / len(EVAL_PROMPTS)
    print(f"\n  平均得分: {avg_score:.2f}")

    return {
        "label": label,
        "avg_score": round(avg_score, 2),
        "results": results,
    }


def print_comparison_table(all_results: list):
    """打印对比表格"""
    print(f"\n{'='*70}")
    print("对比结果")
    print(f"{'='*70}")

    # 表头
    labels = [r["label"] for r in all_results]
    header = f"{'类别':<8} {'Prompt':<20}" + "".join(f" {l:<12}" for l in labels)
    print(header)
    print("-" * len(header))

    # 每个 prompt 的得分
    for i, item in enumerate(EVAL_PROMPTS):
        row = f"{item['category']:<8} {item['prompt']:<20}"
        for r in all_results:
            score = r["results"][i]["score"]["total"]
            row += f" {score:<12.2f}"
        print(row)

    # 平均分
    print("-" * len(header))
    avg_row = f"{'平均':<8} {'':<20}"
    for r in all_results:
        avg_row += f" {r['avg_score']:<12.2f}"
    print(avg_row)


def main():
    parser = argparse.ArgumentParser(description="结构化评测")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="模型 checkpoint 路径列表")
    parser.add_argument("--labels", nargs="+", default=None, help="模型标签（默认用文件名）")
    parser.add_argument("--output", type=str, default=None, help="输出 JSON 路径")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    # 加载 tokenizer
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load("tokenizer/bpe.model")

    # 确定标签
    labels = args.labels
    if labels is None:
        labels = [Path(cp).parent.name for cp in args.checkpoints]

    # 评测每个模型
    all_results = []
    for ckpt, label in zip(args.checkpoints, labels):
        model = load_model(ckpt, device)
        result = benchmark_model(model, tokenizer, label, device)
        all_results.append(result)
        del model  # 释放显存
        torch.cuda.empty_cache()

    # 打印对比表格
    print_comparison_table(all_results)

    # 保存结果
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存: {output_path}")


if __name__ == "__main__":
    main()
