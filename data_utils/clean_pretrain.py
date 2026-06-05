"""
预训练语料清洗

从原始 JSONL 文件中提取文本，清洗后输出纯文本文件（一行一句）。

清洗步骤：
  1. 提取 text 字段
  2. 过滤过短的文本（< 10 字）
  3. 过滤非中文占比过高的文本
  4. 去除特殊标记（<|im_start|> 等）
  5. 按比例切分 train/valid
"""

import json
import re
import argparse
from pathlib import Path
from collections import Counter
from typing import Optional


def is_chinese_char(ch: str) -> bool:
    """判断是否为中文字符"""
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF or   # CJK 统一汉字
        0x3400 <= cp <= 0x4DBF or   # CJK 扩展 A
        0x20000 <= cp <= 0x2A6DF    # CJK 扩展 B
    )


def clean_text(text: str) -> Optional[str]:
    """清洗单条文本，返回 None 表示应丢弃"""
    # 去除特殊标记
    text = re.sub(r"<\|im_start\|>|<\|im_end\|>|<\|think\|>|<\|/think\|>", "", text)
    text = text.strip()

    # 过滤过短文本
    if len(text) < 10:
        return None

    # 过滤中文占比过低的文本（< 30%）
    chinese_count = sum(1 for ch in text if is_chinese_char(ch))
    if chinese_count / len(text) < 0.3:
        return None

    # 过滤重复行过多的文本（常见于爬虫数据）
    lines = text.split("\n")
    lines = [l.strip() for l in lines if l.strip()]
    if len(lines) > 5:
        counts = Counter(lines)
        most_common_ratio = counts.most_common(1)[0][1] / len(lines)
        if most_common_ratio > 0.5:
            return None

    return text


def clean_corpus(input_path: Path, output_dir: Path, valid_ratio: float = 0.02):
    """清洗语料并切分 train/valid

    参数：
        input_path: 原始 JSONL 文件路径（每行一个 {"text": "..."} ）
        output_dir: 输出目录
        valid_ratio: 验证集比例
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    texts = []
    total, kept, dropped = 0, 0, 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            try:
                data = json.loads(line)
                raw_text = data.get("text", "")
            except json.JSONDecodeError:
                dropped += 1
                continue

            cleaned = clean_text(raw_text)
            if cleaned:
                texts.append(cleaned)
                kept += 1
            else:
                dropped += 1

    print(f"清洗完成: 总计 {total} 条, 保留 {kept} 条 ({kept/total*100:.1f}%), 丢弃 {dropped} 条")

    # 切分 train/valid
    split_idx = int(len(texts) * (1 - valid_ratio))
    train_texts = texts[:split_idx]
    valid_texts = texts[split_idx:]

    # 写入文件
    train_path = output_dir / "train.txt"
    valid_path = output_dir / "valid.txt"

    with open(train_path, "w", encoding="utf-8") as f:
        f.write("\n".join(train_texts))

    with open(valid_path, "w", encoding="utf-8") as f:
        f.write("\n".join(valid_texts))

    # Tokenizer 训练语料（取前 15MB）
    tokenizer_path = output_dir / "tokenizer_corpus.txt"
    all_text = "\n".join(texts)
    sample = all_text[:15 * 1024 * 1024]
    with open(tokenizer_path, "w", encoding="utf-8") as f:
        f.write(sample)

    # 写 metadata
    metadata = {
        "total_raw": total,
        "kept": kept,
        "dropped": dropped,
        "train_lines": len(train_texts),
        "valid_lines": len(valid_texts),
        "train_chars": sum(len(t) for t in train_texts),
        "valid_chars": sum(len(t) for t in valid_texts),
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"  训练集: {len(train_texts)} 行 → {train_path}")
    print(f"  验证集: {len(valid_texts)} 行 → {valid_path}")
    print(f"  Tokenizer 语料: {len(sample)} 字节 → {tokenizer_path}")

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="原始 JSONL 文件路径")
    parser.add_argument("--output-dir", type=Path, default=Path("data/pretrain_clean"))
    parser.add_argument("--valid-ratio", type=float, default=0.02)
    args = parser.parse_args()

    clean_corpus(args.input, args.output_dir, args.valid_ratio)
