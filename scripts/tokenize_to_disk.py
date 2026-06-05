"""预 tokenize 脚本：把 JSONL 转成 .npy 文件

问题：
  原来的 PretrainDataset 在 __init__ 中一次性 tokenize 所有文本，
  127 万条文本 → 3.95 亿 token → Python list 占 ~11GB 内存 → OOM

解决方案（参考 MicroLM）：
  1. 提前 tokenize 所有文本
  2. 用 numpy uint16 存储（2 字节/token，vs Python int 28 字节/token）
  3. 用 open_memmap 写入磁盘，训练时内存映射按需加载

用法：
  python scripts/tokenize_to_disk.py
  python scripts/tokenize_to_disk.py --max-lines 50000  # 只处理前 5 万行（测试用）
"""

import json
import argparse
from pathlib import Path

import numpy as np
import sentencepiece as spm


def main():
    parser = argparse.ArgumentParser(description="预 tokenize JSONL → .npy")
    parser.add_argument("--data-path", type=Path, default=Path("data/minimind_dataset/pretrain_t2t_mini.jsonl"))
    parser.add_argument("--tokenizer-path", type=str, default="tokenizer/bpe.model")
    parser.add_argument("--output-dir", type=Path, default=Path("data/pretrain_tokenized"))
    parser.add_argument("--max-lines", type=int, default=None, help="限制处理行数（测试用）")
    args = parser.parse_args()

    # 加载 tokenizer
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(args.tokenizer_path)
    print(f"词表大小: {tokenizer.GetPieceSize()}")

    # 读取 JSONL
    print(f"读取 {args.data_path} ...")
    items = []
    with open(args.data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
            if args.max_lines and len(items) >= args.max_lines:
                break
    texts = [item["text"] for item in items]
    print(f"加载 {len(texts)} 条文本")

    # 统计总 token 数（第一遍遍历）
    print("统计 token 数量...")
    total_tokens = 0
    for text in texts:
        tokens = tokenizer.encode(text)
        total_tokens += len(tokens)
    print(f"总 token 数: {total_tokens}")

    # 写入 .npy 文件（内存映射）
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "train_ids.npy"

    print(f"写入 {out_path} ...")
    array = np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=np.uint16,
        shape=(total_tokens,),
    )

    offset = 0
    for i, text in enumerate(texts):
        tokens = tokenizer.encode(text)
        array[offset:offset + len(tokens)] = tokens
        offset += len(tokens)
        if (i + 1) % 100000 == 0:
            print(f"  已处理 {i + 1}/{len(texts)} 条")

    array.flush()
    print(f"写入完成: {out_path} ({total_tokens} tokens, {out_path.stat().st_size / 1e6:.1f} MB)")

    # 保存 metadata
    metadata = {
        "data_path": str(args.data_path),
        "tokenizer_path": args.tokenizer_path,
        "vocab_size": tokenizer.GetPieceSize(),
        "num_tokens": total_tokens,
        "num_texts": len(texts),
        "dtype": "uint16",
        "token_ids_path": str(out_path),
    }
    metadata_path = args.output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"metadata 保存到: {metadata_path}")


if __name__ == "__main__":
    main()
