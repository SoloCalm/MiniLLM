"""将 ultrafeedback parquet 转换为 jsonl 格式

转换后的格式：
{
    "prompt": "user 的问题",
    "chosen": "chosen 的回答内容",
    "rejected": "rejected 的回答内容"
}

用法：
    python data_utils/convert_ultrafeedback.py --max-samples 10000
"""

import argparse
from pathlib import Path

import pandas as pd
import json


def convert_parquet_to_jsonl(parquet_path: Path, output_path: Path, max_samples: int = None):
    """转换 parquet 为 jsonl"""
    print(f"读取: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    if max_samples:
        df = df.head(max_samples)
        print(f"限制为 {max_samples} 条")

    print(f"总数据量: {len(df)} 条")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, row in df.iterrows():
            # 提取 prompt
            prompt = row["prompt"]

            # 提取 chosen 回答（取 assistant 的内容）
            chosen_messages = row["chosen"]
            chosen_text = ""
            for msg in chosen_messages:
                if msg["role"] == "assistant":
                    chosen_text = msg["content"]
                    break

            # 提取 rejected 回答
            rejected_messages = row["rejected"]
            rejected_text = ""
            for msg in rejected_messages:
                if msg["role"] == "assistant":
                    rejected_text = msg["content"]
                    break

            # 跳过空数据
            if not chosen_text or not rejected_text:
                continue

            # 写入 jsonl
            sample = {
                "prompt": prompt,
                "chosen": chosen_text,
                "rejected": rejected_text,
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

            if (idx + 1) % 10000 == 0:
                print(f"  已处理 {idx + 1} 条")

    print(f"转换完成: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="转换 ultrafeedback 数据")
    parser.add_argument("--data-dir", type=str,
                        default="data/ultrafeedback_binarized/data",
                        help="parquet 数据目录")
    parser.add_argument("--output-dir", type=str,
                        default="data/dpo",
                        help="输出目录")
    parser.add_argument("--max-samples", type=int, default=10000,
                        help="最大样本数（默认 10k）")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # 转换训练集
    convert_parquet_to_jsonl(
        data_dir / "train_prefs-00000-of-00001.parquet",
        output_dir / "train.jsonl",
        args.max_samples,
    )

    # 转换测试集
    convert_parquet_to_jsonl(
        data_dir / "test_prefs-00000-of-00001.parquet",
        output_dir / "test.jsonl",
        min(2000, args.max_samples // 5),
    )


if __name__ == "__main__":
    main()
