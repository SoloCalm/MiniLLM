"""脚本 1：训练 BPE Tokenizer"""
# 直接调用 tokenizer/train_tokenizer.py
# 也可在这里加数据预处理逻辑

import sys
from pathlib import Path

# 把项目根目录加入 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.train_tokenizer import train_bpe_tokenizer, test_tokenizer


def prepare_corpus():
    """准备 tokenizer 训练语料

    从预训练数据中抽取文本，保存为一行一句的纯文本文件。
    sentencepiece 需要纯文本输入。
    """
    import json

    data_path = Path("data/minimind_dataset/pretrain_t2t_mini.jsonl")
    output_path = Path("data/tokenizer_corpus.txt")

    texts = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                texts.append(item["text"])

    with open(output_path, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text + "\n")

    print(f"语料准备完成: {len(texts)} 条文本 → {output_path}")


if __name__ == "__main__":
    prepare_corpus()
    train_bpe_tokenizer(
        corpus_path=Path("data/tokenizer_corpus.txt"),
        vocab_size=6400,
        model_prefix="tokenizer/bpe",
    )
    test_tokenizer("tokenizer/bpe")
