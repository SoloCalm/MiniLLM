"""
BPE Tokenizer 训练

BPE（Byte Pair Encoding）原理：
  1. 从字符级别的词表开始
  2. 统计所有相邻字符对的出现频率
  3. 把频率最高的字符对合并成新 token
  4. 重复步骤 2-3，直到词表达到目标大小

为什么用 BPE：
  - 平衡词表大小和 OOV 率
  - 常见词保留为一个 token，罕见词拆成 subword
  - GPT、LLaMA、Qwen 都用 BPE

本项目用 sentencepiece 库实现。
"""

import argparse                    # 命令行参数解析，让脚本支持 --corpus、--vocab-size 等参数
from pathlib import Path           # 路径处理，比字符串拼接更安全

import sentencepiece as spm        # Google 的分词库，支持 BPE、Unigram 等多种算法


def train_bpe_tokenizer(
    corpus_path: Path,
    vocab_size: int = 6400,
    model_prefix: str = "tokenizer/bpe",
    model_type: str = "bpe",
):
    """训练 BPE Tokenizer

    参数：
        corpus_path: 语料文件路径（纯文本，一行一句）
        vocab_size: 词表大小
        model_prefix: 输出模型前缀
        model_type: "bpe" 或 "unigram"

    生成文件：
        {model_prefix}.model  → 模型文件（加载用）
        {model_prefix}.vocab  → 词表文件（查看用）
    """
    spm.SentencePieceTrainer.train(
        input = str(corpus_path),    # 语料文件路径，必须是纯文本，一行一句
        model_prefix = model_prefix, # 输出文件的前缀，会生成 bpe.model 和 bpe.vocab
        vocab_size = vocab_size,     # 词表大小，6400 意味着最终有 6400 个 token
        model_type = model_type,     # "bpe" 表示用字节对编码，还有 "unigram" 等选项
        character_coverage=0.9995,   # 字符覆盖率：99.95% 的字符会被保留
                                     # 中文需要高覆盖率，否则生僻字会被丢弃
        pad_id=0,                    # <pad> 填充 token，ID=0，用于 batch 中补齐短序列
        bos_id=1,                    # <bos> 句子开始 token，ID=1
        eos_id=2,                    # <eos> 句子结束 token，ID=2
        unk_id=3,                    # <unk> 未知 token，ID=3，遇到不认识的字符用它
    )


def test_tokenizer(model_path: str):
    """测试 Tokenizer 的编码/解码效果"""

    # 第 1 步：加载训练好的 tokenizer 模型
    tokenizer = spm.SentencePieceProcessor()        # 创建处理器实例
    tokenizer.Load(model_path + ".model")            # 加载 .model 文件
    print(f"词表大小: {tokenizer.GetPieceSize()}")   # 打印词表大小，应该是 6400

    # 第 2 步：准备测试句子（短句、长句、中英混合都覆盖）
    test_sentences = [
        "你好，世界！",                                          # 短句，标点
        "大语言模型是人工智能的重要方向。",                          # 中等长度
        "从零训练一个中文大模型需要大量的计算资源和高质量数据。",    # 长句
        "PyTorch 是深度学习的主流框架。",                          # 中英混合
    ]

    # 第 3 步：逐句测试 encode → decode
    total_tokens = 0
    for sentence in test_sentences:
        tokens = tokenizer.encode(sentence)    # 编码：字符串 → token ID 列表
        decoded = tokenizer.decode(tokens)     # 解码：token ID 列表 → 字符串
        total_tokens += len(tokens)            # 累计 token 数量

        # 检查解码结果是否和原文一致（不一致说明 tokenizer 有 bug）
        match = "✓" if decoded == sentence else "✗"
        print(f"  {match} [{len(tokens)} tokens] {sentence}")
        if decoded != sentence:
            print(f"      原文: {sentence}")
            print(f"      解码: {decoded}")

    # 第 4 步：统计编码效率
    # 平均 tokens/句子 越少，说明编码效率越高（一个 token 承载更多信息）
    avg_tokens = total_tokens / len(test_sentences)
    print(f"平均 tokens/句子: {avg_tokens:.1f}")


if __name__ == "__main__":
    # 命令行参数解析，允许在终端中自定义训练参数
    # 用法示例：python tokenizer/train_tokenizer.py --corpus data/wiki.txt --vocab-size 16000
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("data/tokenizer_corpus.txt"))  # 语料路径
    parser.add_argument("--vocab-size", type=int, default=6400)                            # 词表大小
    parser.add_argument("--output-prefix", type=str, default="tokenizer/bpe")              # 输出前缀
    args = parser.parse_args()

    # 第 1 步：训练 tokenizer（生成 bpe.model 和 bpe.vocab）
    train_bpe_tokenizer(args.corpus, args.vocab_size, args.output_prefix)
    # 第 2 步：立即测试，确认编码/解码正常
    test_tokenizer(args.output_prefix)
