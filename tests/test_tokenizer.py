"""
Tokenizer 单元测试
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import sentencepiece as spm


def test_tokenizer_roundtrip():
    """测试 encode → decode 是否一致"""
    # 1. 加载训练好的 tokenizer
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load("tokenizer/bpe.model")

    # 2. 编码一个中文句子
    text = "你好，世界！"
    encoded = tokenizer.Encode(text)

    # 3. 解码回去
    decoded = tokenizer.Decode(encoded)

    # 4. 断言原始文本 == 解码文本
    assert decoded == text, f"Roundtrip failed: {text} → {decoded}"
    print(f"[PASS] test_tokenizer_roundtrip: '{text}' → {encoded} → '{decoded}'")


def test_special_tokens():
    """测试特殊 token 是否正确"""
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load("tokenizer/bpe.model")

    # 检查特殊 token 的 id
    assert tokenizer.bos_id() == 1, f"bos_id should be 1, got {tokenizer.bos_id()}"
    assert tokenizer.eos_id() == 2, f"eos_id should be 2, got {tokenizer.eos_id()}"
    assert tokenizer.pad_id() == 0, f"pad_id should be 0, got {tokenizer.pad_id()}"

    print(f"[PASS] test_special_tokens: bos={tokenizer.bos_id()}, eos={tokenizer.eos_id()}, pad={tokenizer.pad_id()}")


def test_vocab_size():
    """测试词表大小"""
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load("tokenizer/bpe.model")

    vocab_size = tokenizer.GetPieceSize()
    assert vocab_size == 6400, f"vocab_size should be 6400, got {vocab_size}"
    print(f"[PASS] test_vocab_size: {vocab_size}")


if __name__ == "__main__":
    test_tokenizer_roundtrip()
    test_special_tokens()
    test_vocab_size()
