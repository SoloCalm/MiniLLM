"""
Tokenizer 单元测试
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_tokenizer_roundtrip():
    """测试 encode → decode 是否一致"""
    # TODO: 你的代码
    # 1. 加载训练好的 tokenizer
    # 2. 编码一个中文句子
    # 3. 解码回去
    # 4. 断言原始文本 == 解码文本
    print("[SKIP] test_tokenizer_roundtrip: tokenizer 还没训练")


def test_special_tokens():
    """测试特殊 token 是否正确"""
    # TODO: 你的代码
    print("[SKIP] test_special_tokens: tokenizer 还没训练")


if __name__ == "__main__":
    test_tokenizer_roundtrip()
    test_special_tokens()
