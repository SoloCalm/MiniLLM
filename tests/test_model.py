"""
模型架构单元测试

验证每个组件的输出形状和数值正确性。
"""

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.config import ModelConfig
from model.modeling_llm import MiniLLM


def test_model_forward():
    """测试模型前向传播"""
    config = ModelConfig(
        hidden_size=128,      # 用小配置快速测试
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=256,
        max_seq_len=64,
    )
    model = MiniLLM(config)

    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits = model(input_ids)

    # 验证输出形状
    assert logits.shape == (batch_size, seq_len, config.vocab_size), \
        f"Expected {(batch_size, seq_len, config.vocab_size)}, got {logits.shape}"

    print(f"[PASS] test_model_forward: {logits.shape}")


def test_model_generate():
    """测试模型生成"""
    config = ModelConfig(
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=256,
        max_seq_len=64,
    )
    model = MiniLLM(config)
    model.eval()

    prompt = torch.randint(0, config.vocab_size, (1, 8))
    generated = model.generate(prompt, max_new_tokens=10, temperature=0.0)

    assert generated.shape[0] == 1
    assert generated.shape[1] == prompt.shape[1] + 10

    print(f"[PASS] test_model_generate: {generated.shape}")


def test_parameter_count():
    """测试参数量"""
    config = ModelConfig()
    model = MiniLLM(config)
    count = model.count_parameters()
    print(f"[INFO] 参数量: {count / 1e6:.1f}M")


if __name__ == "__main__":
    test_model_forward()
    test_model_generate()
    test_parameter_count()
