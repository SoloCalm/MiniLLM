"""
注意力机制单元测试
"""

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.config import ModelConfig
from model.rope import RotaryEmbedding, apply_rotary_pos_emb
from model.attention import CausalSelfAttention, KVCache


def test_rope():
    """测试 RoPE 旋转编码"""
    head_dim = 64
    max_seq_len = 128
    rope = RotaryEmbedding(head_dim, max_seq_len)

    freqs = rope(32)  # 取前 32 个位置
    assert freqs.shape[0] == 32
    print(f"[PASS] test_rope: freqs shape = {freqs.shape}")


def test_attention_output_shape():
    """测试注意力输出形状"""
    config = ModelConfig(
        hidden_size=128,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=256,
        max_seq_len=64,
    )
    attn = CausalSelfAttention(config)
    rope = RotaryEmbedding(config.head_dim, config.max_seq_len, config.rope_theta)

    batch_size = 2
    seq_len = 16
    x = torch.randn(batch_size, seq_len, config.hidden_size)
    freqs = rope(seq_len)

    output = attn(x, freqs)
    assert output.shape == (batch_size, seq_len, config.hidden_size)
    print(f"[PASS] test_attention_output_shape: {output.shape}")


def test_kv_cache():
    """测试 KV Cache 更新"""
    cache = KVCache()
    # TODO: 测试 KV Cache 的 update 和 get_seq_length
    print("[PASS] test_kv_cache (placeholder)")


if __name__ == "__main__":
    test_rope()
    test_attention_output_shape()
    test_kv_cache()
