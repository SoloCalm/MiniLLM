"""
Smoke Test：一条命令验证环境

检查依赖 → 加载模型 → 前向传播 → 生成 → tokenizer，确认一切正常。

用法：
    python scripts/smoke_test.py
    python scripts/smoke_test.py --checkpoint outputs/dpo/ckpt_final.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def check_imports():
    """检查核心依赖"""
    print("[1/5] 检查依赖...")
    deps = ["torch", "sentencepiece", "numpy"]
    for dep in deps:
        try:
            __import__(dep)
            print(f"  ✅ {dep}")
        except ImportError:
            print(f"  ❌ {dep} 未安装")
            return False
    return True


def check_tokenizer():
    """检查 tokenizer"""
    print("[2/5] 检查 tokenizer...")
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    if not sp.Load("tokenizer/bpe.model"):
        print("  ❌ tokenizer 加载失败")
        return False

    # 编解码测试
    text = "你好世界"
    ids = sp.encode(text)
    decoded = sp.decode(ids)
    if decoded != text:
        print(f"  ❌ 编解码不一致: {text} -> {ids} -> {decoded}")
        return False

    print(f"  ✅ tokenizer 正常 (vocab={sp.get_piece_size()}, encode='{text}' -> {ids})")
    return True


def check_model(checkpoint_path: str):
    """检查模型加载和前向传播"""
    import torch
    from model.config import ModelConfig
    from model.modeling_llm import MiniLLM

    print(f"[3/5] 加载模型: {checkpoint_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  设备: {device}")

    config = ModelConfig()
    model = MiniLLM(config)

    if Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        print(f"  ✅ checkpoint 加载成功")
    else:
        print(f"  ⚠️ checkpoint 不存在，使用随机权重测试")

    model = model.to(device).eval()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  ✅ 模型参数量: {param_count:,} ({param_count/1e6:.1f}M)")
    return model, device


def check_forward(model, device):
    """检查前向传播"""
    print("[4/5] 前向传播测试...")
    import torch
    from model.config import ModelConfig

    config = ModelConfig()
    input_ids = torch.randint(0, config.vocab_size, (1, 32)).to(device)

    with torch.no_grad():
        logits = model(input_ids)

    expected_shape = (1, 32, config.vocab_size)
    if logits.shape != expected_shape:
        print(f"  ❌ 输出 shape 不对: {logits.shape} (expected {expected_shape})")
        return False

    print(f"  ✅ 输出 shape: {logits.shape}")
    return True


def check_generate(model, device):
    """检查生成"""
    print("[5/5] 生成测试...")
    import torch
    import sentencepiece as spm
    from model.config import ModelConfig

    config = ModelConfig()
    sp = spm.SentencePieceProcessor()
    sp.Load("tokenizer/bpe.model")

    input_ids = torch.tensor([[1] + sp.encode("你好") + [2]], dtype=torch.long).to(device)
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=20, temperature=0.7)

    new_tokens = output.shape[1] - input_ids.shape[1]
    print(f"  ✅ 生成 {new_tokens} 个 token (输入 {input_ids.shape[1]} -> 输出 {output.shape[1]})")
    return True


def main():
    parser = argparse.ArgumentParser(description="Smoke Test")
    parser.add_argument("--checkpoint", type=str, default="outputs/dpo/ckpt_final.pt")
    args = parser.parse_args()

    print("=" * 50)
    print("MiniLLM Smoke Test")
    print("=" * 50)
    print()

    passed = 0
    total = 5

    if check_imports():
        passed += 1
    else:
        print("\n❌ 依赖检查失败，请安装: pip install -e '.[all]'")
        sys.exit(1)

    if check_tokenizer():
        passed += 1

    try:
        model, device = check_model(args.checkpoint)
        passed += 1
    except Exception as e:
        print(f"  ❌ 模型加载失败: {e}")
        sys.exit(1)

    if check_forward(model, device):
        passed += 1

    if check_generate(model, device):
        passed += 1

    print()
    print(f"结果: {passed}/{total} 通过")
    if passed == total:
        print("✅ 环境正常，可以开始训练")
    else:
        print("⚠️ 部分检查失败，请查看上方输出")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
