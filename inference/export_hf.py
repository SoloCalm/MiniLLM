"""导出模型为 HuggingFace 格式

将自研 ~38M 模型导出为 HuggingFace 格式，支持：
- config.json：模型配置
- model.safetensors：模型权重
- tokenizer.json：tokenizer 配置

用法：
    python inference/export_hf.py --checkpoint outputs/dpo/ckpt_final.pt --output-dir outputs/hf_model
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json

import torch
import sentencepiece as spm

from model.config import ModelConfig
from model.modeling_llm import MiniLLM


def export_to_huggingface(checkpoint_path: str, output_dir: str):
    """导出模型为 HuggingFace 格式"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载模型
    print(f"加载模型: {checkpoint_path}")
    config = ModelConfig()
    model = MiniLLM(config)

    # checkpoint 包含 model state_dict 和 config
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"模型参数量: {model.count_parameters() / 1e6:.1f}M")

    # 2. 保存 config.json
    hf_config = {
        "architectures": ["MiniLLM"],
        "model_type": "minillm",
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_attention_heads": config.num_heads,
        "num_hidden_layers": config.num_layers,
        "num_key_value_heads": config.num_kv_heads,
        "vocab_size": config.vocab_size,
        "max_position_embeddings": config.max_seq_len,
        "rms_norm_eps": config.rms_norm_eps,
        "rope_theta": config.rope_theta,
        "bos_token_id": config.bos_token_id,
        "eos_token_id": config.eos_token_id,
        "pad_token_id": config.pad_token_id,
    }

    config_path = output_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(hf_config, f, indent=2, ensure_ascii=False)
    print(f"配置已保存: {config_path}")

    # 3. 保存模型权重（pytorch 格式，因为有 weight tying）
    # 注意：模型有 weight tying（tok_emb 和 lm_head 共享权重）
    # safetensors 不支持共享张量，所以使用 pytorch 格式
    pt_path = output_dir / "pytorch_model.bin"
    torch.save(model.state_dict(), pt_path)
    print(f"权重已保存: {pt_path}")

    # 4. 复制 tokenizer
    tokenizer_src = Path("tokenizer/bpe.model")
    if tokenizer_src.exists():
        tokenizer_dst = output_dir / "bpe.model"
        import shutil
        shutil.copy(tokenizer_src, tokenizer_dst)
        print(f"Tokenizer 已保存: {tokenizer_dst}")

    # 5. 创建 tokenizer_config.json
    tokenizer_config = {
        "model_type": "sentencepiece",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
    }

    tokenizer_config_path = output_dir / "tokenizer_config.json"
    with open(tokenizer_config_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
    print(f"Tokenizer 配置已保存: {tokenizer_config_path}")

    print(f"\n导出完成！输出目录: {output_dir}")
    print(f"文件列表:")
    for file in output_dir.iterdir():
        print(f"  - {file.name}")


def main():
    parser = argparse.ArgumentParser(description="导出模型为 HuggingFace 格式")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型 checkpoint 路径")
    parser.add_argument("--output-dir", type=str, default="outputs/hf_model", help="输出目录")
    args = parser.parse_args()

    export_to_huggingface(args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
