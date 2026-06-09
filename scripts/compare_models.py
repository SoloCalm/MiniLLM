"""41M vs 1.5B 生成效果对比

用 20 个固定问题，让两个模型分别生成回答，对比效果。

用法：
    python scripts/compare_models.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import sentencepiece as spm

from model.config import ModelConfig
from model.modeling_llm import MiniLLM


# 20 个固定测试问题
TEST_PROMPTS = [
    "你好",
    "今天天气怎么样",
    "人工智能是什么",
    "中国的首都是哪里",
    "写一首诗",
    "如何学习编程",
    "解释一下量子计算",
    "推荐一本好书",
    "怎样保持健康",
    "什么是机器学习",
    "如何提高工作效率",
    "谈谈你的看法",
    "写一个故事",
    "解释深度学习",
    "什么是大语言模型",
    "如何训练一个模型",
    "谈谈人工智能的未来",
    "写一段代码",
    "解释神经网络",
    "什么是Transformer",
]


def load_mini_model(ckpt_path, device):
    """加载自研 41M 模型"""
    config = ModelConfig()
    model = MiniLLM(config)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()

    return model


def load_qwen_model(adapter_path, device):
    """加载 Qwen2.5-1.5B + QLoRA"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    # 4-bit 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # 加载 LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    return model, tokenizer


def generate_mini_response(model, tokenizer, prompt, max_new_tokens=100, device="cuda"):
    """用 41M 模型生成回答"""
    prompt_ids = tokenizer.encode(prompt)
    bos_id = 1
    eos_id = 2
    input_ids = [bos_id] + prompt_ids + [eos_id]

    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        output = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
        )

    generated_ids = output[0].cpu().tolist()
    response_ids = generated_ids[len(input_ids):]
    response_ids = [tid for tid in response_ids if tid not in [eos_id, 0]]

    if response_ids:
        response = tokenizer.decode(response_ids)
    else:
        response = "(空)"

    return response


def generate_qwen_response(model, tokenizer, prompt, max_new_tokens=100, device="cuda"):
    """用 Qwen 1.5B 模型生成回答"""
    messages = [
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    # 只取生成的部分
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


def main():
    import argparse
    parser = argparse.ArgumentParser(description="41M vs 1.5B 生成效果对比")
    parser.add_argument("--tokenizer-path", type=str, default="tokenizer/bpe.model")  # tokenizer 路径
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 加载 tokenizer
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(args.tokenizer_path)

    # 加载 41M 模型
    print("\n加载 41M 模型...")
    mini_model = load_mini_model("outputs/dpo/ckpt_final.pt", device)
    print("41M 模型加载完成")

    # 加载 Qwen 1.5B 模型
    print("\n加载 Qwen 1.5B + QLoRA 模型...")
    qwen_model, qwen_tokenizer = load_qwen_model("outputs/sft_qlora/lora_adapter", device)
    print("Qwen 1.5B 模型加载完成")

    # 生成对比结果
    results = []
    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"\n[{i+1}/20] {prompt}")

        # 41M 模型生成
        mini_response = generate_mini_response(mini_model, tokenizer, prompt, device=device)
        print(f"  41M: {mini_response[:80]}...")

        # Qwen 1.5B 生成
        qwen_response = generate_qwen_response(qwen_model, qwen_tokenizer, prompt, device=device)
        # 处理编码问题
        try:
            print(f"  1.5B: {qwen_response[:80]}...")
        except UnicodeEncodeError:
            print(f"  1.5B: (编码问题，跳过打印)")

        results.append({
            "prompt": prompt,
            "mini_41m": mini_response,
            "qwen_1_5b": qwen_response,
        })

    # 保存结果
    output_file = Path("results/model_comparison.txt")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("41M vs 1.5B 生成效果对比\n")
        f.write("=" * 60 + "\n\n")

        for r in results:
            f.write(f"Prompt: {r['prompt']}\n")
            f.write(f"41M: {r['mini_41m']}\n")
            f.write(f"1.5B: {r['qwen_1_5b']}\n")
            f.write("-" * 60 + "\n")

    print(f"\n结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
