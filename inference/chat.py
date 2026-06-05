"""
交互式命令行对话

用法：
    python inference/chat.py --checkpoint outputs/dpo/ckpt_final.pt
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import sentencepiece as spm

from model.config import ModelConfig
from model.modeling_llm import MiniLLM


def load_model(checkpoint_path: str, device: str = "cuda"):
    """加载模型"""
    config = ModelConfig()
    model = MiniLLM(config)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()

    return model


def generate_response(
    model: MiniLLM,
    tokenizer,
    user_message: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: str = "cuda",
):
    """生成回答

    训练格式：[BOS] user_text [EOS] assistant_text [EOS]
    推理时：  [BOS] user_text [EOS] → 模型生成 assistant_text
    """
    bos_id = 1
    eos_id = 2

    prompt_ids = tokenizer.encode(user_message)
    input_ids = [bos_id] + prompt_ids + [eos_id]
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        output = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=50,
        )

    generated_ids = output[0].cpu().tolist()
    response_ids = generated_ids[len(input_ids):]
    response_ids = [tid for tid in response_ids if tid not in [eos_id, 0]]

    if response_ids:
        response = tokenizer.decode(response_ids)
    else:
        response = "(模型未生成回答)"

    return response


def chat_loop(model, tokenizer, max_new_tokens: int = 256, temperature: float = 0.7):
    """交互式对话循环（保留最近 5 轮历史）"""
    print("=" * 50)
    print("MiniLLM 对话系统")
    print("  输入 quit 退出")
    print("  输入 clear 清空历史")
    print("=" * 50)

    history = []  # [(user, assistant), ...]

    while True:
        user_input = input("\n你: ").strip()
        if user_input.lower() == "quit":
            print("再见！")
            break

        if user_input.lower() == "clear":
            history.clear()
            print("(历史已清空)")
            continue

        if not user_input:
            continue

        # 构造带历史的 prompt
        prompt = ""
        for h in history[-5:]:  # 最近 5 轮
            prompt += f"{h[0]}\n{h[1]}\n"
        prompt += user_input

        response = generate_response(model, tokenizer, prompt,
                                     max_new_tokens=max_new_tokens,
                                     temperature=temperature)
        print(f"助手: {response}")

        history.append((user_input, response))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="模型 checkpoint 路径")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load("tokenizer/bpe.model")

    print(f"加载模型: {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    print("模型加载完成")

    chat_loop(model, tokenizer, args.max_new_tokens, args.temperature)
