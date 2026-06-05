"""
SFT 数据构造

将原始对话数据转换为 SFT 训练格式。

支持两种输入格式：
  1. messages 格式: [{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}]
  2. instruction 格式: [{"instruction": "...", "input": "...", "output": "..."}]

输出格式（与 data_loader.py 的 SFTDataset 对应）：
  {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}

关键：assistant-only loss mask
  训练时只有 assistant 的回复部分计算 loss，prompt 部分的 labels 设为 -100。
  这确保模型学习"如何回复"，而不是"如何复述问题"。
"""

import json
import argparse
from pathlib import Path


def convert_messages_to_conversations(messages: list) -> dict:
    """将 messages 格式转为 conversations 格式"""
    conversations = []
    for msg in messages:
        role = "human" if msg["role"] == "user" else "gpt"
        conversations.append({"from": role, "value": msg["content"]})
    return {"conversations": conversations}


def convert_instruction_to_conversations(item: dict) -> dict:
    """将 instruction 格式转为 conversations 格式"""
    prompt = item.get("instruction", "")
    input_text = item.get("input", "")
    output_text = item.get("output", "")

    if input_text:
        user_content = f"{prompt}\n{input_text}"
    else:
        user_content = prompt

    return {
        "conversations": [
            {"from": "human", "value": user_content},
            {"from": "gpt", "value": output_text},
        ]
    }


def is_valid_conversation(conv: dict) -> bool:
    """检查对话是否有效"""
    messages = conv.get("conversations", [])
    # 至少有一轮完整的 user-assistant 对话
    if len(messages) < 2:
        return False
    # 第一条必须是 human
    if messages[0].get("from") != "human":
        return False
    # 检查内容不为空
    for msg in messages:
        if not msg.get("value", "").strip():
            return False
    # 过滤过长或过短的对话
    total_len = sum(len(msg["value"]) for msg in messages)
    if total_len < 20 or total_len > 4096:
        return False
    return True


def prepare_sft_data(input_path: Path, output_dir: Path, valid_ratio: float = 0.05):
    """准备 SFT 数据

    参数：
        input_path: 原始 JSONL 文件
        output_dir: 输出目录
        valid_ratio: 验证集比例
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    conversations = []
    total, kept = 0, 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            # 自动检测格式并转换
            if "messages" in data:
                conv = convert_messages_to_conversations(data["messages"])
            elif "conversations" in data:
                conv = {"conversations": data["conversations"]}
            elif "instruction" in data:
                conv = convert_instruction_to_conversations(data)
            else:
                continue

            if is_valid_conversation(conv):
                conversations.append(conv)
                kept += 1

    print(f"SFT 数据准备完成: 总计 {total} 条, 保留 {kept} 条 ({kept/total*100:.1f}%)")

    # 切分 train/valid
    import random
    random.shuffle(conversations)
    split_idx = int(len(conversations) * (1 - valid_ratio))
    train_data = conversations[:split_idx]
    valid_data = conversations[split_idx:]

    # 写入
    for name, data in [("train", train_data), ("valid", valid_data)]:
        path = output_dir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  {name}: {len(data)} 条 → {path}")

    # 统计
    avg_turns = sum(len(c["conversations"]) // 2 for c in conversations) / len(conversations)
    avg_len = sum(
        sum(len(m["value"]) for m in c["conversations"])
        for c in conversations
    ) / len(conversations)

    metadata = {
        "total_raw": total,
        "kept": kept,
        "train": len(train_data),
        "valid": len(valid_data),
        "avg_turns": round(avg_turns, 1),
        "avg_chars": round(avg_len, 0),
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="原始 JSONL 文件路径")
    parser.add_argument("--output-dir", type=Path, default=Path("data/sft_clean"))
    parser.add_argument("--valid-ratio", type=float, default=0.05)
    args = parser.parse_args()

    prepare_sft_data(args.input, args.output_dir, args.valid_ratio)
