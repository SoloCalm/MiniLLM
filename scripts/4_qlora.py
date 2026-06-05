"""脚本 4：QLoRA 微调 Qwen2.5-1.5B

QLoRA = 4-bit 量化基础模型 + LoRA 微调
- 基础模型：4-bit NF4 量化（每个参数 0.5 字节）
- LoRA：16-bit（正常精度）
- 显存需求：~3GB（vs 全参微调 ~12GB）

用法：
    python scripts/4_qlora.py --max-lines 100000 --epochs 1
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from training.data_loader import load_jsonl


def main():
    parser = argparse.ArgumentParser(description="QLoRA 微调 Qwen2.5-1.5B")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--data-path", type=Path, default=Path("data/firefly-train-1.1M/firefly-train-1.1M.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sft_qlora"))
    parser.add_argument("--max-lines", type=int, default=100000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--log-interval", type=int, default=50)
    args = parser.parse_args()

    # ============================================================
    # 步骤 1 - 加载 4-bit 量化模型
    # ============================================================
    print(f"加载模型: {args.model_name}")

    # 4-bit 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                    # 4-bit 量化
        bnb_4bit_quant_type="nf4",            # NF4 量化类型
        bnb_4bit_use_double_quant=True,       # 双重量化（进一步压缩）
        bnb_4bit_compute_dtype=torch.bfloat16,  # 计算时用 bf16
    )

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型（4-bit 量化）
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # 准备模型用于 QLoRA 训练
    model = prepare_model_for_kbit_training(model)

    # ============================================================
    # 步骤 2 - 应用 LoRA
    # ============================================================
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ============================================================
    # 步骤 3 - 加载数据
    # ============================================================
    print(f"加载数据: {args.data_path}")
    items = load_jsonl(args.data_path)
    if args.max_lines:
        items = items[:args.max_lines]
    print(f"数据量: {len(items)} 条")

    # 数据处理：支持多种格式
    texts = []
    for item in items:
        if "instruction" in item:
            # BelleGroup 格式
            prompt = item["instruction"]
            if item.get("input"):
                prompt += "\n" + item["input"]
            response = item.get("output", "")
            texts.append(f"User: {prompt}\nAssistant: {response}")
        elif "conversations" in item:
            # minimind 格式
            parts = []
            for turn in item["conversations"]:
                role = turn["role"]
                content = turn["content"]
                parts.append(f"{role.capitalize()}: {content}")
            texts.append("\n".join(parts))
        elif "input" in item and "target" in item:
            # firefly 格式
            texts.append(f"User: {item['input']}\nAssistant: {item['target']}")

    # tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )

    from datasets import Dataset
    dataset = Dataset.from_dict({"text": texts})
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )

    # 设置 labels（ causal lm 需要）
    def set_labels(examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples

    tokenized_dataset = tokenized_dataset.map(set_labels, batched=True)

    # ============================================================
    # 步骤 4 - 训练
    # ============================================================
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        logging_steps=args.log_interval,
        save_steps=500,
        save_total_limit=3,
        fp16=False,
        bf16=True,
        optim="paged_adamw_8bit",  # 8-bit 优化器，省显存
        gradient_checkpointing=True,  # 梯度检查点，省显存
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    print("开始训练...")
    trainer.train()

    # ============================================================
    # 步骤 5 - 保存 LoRA 权重
    # ============================================================
    model.save_pretrained(str(output_dir / "lora_adapter"))
    tokenizer.save_pretrained(str(output_dir / "lora_adapter"))
    print(f"LoRA adapter 保存到: {output_dir / 'lora_adapter'}")


if __name__ == "__main__":
    main()
