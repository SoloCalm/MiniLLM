"""QLoRA 基线训练脚本

使用 Qwen2.5-1.5B + firefly 数据集进行 QLoRA 微调
目标：作为对比基线，验证大模型微调效果

配置：
- 基础模型：Qwen2.5-1.5B（4-bit NF4 量化）
- LoRA：rank=16, alpha=32
- 数据：firefly-train-1.1M（取前 100k 条）
- 训练：1 epoch，lr=2e-4
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import time
from datetime import datetime

import torch
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
    parser = argparse.ArgumentParser(description="QLoRA 基线训练")
    parser.add_argument("--max-lines", type=int, default=100000,
                        help="最大训练样本数（默认 100k）")
    parser.add_argument("--epochs", type=int, default=1,
                        help="训练轮数（默认 1）")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="每设备 batch size（默认 4）")
    parser.add_argument("--output-dir", type=str,
                        default="outputs/sft_qlora_baseline",
                        help="输出目录")
    args = parser.parse_args()

    # 记录开始时间
    start_time = time.time()
    print(f"=" * 60)
    print(f"QLoRA 基线训练")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"=" * 60)

    # ============================================================
    # 配置参数
    # ============================================================
    model_name = "Qwen/Qwen2.5-1.5B"
    data_path = Path("data/firefly-train-1.1M/firefly-train-1.1M.jsonl")
    output_dir = Path(args.output_dir)

    lora_config = {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
    }

    training_config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": 4,
        "lr": 2e-4,
        "warmup_ratio": 0.03,
        "max_length": 512,
        "bf16": True,
        "gradient_checkpointing": True,
    }

    # 保存配置
    output_dir.mkdir(parents=True, exist_ok=True)
    config_save = {
        "model": model_name,
        "lora": lora_config,
        "training": training_config,
        "data_lines": args.max_lines,
        "start_time": datetime.now().isoformat(),
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_save, f, indent=2, ensure_ascii=False)
    print(f"配置已保存到: {output_dir / 'config.json'}")

    # ============================================================
    # 步骤 1 - 加载 4-bit 量化模型
    # ============================================================
    print(f"\n[1/5] 加载模型: {model_name}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model)
    print(f"模型加载完成，显存占用: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # ============================================================
    # 步骤 2 - 应用 LoRA
    # ============================================================
    print(f"\n[2/5] 应用 LoRA (r={lora_config['r']}, alpha={lora_config['lora_alpha']})")

    lora_config_obj = LoraConfig(
        r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=lora_config["lora_dropout"],
        target_modules=lora_config["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config_obj)
    model.print_trainable_parameters()

    # ============================================================
    # 步骤 3 - 加载数据
    # ============================================================
    print(f"\n[3/5] 加载数据: {data_path}")

    items = load_jsonl(data_path)
    if args.max_lines:
        items = items[:args.max_lines]
    print(f"数据量: {len(items)} 条")

    # 数据处理：firefly 格式
    texts = []
    for item in items:
        if "input" in item and "target" in item:
            texts.append(f"User: {item['input']}\nAssistant: {item['target']}")
        elif "instruction" in item:
            prompt = item["instruction"]
            if item.get("input"):
                prompt += "\n" + item["input"]
            response = item.get("output", "")
            texts.append(f"User: {prompt}\nAssistant: {response}")

    print(f"处理后的对话数: {len(texts)}")

    # tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=training_config["max_length"],
            padding="max_length",
        )

    from datasets import Dataset
    dataset = Dataset.from_dict({"text": texts})
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )

    # 设置 labels
    def set_labels(examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples

    tokenized_dataset = tokenized_dataset.map(set_labels, batched=True)
    print(f"Tokenized 数据量: {len(tokenized_dataset)} 条")

    # ============================================================
    # 步骤 4 - 训练
    # ============================================================
    print(f"\n[4/5] 开始训练...")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=training_config["epochs"],
        per_device_train_batch_size=training_config["batch_size"],
        gradient_accumulation_steps=training_config["grad_accum"],
        learning_rate=training_config["lr"],
        warmup_ratio=training_config["warmup_ratio"],
        logging_steps=50,
        save_steps=500,
        save_total_limit=3,
        fp16=False,
        bf16=training_config["bf16"],
        optim="paged_adamw_8bit",
        gradient_checkpointing=training_config["gradient_checkpointing"],
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()

    # ============================================================
    # 步骤 5 - 保存
    # ============================================================
    print(f"\n[5/5] 保存模型...")

    model.save_pretrained(str(output_dir / "lora_adapter"))
    tokenizer.save_pretrained(str(output_dir / "lora_adapter"))

    # 记录训练结果
    end_time = time.time()
    duration = end_time - start_time
    final_stats = {
        "end_time": datetime.now().isoformat(),
        "duration_seconds": duration,
        "duration_hours": duration / 3600,
        "total_steps": trainer.state.global_step,
        "final_loss": trainer.state.log_history[-1].get("loss") if trainer.state.log_history else None,
    }

    with open(output_dir / "config.json", "r") as f:
        config = json.load(f)
    config.update(final_stats)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"训练完成!")
    print(f"耗时: {duration/3600:.2f} 小时")
    print(f"最终 loss: {final_stats.get('final_loss', 'N/A')}")
    print(f"模型保存到: {output_dir / 'lora_adapter'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
