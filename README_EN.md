# MiniLLM

Dual-track LLM training project on a single **RTX 4050 6GB** GPU:

- **Track A (Self-built 38M):** Hand-written LLaMA2-style Transformer → Pretrain → SFT → LoRA → DPO
- **Track B (Qwen2.5-1.5B):** QLoRA fine-tuning → HuggingFace export + CLI chat

## Key Results

### Track A — Self-built 38M

| Metric | Value |
|--------|-------|
| Architecture | LLaMA2-style (GQA + SwiGLU + RoPE + RMSNorm) |
| Parameters | ~38M (after weight tying) |
| Perplexity | 13.03 |
| DPO Reward Margin (β=0.2) | 0.2236 |
| LoRA r=8 Trainable Params | 887,808 (2.28%) |
| Ablation Experiments | 5 (LR / LoRA rank / FT vs LoRA / DPO β / 38M vs 1.5B) |

### Track B — Qwen2.5-1.5B QLoRA

| Metric | Value |
|--------|-------|
| Base Model | Qwen2.5-1.5B (1.55B params) |
| Quantization | 4-bit NF4 (BitsAndBytes) |
| LoRA Params | 659,456 (0.04% of base) |
| Deployment | HuggingFace export + CLI chat |

## 📊 Experiment Results

Complete experiment results are available in the `results/` directory.

### 1. Pretraining Learning Rate

| LR | Steps | Convergence | Final Loss |
|----|-------|-------------|------------|
| 1e-4 | 10k | Slow | Higher |
| 3e-4 | 50k | Fast | Lower |

**Conclusion:** 3e-4 converges faster with better final performance.

### 2. LoRA Rank

| Rank | Trainable Params | Ratio | GPU Memory |
|------|-----------------|-------|------------|
| r=4 | 443,904 | 1.15% | 1.05 GB |
| r=8 | 887,808 | 2.28% | 1.20 GB |
| r=16 | 1,775,616 | 4.45% | 1.36 GB |

**Conclusion:** r=8 offers the best cost-performance ratio.

### 3. Full Fine-tuning vs LoRA

| Method | Trainable Params | Ratio | Peak Memory |
|--------|-----------------|-------|-------------|
| Full FT | 38,089,216 | 100% | 1.27 GB |
| LoRA | 887,808 | 2.28% | 1.05 GB |

**Conclusion:** LoRA achieves comparable results with 2.28% of parameters, saving 17% GPU memory.

### 4. DPO Beta

| β | Margin | Character |
|---|--------|-----------|
| 0.1 | 0.1222 | Aggressive — deviates far from reference |
| 0.2 | 0.2236 | Balanced — recommended |
| 0.5 | 0.3199 | Conservative — stays close to reference |

**Conclusion:** β=0.2 is the sweet spot with moderate margin and low loss.

### 5. Model Size: 38M vs 1.5B

| Model | Response Rate | Quality |
|-------|--------------|---------|
| MiniLLM 38M | 15% | Mostly empty, limited expressiveness |
| Qwen2.5-1.5B QLoRA | 70% | Has content but with repetition |

**Conclusion:** Larger models with QLoRA fine-tuning produce significantly better results.

### Perplexity

| Model | Perplexity |
|-------|------------|
| DPO Final | 13.03 |

## Model Architecture

| Config | Value |
|--------|-------|
| Architecture | Decoder-only Transformer (LLaMA2-style) |
| Parameters | ~38M (after weight tying) |
| Layers | 12 |
| Hidden Size | 512 |
| Attention Heads | 8Q / 4KV (GQA) |
| FFN | SwiGLU (intermediate=1376) |
| Positional Encoding | RoPE (θ=10000) |
| Normalization | RMSNorm (Pre-norm) |
| Vocab Size | 6,400 (BPE) |
| Max Sequence Length | 1024 |

## ✨ Highlights

- 🎯 **Full Pipeline** - From tokenizer training to model deployment, complete LLM training pipeline
- 💻 **Single GPU** - Runs on RTX 4050 6GB, lowering the barrier to entry
- 📝 **Hand-written** - From-zero Transformer implementation, no black-box libraries
- 📊 **Complete Experiments** - 5 ablation studies with data-driven conclusions
- 📚 **Detailed Documentation** - Bilingual (EN/CN), 11 core code analysis files
- 🎓 **Interview Friendly** - Covers high-frequency interview topics

## Comparison with MiniMind

MiniLLM is an advanced project based on MiniMind, with significant improvements:

| Dimension | MiniLLM | MiniMind |
|-----------|---------|----------|
| **Architecture** | LLaMA2-style (GQA + RoPE + SwiGLU) | Standard Transformer |
| **Attention** | GQA (Grouped-Query Attention) | MHA (Multi-Head Attention) |
| **Position Encoding** | RoPE (Rotary Position Embedding) | Absolute Position Encoding |
| **Activation** | SwiGLU | ReLU/GELU |
| **Normalization** | RMSNorm | LayerNorm |
| **LoRA** | Hand-written implementation | Depends on PEFT library |
| **Ablation Studies** | 5 comprehensive experiments | Basic experiments |
| **Documentation** | 11 core code analysis files | Basic documentation |

### MiniLLM's Core Advantages

1. **Advanced Architecture** - Uses LLaMA2-style, which is the standard for modern LLMs (LLaMA, Qwen, Mistral)
2. **Transparent LoRA** - Hand-written implementation, easy to explain in interviews
3. **Complete Experiments** - 5 ablation studies with scientific methodology
4. **Detailed Documentation** - 11 code analysis files for deep learning
5. **OOM Optimization** - Pre-tokenization + memory mapping solution

## Quick Start

### Prerequisites

```bash
git clone https://github.com/SoloCalm/MiniLLM.git
cd MiniLLM
pip install -e ".[all]"
```

### Download Data

```bash
mkdir -p data
pip install huggingface_hub

python -c "
from huggingface_hub import hf_hub_download

hf_hub_download(repo_id='jingyaogong/minimind_dataset', repo_type='dataset',
                filename='pretrain_t2t_mini.jsonl', local_dir='data/minimind_dataset')
hf_hub_download(repo_id='jingyaogong/minimind_dataset', repo_type='dataset',
                filename='lora_identity.jsonl', local_dir='data/minimind_dataset')
hf_hub_download(repo_id='jingyaogong/minimind_dataset', repo_type='dataset',
                filename='dpo.jsonl', local_dir='data/minimind_dataset')

# Firefly dataset (for QLoRA fine-tuning)
hf_hub_download(repo_id='YeungNLP/firefly-train-1.1M', repo_type='dataset',
                filename='firefly-train-1.1M.jsonl', local_dir='data/firefly-train-1.1M')

print('Data download complete')
"
```

### 1. Train Tokenizer

```bash
python scripts/1_train_tokenizer.py
```

### 2. Pre-tokenize Data (Solves OOM)

```bash
python scripts/tokenize_to_disk.py
```

### 3. Smoke Pretrain (100 steps)

```bash
python scripts/2_pretrain.py \
    --tokenized-data data/pretrain_tokenized/train_ids.npy \
    --max-steps 100 --batch-size 8 --log-interval 10
```

### 4. Full Pretraining

```bash
python scripts/2_pretrain.py \
    --tokenized-data data/pretrain_tokenized/train_ids.npy
```

### 5. SFT Fine-tuning

```bash
python scripts/3_sft.py \
    --pretrained-path outputs/pretrained/ckpt_final.pt \
    --data-path data/minimind_dataset/lora_identity.jsonl
```

### 6. DPO Alignment

```bash
python scripts/5_dpo.py \
    --sft-path outputs/sft/ckpt_final.pt \
    --data-path data/minimind_dataset/dpo.jsonl
```

### 7. QLoRA Baseline (Qwen2.5-1.5B)

```bash
python scripts/4_qlora.py
```

### (Optional) LoRA Comparison Experiment

```bash
python scripts/run_ft_vs_lora.py
```

### 8. Evaluation

```bash
# Perplexity
python eval/perplexity.py --model-path outputs/dpo/ckpt_final.pt

# Structured benchmark (compare multiple checkpoints)
python eval/benchmark.py \
    --checkpoints outputs/pretrained/ckpt_final.pt outputs/sft/ckpt_final.pt outputs/dpo/ckpt_final.pt \
    --labels Pretrain SFT DPO
```

### 9. Deployment

```bash
# Export to HuggingFace format
python inference/export_hf.py \
    --checkpoint outputs/dpo/ckpt_final.pt \
    --output-dir outputs/hf_model

# Interactive chat
python inference/chat.py --checkpoint outputs/dpo/ckpt_final.pt
```

## Project Structure

```
MiniLLM/
├── model/                  # Hand-written Transformer architecture
│   ├── config.py           #   ModelConfig (all hyperparameters)
│   ├── rope.py             #   Rotary Position Embedding
│   ├── attention.py        #   GQA Attention + KV Cache
│   ├── ffn.py              #   SwiGLU Feed-Forward Network
│   ├── block.py            #   RMSNorm + TransformerBlock
│   └── modeling_llm.py     #   MiniLLM: full decoder model + generate()
│
├── training/               # Training modules
│   ├── optimizer.py        #   AdamW + cosine scheduler
│   ├── data_loader.py      #   Pretrain/SFT/DPO datasets + mmap loader
│   ├── pretrain.py         #   Pretraining loop
│   ├── sft.py              #   SFT training loop
│   ├── lora.py             #   LoRA: LoRALinear + apply/merge
│   └── dpo.py              #   DPO loss + training loop
│
├── scripts/                # Execution entry points
│   ├── 1_train_tokenizer.py
│   ├── 2_pretrain.py
│   ├── 3_sft.py
│   ├── 4_qlora.py          #   Qwen2.5-1.5B QLoRA (HF Trainer + PEFT)
│   ├── 5_dpo.py
│   ├── run_pipeline.py     #   Data pipeline (clean→tokenize→split)
│   ├── tokenize_to_disk.py #   JSONL → numpy uint16 mmap (OOM solution)
│   ├── run_lora_rank_ablation.py
│   ├── run_ft_vs_lora.py
│   ├── run_qlora_baseline.py
│   ├── compare_models.py   #   38M vs 1.5B generation comparison
│   ├── kv_cache_benchmark.py
│   ├── serve_vllm.py       #   vLLM service deployment (Linux/WSL2 required)
│   ├── smoke_vllm.py       #   vLLM service verification (Linux/WSL2 required)
│   └── smoke_test.py       #   Environment smoke test
│
├── tokenizer/              # Tokenizer training
│   └── train_tokenizer.py  #   SentencePiece BPE (6400 vocab)
│
├── data_utils/             # Data processing
│   ├── clean_pretrain.py   #   Pretrain corpus cleaning
│   ├── prepare_sft.py      #   SFT data preparation
│   └── convert_ultrafeedback.py  # DPO data conversion
│
├── inference/              # Deployment
│   ├── export_hf.py        #   Export to HuggingFace format
│   └── chat.py             #   CLI interactive chat (multi-turn)
│
├── eval/                   # Evaluation
│   ├── perplexity.py       #   Perplexity computation
│   └── benchmark.py        #   Structured benchmark (multi-model comparison)
│
├── tests/                  # Unit tests
│   ├── test_model.py
│   ├── test_attention.py
│   ├── test_tokenizer.py
│   └── test_mmap.py
│
├── configs/                # Experiment configurations (JSON)
│   ├── pretrain.json       #   Full pretrain (50k steps)
│   ├── pretrain_smoke.json #   Smoke test (100 steps)
│   ├── sft.json            #   Full-parameter SFT
│   ├── sft_lora.json       #   LoRA SFT
│   ├── sft_qlora.json      #   QLoRA (Qwen2.5-1.5B)
│   └── dpo.json            #   DPO alignment
│
├── docs/                   # Documentation
│   ├── 项目总结.md          #   Full project & experiment summary
│   ├── 架构.md             #   Architecture diagrams + data flow
│   ├── 训练笔记.md         #   PyTorch training tutorial
│   ├── DPO理论.md          #   DPO theory + math derivation
│   └── code_analysis/      #   11 core code analysis files
│
├── results/                # Experiment results
│   ├── ablation_results.json
│   ├── lora_rank_ablation.json
│   ├── ft_vs_lora_ablation.json
│   ├── perplexity.json
│   └── compare/            #   Generation comparison outputs
│
├── data/                   # Datasets (gitignored, download separately)
├── outputs/                # Model checkpoints (gitignored)
├── pyproject.toml
├── .gitignore
├── README.md
└── README_EN.md
```

## Problem Solved: OOM During Pretraining

### Root Cause

The original `PretrainDataset` loaded all 395M tokens into a Python list at initialization:

```
395M tokens × 28 bytes/token (Python int) = ~11 GB
```

This exceeded the available system memory on a 6GB GPU setup.

### Solution: Pre-tokenization + Memory Mapping

1. **Pre-tokenize to disk** (`scripts/tokenize_to_disk.py`):
   - Encodes all text with SentencePiece
   - Stores as `numpy.uint16` (2 bytes/token vs 28 bytes for Python int)
   - Writes via `np.lib.format.open_memmap`

2. **Memory-mapped loading** (`training/data_loader.py`):
   - `np.load("train_ids.npy", mmap_mode="r")` — data stays on disk
   - OS loads pages on demand — 395M tokens use only a few MB of RAM

| | Python List | numpy mmap |
|---|---|---|
| Storage format | Python list of int | numpy uint16 array |
| Per-token memory | 28 bytes | 2 bytes |
| 395M tokens | ~11 GB | ~0 MB (791 MB on disk) |
| Loading | Full RAM load | On-demand via mmap |
| Scalability | < 1M tokens | Unlimited |

## Tech Stack

- **Framework:** PyTorch 2.1+
- **Tokenizer:** SentencePiece (BPE, 6400 vocab)
- **Fine-tuning:** Custom LoRA + HuggingFace PEFT/QLoRA
- **Alignment:** Custom DPO implementation
- **Deployment:** CLI chat + HuggingFace export
- **Quantization:** BitsAndBytes 4-bit NF4 (for QLoRA baseline)
- **Experiment Tracking:** Weights & Biases (optional)

## Code Analysis

Detailed code analysis for each core module:

| File | Description |
|------|-------------|
| [01-transformer.py 模型主干](docs/code_analysis/01-transformer.py%20模型主干.md) | Model architecture, config, forward pass, generation |
| [02-rope.py 旋转位置编码](docs/code_analysis/02-rope.py%20旋转位置编码.md) | Rotary Position Embedding implementation |
| [03-attention.py GQA注意力](docs/code_analysis/03-attention.py%20GQA注意力.md) | Grouped-Query Attention + KV Cache |
| [04-ffn.py SwiGLU前馈网络](docs/code_analysis/04-ffn.py%20SwiGLU前馈网络.md) | SwiGLU Feed-Forward Network |
| [05-lora.py LoRA参数高效微调](docs/code_analysis/05-lora.py%20LoRA参数高效微调.md) | LoRA implementation and application |
| [06-sft.py SFT数据协议](docs/code_analysis/06-sft.py%20SFT数据协议.md) | SFT data format and training loop |
| [07-data_loader.py 数据加载与loss](docs/code_analysis/07-data_loader.py%20数据加载与loss.md) | Dataset classes and loss computation |
| [08-generate.py 推理链路](docs/code_analysis/08-generate.py%20推理链路.md) | Inference pipeline and generation |
| [09-dpo.py DPO偏好对齐](docs/code_analysis/09-dpo.py%20DPO偏好对齐.md) | DPO loss and training |
| [10-数据pipeline](docs/code_analysis/10-数据pipeline.md) | 6-step data processing pipeline |
| [11-高频手撕](docs/code_analysis/11-高频手撕.md) | High-frequency interview code implementations |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
