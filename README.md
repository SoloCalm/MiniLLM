# MiniLLM-PostTrain

Dual-track LLM training project on a single **RTX 4050 6GB** GPU:

- **Track A (Self-built 41M):** Hand-written LLaMA2-style Transformer → Pretrain → SFT → LoRA → DPO
- **Track B (Qwen2.5-1.5B):** QLoRA fine-tuning → vLLM service deployment

## Key Results

### Track A — Self-built 41M

| Metric | Value |
|--------|-------|
| Architecture | LLaMA2-style (GQA + SwiGLU + RoPE + RMSNorm) |
| Parameters | ~38M (after weight tying) |
| Perplexity | 13.03 |
| DPO Reward Margin (β=0.2) | 0.2236 |
| LoRA r=8 Trainable Params | 887,808 (2.28%) |
| Ablation Experiments | 5 (LR / LoRA rank / FT vs LoRA / DPO β / 41M vs 1.5B) |

### Track B — Qwen2.5-1.5B QLoRA

| Metric | Value |
|--------|-------|
| Base Model | Qwen2.5-1.5B (1.55B params) |
| Quantization | 4-bit NF4 (BitsAndBytes) |
| LoRA Params | 2.18M (0.14% of base) |
| Deployment | vLLM with PagedAttention |

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

## Quick Start

### Prerequisites

```bash
pip install -e ".[all]"
```

### 1. Smoke Test (Verify Environment)

```bash
python scripts/smoke_test.py
```

### 2. Train Tokenizer

```bash
python scripts/1_train_tokenizer.py
```

### 3. Pre-tokenize Data (Solves OOM)

```bash
python scripts/tokenize_to_disk.py
```

### 4. Smoke Pretrain (100 steps)

```bash
python scripts/2_pretrain.py \
    --tokenized-data data/pretrain_tokenized/train_ids.npy \
    --max-steps 100 --batch-size 8 --log-interval 10
```

### 5. Full Pretraining

```bash
python scripts/2_pretrain.py \
    --tokenized-data data/pretrain_tokenized/train_ids.npy
```

### 6. SFT Fine-tuning

```bash
python scripts/3_sft.py \
    --pretrained-path outputs/pretrained/ckpt_final.pt \
    --data-path data/minimind_dataset/lora_identity.jsonl
```

### 7. LoRA Fine-tuning

```bash
python scripts/3_sft.py \
    --pretrained-path outputs/pretrained/ckpt_final.pt \
    --data-path data/minimind_dataset/lora_identity.jsonl \
    --use-lora --lora-rank 8 --lora-alpha 16
```

### 8. DPO Alignment

```bash
python scripts/5_dpo.py \
    --sft-path outputs/sft/ckpt_final.pt \
    --data-path data/ultrafeedback_binarized/train.jsonl
```

### 9. QLoRA Baseline (Qwen2.5-1.5B)

```bash
python scripts/4_qlora.py
```

### 10. Evaluation

```bash
# Perplexity
python eval/perplexity.py --model-path outputs/dpo/ckpt_final.pt

# Structured benchmark (compare multiple checkpoints)
python eval/benchmark.py \
    --checkpoints outputs/pretrained/ckpt_final.pt outputs/sft/ckpt_final.pt outputs/dpo/ckpt_final.pt \
    --labels Pretrain SFT DPO
```

### 11. Deployment

```bash
# Export to HuggingFace format
python inference/export_hf.py \
    --checkpoint outputs/dpo/ckpt_final.pt \
    --output-dir outputs/hf_model

# vLLM service (Qwen2.5-1.5B QLoRA)
python scripts/serve_vllm.py
python scripts/smoke_vllm.py  # verify service

# Interactive chat
python inference/chat.py --checkpoint outputs/dpo/ckpt_final.pt
```

## Project Structure

```
MyLLM/
├── model/                  # Hand-written Transformer architecture
│   ├── config.py           #   ModelConfig (all hyperparameters)
│   ├── rope.py             #   Rotary Position Embedding
│   ├── attention.py        #   GQA Attention + KV Cache
│   ├── ffn.py              #   SwiGLU Feed-Forward Network
│   ├── block.py            #   RMSNorm + TransformerBlock
│   ├── modeling_llm.py     #   MiniLLM: full decoder model + generate()
│   └── reference/          #   Reference implementations (annotated)
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
│   ├── compare_models.py   #   41M vs 1.5B generation comparison
│   ├── kv_cache_benchmark.py
│   ├── serve_vllm.py       #   vLLM service deployment
│   ├── smoke_vllm.py       #   vLLM service verification
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
│   ├── project_summary.md  #   Full project & experiment summary
│   ├── architecture.md     #   Architecture diagrams + data flow
│   ├── bug_catalog.md      #   15 bugs with root cause analysis
│   ├── training_notes.md   #   PyTorch training tutorial
│   └── dpo_theory.md       #   DPO theory + math derivation
│
├── results/                # Experiment results
│   ├── ablation_results.json
│   ├── lora_rank_ablation.json
│   ├── ft_vs_lora_ablation.json
│   ├── perplexity.json
│   ├── model_comparison.txt
│   └── compare/            #   Generation comparison outputs
│
├── data/                   # Datasets (gitignored, download separately)
├── outputs/                # Model checkpoints (gitignored)
├── pyproject.toml
├── .gitignore
├── README.md
└── README_CN.md
```

## Ablation Experiments

### 1. Pretraining Learning Rate

| LR | Steps | Convergence | Result |
|----|-------|-------------|--------|
| 1e-4 | 10k | Slow | Higher final loss |
| 3e-4 | 50k | Fast | Lower final loss |

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

### 5. Model Size: 41M vs 1.5B

| Model | Response Rate | Quality |
|-------|--------------|---------|
| MiniLLM 41M | 15% | Mostly empty, limited expressiveness |
| Qwen2.5-1.5B QLoRA | 70% | Has content but with repetition |

**Conclusion:** Larger models with QLoRA fine-tuning produce significantly better results.

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
- **Deployment:** vLLM service + CLI chat + HuggingFace export
- **Quantization:** BitsAndBytes 4-bit NF4 (for QLoRA baseline)
- **Experiment Tracking:** Weights & Biases (optional)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
