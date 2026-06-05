# MiniLLM-PostTrain：双轨中文对话模型训练项目

单卡 **RTX 4050 6GB** 上完成双轨并行：

- **轨道 A（自研 41M）：** 手写 LLaMA2 风格 Transformer → 预训练 → SFT → LoRA → DPO
- **轨道 B（Qwen2.5-1.5B）：** QLoRA 微调 → vLLM 服务化部署

## 核心指标

### 轨道 A — 自研 41M

| 指标 | 数值 |
|------|------|
| 架构 | LLaMA2 风格（GQA + SwiGLU + RoPE + RMSNorm） |
| 参数量 | ~38M（权重共享后） |
| Perplexity | 13.03 |
| DPO Reward Margin（β=0.2） | 0.2236 |
| LoRA r=8 可训练参数 | 887,808（2.28%） |
| 消融实验 | 5 组（学习率 / LoRA rank / 全参 vs LoRA / DPO β / 41M vs 1.5B） |

### 轨道 B — Qwen2.5-1.5B QLoRA

| 指标 | 数值 |
|------|------|
| 基座模型 | Qwen2.5-1.5B（1.55B 参数） |
| 量化 | 4-bit NF4（BitsAndBytes） |
| LoRA 参数 | 2.18M（基座的 0.14%） |
| 部署 | vLLM + PagedAttention |

## 模型架构

| 配置 | 值 |
|------|-----|
| 架构 | Decoder-only Transformer（LLaMA2 风格） |
| 参数量 | ~38M（权重共享后） |
| 层数 | 12 |
| 隐藏维度 | 512 |
| 注意力头 | 8Q / 4KV（GQA） |
| FFN | SwiGLU（中间层 1376） |
| 位置编码 | RoPE（θ=10000） |
| 归一化 | RMSNorm（Pre-norm） |
| 词表大小 | 6,400（BPE） |
| 最大序列长度 | 1024 |

## 快速开始

### 环境准备

```bash
pip install -e ".[all]"
```

### 1. 环境验证

```bash
python scripts/smoke_test.py
```

### 2. 训练 Tokenizer

```bash
python scripts/1_train_tokenizer.py
```

### 3. 预 tokenize 数据（解决 OOM 问题）

```bash
python scripts/tokenize_to_disk.py
```

### 4. Smoke Test（100 步验证）

```bash
python scripts/2_pretrain.py \
    --tokenized-data data/pretrain_tokenized/train_ids.npy \
    --max-steps 100 --batch-size 8 --log-interval 10
```

### 5. 正式预训练

```bash
python scripts/2_pretrain.py \
    --tokenized-data data/pretrain_tokenized/train_ids.npy
```

### 6. SFT 微调

```bash
python scripts/3_sft.py \
    --pretrained-path outputs/pretrained/ckpt_final.pt \
    --data-path data/minimind_dataset/lora_identity.jsonl
```

### 7. LoRA 微调

```bash
python scripts/3_sft.py \
    --pretrained-path outputs/pretrained/ckpt_final.pt \
    --data-path data/minimind_dataset/lora_identity.jsonl \
    --use-lora --lora-rank 8 --lora-alpha 16
```

### 8. DPO 偏好对齐

```bash
python scripts/5_dpo.py \
    --sft-path outputs/sft/ckpt_final.pt \
    --data-path data/ultrafeedback_binarized/train.jsonl
```

### 9. QLoRA 基线（Qwen2.5-1.5B）

```bash
python scripts/4_qlora.py
```

### 10. 评估

```bash
# 困惑度
python eval/perplexity.py --model-path outputs/dpo/ckpt_final.pt

# 结构化评测（多模型对比）
python eval/benchmark.py \
    --checkpoints outputs/pretrained/ckpt_final.pt outputs/sft/ckpt_final.pt outputs/dpo/ckpt_final.pt \
    --labels Pretrain SFT DPO
```

### 11. 部署

```bash
# 导出 HuggingFace 格式
python inference/export_hf.py \
    --checkpoint outputs/dpo/ckpt_final.pt \
    --output-dir outputs/hf_model

# vLLM 服务（Qwen2.5-1.5B QLoRA）
python scripts/serve_vllm.py
python scripts/smoke_vllm.py  # 验证服务

# 命令行对话
python inference/chat.py --checkpoint outputs/dpo/ckpt_final.pt
```

## 项目结构

```
MyLLM/
├── model/                  # 手写 Transformer 架构
│   ├── config.py           #   ModelConfig（所有超参数）
│   ├── rope.py             #   旋转位置编码 RoPE
│   ├── attention.py        #   GQA 注意力 + KV Cache
│   ├── ffn.py              #   SwiGLU 前馈网络
│   ├── block.py            #   RMSNorm + TransformerBlock
│   ├── modeling_llm.py     #   MiniLLM：完整 decoder 模型 + generate()
│   └── reference/          #   参考实现（带中文注释）
│
├── training/               # 训练模块
│   ├── optimizer.py        #   AdamW + 余弦退火调度器
│   ├── data_loader.py      #   Pretrain/SFT/DPO 数据集 + mmap 加载器
│   ├── pretrain.py         #   预训练循环
│   ├── sft.py              #   SFT 训练循环
│   ├── lora.py             #   LoRA：LoRALinear + apply/merge
│   └── dpo.py              #   DPO 损失 + 训练循环
│
├── scripts/                # 执行入口
│   ├── 1_train_tokenizer.py
│   ├── 2_pretrain.py
│   ├── 3_sft.py
│   ├── 4_qlora.py          #   Qwen2.5-1.5B QLoRA（HF Trainer + PEFT）
│   ├── 5_dpo.py
│   ├── run_pipeline.py     #   数据 pipeline（清洗→tokenize→切分）
│   ├── tokenize_to_disk.py #   JSONL → numpy uint16 mmap（OOM 解决方案）
│   ├── run_lora_rank_ablation.py
│   ├── run_ft_vs_lora.py
│   ├── run_qlora_baseline.py
│   ├── compare_models.py   #   41M vs 1.5B 生成对比
│   ├── kv_cache_benchmark.py
│   ├── serve_vllm.py       #   vLLM 服务部署
│   ├── smoke_vllm.py       #   vLLM 服务验证
│   └── smoke_test.py       #   环境验证
│
├── tokenizer/              # Tokenizer 训练
│   └── train_tokenizer.py  #   SentencePiece BPE（6400 词表）
│
├── data_utils/             # 数据处理
│   ├── clean_pretrain.py   #   预训练语料清洗
│   ├── prepare_sft.py      #   SFT 数据准备
│   └── convert_ultrafeedback.py  # DPO 数据转换
│
├── inference/              # 部署
│   ├── export_hf.py        #   导出 HuggingFace 格式
│   └── chat.py             #   命令行对话（支持多轮历史）
│
├── eval/                   # 评估
│   ├── perplexity.py       #   困惑度计算
│   └── benchmark.py        #   结构化评测（多模型对比）
│
├── tests/                  # 单元测试
│   ├── test_model.py
│   ├── test_attention.py
│   ├── test_tokenizer.py
│   └── test_mmap.py
│
├── configs/                # 实验配置（JSON）
│   ├── pretrain.json       #   完整预训练（50k 步）
│   ├── pretrain_smoke.json #   Smoke test（100 步）
│   ├── sft.json            #   全参 SFT
│   ├── sft_lora.json       #   LoRA SFT
│   ├── sft_qlora.json      #   QLoRA（Qwen2.5-1.5B）
│   └── dpo.json            #   DPO 对齐
│
├── docs/                   # 文档
│   ├── project_summary.md  #   完整项目与实验总结
│   ├── architecture.md     #   架构图 + 数据流
│   ├── bug_catalog.md      #   15 个 bug 的根因分析与解决方案
│   ├── training_notes.md   #   PyTorch 训练教程
│   └── dpo_theory.md       #   DPO 理论 + 数学推导
│
├── results/                # 实验结果
│   ├── ablation_results.json
│   ├── lora_rank_ablation.json
│   ├── ft_vs_lora_ablation.json
│   ├── perplexity.json
│   ├── model_comparison.txt
│   └── compare/            #   生成对比输出
│
├── data/                   # 数据集（gitignore，需自行下载）
├── outputs/                # 模型 checkpoint（gitignore）
├── pyproject.toml
├── .gitignore
├── README.md
└── README_CN.md
```

## 消融实验

### 1. 预训练学习率

| 学习率 | 步数 | 收敛速度 | 结果 |
|--------|------|----------|------|
| 1e-4 | 10k | 较慢 | 最终 loss 较高 |
| 3e-4 | 50k | 较快 | 最终 loss 较低 |

**结论：** 3e-4 收敛更快，最终效果更好。

### 2. LoRA Rank

| Rank | 可训练参数 | 占比 | 显存 |
|------|------------|------|------|
| r=4 | 443,904 | 1.15% | 1.05 GB |
| r=8 | 887,808 | 2.28% | 1.20 GB |
| r=16 | 1,775,616 | 4.45% | 1.36 GB |

**结论：** r=8 性价比最高。

### 3. 全参微调 vs LoRA

| 方法 | 可训练参数 | 占比 | 峰值显存 |
|------|------------|------|----------|
| 全参 SFT | 38,089,216 | 100% | 1.27 GB |
| LoRA SFT | 887,808 | 2.28% | 1.05 GB |

**结论：** LoRA 用 2.28% 参数达到全参效果，显存省 17%。

### 4. DPO β 值

| β | Margin | 特点 |
|---|--------|------|
| 0.1 | 0.1222 | 激进，偏离参考模型较远 |
| 0.2 | 0.2236 | 平衡，推荐值 |
| 0.5 | 0.3199 | 保守，保持接近参考模型 |

**结论：** β=0.2 是平衡点，margin 适中，loss 较低。

### 5. 模型规模对比：41M vs 1.5B

| 模型 | 回复率 | 质量 |
|------|--------|------|
| MiniLLM 41M | 15% | 大部分为空，表达能力有限 |
| Qwen2.5-1.5B QLoRA | 70% | 有内容，但存在重复 |

**结论：** 大模型 + QLoRA 微调效果显著更好。

## 解决的核心问题：预训练 OOM

### 问题原因

原始 `PretrainDataset` 在初始化时将 3.95 亿 token 一次性加载到 Python list：

```
3.95 亿 token × 28 字节/个（Python int）= ~11 GB
```

6GB 显卡的系统内存通常只有 16GB，扣除系统和其他进程后不够用。

### 解决方案：预 tokenize + 内存映射

**第 1 步：预 tokenize 存磁盘**（`scripts/tokenize_to_disk.py`）

- 用 SentencePiece 编码所有文本
- 用 `numpy.uint16` 存储（2 字节/token vs Python int 28 字节）
- 用 `np.lib.format.open_memmap` 写入磁盘

**第 2 步：训练时内存映射加载**（`training/data_loader.py`）

- `np.load("train_ids.npy", mmap_mode="r")` — 数据留在磁盘
- 操作系统按需加载被访问的数据页 — 10 亿 token 也只需几 MB 内存

| | Python list | numpy mmap |
|---|---|---|
| 存储格式 | Python list of int | numpy uint16 数组 |
| 每个 token 占内存 | 28 字节 | 2 字节 |
| 3.95 亿 token 占内存 | ~11 GB | ~0 MB（磁盘上 791 MB） |
| 加载方式 | 全部加载到 RAM | 内存映射，按需加载 |
| 适用规模 | < 100 万 token | 任意规模 |

## 技术栈

- **框架：** PyTorch 2.1+
- **Tokenizer：** SentencePiece（BPE，6400 词表）
- **微调：** 自实现 LoRA + HuggingFace PEFT/QLoRA
- **对齐：** 自实现 DPO
- **部署：** vLLM 服务 + 命令行对话 + HuggingFace 导出
- **量化：** BitsAndBytes 4-bit NF4（QLoRA 基线）
- **实验追踪：** Weights & Biases（可选）

## 许可

本项目基于 [MIT 许可证](LICENSE) 开源。
