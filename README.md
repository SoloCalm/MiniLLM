# MiniLLM：从零搭建一个能对话、能推理的完整 LLM 链路

用 PyTorch 手写 Transformer，跑通预训练、SFT、LoRA、DPO，每一步都能复现。

**单卡 RTX 4050 6GB 即可运行** · 完整训练流程 · 5 组消融实验 · 11 个核心代码解析

---

## 双轨并行

| 维度 | 自研 MiniLLM 41M | Qwen2.5-1.5B QLoRA |
|------|------------------|---------------------|
| **架构** | 手写 LLaMA2 风格 | HuggingFace Transformers |
| **参数量** | ~38M | 1.55B |
| **训练方式** | 从零预训练 → SFT → DPO | QLoRA 微调 |
| **Tokenizer** | SentencePiece BPE（6400 词表） | Qwen 原生 tokenizer |
| **部署** | CLI 对话 + HuggingFace 导出 | vLLM + PagedAttention |
| **可训练参数** | 887,808（2.28%） | 2.18M（0.14%） |

---

## 架构流程

```
┌─────────────────────────────────────────────────────────────┐
│                    数据准备阶段                              │
│  原始文本 → SentencePiece tokenize → numpy uint16 mmap      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    预训练阶段                                │
│  随机初始化 → Next Token Prediction → 学习语言知识           │
│  50,000 步, lr=3e-4 → outputs/pretrained/ckpt_final.pt     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    SFT 微调阶段                              │
│  加载预训练模型 → 指令数据 → 学习对话能力                     │
│  lr=1e-5, 3 epochs → outputs/sft/ckpt_final.pt            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    DPO 对齐阶段                              │
│  加载 SFT 模型 → 偏好数据 → 学习人类偏好                     │
│  lr=5e-7, β=0.2 → outputs/dpo/ckpt_final.pt               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    评估 & 部署                               │
│  困惑度评估 → 多模型对比 → 导出 HuggingFace → CLI 对话       │
└─────────────────────────────────────────────────────────────┘
```

---

## 核心成果

### 自研 MiniLLM — 38M 参数

| 指标 | 数值 |
|------|------|
| 架构 | LLaMA2 风格（GQA + SwiGLU + RoPE + RMSNorm） |
| 参数量 | ~38M（权重共享后） |
| Perplexity | 13.03 |
| DPO Reward Margin（β=0.2） | 0.2236 |
| LoRA r=8 可训练参数 | 887,808（2.28%） |
| 显存占用 | 1.05-1.27 GB |
| 消融实验 | 5 组（学习率 / LoRA rank / 全参 vs LoRA / DPO β / 41M vs 1.5B） |

### QLoRA 基线 — Qwen2.5-1.5B-Instruct

| 指标 | 数值 |
|------|------|
| 基座模型 | Qwen2.5-1.5B（1.55B 参数） |
| 量化 | 4-bit NF4（BitsAndBytes） |
| LoRA 参数 | 2.18M（基座的 0.14%） |
| 部署 | vLLM + PagedAttention |

---

## 详细文档

| 文档 | 说明 |
|------|------|
| [DPO 理论](docs/DPO理论.md) | DPO 数学推导 + 与 PPO 对比 |
| [模型架构](docs/架构.md) | Transformer 架构图 + 数据流 |
| [训练笔记](docs/训练笔记.md) | PyTorch 训练教程 + 常见问题 |
| [项目总结](docs/项目总结.md) | 完整项目与实验总结 |

---

## 核心代码解析

| 文件 | 说明 |
|------|------|
| [01-transformer.py 模型主干](docs/code_analysis/01-transformer.py%20模型主干.md) | 模型架构、配置、前向传播、生成 |
| [02-rope.py 旋转位置编码](docs/code_analysis/02-rope.py%20旋转位置编码.md) | RoPE 旋转位置编码实现 |
| [03-attention.py GQA注意力](docs/code_analysis/03-attention.py%20GQA注意力.md) | GQA 分组查询注意力 + KV Cache |
| [04-ffn.py SwiGLU前馈网络](docs/code_analysis/04-ffn.py%20SwiGLU前馈网络.md) | SwiGLU 前馈网络 |
| [05-lora.py LoRA参数高效微调](docs/code_analysis/05-lora.py%20LoRA参数高效微调.md) | LoRA 实现与应用 |
| [06-sft.py SFT数据协议](docs/code_analysis/06-sft.py%20SFT数据协议.md) | SFT 数据格式与训练循环 |
| [07-data_loader.py 数据加载与loss](docs/code_analysis/07-data_loader.py%20数据加载与loss.md) | 数据集类与损失计算 |
| [08-generate.py 推理链路](docs/code_analysis/08-generate.py%20推理链路.md) | 推理流程与生成 |
| [09-dpo.py DPO偏好对齐](docs/code_analysis/09-dpo.py%20DPO偏好对齐.md) | DPO 损失与训练 |
| [10-数据pipeline](docs/code_analysis/10-数据pipeline.md) | 6 步数据处理流程 |
| [11-高频手撕](docs/code_analysis/11-高频手撕.md) | 面试高频代码实现 |

---

## 快速开始

### 安装

```bash
git clone https://github.com/SoloCalm/MiniLLM.git
cd MiniLLM
pip install -e ".[all]"
```

### 验证

```bash
python scripts/smoke_test.py
```

### 数据准备

```bash
# 下载数据集（MiniMind）
# 参考 data/minimind_dataset/README.md

# 预 tokenize（解决 OOM 问题）
python scripts/tokenize_to_disk.py
```

### 训练流程

```bash
# 步骤 1：训练 Tokenizer
python scripts/1_train_tokenizer.py

# 步骤 2：Smoke Test（100 步验证）
python scripts/2_pretrain.py \
    --tokenized-data data/pretrain_tokenized/train_ids.npy \
    --max-steps 100 --batch-size 8 --log-interval 10

# 步骤 3：正式预训练（50,000 步）
python scripts/2_pretrain.py \
    --tokenized-data data/pretrain_tokenized/train_ids.npy

# 步骤 4：SFT 微调
python scripts/3_sft.py \
    --pretrained-path outputs/pretrained/ckpt_final.pt \
    --data-path data/minimind_dataset/lora_identity.jsonl

# 步骤 5：LoRA 微调
python scripts/3_sft.py \
    --pretrained-path outputs/pretrained/ckpt_final.pt \
    --data-path data/minimind_dataset/lora_identity.jsonl \
    --use-lora --lora-rank 8 --lora-alpha 16

# 步骤 6：DPO 偏好对齐
python scripts/5_dpo.py \
    --sft-path outputs/sft/ckpt_final.pt \
    --data-path data/ultrafeedback_binarized/train.jsonl

# 步骤 7：QLoRA 基线（Qwen2.5-1.5B）
python scripts/4_qlora.py
```

### 评估

```bash
# 困惑度
python eval/perplexity.py --model-path outputs/dpo/ckpt_final.pt

# 结构化评测（多模型对比）
python eval/benchmark.py \
    --checkpoints outputs/pretrained/ckpt_final.pt outputs/sft/ckpt_final.pt outputs/dpo/ckpt_final.pt \
    --labels Pretrain SFT DPO
```

### 部署

```bash
# 导出 HuggingFace 格式
python inference/export_hf.py \
    --checkpoint outputs/dpo/ckpt_final.pt \
    --output-dir outputs/hf_model

# 命令行对话
python inference/chat.py --checkpoint outputs/dpo/ckpt_final.pt

# vLLM 服务（Qwen2.5-1.5B QLoRA）
python scripts/serve_vllm.py
python scripts/smoke_vllm.py  # 验证服务
```

---

## 项目结构

```
MiniLLM/
├── model/                  # 手写 Transformer 架构
│   ├── config.py           #   ModelConfig（所有超参数）
│   ├── rope.py             #   旋转位置编码 RoPE
│   ├── attention.py        #   GQA 注意力 + KV Cache
│   ├── ffn.py              #   SwiGLU 前馈网络
│   ├── block.py            #   RMSNorm + TransformerBlock
│   └── modeling_llm.py     #   MiniLLM：完整 decoder 模型 + generate()
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
│   ├── DPO理论.md          #   DPO 理论 + 数学推导
│   ├── 架构.md             #   架构图 + 数据流
│   ├── 训练笔记.md         #   PyTorch 训练教程
│   ├── 项目总结.md         #   完整项目与实验总结
│   └── code_analysis/      #   11 个核心代码解析文件
│
├── results/                # 实验结果
│   ├── ablation_results.json
│   ├── lora_rank_ablation.json
│   ├── ft_vs_lora_ablation.json
│   ├── perplexity.json
│   └── compare/            #   生成对比输出
│
├── data/                   # 数据集（gitignore，需自行下载）
├── outputs/                # 模型 checkpoint（gitignore）
├── pyproject.toml
├── .gitignore
├── README.md
└── README_EN.md
```

---

## 消融实验

### 1. 预训练学习率

| 学习率 | 步数 | 收敛速度 | 最终 Loss |
|--------|------|----------|-----------|
| 1e-4 | 10k | 较慢 | 较高 |
| 3e-4 | 50k | 较快 | 较低 |

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

### 困惑度

| 模型 | Perplexity |
|------|------------|
| DPO 最终模型 | 13.03 |

---

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

---

## 技术栈

- **框架：** PyTorch 2.1+
- **Tokenizer：** SentencePiece（BPE，6400 词表）
- **微调：** 自实现 LoRA + HuggingFace PEFT/QLoRA
- **对齐：** 自实现 DPO
- **部署：** vLLM 服务 + 命令行对话 + HuggingFace 导出
- **量化：** BitsAndBytes 4-bit NF4（QLoRA 基线）
- **实验追踪：** Weights & Biases（可选）

---

## 许可

本项目基于 [MIT 许可证](LICENSE) 开源。
