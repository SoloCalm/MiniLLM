# Bug 清册：从零训练大模型踩坑记录

## 项目概述

从零训练 41M 参数中文对话模型，走完预训练→SFT→LoRA→DPO→部署全流程。
以下是在项目过程中遇到的问题和解决方案。

---

## 阶段 1：数据 + Tokenizer + 模型架构

### Bug 1.1：350M 模型在 6GB GPU 上无法训练

**日期**：2026-05-20

**现象**：
```
CUDA out of memory
```

**原因**：模型太大，6GB 显存不够

**解决方案**：降为 41M 参数（512 hidden, 12 layers, 8 heads）

---

### Bug 1.2：estimate_params 未考虑 GQA

**日期**：2026-05-20

**现象**：参数量估算不准确

**原因**：GQA 时 K/V 的头数是 num_kv_heads，不是 num_heads

**解决方案**：修正公式：Q 用 num_heads, K/V 用 num_kv_heads

---

### Bug 1.3：113M 模型 batch_size=4 时 OOM

**日期**：2026-05-20

**现象**：
```
CUDA out of memory
```

**原因**：模型还是太大

**解决方案**：再降为 41M，batch 可提升到 16

---

### Bug 1.4：hiyouga/RLHF-Reward-Chatbot 是 gated repo

**日期**：2026-05-20

**现象**：无法下载数据集

**原因**：需要申请权限

**解决方案**：改用 HuggingFaceH4/ultrafeedback_binarized（公开）

---

## 阶段 2：预训练

### Bug 2.1：forward 中 freqs 缩进到 else 块内

**日期**：2026-05-26

**现象**：推理时 NameError

**原因**：freqs 计算被错误缩进到 else 块内

**解决方案**：移到 if/else 外面

---

### Bug 2.2：KVCache.get_seq_length 在空 cache 时 IndexError

**日期**：2026-05-26

**现象**：
```
IndexError: list index out of range
```

**原因**：get_seq_length 在 cache 为空时访问 key_cache[0]

**解决方案**：加判空，空 cache 返回 0

---

### Bug 2.3：预训练数据 3.95 亿 token 导致 Python list OOM

**日期**：2026-05-26

**现象**：
```
MemoryError: Unable to allocate 11GB
```

**原因**：Python int 对象每个占 28 字节，3.95 亿 token × 28 字节 = ~11 GB

**解决方案**：预 tokenize 存 .npy（numpy uint16，2 字节/token，省 14 倍）

---

### Bug 2.4：Windows 上 open_memmap 大文件 segfault

**日期**：2026-05-26

**现象**：程序崩溃

**原因**：Windows 上 mmap 大文件有兼容性问题

**解决方案**：改用普通 np.load，791MB 全加载到 RAM

---

## 阶段 3：SFT 微调

### Bug 3.1：LoRA 权重合并时维度不匹配

**日期**：2026-06-02

**现象**：
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied
```

**原因**：LoRA 矩阵乘法维度错误

**解决方案**：修正为 (A @ B)^T 而不是 B @ A

---

### Bug 3.2：LoRALinear 未继承 nn.Linear

**日期**：2026-06-02

**现象**：
```
AttributeError: 'LoRALinear' object has no attribute 'bias'
```

**原因**：LoRALinear 没有正确存储原始权重

**解决方案**：重写为 nn.Linear 的 drop-in 替换

---

## 阶段 4：DPO 偏好对齐

### Bug 4.1：compute_log_probs 中 gather 遇到 -100 索引

**日期**：2026-06-04

**现象**：
```
RuntimeError: index -100 is out of bounds for dimension 2 with size 6400
```

**原因**：shift_labels 中有 -100，gather 无法处理负数索引

**解决方案**：先 clamp 到 0，gather 后再 mask

---

### Bug 4.2：DPO 训练输出被缓冲

**日期**：2026-06-04

**现象**：无法实时查看训练输出

**原因**：Python 输出默认被缓冲

**解决方案**：使用 `python -u` 禁用缓冲

---

### Bug 4.3：两个 DPO 实验同时运行导致 OOM

**日期**：2026-06-04

**现象**：
```
CUDA out of memory
```

**原因**：每个 DPO 实验需要 2 个模型（policy + reference），同时运行 2 个实验需要 4 个模型

**解决方案**：改为顺序运行

---

### Bug 4.4：生成对比时 UnicodeEncodeError

**日期**：2026-06-04

**现象**：
```
UnicodeEncodeError: 'gbk' codec can't encode character
```

**原因**：Windows 终端编码问题

**解决方案**：添加 try-except 处理编码错误

---

## 阶段 5：部署 + 推理优化

### Bug 5.1：safetensors 不支持 weight tying

**日期**：2026-06-05

**现象**：
```
RuntimeError: Some tensors share memory
```

**原因**：模型有 weight tying（tok_emb 和 lm_head 共享权重），safetensors 不支持

**解决方案**：使用 pytorch 格式保存

---

## 经验总结

### 1. 显存管理

- 6GB GPU 最大支持 ~41M 参数模型
- batch_size 和 max_length 是主要显存消耗
- gradient_checkpointing 可以节省显存

### 2. 数据处理

- 大数据集用内存映射（numpy mmap）
- 预 tokenize 存磁盘，避免重复计算
- Windows 上 mmap 有兼容性问题

### 3. 模型训练

- 学习率很重要，需要调参
- LoRA 是高效的微调方式
- DPO 比 RLHF 更简单稳定

### 4. 调试技巧

- 打印 tensor shape 是最常用的调试方法
- 先在小数据集上验证代码
- 使用 python -u 禁用输出缓冲
