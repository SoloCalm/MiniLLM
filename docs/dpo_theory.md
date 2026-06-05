# DPO（Direct Preference Optimization）理论学习

## 一、为什么需要 DPO？

### 1.1 LLM 对齐问题

预训练 + SFT 只教模型"怎么说"，没教模型"说什么更好"。

```
用户问：如何偷东西？
SFT 模型：以下是偷东西的步骤...  ← 回答了，但不安全
```

我们需要让模型学会**偏好**：什么回答好，什么回答不好。

### 1.2 传统方法：RLHF

RLHF（Reinforcement Learning from Human Feedback）流程：

```
步骤 1：收集偏好数据
  Prompt → 模型生成多个回答 → 人工标注哪个更好
  (prompt, chosen, rejected)

步骤 2：训练 Reward Model
  学习一个打分函数 r(prompt, response)
  让 chosen 得分 > rejected 得分

步骤 3：PPO 强化学习
  用 reward model 的分数作为奖励信号
  通过 PPO 算法优化策略模型
```

**RLHF 的问题：**
- 需要训练 3 个模型（Policy + Reward + Reference）
- PPO 训练不稳定，超参数敏感
- Reward model 可能被 hack（reward hacking）
- 计算开销大，需要大量 GPU

---

## 二、DPO 核心思想

### 2.1 关键洞察

DPO 论文的核心发现：**不需要单独训练 Reward Model！**

Reward 可以直接从偏好数据中隐式学习：

```
传统 RLHF：
  训练 Reward Model → 用 Reward 优化 Policy

DPO：
  直接用偏好数据优化 Policy（Reward 隐式包含在策略中）
```

### 2.2 数学推导

#### RLHF 的目标函数

```
max E[r(x, y)] - β * KL[π_θ || π_ref]

其中：
- r(x, y): Reward 函数
- π_θ: 当前策略（要优化的模型）
- π_ref: 参考策略（SFT 模型，防止偏离太远）
- β: KL 散度系数（控制偏离程度）
```

#### 最优策略的闭式解

经过数学推导（详见论文），最优策略为：

```
π*(y|x) = π_ref(y|x) * exp(r(x,y) / β) / Z(x)

其中 Z(x) 是配分函数（归一化常数）
```

#### 反解 Reward

从上面的公式反解出 Reward：

```
r(x, y) = β * log(π*(y|x) / π_ref(y|x)) + β * log(Z(x))
```

#### 代入 Bradley-Terry 模型

偏好概率（chosen 比 rejected 好的概率）：

```
p(chosen > rejected) = σ(r(x, y_w) - r(x, y_l))

其中：
- σ: sigmoid 函数
- y_w: chosen（更好的回答）
- y_l: rejected（更差的回答）
```

#### DPO Loss

把 Reward 代入偏好概率，得到 DPO Loss：

```
L_DPO = -E[log σ(β * log(π_θ(y_w|x) / π_ref(y_w|x))
                - β * log(π_θ(y_l|x) / π_ref(y_l|x)))]
```

---

## 三、DPO 直觉理解

### 3.1 Loss 含义

```
L_DPO = -log σ(β * (log_ratio_w - log_ratio_l))

其中：
- log_ratio_w = log(π_θ(y_w|x) / π_ref(y_w|x))  # chosen 的偏好变化
- log_ratio_l = log(π_θ(y_l|x) / π_ref(y_l|x))  # rejected 的偏好变化
```

**目标：** 让 chosen 的偏好变化 **大于** rejected 的偏好变化

```
如果 log_ratio_w > log_ratio_l：
  → σ(...) 接近 1
  → -log σ(...) 接近 0
  → Loss 小 ✓

如果 log_ratio_w < log_ratio_l：
  → σ(...) 接近 0
  → -log σ(...) 很大
  → Loss 大 ✗
```

### 3.2 梯度直觉

```
∇L_DPO ∝ -β * (π_ref(y_w|x)/π_θ(y_w|x) * ∇π_θ(y_w|x)
               - π_ref(y_l|x)/π_θ(y_l|x) * ∇π_θ(y_l|x))
```

**梯度作用：**
- 增加 chosen 的概率（正梯度）
- 减少 rejected 的概率（负梯度）
- 用 importance weight 调整幅度

### 3.3 β 的作用

| β 值 | 效果 |
|------|------|
| β 小（如 0.1） | 更激进地优化偏好，可能偏离参考模型较远 |
| β 大（如 0.5） | 更保守，保持接近参考模型，偏好学习较慢 |
| β 中等（如 0.2） | 平衡偏好学习和模型稳定性（推荐） |

---

## 四、DPO 实现要点

### 4.1 Reference Model

```python
# Reference Model 是 SFT 模型的冻结副本
# 用于计算 log_ratio = log(π_θ / π_ref)
reference_model = copy.deepcopy(policy_model)
reference_model.eval()
for param in reference_model.parameters():
    param.requires_grad = False
```

### 4.2 偏好数据格式

```json
{
  "prompt": "如何学习编程？",
  "chosen": "建议从 Python 开始，每天练习1小时...",
  "rejected": "随便学学就行了，不用太认真..."
}
```

### 4.3 DPO Loss 计算

```python
def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             reference_chosen_logps, reference_rejected_logps,
             beta=0.2):
    """
    计算 DPO Loss

    参数：
        policy_chosen_logps: π_θ(y_w|x) 的 log 概率
        policy_rejected_logps: π_θ(y_l|x) 的 log 概率
        reference_chosen_logps: π_ref(y_w|x) 的 log 概率
        reference_rejected_logps: π_ref(y_l|x) 的 log 概率
        beta: KL 散度系数
    """
    # 计算 log ratio
    chosen_log_ratio = policy_chosen_logps - reference_chosen_logps
    rejected_log_ratio = policy_rejected_logps - reference_rejected_logps

    # DPO Loss
    logits = beta * (chosen_log_ratio - rejected_log_ratio)
    loss = -F.logsigmoid(logits).mean()

    return loss
```

### 4.4 训练循环

```python
for batch in dataloader:
    prompt = batch["prompt"]
    chosen = batch["chosen"]
    rejected = batch["rejected"]

    # 1. 计算 Policy 模型的 log 概率
    policy_chosen_logps = compute_log_probs(policy_model, prompt, chosen)
    policy_rejected_logps = compute_log_probs(policy_model, prompt, rejected)

    # 2. 计算 Reference 模型的 log 概率（不需要梯度）
    with torch.no_grad():
        ref_chosen_logps = compute_log_probs(reference_model, prompt, chosen)
        ref_rejected_logps = compute_log_probs(reference_model, prompt, rejected)

    # 3. 计算 DPO Loss
    loss = dpo_loss(policy_chosen_logps, policy_rejected_logps,
                    ref_chosen_logps, ref_rejected_logps, beta=0.2)

    # 4. 反向传播
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

---

## 五、DPO vs RLHF 对比

| 维度 | RLHF | DPO |
|------|------|-----|
| 需要 Reward Model | ✅ 是 | ❌ 否 |
| 训练模型数量 | 3 个 | 2 个（Policy + Reference） |
| 训练稳定性 | PPO 不稳定 | 稳定（类似 SFT） |
| 超参数敏感度 | 高 | 低 |
| 计算开销 | 大 | 小 |
| 理论最优性 | 是 | 近似最优 |
| 实际效果 | 好 | 好（通常可比） |

---

## 六、DPO 的局限性

### 6.1 Offline 学习

DPO 是 offline 算法，只从固定数据集学习：
- 无法探索新策略
- 数据质量决定上限

### 6.2 Distribution Shift

训练过程中，π_θ 会偏离 π_ref：
- 如果偏离太远，log_ratio 估计不准
- 需要定期更新 reference model（可选）

### 6.3 偏好数据依赖

- 需要高质量的 (chosen, rejected) 对
- 标注不一致会影响效果

---

## 七、关键超参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| β | 0.1 ~ 0.5 | KL 散度系数，0.2 最常用 |
| lr | 5e-7 ~ 5e-6 | 比 SFT 小 10-100 倍 |
| epochs | 1 ~ 3 | 不要太多，防止过拟合 |
| batch_size | 4 ~ 8 | 根据显存调整 |
| max_length | 512 | 和 SFT 保持一致 |

---

## 八、学习检查清单

- [ ] 理解为什么需要 DPO（RLHF 的问题）
- [ ] 理解 DPO 的数学推导（Reward 反解）
- [ ] 理解 Loss 函数的直觉含义
- [ ] 理解 β 的作用
- [ ] 理解 Reference Model 的作用
- [ ] 能手写 DPO Loss 计算代码
- [ ] 理解 DPO vs RLHF 的优劣

---

## 参考资源

1. **DPO 原论文**: [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
2. **RLHF 原论文**: [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
3. **实现参考**: [TRL 库 DPO 实现](https://github.com/huggingface/trl)
4. **可视化理解**: [Lilian Weng - RLHF](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/)
