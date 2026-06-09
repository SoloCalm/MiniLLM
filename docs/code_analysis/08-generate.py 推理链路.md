# 08-generate.py 推理链路

## 逐段源码与解析

### 1. 模型加载 Load Model (L19-29)

```python
def load_model(checkpoint_path: str, device: str = "cuda"):
    """加载模型"""
    config = ModelConfig()
    model = MiniLLM(config)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()

    return model
```

---

### 2. 生成回答 Generate Response (L32-71)

```python
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
```

**推理流程：**
```
用户输入: "你好"
    ↓
Tokenize: [101, 102]
    ↓
添加特殊 token: [1, 101, 102, 2]
    ↓
模型生成: [1, 101, 102, 2, 201, 202, 301, 302, 2]
    ↓
提取回答: [201, 202, 301, 302]
    ↓
Detokenize: "你好！有什么我可以帮助你的吗？"
```

---

### 3. 交互式对话 Chat Loop (L74-110)

```python
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
```

**多轮对话示例：**
```
历史:
    user: 你好
    assistant: 你好！有什么我可以帮助你的吗？
    user: 今天天气怎么样？
    assistant: 今天天气很好，阳光明媚。

构造 prompt:
    你好\n你好！有什么我可以帮助你的吗？\n今天天气怎么样？\n今天天气很好，阳光明媚。\n最近有什么新闻？

模型生成回答
```

---

## 推理参数详解

### 1. Temperature
- **temperature=0**：贪心解码，每次选概率最高的 token
- **temperature=0.7**：适度随机，生成更自然
- **temperature>1**：更随机，可能生成不相关内容

### 2. Top-K
- 只从概率最高的 K 个 token 中采样
- 过滤掉概率很低的 token
- 避免生成不相关的内容

### 3. Top-P (Nucleus Sampling)
- 从累积概率超过 P 的最小 token 集合中采样
- 动态调整候选 token 数量
- 比 Top-K 更灵活

### 4. 推荐参数
```python
temperature=0.7  # 适度随机
top_p=0.9        # 过滤掉概率最低的 10%
top_k=50         # 只从 top 50 中采样
```

---

## KV Cache 加速推理

### 不使用 KV Cache
```python
# 每次都用完整序列前向传播
for _ in range(max_new_tokens):
    logits = model(generated)  # O(n²) 复杂度
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = torch.cat([generated, next_token], dim=1)
```

### 使用 KV Cache
```python
# 只传最后一个 token，利用缓存的 K/V
kv_cache = KVCache()
for _ in range(max_new_tokens):
    logits = model(generated[:, -1:], kv_cache)  # O(n) 复杂度
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = torch.cat([generated, next_token], dim=1)
```

**加速效果：**
```
生成 100 个 token：
- 不用 KV Cache：100 次完整前向传播
- 用 KV Cache：1 次完整前向传播 + 99 次单 token 前向传播

加速比：约 10-100 倍
```
