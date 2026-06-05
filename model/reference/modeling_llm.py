"""参考代码 - modeling_llm.py
MiniLLM 完整模型定义 — 将所有组件组装成可训练/推理的 LLM

模型组成：
1. Token Embedding: token ID → 向量
2. RoPE: 旋转位置编码（不用额外的 position embedding）
3. N 层 Transformer Block: 注意力 + FFN 交替
4. RMSNorm: 最终归一化
5. LM Head: 隐藏层 → 词表 logits（与 embedding 共享权重）

权重共享 (Weight Tying)：
- lm_head.weight = tok_emb.weight
- 优势：减少参数量（vocab * hidden）、embedding 梯度更密集
- 很多模型都这样做：GPT-2、LLaMA、MiniLLM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import ModelConfig
from .block import TransformerBlock, RMSNorm
from .attention import KVCache
from .rope import RotaryEmbedding


class MiniLLM(nn.Module):
    """MiniLLM: 41M 参数的 Decoder-only Transformer

    架构总览：
    input_ids → Embedding → [TransformerBlock × 12] → RMSNorm → LM Head → logits

    推理时使用自回归生成：每次取 logits 最后一个位置，
    采样得到下一个 token，拼接到输入，重复直到生成 EOS。
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # TODO 1: 创建 5 个组件
        # Token Embedding: 将词表中的 token ID 映射为 hidden_size 维向量
        self.tok_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        # Transformer 层堆叠: 每层包含 Attention + FFN，共 num_layers 层
        self.layers = nn.ModuleList([TransformerBlock(config, i) for i in range(config.num_layers)])
        # 最终 RMSNorm: 对最后一层的输出做归一化
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        # LM Head: 将 hidden 向量投影到词表维度，输出每个 token 的 logits
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # RoPE: 旋转位置编码器，预计算频率后注册为 buffer
        self.rope = RotaryEmbedding(config.head_dim, config.max_seq_len, config.rope_theta)

        # TODO 2: 权重绑定 — lm_head 和 tok_emb 共享权重
        # 直觉：embedding 是"查表"，lm_head 是"反查表"，逻辑上互逆
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, input_ids, kv_cache=None):
        """前向传播: token IDs → logits

        Args:
            input_ids: token ID 序列, shape (batch, seq_len)
            kv_cache: KV 缓存（推理时使用，训练时为 None）

        Returns:
            logits: shape (batch, seq_len, vocab_size)，每个 token 对词表的概率分布
        """
        batch_size, seq_len = input_ids.shape

        # 步骤 1: Token Embedding
        #   (batch, seq_len) → (batch, seq_len, hidden_size)
        x = self.tok_emb(input_ids)

        # 步骤 2: 计算 RoPE 频率（考虑 KV Cache 偏移）
        #   prefill 阶段: cache_len=0, freqs = [0, 1, 2, ..., seq_len-1]
        #   decode 阶段: cache_len=已有长度, freqs = [cache_len, cache_len+1, ...]
        #   只取新 token 对应的频率子集 freqs[cache_len:]
        if kv_cache is not None:
            cache_len = kv_cache.get_seq_length(0)
        else:
            cache_len = 0
        freqs = self.rope(cache_len + seq_len)
        freqs = freqs[cache_len:]

        # 步骤 3: 逐层经过 TransformerBlock
        #   每层: x = x + Attention(RMSNorm(x)) + FFN(RMSNorm(x))
        for layer in self.layers:
            x = layer(x, freqs, kv_cache)

        # 步骤 4-5: 最终 RMSNorm + LM Head
        #   (batch, seq_len, hidden) → (batch, seq_len, vocab_size)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=256, temperature=0.7, top_p=0.9, top_k=50):
        """自回归文本生成

        生成流程：
        1. Prefill: 用完整 prompt 做一次 forward，缓存所有 K/V
        2. Decode: 每步只输入上一步生成的 1 个 token，利用 KV Cache 高效生成
        3. 采样策略: Temperature + Top-K + Top-P 组合

        Args:
            input_ids: prompt token IDs, shape (1, prompt_len)
            max_new_tokens: 最多生成多少新 token
            temperature: 温度系数（0=贪心采样，0.7=平衡，>1=更随机）
            top_p: Nucleus sampling 阈值（只从累积概率 > top_p 的最小集合中采样）
            top_k: 只从概率最大的 k 个 token 中采样

        采样策略详解:
        - Temperature: logits / temperature，越大越平滑（随机性高），越小越尖锐（确定性高）
        - Top-K: 将概率最小的 (vocab - k) 个 token 概率设为 0
        - Top-P (Nucleus): 按概率排序，累积到 P 的最小集合保留，其余设为 0
        通常 Top-K 和 Top-P 组合使用效果最好。
        """
        kv_cache = KVCache()
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            # Prefill: 用完整已有序列做 forward
            # Decode: 只用最后一个 token（KV Cache 保存了历史信息）
            if kv_cache.get_seq_length(0) == 0:
                logits = self.forward(generated, kv_cache)
            else:
                logits = self.forward(generated[:, -1:], kv_cache)

            # 只取最后一个位置的 logits（预测下一个 token）
            next_logits = logits[:, -1, :]

            # ===== Temperature 缩放 =====
            if temperature > 0:
                # 除以 temperature：>1 更平滑，<1 更尖锐
                next_logits = next_logits / temperature
            else:
                # temperature=0: 贪心解码，直接取概率最大的 token
                next_token = next_logits.argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
                if next_token.item() == self.config.eos_token_id:
                    break
                continue

            # ===== Top-K 过滤 =====
            if top_k > 0:
                # 取第 k 大的值作为阈值，小于它的全设为 -inf
                top_k_values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < top_k_values[:, -1:]] = float("-inf")

            # ===== Top-P (Nucleus Sampling) =====
            if top_p < 1.0:
                # 1. 按 logits 降序排列
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                # 2. 计算累积概率
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # 3. 标记需要移除的 token: 累积概率超过 top_p 的
                sorted_indices_to_remove = cumulative_probs > top_p
                # 4. 至少保留 1 个 token（右移一位，第一个永远保留）
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                # 5. 映射回原始索引并设为 -inf
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_logits[indices_to_remove] = float("-inf")

            # ===== 采样 =====
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # 从分布中随机采样
            generated = torch.cat([generated, next_token], dim=1)
            # 遇到 EOS 则停止生成
            if next_token.item() == self.config.eos_token_id:
                break
        return generated

    def count_parameters(self):
        """统计可训练参数量（不含冻结的参数）"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
