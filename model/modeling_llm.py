"""
完整 LLM 模型：将所有组件组装成 Decoder-only Transformer

结构：
  Token Embeddings
      ↓
  [TransformerBlock × num_layers]
      ↓
  RMSNorm
      ↓
  LM Head（线性层，输出词表大小的概率分布）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .block import TransformerBlock, RMSNorm
from .attention import KVCache
from .rope import RotaryEmbedding


class MiniLLM(nn.Module):
    """Decoder-only Transformer（LLaMA2 架构）"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # 1. Token Embedding 层，将 token ids 转换为 hidden_size 维的向量表示
        self.tok_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # 2. TransformerBlock 层，每个层包含一个 Self-Attention 层和一个 FeedForward 层
        self.layers = nn.ModuleList([TransformerBlock(config, i) for i in range(config.num_layers)])
        
        # 3. RMSNorm 层，对 TransformerBlock 输出进行归一化
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        # 4. LM Head 层，将归一化后的 hidden_size 维向量转换为 vocab_size 维的概率分布
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # 5. Rotary Embedding 层，用于计算 RoPE 频率
        self.rope = RotaryEmbedding(config.head_dim, config.max_seq_len, config.rope_theta)

        # 权重绑定：lm_head 和 tok_emb 共享权重，减少 ~3.3M 参数
        self.lm_head.weight = self.tok_emb.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_cache: KVCache = None,
    ) -> torch.Tensor:
        """前向传播

        参数：
            input_ids: (batch, seq_len) token ids
            kv_cache: KV Cache（推理时传入，训练时为 None）

        返回：
            logits: (batch, seq_len, vocab_size)

        流程：
        1. Token Embedding: input_ids → (batch, seq_len, hidden_size)
        2. 计算 RoPE 频率（考虑 KV Cache 偏移）
        3. 逐层经过 TransformerBlock
        4. 最终 RMSNorm
        5. LM Head: (batch, seq_len, hidden_size) → (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape

        # 1. Token Embedding：将 token ids 转换为 hidden_size 维的向量表示
        x = self.tok_emb(input_ids)

        # 2. 计算 RoPE 频率（需要考虑 KV Cache 中已有的长度作为偏移）
        if kv_cache is not None:
            cache_len = kv_cache.get_seq_length(0)
        else:
            cache_len = 0
        freqs = self.rope(cache_len + seq_len)
        freqs = freqs[cache_len:]  # 只取当前 token 对应的频率

        # 3. 逐层经过 TransformerBlock
        for layer in self.layers:
            x = layer(x, freqs, kv_cache)

        # 4. 最终 RMSNorm + LM Head
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> torch.Tensor:
        """自回归生成

        参数：
            input_ids: (1, prompt_len) 提示 token
            max_new_tokens: 最大生成 token 数
            temperature: 温度（0=贪心，>1=更随机）
            top_p: nucleus sampling 阈值
            top_k: top-k sampling 的 k

        核心循环：
        1. 前向传播，取最后一个位置的 logits
        2. temperature 缩放
        3. top-k 过滤（把概率最低的 k 以外的 token 设为 -inf）
        4. top-p nucleus sampling（累积概率超过 p 的截断）
        5. 采样下一个 token
        6. 追加到 generated，重复直到遇到 EOS 或达到 max_new_tokens

        KV Cache 的作用：
        - 第一次传完整 prompt
        - 后续只传新生成的 1 个 token，用 cache 保存历史 K,V
        - 避免对历史 token 重复计算 attention
        """
        kv_cache = KVCache()
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            # 首次用完整输入填充 KV Cache，后续只传新 token
            if kv_cache.get_seq_length(0) == 0:
                logits = self.forward(generated, kv_cache)
            else:
                logits = self.forward(generated[:, -1:], kv_cache)

            next_logits = logits[:, -1, :]

            # temperature 缩放
            if temperature > 0:
                next_logits = next_logits / temperature

            # top-k 过滤
            if top_k > 0:
                top_k_values, top_k_indices = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < top_k_values[:, -1:]] = float('-inf')

            # top-p 过滤（标准 nucleus sampling）
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # 标记需要移除的 token：累积概率超过 top_p 的
                sorted_indices_to_remove = cumulative_probs > top_p
                # 至少保留 1 个 token（右移一位，第一个永远保留）
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                # 映射回原始索引并设为 -inf
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_logits[indices_to_remove] = float('-inf')

            # 采样
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

            # 拼接到 generated
            generated = torch.cat([generated, next_token], dim=1)

            # 遇到 EOS 就停
            if next_token.item() == self.config.eos_token_id:
                break

        return generated

    def count_parameters(self) -> int:
        """统计可训练参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
