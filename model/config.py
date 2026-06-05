"""
模型配置：定义 LLM 的所有超参数

所有模块都从这里读取配置，修改一处即可全局生效。
"""

class ModelConfig:
    """LLM 模型配置（41M 参数，适配 6GB GPU）"""

    # ============================================================
    # 模型架构参数
    # ============================================================

    vocab_size: int = 6400           # 词表大小（BPE）
    hidden_size: int = 512           # 隐藏层维度
    num_layers: int = 12             # Transformer 层数
    num_heads: int = 8               # Q 注意力头数
    num_kv_heads: int = 4            # KV 注意力头数（GQA：KV 头数 < Q 头数）
    intermediate_size: int = 1376    # FFN 中间层维度
    max_seq_len: int = 1024          # 最大序列长度
    rope_theta: float = 10000.0      # RoPE 基础频率
    rms_norm_eps: float = 1e-6       # RMSNorm 的 epsilon

    # ============================================================
    # 特殊 token
    # ============================================================

    bos_token_id: int = 1            # <bos> 开始 token
    eos_token_id: int = 2            # <eos> 结束 token
    pad_token_id: int = 0            # <pad> 填充 token

    # ============================================================
    # 训练参数（默认值，可通过命令行覆盖）
    # ============================================================

    # 预训练
    pretrain_lr: float = 3e-4
    pretrain_warmup_steps: int = 1000
    pretrain_batch_size: int = 16
    pretrain_grad_accum: int = 4
    pretrain_weight_decay: float = 0.1
    pretrain_max_steps: int = 50000

    # SFT
    sft_lr: float = 1e-5
    sft_epochs: int = 3
    sft_batch_size: int = 8
    sft_grad_accum: int = 2
    sft_max_length: int = 512

    # DPO
    dpo_lr: float = 5e-7
    dpo_epochs: int = 2
    dpo_beta: float = 0.2
    dpo_batch_size: int = 4
    dpo_max_length: int = 512

    # ============================================================
    # 训练基础设施
    # ============================================================

    bf16: bool = True
    max_grad_norm: float = 1.0
    log_interval: int = 100
    save_interval: int = 1000
    eval_interval: int = 1000

    # ============================================================
    # 推理参数
    # ============================================================

    default_max_new_tokens: int = 256
    default_temperature: float = 0.7
    default_top_p: float = 0.9
    default_top_k: int = 50

    def __init__(self, **kwargs):
        """支持通过 kwargs 覆盖任意配置项

        用法: ModelConfig(hidden_size=768, num_layers=24)
        提示: 遍历 kwargs，如果属性存在就 setattr，否则报错
        """
        # TODO: 实现 kwargs 覆盖逻辑
        # 提示: 用 hasattr() 检查属性是否存在，用 setattr() 设置值
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config key: {key}")

    @property
    def head_dim(self):
        """每个注意力头的维度 = hidden_size // num_heads"""
        # TODO: 返回 head_dim
        return self.hidden_size // self.num_heads

    def __repr__(self):
        return f"ModelConfig(hidden={self.hidden_size}, layers={self.num_layers}, heads={self.num_heads}, params≈{self.estimate_params()/1e6:.1f}M)"

    def estimate_params(self):
        """估算模型总参数量

        分四部分计算：
        1. Embedding 层: vocab_size * hidden_size
        2. Transformer 层（共 num_layers 层）:
           - 注意力: Q/K/V/O 四个投影矩阵
             注意：GQA 时 K、V 的头数是 num_kv_heads，不是 num_heads
           - FFN: gate/up/down 三个投影矩阵
           - 2 个 RMSNorm: 每个有 hidden_size 个参数
        3. LM Head: hidden_size * vocab_size（权重共享时已算在 embedding 里）
        4. Final RMSNorm: hidden_size

        思考题：
        - Q 投影的参数量是 hidden_size * (num_heads * head_dim)，为什么？
        - K 投影的参数量和 Q 一样吗？为什么 GQA 时不同？
        - FFN 为什么是 3 * hidden_size * intermediate_size？
        """
        # TODO: 实现参数量估算
        # 提示: 先算 embedding，再算每层的参数，乘以层数，最后加 output 和 final_norm
        embedding_params = self.vocab_size * self.hidden_size
        head_dim = self.head_dim
        # GQA 关键：Q 有 num_heads 个投影，K/V 只有 num_kv_heads 个
        attn_per_layer = (
            self.hidden_size * self.num_heads * head_dim       # Q: hidden → heads * head_dim
            + self.hidden_size * self.num_kv_heads * head_dim  # K: hidden → kv_heads * head_dim（GQA: 比 Q 少）
            + self.hidden_size * self.num_kv_heads * head_dim  # V: 同 K
            + self.num_heads * head_dim * self.hidden_size     # O: heads * head_dim → hidden
        )
        ffn_per_layer = 3 * self.hidden_size * self.intermediate_size  # gate + up + down
        norm_per_layer = 2 * self.hidden_size                           # 2 个 RMSNorm
        per_layer = attn_per_layer + ffn_per_layer + norm_per_layer
        layers = self.num_layers * per_layer
        output = self.hidden_size * self.vocab_size      # LM Head
        final_norm = self.hidden_size
        return embedding_params + layers + output + final_norm
