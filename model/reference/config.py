"""参考代码 - config.py
对照 model/config.py 中的 TODO 逐个参考
"""

class ModelConfig:
    """模型超参数配置类

    使用类属性作为默认值，通过 __init__ 接受 kwargs 覆盖。
    这种模式的好处：配置清晰可见、IDE 能自动补全、序列化方便。
    """
    # ===== 模型结构 =====
    vocab_size: int = 6400            # 词表大小，自训练 SentencePiece tokenizer
    hidden_size: int = 512            # 隐藏层维度（所有 Transformer 层的宽度）
    num_layers: int = 12              # Transformer Block 层数
    num_heads: int = 8                # Query 注意力头数
    num_kv_heads: int = 4             # Key/Value 注意力头数（GQA：KV 头数 < Q 头数，节省显存）
    intermediate_size: int = 1376     # FFN 中间层维度（SwiGLU 的 gate/up 投影宽度）
    max_seq_len: int = 1024           # 最大序列长度
    rope_theta: float = 10000.0       # RoPE 基础频率（越大支持越长序列）
    rms_norm_eps: float = 1e-6        # RMSNorm 防除零的小常数

    # ===== 特殊 token =====
    bos_token_id: int = 1             # 句子开始 token
    eos_token_id: int = 2             # 句子结束 token
    pad_token_id: int = 0             # 填充 token

    # ===== 预训练超参 =====
    pretrain_lr: float = 3e-4         # 预训练学习率（通常较大，因为从随机初始化开始）
    pretrain_warmup_steps: int = 1000 # 学习率 warmup 步数
    pretrain_batch_size: int = 16
    pretrain_grad_accum: int = 4      # 梯度累积步数（有效 batch = 16*4 = 64）
    pretrain_weight_decay: float = 0.1
    pretrain_max_steps: int = 50000

    # ===== SFT 超参 =====
    sft_lr: float = 1e-5              # SFT 学习率（远小于预训练，避免破坏已学知识）
    sft_epochs: int = 3
    sft_batch_size: int = 8
    sft_grad_accum: int = 2
    sft_max_length: int = 512

    # ===== DPO 超参 =====
    dpo_lr: float = 5e-7              # DPO 学习率（极小，只做偏好微调）
    dpo_epochs: int = 2
    dpo_beta: float = 0.2             # DPO 温度系数（越大越保守，越依赖偏好差距）
    dpo_batch_size: int = 4
    dpo_max_length: int = 512

    # ===== 训练控制 =====
    bf16: bool = True                 # 使用 bfloat16 混合精度
    max_grad_norm: float = 1.0        # 梯度裁剪阈值
    log_interval: int = 100           # 每 N 步打印 loss
    save_interval: int = 1000         # 每 N 步保存 checkpoint
    eval_interval: int = 1000         # 每 N 步验证

    # ===== 推理默认值 =====
    default_max_new_tokens: int = 256
    default_temperature: float = 0.7  # 温度：0 = 贪心，>1 = 更随机
    default_top_p: float = 0.9        # Nucleus sampling 阈值
    default_top_k: int = 50           # Top-K 过滤

    # ================================================================
    # TODO 1: __init__ — 参考下面的实现
    # ================================================================
    def __init__(self, **kwargs):
        """用 kwargs 覆盖类属性默认值

        例如 ModelConfig(hidden_size=1024, num_layers=24) 只修改这两个参数，
        其余保持类定义的默认值。这种模式比 dataclass 更灵活。
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config key: {key}")

    # ================================================================
    # TODO 2: head_dim — 参考下面的实现
    # ================================================================
    @property
    def head_dim(self):
        """每个注意力头的维度 = hidden_size / num_heads

        例如 hidden=512, heads=8 → head_dim=64
        每个 head 独立做注意力计算，维度更小所以计算量更低。
        """
        return self.hidden_size // self.num_heads

    def __repr__(self):
        return f"ModelConfig(hidden={self.hidden_size}, layers={self.num_layers}, heads={self.num_heads}, params≈{self.estimate_params()/1e6:.1f}M)"

    # ================================================================
    # TODO 3: estimate_params — 参考下面的实现
    # ================================================================
    def estimate_params(self):
        """估算模型总参数量（不含词表 embedding 时约 41M）

        参数分布：
        - Embedding: vocab * hidden（最大单模块）
        - 每层 Attention: Q/K/V/O 四个线性投影（GQA 让 K/V 更小）
        - 每层 FFN: gate + up + down 三个矩阵
        - 每层 2 个 RMSNorm: 各 hidden_size 个参数
        """
        embedding = self.vocab_size * self.hidden_size
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
        output = self.hidden_size * self.vocab_size      # LM head（权重共享时实际不额外占参数）
        final_norm = self.hidden_size
        return embedding + layers + output + final_norm
