"""
模型配置：定义 LLM 的所有超参数

所有模块都从这里读取配置，修改一处即可全局生效。

设计原则：
1. 类属性作为默认值，清晰可见
2. __init__ 支持 kwargs 覆盖，方便实验
3. 参数量估算函数，方便验证设计
"""


class ModelConfig:
    """LLM 模型配置（~38M 参数，适配 6GB GPU）

    使用类属性作为默认值，通过 __init__ 接受 kwargs 覆盖。
    这种模式的好处：配置清晰可见、IDE 能自动补全、序列化方便。

    用法示例：
        config = ModelConfig()  # 使用所有默认值
        config = ModelConfig(hidden_size=768, num_layers=24)  # 只修改这两个参数
    """

    # ============================================================
    # 模型架构参数
    # ============================================================

    vocab_size: int = 6400           # 词表大小（BPE分词器，自训练6400词表）
    hidden_size: int = 512           # 隐藏层维度（所有 Transformer 层的宽度）
    num_layers: int = 12             # Transformer Block 层数
    num_heads: int = 8               # Query 注意力头数
    num_kv_heads: int = 4            # Key/Value 注意力头数（GQA：KV 头数 < Q 头数，节省显存）
    intermediate_size: int = 1376    # FFN 中间层维度（SwiGLU 的 gate/up 投影宽度）
    max_seq_len: int = 1024          # 最大序列长度
    rope_theta: float = 10000.0      # RoPE 基础频率（越大支持越长序列）
    rms_norm_eps: float = 1e-6       # RMSNorm 防除零的小常数

    # ============================================================
    # 特殊 token
    # ============================================================

    bos_token_id: int = 1            # <bos> 开始 token
    eos_token_id: int = 2            # <eos> 结束 token
    pad_token_id: int = 0            # <pad> 填充 token

    # ============================================================
    # 预训练参数（默认值，可通过命令行覆盖）
    # ============================================================

    # 预训练：从随机初始化开始，需要较大的学习率
    pretrain_lr: float = 3e-4        # 预训练学习率（通常较大，因为从随机初始化开始）
    pretrain_warmup_steps: int = 1000 # 学习率 warmup 步数
    pretrain_batch_size: int = 16    # 批大小
    pretrain_grad_accum: int = 4     # 梯度累积步数（有效 batch = 16*4 = 64）
    pretrain_weight_decay: float = 0.1 # 权重衰减
    pretrain_max_steps: int = 50000  # 最大训练步数

    # SFT：已有好的表征，需要较小的学习率避免破坏语言知识
    sft_lr: float = 1e-5             # SFT 学习率（远小于预训练，避免破坏已学知识）
    sft_epochs: int = 3              # 训练轮数
    sft_batch_size: int = 8          # 批大小
    sft_grad_accum: int = 2          # 梯度累积步数
    sft_max_length: int = 512        # 最大序列长度

    # DPO：学习微妙的偏好差异，需要极小的学习率
    dpo_lr: float = 5e-7             # DPO 学习率（极小，只做偏好微调）
    dpo_epochs: int = 2              # 训练轮数
    dpo_beta: float = 0.2            # DPO 温度系数（越大越保守，越依赖偏好差距）
    dpo_batch_size: int = 4          # 批大小
    dpo_max_length: int = 512        # 最大序列长度

    # ============================================================
    # 训练基础设施
    # ============================================================

    bf16: bool = True                # 使用 bfloat16 混合精度
    max_grad_norm: float = 1.0       # 梯度裁剪阈值
    log_interval: int = 100          # 每 N 步打印 loss
    save_interval: int = 1000        # 每 N 步保存 checkpoint
    eval_interval: int = 1000        # 每 N 步验证

    # ============================================================
    # 推理参数
    # ============================================================

    default_max_new_tokens: int = 256  # 默认最大生成 token 数
    default_temperature: float = 0.7   # 温度：0 = 贪心，>1 = 更随机
    default_top_p: float = 0.9         # Nucleus sampling 阈值
    default_top_k: int = 50            # Top-K 过滤

    def __init__(self, **kwargs):
        """支持通过 kwargs 覆盖任意配置项

        用法: ModelConfig(hidden_size=768, num_layers=24)
        逻辑: 遍历 kwargs，如果属性存在就 setattr，否则报错

        这种模式比 dataclass 更灵活，可以在运行时动态修改配置。
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config key: {key}")

    @property
    def head_dim(self):
        """每个注意力头的维度 = hidden_size // num_heads

        例如 hidden=512, heads=8 → head_dim=64
        每个 head 独立做注意力计算，维度更小所以计算量更低。

        注意：head_dim 必须是偶数，因为 RoPE 需要将维度分成二维平面对。
        """
        return self.hidden_size // self.num_heads

    def __repr__(self):
        """打印配置概要"""
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

        参数量计算公式：
        - Q 投影: hidden_size × (num_heads × head_dim) = 512 × 512 = 262144
        - K 投影: hidden_size × (num_kv_heads × head_dim) = 512 × 256 = 131072（GQA: 比 Q 少）
        - V 投影: 同 K
        - O 投影: (num_heads × head_dim) × hidden_size = 512 × 512 = 262144
        - FFN: 3 × hidden_size × intermediate_size = 3 × 512 × 1376 = 2113536
        """
        # Embedding 层参数
        embedding_params = self.vocab_size * self.hidden_size

        head_dim = self.head_dim

        # 每层 Transformer 的参数
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

        # 所有 Transformer 层的总参数
        layers = self.num_layers * per_layer

        # 输出层（权重共享时不额外占参数，但为了完整性还是加上）
        output = self.hidden_size * self.vocab_size      # LM Head

        # 最终 RMSNorm
        final_norm = self.hidden_size

        return embedding_params + layers + output + final_norm
