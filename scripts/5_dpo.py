"""脚本 5：DPO 偏好对齐

DPO（Direct Preference Optimization）的核心：
  不需要训练 Reward Model，直接用偏好数据优化策略模型。

用法：
    # 主模型 DPO（β=0.2）
    python scripts/5_dpo.py --sft-path outputs/sft/ckpt_final.pt --beta 0.2

    # β 消融实验
    python scripts/5_dpo.py --sft-path outputs/sft/ckpt_final.pt --beta 0.1 --output-dir outputs/dpo_beta0.1
    python scripts/5_dpo.py --sft-path outputs/sft/ckpt_final.pt --beta 0.5 --output-dir outputs/dpo_beta0.5

参数：
    --sft-path: SFT 模型路径（必填）
    --data-path: DPO 数据路径（默认 data/minimind_dataset/dpo.jsonl）
    --output-dir: 输出目录（默认 outputs/dpo）
    --lr: 学习率（默认 5e-7）
    --epochs: 训练轮数（默认 2）
    --beta: DPO 温度参数（默认 0.2）
    --batch-size: batch 大小（默认 2）
    --max-length: 最大序列长度（默认 512）
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.dpo import main

if __name__ == "__main__":
    main()
