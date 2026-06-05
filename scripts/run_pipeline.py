"""
数据处理 pipeline

串联清洗→tokenize→切分，支持选择性执行。

用法：
    # 跑完整 pipeline
    python scripts/run_pipeline.py

    # 只跑清洗
    python scripts/run_pipeline.py --stage clean

    # 只跑 tokenize
    python scripts/run_pipeline.py --stage tokenize

    # 只跑 SFT 数据准备
    python scripts/run_pipeline.py --stage sft
"""

import argparse
import subprocess
import sys
from pathlib import Path


STAGES = {
    "clean": {
        "desc": "预训练语料清洗",
        "cmd": [sys.executable, "data_utils/clean_pretrain.py"],
    },
    "tokenize": {
        "desc": "预 tokenize 到磁盘",
        "cmd": [sys.executable, "scripts/tokenize_to_disk.py"],
    },
    "sft": {
        "desc": "SFT 数据准备",
        "cmd": [sys.executable, "data_utils/prepare_sft.py"],
    },
    "dpo": {
        "desc": "DPO 数据转换",
        "cmd": [sys.executable, "data_utils/convert_ultrafeedback.py"],
    },
}


def main():
    parser = argparse.ArgumentParser(description="数据处理 pipeline")
    parser.add_argument("--stage", type=str, default=None,
                        choices=list(STAGES.keys()),
                        help="指定执行阶段（默认全部）")
    args = parser.parse_args()

    stages = [args.stage] if args.stage else list(STAGES.keys())

    print("=" * 50)
    print("数据处理 Pipeline")
    print("=" * 50)
    print(f"执行阶段: {' → '.join(stages)}")
    print()

    for stage_name in stages:
        stage = STAGES[stage_name]
        print(f"[{stage_name}] {stage['desc']}...")
        print(f"  命令: {' '.join(stage['cmd'])}")
        print()

        result = subprocess.run(stage["cmd"], cwd=str(Path(__file__).parent.parent))
        if result.returncode != 0:
            print(f"  ❌ 阶段 {stage_name} 失败 (exit code {result.returncode})")
            sys.exit(1)

        print(f"  ✅ {stage_name} 完成")
        print()

    print("=" * 50)
    print("Pipeline 完成")
    print("=" * 50)


if __name__ == "__main__":
    main()
