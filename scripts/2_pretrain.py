"""脚本 2：预训练"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.pretrain import main

if __name__ == "__main__":
    main()
