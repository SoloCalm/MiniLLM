"""脚本 3：SFT 微调"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.sft import main

if __name__ == "__main__":
    main()
