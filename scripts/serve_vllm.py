"""
vLLM 服务化部署

用 vLLM 启动 Qwen2.5-1.5B QLoRA 模型的 API 服务。
vLLM 支持 PagedAttention，推理效率远高于原生 PyTorch。

用法：
    # 启动服务（默认 http://localhost:8000）
    python scripts/serve_vllm.py

    # 自定义端口
    python scripts/serve_vllm.py --port 8080

    # 指定 adapter 路径
    python scripts/serve_vllm.py --adapter-path outputs/sft_qlora/lora_adapter

依赖：
    pip install vllm
"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="vLLM 部署 Qwen2.5-1.5B QLoRA")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B",
                        help="基础模型名称")
    parser.add_argument("--adapter-path", type=str, default="outputs/sft_qlora/lora_adapter",
                        help="LoRA adapter 路径")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=2048,
                        help="最大序列长度")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Tensor 并行数（单卡设为 1）")
    args = parser.parse_args()

    print("=" * 50)
    print("vLLM 部署 Qwen2.5-1.5B QLoRA")
    print("=" * 50)
    print(f"基础模型: {args.model_name}")
    print(f"Adapter:  {args.adapter_path}")
    print(f"地址:     http://{args.host}:{args.port}")
    print()

    # 检查 vllm 是否安装
    try:
        import vllm
        print(f"vLLM 版本: {vllm.__version__}")
    except ImportError:
        print("错误: 未安装 vllm，请运行: pip install vllm")
        sys.exit(1)

    # 检查 adapter 是否存在
    from pathlib import Path
    if not Path(args.adapter_path).exists():
        print(f"错误: adapter 路径不存在: {args.adapter_path}")
        print("请先运行 QLoRA 训练: python scripts/4_qlora.py")
        sys.exit(1)

    # 启动 vLLM
    print("启动 vLLM 服务...")
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model_name,
        "--adapter", args.adapter_path,
        "--host", args.host,
        "--port", str(args.port),
        "--max-model-len", str(args.max_model_len),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--trust-remote-code",
    ]

    print(f"命令: {' '.join(cmd)}")
    print()
    print("服务启动后，可以用以下方式调用：")
    print()
    print("  # curl 调用")
    print(f'  curl http://localhost:{args.port}/v1/chat/completions \\')
    print(f'    -H "Content-Type: application/json" \\')
    print(f'    -d \'{{"model": "{args.model_name}", "messages": [{{"role": "user", "content": "你好"}}]}}\'')
    print()
    print("  # Python 调用")
    print("  from openai import OpenAI")
    print(f'  client = OpenAI(base_url="http://localhost:{args.port}/v1", api_key="none")')
    print('  resp = client.chat.completions.create(model="qwen", messages=[{"role":"user","content":"你好"}])')
    print()

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n服务已停止")
    except subprocess.CalledProcessError as e:
        print(f"服务启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
