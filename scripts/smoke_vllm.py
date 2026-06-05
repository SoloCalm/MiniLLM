"""
vLLM 功能验证

验证 vLLM 服务是否正常工作。

用法：
    # 先启动服务: python scripts/serve_vllm.py
    # 再运行验证: python scripts/smoke_vllm.py
"""

import json
import sys
import urllib.request


def test_chat_completions(base_url: str = "http://localhost:8000"):
    """测试 /v1/chat/completions 接口"""
    url = f"{base_url}/v1/chat/completions"
    payload = json.dumps({
        "model": "Qwen/Qwen2.5-1.5B",
        "messages": [{"role": "user", "content": "你好"}],
        "max_tokens": 50,
        "temperature": 0.7,
    }).encode("utf-8")

    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            content = result["choices"][0]["message"]["content"]
            print(f"  ✅ chat/completions 正常")
            print(f"     回复: {content[:80]}")
            return True
    except Exception as e:
        print(f"  ❌ chat/completions 失败: {e}")
        return False


def test_models(base_url: str = "http://localhost:8000"):
    """测试 /v1/models 接口"""
    url = f"{base_url}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            models = [m["id"] for m in result["data"]]
            print(f"  ✅ /v1/models 正常")
            print(f"     可用模型: {models}")
            return True
    except Exception as e:
        print(f"  ❌ /v1/models 失败: {e}")
        return False


def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    print(f"验证 vLLM 服务: {base_url}")
    print()

    passed = 0
    total = 2

    print("[1/2] 测试 /v1/models")
    if test_models(base_url):
        passed += 1

    print("[2/2] 测试 /v1/chat/completions")
    if test_chat_completions(base_url):
        passed += 1

    print()
    print(f"结果: {passed}/{total} 通过")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
