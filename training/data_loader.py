"""
数据加载器

支持三种数据格式：
1. 预训练：纯文本，每行一个 JSON {"text": "..."}
2. SFT：指令数据，{"messages": [...]} 或 {"instruction": ..., "input": ..., "output": ...}
3. DPO：偏好数据，{"prompt": ..., "chosen": ..., "rejected": ...}
"""

import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def load_jsonl(path: Path) -> List[Dict]:
    """读取 jsonl 文件"""
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# ============================================================
# 预训练数据集
# ============================================================

class PretrainDataset(Dataset):
    """预训练数据集

    数据格式：每行 {"text": "一段文本..."}
    处理方式：把所有文本拼接，按 max_length 切成固定长度的块

    为什么不一条一条训练：
    - 预训练数据每条长度不一，直接训练效率低
    - 拼接后切块，每个样本都是 max_length，batch 训练更高效
    - 这是 LLM 预训练的标准做法
    """

    def __init__(self, data_path: Path, tokenizer, max_length: int = 1024, max_lines: int = None):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 1. 读取文本（可限制行数，避免 OOM）
        items = load_jsonl(data_path)
        if max_lines:
            items = items[:max_lines]
        texts = [item["text"] for item in items]
        print(f"加载 {len(texts)} 条文本")

        # 2. 拼接所有文本，一起 tokenize
        all_tokens = []
        for text in texts:
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)
        print(f"总 token 数: {len(all_tokens)}")

        # 3. 按 max_length 切块
        self.samples = []
        for i in range(0, len(all_tokens), max_length):
            chunk = all_tokens[i:i + max_length]
            if len(chunk) > 1:  # 至少 2 个 token 才有意义
                self.samples.append(chunk)
        print(f"训练样本数: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """返回一个训练样本

        返回：
            {
                "input_ids": (max_length,) LongTensor,
                "labels": (max_length,) LongTensor,  # 和 input_ids 一样（next token prediction）
            }
        """
        chunk = self.samples[idx]
        input_ids = torch.tensor(chunk, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "labels": input_ids,   # 和 input_ids 一样，loss 内部右移
        }


# ============================================================
# 预训练数据集（内存映射版，推荐使用）
# ============================================================

class PretrainDatasetMmap(Dataset):
    """预训练数据集（numpy 版，推荐使用）

    解决原版 PretrainDataset 的 OOM 问题：

    问题：
      原版在 __init__ 中一次性 tokenize 所有文本到 Python list，
      127 万条文本 → 3.95 亿 token → Python int 占 28 字节 → ~11GB 内存 → OOM

    解决方案（参考 MicroLM）：
      1. 用 scripts/tokenize_to_disk.py 提前 tokenize，存成 .npy 文件
      2. numpy uint16 存储：2 字节/token（vs Python int 28 字节，省 14 倍）
      3. numpy 加载到内存：3.95 亿 token × 2 字节 = ~791 MB（可接受）

    和 MicroLM 的区别：
      MicroLM 用 mmap_mode="r"（内存映射，数据留在磁盘）
      我们用普通 np.load（全部加载到内存，但 numpy 比 Python list 省 14 倍）
      原因：Windows 上 mmap 大文件有兼容性问题

    用法：
      先运行预 tokenize：
        python scripts/tokenize_to_disk.py
      然后训练时指定 .npy 路径：
        python scripts/2_pretrain.py --tokenized-data data/pretrain_tokenized/train_ids.npy
    """

    def __init__(self, token_ids_path: Path, max_length: int = 1024):
        self.max_length = max_length

        # numpy 加载：uint16 只占 2 字节/token，比 Python list 省 14 倍
        self.token_ids = np.load(token_ids_path).astype(np.int64)
        self.total_tokens = len(self.token_ids)

        print(f"加载 token ids: {self.total_tokens} tokens ({self.total_tokens * 2 / 1e6:.1f} MB)")

    def __len__(self):
        # 可用的训练样本数
        return (self.total_tokens - 1) // self.max_length

    def __getitem__(self, idx):
        """按索引取第 idx 个固定长度的片段"""
        start = idx * self.max_length
        chunk = self.token_ids[start:start + self.max_length]

        input_ids = torch.from_numpy(chunk)
        return {
            "input_ids": input_ids,
            "labels": input_ids,
        }


# ============================================================
# SFT 数据集
# ============================================================

class SFTDataset(Dataset):
    """SFT 数据集

    数据格式：{"instruction": ..., "input": ..., "output": ...}
    或：{"conversations": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}

    关键点：
    - 用 chat_template 把对话转成 token 序列
    - labels 中 prompt 部分设为 -100（只对 assistant 回复计算 loss）
    """

    def __init__(self, data_path: Path, tokenizer, max_length: int = 512, max_lines: int = None):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 1. 读取数据（可限制行数）
        items = load_jsonl(data_path)
        if max_lines:
            items = items[:max_lines]
        print(f"加载 {len(items)} 条 SFT 数据")

        # 2. 对每条数据预处理
        self.samples = []
        for item in items:
            sample = self.preprocess(item)
            if sample is not None:
                self.samples.append(sample)

        print(f"有效样本数: {len(self.samples)}")

    def preprocess(self, item: dict) -> dict:
        """将一条原始数据转成 input_ids 和 labels

        两种数据格式：
        1. minimind: {"conversations": [{"role": "user", ...}, {"role": "assistant", ...}]}
        2. BelleGroup: {"instruction": ..., "input": ..., "output": ...}
        """
        # 构造 user 部分和 assistant 部分的文本
        if "conversations" in item:
            # minimind 格式
            user_text = ""
            assistant_text = ""
            for turn in item["conversations"]:
                if turn["role"] == "user":
                    user_text = turn["content"]
                elif turn["role"] == "assistant":
                    assistant_text = turn["content"]
        elif "instruction" in item:
            # BelleGroup 格式
            user_text = item["instruction"]
            if item.get("input"):
                user_text += "\n" + item["input"]
            assistant_text = item["output"]
        else:
            return None

        if not user_text or not assistant_text:
            return None

        # 3. tokenize 两部分
        user_ids = self.tokenizer.encode(user_text)
        assistant_ids = self.tokenizer.encode(assistant_text)

        # 4. 拼接：[BOS] user tokens [EOS] assistant tokens [EOS]
        #    注意：SentencePiece 的 encode 默认不加 BOS/EOS，需要手动加
        bos_id = 1  # <s>
        eos_id = 2  # </s>

        input_ids = [bos_id] + user_ids + [eos_id] + assistant_ids + [eos_id]

        # 5. 构造 labels：prompt 部分设为 -100
        #    prompt = [BOS] user tokens [EOS]（包括 user 部分和分隔符）
        prompt_len = 1 + len(user_ids) + 1  # BOS + user + EOS
        labels = [-100] * prompt_len + assistant_ids + [eos_id]

        # 6. 截断
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]

        return {
            "input_ids": input_ids,
            "labels": labels,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """返回：
            {
                "input_ids": ...,
                "labels": ...,  # prompt 部分为 -100
            }
        """
        sample = self.samples[idx]
        return {
            "input_ids": torch.tensor(sample["input_ids"], dtype=torch.long),
            "labels": torch.tensor(sample["labels"], dtype=torch.long),
        }


# ============================================================
# DPO 数据集
# ============================================================

class DPODataset(Dataset):
    """DPO 偏好数据集

    数据格式：{"prompt": "...", "chosen": "...", "rejected": "..."}

    每条数据包含一个问题的两个回答：
    - chosen: 人类偏好的回答
    - rejected: 人类不偏好的回答

    DPO 训练时需要同时计算两个回答的 loss

    数据流：
    原始数据 → tokenize → 拼接 → 构造 labels → 返回 4 个 tensor
    """

    def __init__(self, data_path, tokenizer, max_length: int = 512):
        """
        参数：
            data_path: jsonl 文件路径
            tokenizer: SentencePiece 分词器
            max_length: 最大序列长度（超过会截断）
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 读取 jsonl 文件，每行是一个 json 对象
        # 返回 list: [{"prompt": "...", "chosen": "...", "rejected": "..."}, ...]
        self.samples = load_jsonl(Path(data_path))
        print(f"加载 {len(self.samples)} 条 DPO 数据")

    def __len__(self):
        """返回数据集大小"""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        返回第 idx 条数据的 4 个 tensor

        返回：
            {
                "chosen_input_ids": [BOS] prompt [EOS] chosen [EOS],
                "chosen_labels":    [-100...-100]      chosen [EOS],
                "rejected_input_ids": [BOS] prompt [EOS] rejected [EOS],
                "rejected_labels":    [-100...-100]      rejected [EOS],
            }
        """
        # 1. 获取原始数据
        sample = self.samples[idx]
        prompt = sample["prompt"]      # 用户问题
        chosen = sample["chosen"]      # 偏好的回答
        rejected = sample["rejected"]  # 不偏好的回答

        # 2. Tokenize（文本 → token ids）
        # 例如: "如何学习" → [101, 2023, 102, 506]
        prompt_ids = self.tokenizer.encode(prompt)
        chosen_ids = self.tokenizer.encode(chosen)
        rejected_ids = self.tokenizer.encode(rejected)

        # 3. 特殊 token 的 id
        bos_id = 1  # <s> 句子开始
        eos_id = 2  # </s> 句子结束

        # 4. 拼接 input_ids
        # 格式: [BOS] prompt [EOS] chosen/rejected [EOS]
        #
        # 为什么这样拼接？
        # - [BOS]: 告诉模型句子开始
        # - prompt: 用户问题
        # - [EOS]: 告诉模型 prompt 结束，开始生成回答
        # - chosen/rejected: 模型要生成的回答
        # - [EOS]: 告诉模型回答结束
        chosen_input_ids = [bos_id] + prompt_ids + [eos_id] + chosen_ids + [eos_id]
        rejected_input_ids = [bos_id] + prompt_ids + [eos_id] + rejected_ids + [eos_id]

        # 5. 构造 labels（prompt 部分设为 -100）
        #
        # 为什么 prompt 部分是 -100？
        # - DPO 只关心"回答"的质量，不关心"问题"的理解
        # - 训练时 loss 只在 chosen/rejected 部分计算
        # - prompt 部分不参与 loss 计算，所以设为 -100
        # - PyTorch 的 cross_entropy 会自动忽略 label=-100 的位置
        prompt_len = 1 + len(prompt_ids) + 1  # BOS + prompt + EOS 的长度

        # labels 格式: [-100...-100] chosen_ids [EOS]
        #              └── prompt 部分 ──┘  └─ 回答部分 ─┘
        chosen_labels = [-100] * prompt_len + chosen_ids + [eos_id]
        rejected_labels = [-100] * prompt_len + rejected_ids + [eos_id]

        # 6. 截断（防止序列过长导致 OOM）
        if len(chosen_input_ids) > self.max_length:
            chosen_input_ids = chosen_input_ids[:self.max_length]
            chosen_labels = chosen_labels[:self.max_length]

        if len(rejected_input_ids) > self.max_length:
            rejected_input_ids = rejected_input_ids[:self.max_length]
            rejected_labels = rejected_labels[:self.max_length]

        # 7. 返回 4 个 tensor
        return {
            "chosen_input_ids": torch.tensor(chosen_input_ids, dtype=torch.long),
            "chosen_labels": torch.tensor(chosen_labels, dtype=torch.long),
            "rejected_input_ids": torch.tensor(rejected_input_ids, dtype=torch.long),
            "rejected_labels": torch.tensor(rejected_labels, dtype=torch.long),
        }


# ============================================================
# 数据加载器工厂
# ============================================================

def create_dataloader(dataset, batch_size: int, shuffle: bool = True):
    """创建 DataLoader

    使用自定义的 collate_fn 处理动态 padding
    （不同样本长度不同时，pad 到 batch 内最长）
    """
    def collate_fn(batch):
        """动态 padding：pad 到 batch 内最长序列"""
        keys = batch[0].keys()
        padded = {}
        for key in keys:
            sequences = [item[key] for item in batch]
            # 找 batch 内最大长度
            max_len = max(seq.shape[0] for seq in sequences)
            # pad 到最大长度
            padded_seqs = []
            for seq in sequences:
                if seq.shape[0] < max_len:
                    pad_len = max_len - seq.shape[0]
                    pad = torch.full((pad_len,), 0, dtype=seq.dtype)  # pad_token_id=0
                    padded_seq = torch.cat([seq, pad])
                else:
                    padded_seq = seq
                padded_seqs.append(padded_seq)
            padded[key] = torch.stack(padded_seqs)
        return padded

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )
