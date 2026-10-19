# ============================================================
# Yuki - Demo 字符级分词器
# 不依赖任何第三方库：统计语料里的字符，按词频保留最常用的若干字
# 特殊 token：<pad>=0, <unk>=1, <bos>=2, <eos>=3
# ============================================================

import json
import os
from collections import Counter

# 0~3: 基础特殊 token；4~5: 对话角色 token
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", "<|user|>", "<|yuki|>"]
PAD_ID, UNK_ID, BOS_ID, EOS_ID, USER_ID, YUKI_ID = 0, 1, 2, 3, 4, 5


class CharTokenizer:
    """字符级分词器：一个字符 = 一个 token（中文天然友好）。"""

    def __init__(self, chars, max_vocab_size: int = 4200):
        self.max_vocab_size = int(max_vocab_size)
        self.itos = list(SPECIAL_TOKENS) + list(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}

    # ---------- 构建 ----------
    @classmethod
    def build(cls, texts, max_vocab_size: int = 4200,
              base: "CharTokenizer" = None) -> "CharTokenizer":
        """从语料构建字符级分词器。

        base=None   ：全新构建，字符按词频降序排 id。
        base=<旧分词器>：**只追加模式** —— 旧字符的 id 一个都不动，新字符按词频
                      追加到末尾。这样旧 checkpoint 的嵌入行不会错位，语料扩充后
                      可以热启动/续训，而不是从头重训。
        """
        counter = Counter()
        for t in texts:
            counter.update(t)
        # 丢掉控制字符（保留换行与制表）；按词频降序、同频按字符升序，保证可复现
        items = [(c, n) for c, n in counter.items() if c.isprintable() or c in "\n\t"]
        items.sort(key=lambda kv: (-kv[1], kv[0]))

        if base is not None:
            old_chars = list(base.itos[len(SPECIAL_TOKENS):])
            known = set(old_chars)
            room = max(0, int(max_vocab_size) - len(SPECIAL_TOKENS) - len(old_chars))
            fresh = [c for c, _ in items if c not in known]
            added = fresh[:room]
            tok = cls(old_chars + added, max_vocab_size)
            tok.total_chars = sum(counter.values())
            tok.unique_chars = len(items)
            tok.base_size = len(old_chars)
            tok.added_chars = len(added)
            tok.oov_candidates = len(fresh) - len(added)   # 超出词表上限、只能当 <unk> 的
            return tok

        limit = max(1, int(max_vocab_size) - len(SPECIAL_TOKENS))
        chars = [c for c, _ in items[:limit]]
        tok = cls(chars, max_vocab_size)
        tok.total_chars = sum(counter.values())
        tok.unique_chars = len(items)
        tok.base_size = 0
        tok.added_chars = len(chars)
        tok.oov_candidates = max(0, len(items) - len(chars))
        return tok

    # ---------- 属性 ----------
    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    # ---------- 编解码 ----------
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list:
        ids = [self.stoi.get(c, UNK_ID) for c in text]
        if add_bos:
            ids.insert(0, BOS_ID)
        if add_eos:
            ids.append(EOS_ID)
        return ids

    def decode(self, ids) -> str:
        """解码成文本：所有特殊 token 都不回显，遇到 <eos> 停止。"""
        out = []
        n_special = len(SPECIAL_TOKENS)
        for i in ids:
            i = int(i)
            if i == EOS_ID:
                break
            if i < n_special:
                if i == UNK_ID:
                    out.append("?")
                continue
            out.append(self.itos[i] if i < len(self.itos) else "?")
        return "".join(out)

    # ---------- 持久化 ----------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "type": "char",
            "max_vocab_size": self.max_vocab_size,
            "total_chars": getattr(self, "total_chars", None),
            "unique_chars": getattr(self, "unique_chars", None),
            "chars": self.itos[len(SPECIAL_TOKENS):],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        tok = cls(payload["chars"], payload.get("max_vocab_size", 4200))
        tok.total_chars = payload.get("total_chars")
        tok.unique_chars = payload.get("unique_chars")
        return tok
