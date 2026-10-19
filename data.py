# ============================================================
# Yuki - Demo 数据管线
# ------------------------------------------------------------
# 支持两种语料形态（自动检测，也可以用 mode 强制）：
#   1) 对话语料：以「用户：」/「Yuki：」开头的行
#      **空行 = 对话分隔符**：空行后的「用户」行开一段新对话；
#      紧跟着上一句 Yuki 的「用户」行算同一段的下一轮（这就是多轮对话）
#   2) 普通语料：每个非空行算一篇文档
# 可选清洗：去掉行首序号（"1 今天也要开心" -> "今天也要开心"）
# 切块：把所有文档的 token 拼成一条长序列，再按 seq_len 切成 (x, y) 对
# 注意：清洗只在内存里做，不会改动 data/ 下的原始语料文件
# ============================================================

import glob
import os
import re

import numpy as np

from tokenizer import BOS_ID, EOS_ID, USER_ID, YUKI_ID

# 行首序号：数字 + 可选分隔符 + 至少一个空白
# 匹配 "1 句子" / "12 句子" / "3. 句子" / "4、句子"；不会误伤 "2026年..."
LINE_NUMBER_RE = re.compile(r"^\s*\d+\s*[.、,，:：)）\]]?\s+")

# 角色前缀
ROLE_USER_RE = re.compile(r"^\s*(用户|清久酱|user|User|USER|Q)\s*[:：]\s*(.*)$")
ROLE_YUKI_RE = re.compile(r"^\s*(Yuki|yuki|YUKI|助手|AI|assistant|A)\s*[:：]\s*(.*)$")


def load_texts(data_dir: str):
    """读取目录下所有 .txt，返回 (文件列表, 原始文本列表)。"""
    data_dir = os.path.abspath(data_dir)
    files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    if not files:
        raise FileNotFoundError(
            f"在 {data_dir} 下没有找到任何 .txt 语料文件。\n"
            f"把你的语料放进这个目录即可（可以放多个 txt，会自动合并）。"
        )
    texts = []
    for f in files:
        with open(f, "r", encoding="utf-8", errors="ignore") as fh:
            texts.append(fh.read().replace("\r\n", "\n").replace("\r", "\n"))
    return files, texts


def detect_mode(texts) -> str:
    """出现 2 行以上角色前缀就认为是对话语料。"""
    hits = 0
    for t in texts:
        for line in t.split("\n"):
            if ROLE_USER_RE.match(line) or ROLE_YUKI_RE.match(line):
                hits += 1
                if hits >= 2:
                    return "dialogue"
    return "plain"


def _clean_line(line: str, strip_line_numbers: bool):
    """去掉行首序号，返回 (新行, 是否改动了)。"""
    if not strip_line_numbers:
        return line, False
    cleaned = LINE_NUMBER_RE.sub("", line).strip()
    if cleaned != line:
        return cleaned, True
    return line, False


def build_corpus(texts, strip_line_numbers: bool = True,
                 doc_per_line: bool = True, mode: str = "auto"):
    """把原始文本整理成文档列表。

    返回 (documents, stats)：
      documents: [ [(special_token_id_or_None, text), ...], ... ]
                 一段对话 / 一篇文档 = 一个列表
      stats:     行数 / 去掉序号的行数 / 文档数 / 对话轮数 / 实际使用的模式
    """
    if mode == "auto":
        mode = detect_mode(texts)

    documents = []
    n_lines = 0
    n_stripped = 0
    n_turns = 0

    if mode == "dialogue":
        current = None
        prev_blank = True                  # 文件开头视为「空行之后」
        for t in texts:
            for raw in t.split("\n"):
                line = raw.strip()
                if not line:
                    prev_blank = True      # 空行 = 对话分隔符，本身不进语料
                    continue
                n_lines += 1
                line, changed = _clean_line(line, strip_line_numbers)
                n_stripped += int(changed)
                if not line:
                    continue

                m_user = ROLE_USER_RE.match(line)
                m_yuki = ROLE_YUKI_RE.match(line)
                if m_user:
                    # 空行之后的「用户」行 = 新的一段对话
                    # 紧跟上一句 Yuki 的「用户」行 = 同一段对话的下一轮（多轮）
                    if current is None or prev_blank:
                        if current:
                            documents.append(current)
                        current = [(USER_ID, m_user.group(2).strip())]
                    else:
                        current.append((USER_ID, m_user.group(2).strip()))
                elif m_yuki:
                    if current is None:       # 允许对话以 Yuki 开头
                        current = []
                    current.append((YUKI_ID, m_yuki.group(2).strip()))
                else:
                    if current:               # 非角色行：结束当前对话，单独成篇
                        documents.append(current)
                        current = None
                    documents.append([(None, line)])
                prev_blank = False
        if current:
            documents.append(current)
    else:
        for t in texts:
            lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
            n_lines += len(lines)
            if not doc_per_line:
                body_lines = []
                for line in lines:
                    line, changed = _clean_line(line, strip_line_numbers)
                    n_stripped += int(changed)
                    if line:
                        body_lines.append(line)
                if body_lines:
                    documents.append([(None, "\n".join(body_lines))])
                continue
            for line in lines:
                line, changed = _clean_line(line, strip_line_numbers)
                n_stripped += int(changed)
                if line:
                    documents.append([(None, line)])

    for doc in documents:
        n_turns += len(doc)
    stats = {"mode": mode, "lines": n_lines, "stripped": n_stripped,
             "docs": len(documents), "turns": n_turns}
    return documents, stats


def corpus_texts(documents):
    """把文档展开成纯文本列表（给分词器统计字符用）。"""
    return [text for doc in documents for _, text in doc]


def encode_corpus(tokenizer, documents) -> np.ndarray:
    """编码：每篇文档 [<bos>] + 各轮（角色 token + 文本）+ [<eos>]，再拼成一条长序列。"""
    ids = []
    for doc in documents:
        ids.append(BOS_ID)
        for special_id, text in doc:
            if special_id is not None:
                ids.append(special_id)
            ids.extend(tokenizer.encode(text))
        ids.append(EOS_ID)
    return np.asarray(ids, dtype=np.int64)


def make_chunks(ids: np.ndarray, seq_len: int):
    """切成 (x, y)：y 是 x 右移一位，即标准的自回归下一 token 预测。"""
    n = (len(ids) - 1) // seq_len
    if n <= 0:
        raise ValueError(
            f"语料太短：共 {len(ids)} 个 token，至少需要 {seq_len + 1} 个才能切出一条训练样本。"
        )
    arr = ids[: n * seq_len + 1]
    x = arr[:-1].reshape(n, seq_len)
    y = arr[1:].reshape(n, seq_len)
    return x, y


def batch_iterator(x: np.ndarray, y: np.ndarray, batch_size: int, rng):
    """无限循环产出 mini-batch：每轮打乱样本顺序，末尾不足一个 batch 的丢掉。"""
    n = len(x)
    while True:
        order = rng.permutation(n)
        for s in range(0, n - batch_size + 1, batch_size):
            b = order[s:s + batch_size]
            yield x[b], y[b]


def epoch_batches(documents, tokenizer, seq_len: int, batch_size: int, rng):
    """每个 epoch 重新打乱文档顺序，再重新编码、切块、分批（无限循环）。

    为什么必须每轮重排：
      如果把对话按固定顺序拼成一整条长序列，模型会走捷径 —— 用「上一段对话」
      推断自己在语料里的位置，然后直接吐出下一段的答案。这样训练 loss 会很低，
      但模型其实没学会「看问题选答案」；等到真正生成时只给它一个问题（没有前文），
      它就不知道自己在第几段，只能瞎猜。
      每轮重排之后，前后文变成随机搭配，模型只能靠问题本身来选答案。
    """
    while True:
        order = rng.permutation(len(documents))
        shuffled = [documents[i] for i in order]
        ids = encode_corpus(tokenizer, shuffled)
        x, y = make_chunks(ids, seq_len)
        idx = rng.permutation(len(x))
        for s in range(0, len(x) - batch_size + 1, batch_size):
            b = idx[s:s + batch_size]
            yield x[b], y[b]
