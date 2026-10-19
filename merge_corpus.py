# ============================================================
# 把基础对话语料与增强片段合并成新语料，并做一致性校验
# ------------------------------------------------------------
# 输入:
#   data/Demo_chat.txt          基础语料（500 组「用户：/Yuki：」）
#   data/_parts/para_*.txt      同义问法增强（每个答案 3 种新问法）
#   data/_parts/chat_extra.txt  寒暄 / 日常闲聊补充
#   data/_parts/extra_*.txt     其它补充片段（可选，自动收集）
# 输出:
#   data/Demo_chat_v2.txt       合并后的语料
#
# 校验内容:
#   1. 每个答案的 3 种同义问法是否与基础语料的答案一一对应（逐字）
#   2. 总数是否符合预期
#   3. 是否存在「同一个问句对应不同答案」的冲突（会污染训练）
# 用法: python merge_corpus.py
# ============================================================

import glob
import os
from collections import Counter

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PARTS = os.path.join(BASE, "_parts")
OUT = os.path.join(BASE, "Demo_chat_v2.txt")

VARIANTS_PER_ANSWER = 3


def parse_pairs(path):
    """读一个「用户：/Yuki：」格式的文件，返回 [(问句, 答案), ...]。"""
    pairs, user = [], None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s.startswith("用户："):
                user = s[3:].strip()
            elif s.startswith("Yuki："):
                if user is None:
                    raise ValueError(f"{path}: 出现没有对应问句的 Yuki 行")
                pairs.append((user, s[5:].strip()))
                user = None
    return pairs


def main():
    base_path = os.path.join(BASE, "Demo_chat.txt")
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"找不到基础语料 {base_path}")
    base = parse_pairs(base_path)
    print(f"[基础] {os.path.basename(base_path)}: {len(base)} 组")

    # ---------- 同义问法增强 ----------
    para_files = sorted(glob.glob(os.path.join(PARTS, "para_*.txt")))
    if not para_files:
        raise FileNotFoundError(f"{PARTS} 下没有 para_*.txt")
    variants = []
    for p in para_files:
        part = parse_pairs(p)
        print(f"[增强] {os.path.basename(p)}: {len(part)} 组")
        variants.extend(part)

    expect = len(base) * VARIANTS_PER_ANSWER
    ok = True
    if len(variants) != expect:
        print(f"  !! 同义问法组数不对：期望 {expect}，实际 {len(variants)}")
        ok = False
    else:
        # 按 3 个一组，答案必须与基础语料逐字一致
        bad = []
        for i, (user, ans) in enumerate(base):
            for j in range(VARIANTS_PER_ANSWER):
                q2, a2 = variants[i * VARIANTS_PER_ANSWER + j]
                if a2 != ans:
                    bad.append((i + 1, j + 1, ans, a2))
        if bad:
            print(f"  !! 有 {len(bad)} 组答案对不上，前 3 条：")
            for i, j, a, b in bad[:3]:
                print(f"     第 {i} 组第 {j} 个问法：期望 {a!r} 实际 {b!r}")
            ok = False
        else:
            print(f"  OK 全部 {len(base)} 个答案 × {VARIANTS_PER_ANSWER} 种问法，逐字一致")

    # ---------- 寒暄 / 其它补充 ----------
    extra_files = sorted(glob.glob(os.path.join(PARTS, "chat_extra.txt"))
                         + glob.glob(os.path.join(PARTS, "extra_*.txt")))
    extras = []
    for p in extra_files:
        part = parse_pairs(p)
        print(f"[补充] {os.path.basename(p)}: {len(part)} 组")
        extras.extend(part)

    merged = base + variants + extras
    print(f"[合并] 共 {len(merged)} 组")

    # ---------- 冲突检查：同一个问句不能对应不同答案 ----------
    by_q = {}
    for q, a in merged:
        by_q.setdefault(q, set()).add(a)
    conflicts = {q: v for q, v in by_q.items() if len(v) > 1}
    dup_q = sum(1 for q, c in Counter(q for q, _ in merged).items()
                if c > 1 and len(by_q[q]) == 1)
    if conflicts:
        print(f"  !! 有 {len(conflicts)} 个问句对应了多个不同答案（会污染训练），前 3 条：")
        for q, v in list(conflicts.items())[:3]:
            print(f"     {q!r} -> {v}")
        ok = False
    else:
        print(f"  OK 无「同问不同答」冲突；重复问句（答案相同）{dup_q} 个")

    # ---------- 写出 ----------
    with open(OUT, "w", encoding="utf-8") as f:
        for q, a in merged:
            f.write(f"用户：{q}\nYuki：{a}\n\n")

    chars = sum(len(q) + len(a) for q, a in merged)
    uniq = len(set("".join(q + a for q, a in merged)))
    tokens = chars + len(merged) * 4          # <bos> + <|user|> + <|yuki|> + <eos>
    V = min(uniq + 6, 4200)
    D, F, L = 64, 256, 5
    params = V * D + L * (4 * D * D + 2 * D * F + 2 * D) + D
    print("-" * 56)
    print(f"输出: {OUT}")
    print(f"有效字符 = {chars:,}   不同字符 = {uniq:,}")
    print(f"tokens 估算 = {tokens:,}   训练样本(seq=128) = {(tokens - 1) // 128:,}")
    print(f"词表 V = {V}   参数量 = {params:,} ({params / 1e4:.1f}W)")
    print("校验结论 =", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
