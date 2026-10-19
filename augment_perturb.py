# ============================================================
# 字符级扰动增强 —— 治「删一个字就崩」
# ------------------------------------------------------------
# 做法：给问句生成若干扰动版本，**答案一字不改**。
# 这样模型学到的是「意思对就答这句」，而不是「必须逐字对上」。
#
# 扰动算子（随机选一个，保证结果与原句不同）：
#   del    删掉 1 个字         （对应评测里的「删 1 个字」）
#   add    随机插入 1 个字      （对应评测里的「多 1 个字」）
#   swap   交换相邻 2 个字
#   dup    重复 1 个字（"我我"）
#   prep   前面加口语词（那个/嗯/就是/欸/话说）
#   strip  去掉句尾标点
# 插入用的字按语料字频加权采样，保证「像人打错的」而不是乱码。
#
# 剂量（--per-turn）就是「每个问句生成几种不同的错法」：
#   V2.3 用了 1，V2.4 用了 2（合起来就是历史上那两份 perturb 片段）。
#   同一个问句的多个扰动默认**互不重复**，也会避开 --avoid 里已有的问句。
#
# 多轮（--include-multiturn）：
#   默认只动单轮段（V2.3 / V2.4 就是这么做的）。
#   加上这个开关后，多轮段的**最后一轮**也会被扰动，前面几轮原样保留当上下文 ——
#   这样模型能学到「聊了好几轮之后，用户打了个错字也照样听得懂」。
#
# 用法:
#   # 单轮，每问 2 种错法
#   python augment_perturb.py --input data/Demo_chat_v22.txt \
#                             --out data/_parts6/perturb3.txt --per-turn 2
#   # 单轮 + 多轮末轮，且避开 v24 里已有的问句
#   python augment_perturb.py --input data/_raw/Demo_chat_v22.txt \
#       --out data/_parts6/perturb3.txt --per-turn 2 --seed 20261017 \
#       --include-multiturn --avoid data/Demo_chat_v24.txt
# ============================================================

import argparse
import os
import random
from collections import Counter

FILLERS = ["那个", "嗯", "就是", "欸", "话说", "对了"]
TAIL_PUNCT = "。！？~，、"


def parse_conversations(path):
    convs, cur = [], []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                if cur:
                    convs.append(cur)
                    cur = []
                continue
            if s.startswith("用户："):
                cur.append([s[3:].strip(), None])
            elif s.startswith("Yuki："):
                if not cur:
                    cur.append(["", None])
                cur[-1][1] = s[5:].strip()
    if cur:
        convs.append(cur)
    return convs


def collect_questions(path):
    """读一个语料文件里所有问句（单轮 + 多轮的每一轮都算）。"""
    qs = set()
    for c in parse_conversations(path):
        for q, a in c:
            if q and a:
                qs.add(q)
    return qs


def perturb(q, rng, chars, avoid=None, tries=14):
    """对问句施加一次随机扰动；保证非空、与原句不同、且不在 avoid 里。"""
    ops = ["del", "add", "swap", "dup", "prep", "strip"]
    for _ in range(tries):
        op = rng.choice(ops)
        if op == "del" and len(q) > 2:
            i = rng.randrange(len(q))
            out = q[:i] + q[i + 1:]
        elif op == "add":
            i = rng.randrange(len(q) + 1)
            out = q[:i] + rng.choice(chars) + q[i:]
        elif op == "swap" and len(q) > 2:
            i = rng.randrange(len(q) - 1)
            out = q[:i] + q[i + 1] + q[i] + q[i + 2:]
        elif op == "dup":
            i = rng.randrange(len(q))
            out = q[:i + 1] + q[i] + q[i + 1:]
        elif op == "prep":
            out = rng.choice(FILLERS) + q
        elif op == "strip":
            out = q.rstrip(TAIL_PUNCT)
        else:
            continue
        out = out.strip()
        if out and out != q and (avoid is None or out not in avoid):
            return out, op
    return None, None


def main():
    ap = argparse.ArgumentParser(description="字符级扰动增强")
    ap.add_argument("--input", default="data/Demo_chat_v22.txt")
    ap.add_argument("--out", default="data/_parts4/perturb.txt")
    ap.add_argument("--per-turn", type=int, default=1,
                    help="每个问句生成几个**互不重复**的扰动版本")
    ap.add_argument("--seed", type=int, default=20261016)
    ap.add_argument("--include-multiturn", action="store_true",
                    help="多轮段的最后一轮也加扰动（前几轮原样保留当上下文）")
    ap.add_argument("--avoid", nargs="*", default=None,
                    help="不要生成这些语料里已有的问句（避免和旧扰动集重复）")
    ap.add_argument("--no-dedup", action="store_true",
                    help="允许同一个问句生成重复的扰动（默认不重复）")
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    src = a.input if os.path.isabs(a.input) else os.path.join(here, a.input)
    dst = a.out if os.path.isabs(a.out) else os.path.join(here, a.out)

    convs = parse_conversations(src)
    single = [c for c in convs if len(c) == 1 and c[0][0] and c[0][1]]
    multi = [c for c in convs if len(c) >= 2]
    print(f"[输入] {os.path.basename(src)}: {len(convs)} 段，"
          f"单轮段 {len(single)}，多轮段 {len(multi)}")

    # 待扰动的目标：(前几轮上下文, 问句, 答案)
    targets = [([], c[0][0], c[0][1]) for c in single]
    if a.include_multiturn:
        for c in multi:
            q, ans = c[-1]
            if q and ans:
                targets.append(([(u, x) for u, x in c[:-1]], q, ans))
        print(f"[多轮] 末轮也加入扰动：+{len(targets) - len(single)} 个目标")
    else:
        print(f"[多轮] {len(multi)} 段（默认不动多轮，加 --include-multiturn 打开）")

    # 用字频加权采样「插入什么字」，让扰动更像人打错
    freq = Counter()
    for c in convs:
        for q, ans in c:
            freq.update(q or "")
            freq.update(ans or "")
    chars = list(freq)
    weights = [freq[c] for c in chars]

    # --avoid：已有的问句不重复生成；生成过的也记进去，避免跨问句撞车
    avoid = set()
    for p in (a.avoid or []):
        ap_ = p if os.path.isabs(p) else os.path.join(here, p)
        if os.path.exists(ap_):
            got = collect_questions(ap_)
            avoid |= got
            print(f"[避免] {os.path.basename(ap_)}: {len(got)} 个已有问句")
        else:
            print(f"[避免] 警告：找不到 {ap_}，忽略")

    rng = random.Random(a.seed)
    out_segs, op_count = [], Counter()
    n_single = n_multi = 0
    short = 0
    for prefix, q, ans in targets:
        made, seen = 0, set()
        attempts = 0
        while made < a.per_turn and attempts < a.per_turn * 10 + 20:
            attempts += 1
            pq, op = perturb(q, rng, chars, avoid=avoid)
            if not pq:
                continue
            if not a.no_dedup and pq in seen:
                continue
            seen.add(pq)
            avoid.add(pq)
            out_segs.append(prefix + [(pq, ans)])
            op_count[op] += 1
            made += 1
        if made < a.per_turn:
            short += 1
        if prefix:
            n_multi += made
        else:
            n_single += made

    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        for seg in out_segs:
            for q, ans in seg:
                f.write(f"用户：{q}\nYuki：{ans}\n")
            f.write("\n")

    # ---------- 自查 ----------
    orig = {q for _, q, _ in targets}
    seg_keys = [tuple(seg) for seg in out_segs]
    dup = len(seg_keys) - len(set(seg_keys))
    print("-" * 60)
    print(f"[输出] {os.path.basename(dst)}: {len(out_segs)} 段"
          f"（单轮 {n_single}，多轮末轮 {n_multi}）")
    print(f"  目标 {len(targets)} 个，每个要 {a.per_turn} 种错法"
          + (f"，其中 {short} 个没凑够" if short else "，全部凑够"))
    print(f"  算子分布: {dict(op_count)}")
    print(f"  段级重复: {dup}   扰动后与原句完全相同: "
          f"{sum(1 for seg in out_segs if seg[-1][0] in orig)}")
    for seg in out_segs[:2]:
        pre = " / ".join(q for q, _ in seg[:-1])
        tag = f"[多轮: {pre}] " if pre else ""
        print(f"  例: {tag}{seg[-1][0]!r} -> {seg[-1][1]!r}")


if __name__ == "__main__":
    main()
