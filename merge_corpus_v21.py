# ============================================================
# 合并对话语料（通用版）：base + parts -> out，并做校验
# ------------------------------------------------------------
# 默认（不传参数时）等价于最初构建 V2.1 的用法：
#     base  = data/Demo_chat_v21.txt      （上一代的语料，作为基底）
#     parts = data/_parts3                （这一代新增的片段）
#     out   = data/Demo_chat_v22.txt
#
# 做三件事：
#   1. 解析成「一段一段的对话」，支持多轮（空行 = 对话分隔符）
#   2. 自动处理「同一个问句对应不同答案」的冲突：保留先出现的（base 优先），
#      丢掉后面的冲突轮次并报告 —— 这种冲突会污染训练，必须清掉
#   3. 可选校验：如果 parts 里有 chat_para.txt，检查它的答案是否与
#      _parts/chat_extra.txt 逐字对应（当初的寒暄增强就是这么验的）
#
# 可选：切留出集（--holdout）——
#   把一部分「整段」从训练语料里拿出来，单独写一个文件，供 eval_chat.py 评「留出分」。
#   训练脚本只读 data/ 顶层的 *.txt，留出集放在 data/_holdout/ 下，天然不会被训到。
#
# 用法:
#   python merge_corpus_v21.py
#   python merge_corpus_v21.py --base data/Demo_chat_v21.txt --parts data/_parts3 \
#                              --out data/Demo_chat_v22.txt
#   python merge_corpus_v21.py --base data/Demo_chat_v24.txt --parts data/_parts6 \
#                              --out data/Demo_chat_v25.txt --holdout 0.1
# ============================================================

import argparse
import glob
import os
import random
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(HERE, "data")

DEF_BASE = os.path.join(BASE_DIR, "Demo_chat_v21.txt")
DEF_PARTS = os.path.join(BASE_DIR, "_parts3")
DEF_OUT = os.path.join(BASE_DIR, "Demo_chat_v22.txt")

# 模型结构（用于估算参数量）：D=64 L=9 F=162 SwiGLU -> ~52W
D, F, L, MLP_MATS = 64, 162, 9, 3


def parse_conversations(path):
    """读对话文件 -> [[(问, 答), ...], ...]，空行分隔一段对话。"""
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


def iter_turns(convs):
    for c in convs:
        for q, a in c:
            if q is not None and a is not None:
                yield q, a


def write_convs(path, convs):
    """按「用户：/Yuki：」格式写文件，段与段之间空一行。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in convs:
            for q, ans in c:
                f.write(f"用户：{q}\nYuki：{ans}\n")
            f.write("\n")


def split_holdout(kept, frac, mode, seed):
    """把对话段切成 (train, holdout)，两者不重叠。

    mode="segment"：按段随机留出。口径最直接；因为同一个「答案家族」的其他变体
                    仍在训练集里，留出分实际测的是「同话题的新变体/新错法」——
                    正好是我们迭代想提升的那个能力（字面鲁棒性）。
    mode="answer" ：按「答案家族」整族留出（一个答案的所有变体问法要么全进训练、
                    要么全进留出）。这是「没见过的话题」的天花板分，
                    因为答案串是任意的，预期会很低、也很慢才涨，当诊断用。
    """
    n = len(kept)
    n_ho = int(round(n * frac))
    if n_ho <= 0:
        return kept, []
    rng = random.Random(seed)

    if mode == "segment":
        order = list(range(n))
        rng.shuffle(order)
        ho = set(order[:n_ho])
        return ([c for i, c in enumerate(kept) if i not in ho],
                [kept[i] for i in sorted(ho)])

    # answer 模式：按答案家族分组，随机挑家族整族留出，直到段数接近目标
    groups = {}
    for i, c in enumerate(kept):
        groups.setdefault(tuple(sorted(a for _, a in c)), []).append(i)
    keys = sorted(groups.keys())
    rng.shuffle(keys)
    ho, cnt = set(), 0
    for k in keys:
        if cnt >= n_ho:
            break
        ho.update(groups[k])
        cnt += len(groups[k])
    return ([c for i, c in enumerate(kept) if i not in ho],
            [kept[i] for i in sorted(ho)])


def strip_leaked(holdout, train):
    """丢掉留出集里「问句与训练集重题」的整段。

    这种题在训练集里有一模一样的问句（答案也相同），必然被背过，
    留在留出集里只会把分数抬高、测不出任何东西 —— 整段剔除，宁可小一点。
    """
    train_qs = {q for c in train for q, _ in c}
    clean = [c for c in holdout if not any(q in train_qs for q, _ in c)]
    return clean, len(holdout) - len(clean)


def main():
    ap = argparse.ArgumentParser(description="合并对话语料并校验")
    ap.add_argument("--base", default=DEF_BASE, help="基底语料（已校验过的上一代）")
    ap.add_argument("--parts", default=DEF_PARTS, help="新增片段目录（*_*.txt，_ 开头忽略）")
    ap.add_argument("--no-parts", action="store_true",
                    help="不读片段目录（只做 base -> out，可用来「只给现有语料切留出集」）")
    ap.add_argument("--out", default=DEF_OUT, help="训练语料输出文件")
    ap.add_argument("--holdout", type=float, default=0.0,
                    help="留出集比例（0=不切分，例如 0.1）。这部分的整段会被排除出训练语料")
    ap.add_argument("--holdout-mode", choices=["segment", "answer"], default="segment",
                    help="留出单位：segment=按段随机（默认，测同话题新变体）/"
                         "answer=按答案家族整族留出（测没见过的话题，作天花板诊断）")
    ap.add_argument("--holdout-seed", type=int, default=20261016,
                    help="抽留出集的随机种子；固定它，各代之间的留出集才可比")
    ap.add_argument("--holdout-out", default=None,
                    help="留出集输出路径（默认 <data>/_holdout/<out 的文件名>）")
    ap.add_argument("--keep-leaked", action="store_true",
                    help="保留留出集里与训练集重题的段（默认整段剔除）")
    ap.add_argument("--refresh-holdout", action="store_true",
                    help="即使留出集文件已存在也重新抽一份（会破坏跨代可比性，慎用）")
    a = ap.parse_args()

    base_path = a.base if os.path.isabs(a.base) else os.path.join(HERE, a.base)
    parts_dir = a.parts if os.path.isabs(a.parts) else os.path.join(HERE, a.parts)
    out_path = a.out if os.path.isabs(a.out) else os.path.join(HERE, a.out)

    if not os.path.exists(base_path):
        raise FileNotFoundError(f"找不到基底语料 {base_path}")
    base = parse_conversations(base_path)
    print(f"[基底] {os.path.basename(base_path)}: {len(base)} 段 / "
          f"{sum(len(c) for c in base)} 轮")

    if a.no_parts:
        parts = []
        print("[新增] --no-parts：本次不读片段目录")
    else:
        parts = [p for p in sorted(glob.glob(os.path.join(parts_dir, "*.txt")))
                 if not os.path.basename(p).startswith("_")]
        if not parts:
            raise FileNotFoundError(f"{parts_dir} 下没有可用的 txt")
    new_convs = []
    for p in parts:
        c = parse_conversations(p)
        turns = sum(len(x) for x in c)
        print(f"[新增] {os.path.basename(p)}: {len(c)} 段 / {turns} 轮")
        new_convs.extend(c)

    # ---------- 校验 chat_para（如果这一批里有）----------
    para_path = os.path.join(parts_dir, "chat_para.txt")
    extra_path = os.path.join(BASE_DIR, "_parts", "chat_extra.txt")
    if os.path.exists(para_path) and os.path.exists(extra_path):
        src_answers = [x for _, x in iter_turns(parse_conversations(extra_path))]
        got = [x for _, x in iter_turns(parse_conversations(para_path))]
        exp = [x for x in src_answers for _ in range(3)]
        if got == exp:
            print(f"  OK chat_para 的 {len(got)} 条答案与 chat_extra 的 "
                  f"{len(src_answers)} 个答案逐字一致（各 3 份）")
        else:
            print(f"  !! chat_para 答案对不上：期望 {len(exp)} 条，实际 {len(got)} 条")
            for i, (x, y) in enumerate(zip(exp, got)):
                if x != y:
                    print(f"     第 {i + 1} 条 期望 {x!r} 实际 {y!r}")
                    break

    # ---------- 冲突处理：同问不同答 ----------
    # 注意：不是「抽掉冲突的那一轮」，而是「整段丢掉」。
    # 多轮段落里抽掉中间一轮会留下「上一句答完、下一句话题突变」的断点，
    # 那是在教模型学错误的承接方式 —— 宁可少一段数据。
    convs = base + new_convs
    first_answer = {}
    kept, dropped, dropped_segs = [], [], 0
    for c in convs:
        conflict = False
        for q, ans in c:
            if q is None or ans is None:
                continue
            if q in first_answer and first_answer[q] != ans:
                conflict = True
                dropped.append((q, ans, first_answer[q]))
        if conflict:
            dropped_segs += 1
            continue
        for q, ans in c:
            if q is not None and ans is not None:
                first_answer.setdefault(q, ans)
        kept.append([(q, ans) for q, ans in c if q is not None and ans is not None])
    kept = [c for c in kept if c]

    if dropped:
        print(f"  !! 有 {dropped_segs} 段与已有语料冲突（同问不同答），已整段丢弃；"
              f"涉及 {len(dropped)} 轮，前 3 条：")
        for q, ans, old in dropped[:3]:
            print(f"     {q!r}: 新答案 {ans!r}（先出现的 {old!r}）")
    else:
        print("  OK 无「同问不同答」冲突")

    # ---------- 留出集（固定不变，跨代可比）----------
    # 关键点：留出集只在**第一次**按种子抽出来，之后就固定成一份文件，
    # 以后每次 merge 都从训练语料里「按内容排除」它。
    # 否则语料一变大、重新抽一次，各代的留出集就不是同一批题了，分数没法比。
    train, holdout, holdout_path = kept, [], None
    if a.holdout:
        if not (0.0 < a.holdout < 1.0):
            raise ValueError(f"--holdout 要在 (0,1) 之间，收到{a.holdout}")
        holdout_path = a.holdout_out or os.path.join(BASE_DIR, "_holdout", "holdout.txt")
        if not os.path.isabs(holdout_path):
            holdout_path = os.path.join(HERE, holdout_path)

        reused = os.path.exists(holdout_path) and not a.refresh_holdout
        removed = 0
        if reused:
            holdout = []
            for c in parse_conversations(holdout_path):
                turns = [(q, ans) for q, ans in c if q and ans]
                if turns:
                    holdout.append(turns)
            ho_keys = {tuple(c) for c in holdout}
            n_before = len(train)
            train = [c for c in train if tuple(c) not in ho_keys]
            removed = n_before - len(train)
        else:
            train, holdout = split_holdout(kept, a.holdout, a.holdout_mode, a.holdout_seed)
            if not a.keep_leaked:
                holdout, leaked = strip_leaked(holdout, train)
                if leaked:
                    print(f"[留出] 剔除留出集里与训练集重题的 {leaked} 段")

        tr_q = {q for c in train for q, _ in c}
        tr_a = {x for c in train for _, x in c}
        # 「被考问句」= 每段的最后一轮（单轮段就是它唯一的一轮，多轮段就是评测要生成的末轮）
        ho_test = {c[-1][0] for c in holdout if c}
        ho_ctx = {q for c in holdout for q, _ in c[:-1]}
        tested_leak = sorted(ho_test & tr_q)
        ctx_leak = sorted(ho_ctx & tr_q)
        covered = sum(1 for c in holdout if any(x in tr_a for _, x in c))
        print("-" * 60)
        print(f"[留出] {'复用固定留出集' if reused else '首次创建留出集'}"
              f"  模式={a.holdout_mode} 种子={a.holdout_seed} 目标={a.holdout:.0%}")
        print(f"[留出] 留出 {len(holdout):,} 段 / {sum(len(c) for c in holdout):,} 轮"
              f"  ->  {os.path.relpath(holdout_path, HERE)}"
              + (f"（本次从训练语料排除 {removed:,} 段）" if reused else ""))
        print(f"[留出] 训练 {len(train):,} 段 / {sum(len(c) for c in train):,} 轮"
              f"  ->  {os.path.relpath(out_path, HERE)}")
        print(f"[留出] 被考问句 {len(ho_test):,} 条中，训练集里也有的有 {len(tested_leak)} 条"
              + ("（OK，没有泄漏）" if not tested_leak else "  <<< 这是泄漏，要查！"))
        if tested_leak:
            print(f"       !! 前 3 条: {tested_leak[:3]}")
        print(f"[留出] 多轮『上下文』问句与训练集重叠 {len(ctx_leak)} 条"
              f"（不算泄漏：只当上下文喂进去，不直接计分）")
        print(f"[留出] 留出段中「答案在训练集里见过」的占 {covered}/{max(len(holdout), 1)}"
              f"（这就是留出分的真实含义：同话题的新变体）")
        ho_only = (set("".join(q + a for c in holdout for q, a in c))
                   - set("".join(q + a for c in train for q, a in c)))
        if ho_only:
            sample = "、".join(sorted(ho_only)[:12]) + ("…" if len(ho_only) > 12 else "")
            print(f"[留出] 注意：留出集里有 {len(ho_only)} 个字符不在训练语料中（{sample}）\n"
                  "        从头训练会把它们当成 <unk> -> 留出分偏低；"
                  "续训/继承权重不受影响（词表只追加）。\n"
                  "        想避免：从头训练时带上完整词表（--keep-tokenizer + 旧的 tokenizer.json）。")
        print("[留出] 提醒：若这一代是 --resume/--init-from 继承自「切分前」的模型，"
              "它已经见过留出集，分数偏乐观；\n"
              "        只有从头训练出来的模型，留出分才真正干净。")

    # ---------- 统计（以训练语料为准，因为分词器只从它统计）----------
    all_turns = list(iter_turns(train))
    dist = Counter(len(c) for c in train)
    chars = sum(len(q) + len(a) for q, a in all_turns)
    uniq = len(set("".join(q + a for q, a in all_turns)))
    tokens = chars + sum(2 + 2 * len(c) for c in train)
    V = min(uniq + 6, 4200)
    params = V * D + L * (4 * D * D + MLP_MATS * D * F + 2 * D) + D

    write_convs(out_path, train)
    if holdout_path:
        write_convs(holdout_path, holdout)

    print("-" * 60)
    print(f"输出: {out_path}")
    print(f"对话 {len(train):,} 段 / {len(all_turns):,} 轮"
          + (f"（已排除留出 {len(holdout):,} 段）" if holdout_path else ""))
    print("轮数分布: " + ", ".join(f"{k}轮×{v}" for k, v in sorted(dist.items())))
    multi = sum(v for k, v in dist.items() if k >= 2)
    print(f"其中多轮段（≥2 轮）: {multi:,} 段")
    print(f"有效字符 = {chars:,}   不同字符 = {uniq:,}")
    print(f"tokens 估算 = {tokens:,}   训练样本(seq=128) = {(tokens - 1) // 128:,}")
    print(f"词表 V = {V}   参数量 = {params:,} ({params / 1e4:.1f}W)  [D={D} L={L} F={F}]")


if __name__ == "__main__":
    main()
