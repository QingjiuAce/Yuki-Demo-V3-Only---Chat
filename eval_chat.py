# ============================================================
# Yuki - Demo 对话评测（只看「生成时准确率」，不看 loss）
# ------------------------------------------------------------
# 测试组怎么来（按优先级）：
#   1) --set NAME=GLOB     可重复，显式指定（V3 新语料就用这个）
#   2) --sets-file FILE    每行一句 NAME=GLOB，方便把一组测试存成文件
#   3) 都不给时           沿用 V2 的遗留分组（只保留真实存在的）；
#                          一个都没有就退化成「从训练语料抽样」当记忆分组
#   4) 留出集              --holdout-dir（默认 data/_holdout/）-> 「留出分」
#
# 每个测试组会自动分流：单轮段 -> 直接生成比对；多轮段 -> 用金标准前文测末轮。
# robust（删 1 字 / 多 1 字）会自动从「干净问句集」抽前 N 条现造变体 ——
#   这是唯一「真·没见过」的题，字面鲁棒性就看这两格。
# 缺任何文件都只会跳过并提示，**不会崩**（V2 版会因为写死路径直接 FileNotFoundError）。
#
# ⚠️ 除 holdout 外，其余各组都来自训练语料 —— 它们是「记忆分」。
#    只有 holdout 是没训过的题；判断有没有真变强，先看 robust，再看留出分。
#
# 用法:
#   python eval_chat.py                            # 遗留/V2 分组 + 留出集
#   python eval_chat.py --set base=data/my.txt --set para=data/para_*.txt
#   python eval_chat.py --sets-file data/eval_sets.txt
#   python eval_chat.py --limit 100                # 每组只测前 N 条（快速看趋势）
#   python eval_chat.py --json runs/eval.json      # 顺便导出明细（给 run_iteration.py 读）
# ============================================================

import argparse
import glob
import json
import os

import numpy as np

from chat import generate_reply
from generate import load_model

HERE = os.path.dirname(os.path.abspath(__file__))


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(HERE, path)


# ---------------- 语料读取 ----------------

def parse_pairs(path, limit=None):
    """单轮文件 -> [(问, 答), ...]"""
    pairs, user = [], None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s.startswith("用户："):
                user = s[3:].strip()
            elif s.startswith("Yuki："):
                if user is not None:
                    pairs.append((user, s[5:].strip()))
                    user = None
    return pairs[:limit] if limit else pairs


def parse_conversations(path, limit=None):
    """对话文件（空行分段）-> [[(问, 答), ...], ...]"""
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
    return convs[:limit] if limit else convs


def load_segments(path):
    """读一个语料文件 -> 段列表 [[(q, a), ...], ...]。

    兼容两种写法：
      - 空行分段（merge / augment 的输出）：多轮会被正确还原；
      - 通篇没有空行、只是「用户/Yuki」成对罗列：按对拆成单轮段，
        否则会被误判成「一段超长的多轮对话」。
    """
    segs = [[(q, a) for q, a in c if q and a] for c in parse_conversations(path)]
    segs = [s for s in segs if s]
    pairs = parse_pairs(path)
    if len(segs) == 1 and len(pairs) > 1:
        return [[p] for p in pairs]
    return segs


def auto_corpus(data_dir):
    """data/ 顶层唯一的训练语料（多份就返回 None，避免乱猜）。"""
    files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    return files[0] if len(files) == 1 else None


def collect_sets(args, data_dir):
    """收集要评的测试组 -> [(名字, [路径, ...]), ...]。任何缺失都只跳过、不报错。"""
    specs = []

    if args.sets_file:
        p = resolve(args.sets_file)
        if not os.path.exists(p):
            print(f"[测试组] 找不到 {p}，忽略")
        else:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s and not s.startswith("#") and "=" in s:
                        specs.append(s)

    if args.set:
        specs.extend(args.set)

    if specs:
        out = []
        for spec in specs:
            name, _, pat = spec.partition("=")
            name, pat = name.strip(), pat.strip()
            if not name or not pat:
                print(f"[测试组] 跳过写法不对的一句: {spec!r}")
                continue
            paths = sorted(glob.glob(resolve(pat)))
            if not paths:
                print(f"[测试组] {name}: 没匹配到任何文件（{pat}），跳过")
                continue
            out.append((name, paths))
        return out

    # 没有显式指定 -> 用 V2 的遗留分组，只保留真实存在的
    legacy = [
        ("base       原句", [os.path.join(data_dir, "_raw", "Demo_chat.txt")]),
        ("para       同义问法", sorted(glob.glob(os.path.join(data_dir, "_parts", "para_*.txt")))),
        ("chitchat   寒暄", [os.path.join(data_dir, "_parts", "chat_extra.txt")]),
        ("chat_var   寒暄变体", [os.path.join(data_dir, "_parts2", "chat_para.txt")]),
        ("numeric    数字", [os.path.join(data_dir, "_parts2", "numeric.txt")]),
        ("multiturn  多轮", sorted(glob.glob(os.path.join(data_dir, "_parts2", "multiturn_*.txt"))
                                  + glob.glob(os.path.join(data_dir, "_parts3", "longmt_*.txt")))),
    ]
    out = [(n, [p for p in ps if os.path.exists(p)]) for n, ps in legacy]
    out = [(n, ps) for n, ps in out if ps]
    if out:
        return out

    # 连遗留分组都没有 -> 退化成从训练语料抽样（V3 换新语料时的兜底）
    corpus = resolve(args.eval_corpus) if args.eval_corpus else auto_corpus(data_dir)
    if args.eval_corpus and not os.path.exists(corpus):
        print(f"[测试组] --eval-corpus 找不到: {corpus}")
        return []
    if corpus:
        print("[测试组] 没找到任何遗留测试集，退化成「从训练语料抽样」当记忆分组")
        return [("corpus     训练语料抽样", [corpus])]
    print("[测试组] 一份测试集都没找到。用 --set NAME=GLOB 指定，或把语料放进 data/。")
    return []


# ---------------- 评测 ----------------

def accuracy(model, tok, cfg, pairs, history=None, seed=0):
    """贪心解码完全答对率。history 用于多轮（所有题共用同一段前文）。"""
    ok = 0
    for q, a in pairs:
        got = generate_reply(model, tok, q, history, temperature=0,
                             max_new_tokens=24, seed=seed)
        ok += int(got == a)
    return ok, len(pairs)


def run_multi(model, tok, cfg, segments, seed=0):
    """多轮：带金标准前文，逐段测最后一轮。"""
    ok = tot = 0
    for seg in segments:
        if len(seg) < 2:
            continue
        hist = seg[:-1]
        q_last, a_last = seg[-1]
        got = generate_reply(model, tok, q_last, hist, temperature=0,
                             max_new_tokens=24, seed=seed)
        ok += int(got == a_last)
        tot += 1
    return ok, tot


def main():
    ap = argparse.ArgumentParser(description="Yuki - Demo 对话评测")
    ap.add_argument("--out-dir", type=str, default="checkpoints")
    ap.add_argument("--limit", type=int, default=None, help="每组只测前 N 条")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", action="append", default=None, metavar="NAME=GLOB",
                    help="测试组，可重复：--set base=data/my.txt --set para=data/para_*.txt")
    ap.add_argument("--sets-file", type=str, default=None,
                    help="从文件读测试组（每行 NAME=GLOB，# 开头为注释）")
    ap.add_argument("--eval-corpus", type=str, default=None,
                    help="没有 --set 时的兜底：从这份语料抽样当记忆分组（默认自动找 data/ 顶层唯一的 txt）")
    ap.add_argument("--holdout-dir", type=str, default=None,
                    help="留出集目录（默认 data/_holdout）")
    ap.add_argument("--robust-src", type=str, default=None,
                    help="robust 从哪份语料抽干净问句（默认取第一个测试组）")
    ap.add_argument("--robust-n", type=int, default=200, help="robust 抽多少条（默认 200）")
    ap.add_argument("--json", type=str, default=None,
                    help="把逐组结果写成 json（供 run_iteration.py 读取）")
    args = ap.parse_args()

    try:
        model, tok, cfg = load_model(args.out_dir)
    except FileNotFoundError as e:
        print(f"[评测] 加载不了模型：{e}")
        print("        先用 train.py 训练一份，或者用 --out-dir 指向别的 checkpoint 目录")
        return 1

    print("=" * 74)
    print(f" Yuki - Demo 评测 | 词表 {tok.vocab_size} | 参数量 {model.num_params():,} "
          f"| D={cfg.hidden_dim} L={cfg.num_layers} F={cfg.ff_dim} mlp={cfg.mlp}")
    print("=" * 74)

    data = os.path.join(HERE, "data")
    rows = []

    # ---------- 测试组（记忆分）----------
    sets = collect_sets(args, data)
    for name, paths in sets:
        segs = []
        for p in paths:
            segs.extend(load_segments(p))
        single = [s[0] for s in segs if len(s) == 1]
        multi = [s for s in segs if len(s) >= 2]
        if args.limit:
            single, multi = single[:args.limit], multi[:args.limit]
        if single:
            ok, n = accuracy(model, tok, cfg, single, seed=args.seed)
            rows.append((f"{name} 单轮", ok, n, "mem"))
        if multi:
            ok, n = run_multi(model, tok, cfg, multi, seed=args.seed)
            rows.append((f"{name} 多轮末轮", ok, n, "mem"))

    # ---------- robust：真·没见过的扰动 ----------
    pool = []
    if args.robust_src:
        src = resolve(args.robust_src)
        if os.path.exists(src):
            pool = parse_pairs(src)
        else:
            print(f"[robust] --robust-src 找不到: {src}")
    else:
        for name, paths in sets:
            for p in paths:
                pool = [s[0] for s in load_segments(p) if len(s) == 1]
                if pool:
                    break
            if pool:
                break
    pool = pool[:args.robust_n]

    if pool:
        rng = np.random.default_rng(0)

        def del1(s):
            if len(s) <= 2:
                return s
            i = int(rng.integers(0, len(s)))
            return s[:i] + s[i + 1:]

        for label, mut in (("robust     删 1 个字", del1),
                           ("robust     多 1 个字", lambda s: s + "呀")):
            ok = 0
            for q, a in pool:
                qq = mut(q)
                if qq and generate_reply(model, tok, qq, temperature=0,
                                         max_new_tokens=24, seed=args.seed) == a:
                    ok += 1
            rows.append((label, ok, len(pool), "robust"))
    else:
        print("[robust] 没有可用的干净问句集，跳过（可用 --robust-src 指定）")

    # ---------- 留出集：唯一可信的「真会」分数 ----------
    ho_dir = args.holdout_dir or os.path.join(data, "_holdout")
    ho_dir = ho_dir if os.path.isabs(ho_dir) else os.path.join(HERE, ho_dir)
    ho_paths = sorted(glob.glob(os.path.join(ho_dir, "*.txt")))
    if ho_paths:
        segs = []
        for p in ho_paths:
            segs.extend(load_segments(p))
        single = [s[0] for s in segs if len(s) == 1]
        multi = [s for s in segs if len(s) >= 2]
        if args.limit:
            single, multi = single[:args.limit], multi[:args.limit]
        if single:
            ok, n = accuracy(model, tok, cfg, single, seed=args.seed)
            rows.append(("holdout    留出集 单轮", ok, n, "holdout"))
        if multi:
            ok, n = run_multi(model, tok, cfg, multi, seed=args.seed)
            rows.append(("holdout    留出集 多轮末轮", ok, n, "holdout"))
    else:
        print(f"[留出] 没找到留出集目录 {ho_dir} —— 先跑 "
              f"merge_corpus_v21.py --holdout 0.1 生成它")

    # ---------- 汇总 ----------
    if not rows:
        print("\n[评测] 一组都没跑成：既没有测试集、也没有留出集。")
        return 1

    print(f"{'测试项':<30}{'答对':>8}{'总数':>8}{'准确率':>10}")
    print("-" * 74)
    for name, ok, n, _ in rows:
        print(f"{name:<30}{ok:>8}{n:>8}{100.0 * ok / max(n, 1):>9.1f}%")
    print("-" * 74)

    def group_avg(kind):
        xs = [100.0 * ok / n for _, ok, n, k in rows if k == kind and n]
        return sum(xs) / len(xs) if xs else None

    mem, ho = group_avg("mem"), group_avg("holdout")
    if mem is not None:
        print(f"记忆分（训练见过的组 · 宏平均）  {mem:6.1f}%")
    if ho is not None:
        print(f"留出分（没训过的题 · 宏平均）    {ho:6.1f}%     <- 只认这个")
    if mem is not None and ho is not None:
        print(f"记忆 − 留出 的差距               {mem - ho:6.1f} 个点"
              f"（差距越小越说明是真会，而不是背得熟）")
        if ho > mem:
            print("  ⚠️ 留出分居然高于记忆分 —— 几乎一定是留出集被污染了：\n"
                  "     这个 checkpoint 训练时见过的语料里包含了现在这批留出集\n"
                  "     （典型情况：留出集是在它训完之后才切出来的）。\n"
                  "     这个留出分不可信；等一次从头训练（训练语料已排除留出集）再看。")

    if args.json:
        jp = args.json if os.path.isabs(args.json) else os.path.join(HERE, args.json)
        os.makedirs(os.path.dirname(os.path.abspath(jp)), exist_ok=True)
        payload = {
            "out_dir": args.out_dir,
            "vocab_size": tok.vocab_size,
            "params": model.num_params(),
            "seed": args.seed,
            "rows": [{"name": nm, "ok": ok, "n": n, "kind": k,
                      "acc": (ok / n if n else 0.0)} for nm, ok, n, k in rows],
            "memory_avg": mem,
            "holdout_avg": ho,
        }
        with open(jp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[json] 已写入 {jp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
