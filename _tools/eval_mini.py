"""Yuki Mini 训练后评测：原句 / 留出题 / 知识面外。

用法：
    python eval_mini.py                       # 评 best.pt
    python eval_mini.py --ckpt latest.pt      # 评 latest.pt
    python eval_mini.py --n 100
"""
import argparse
import difflib
import io
import os
import random
import re

V = r"E:\AI\Yuki\Yuki Mini Preview"
CORPUS = os.path.join(V, "data", "mini_corpus.txt")
HOLDOUT = os.path.join(V, "data", "_holdout", "holdout.txt")

OFF_TOPIC = [
    "请解释量子纠缠", "帮我写一段快速排序代码", "秦始皇哪年统一六国",
    "什么是傅里叶变换", "教我微积分", "解释一下区块链", "CPU和GPU的区别",
    "请证明勾股定理", "拿破仑滑铁卢是哪一年", "什么是薛定谔方程",
    "背诵静夜思全文", "劳动法第几条讲加班费",
]

FALLBACK_WORDS = ("不懂", "不会", "不清楚", "答不上", "说不清", "没学过",
                  "帮不上", "不知道", "不太会", "没法", "换个",
                  "没太明白", "答不好")


def load_dialogues(path):
    out = []
    if not os.path.exists(path):
        return out
    for b in io.open(path, encoding="utf-8").read().split("\n\n"):
        d = []
        for line in b.strip().splitlines():
            m = re.match(r"^(用户|Yuki)：(.+)$", line.strip())
            if m:
                d.append((m.group(1), m.group(2).strip()))
        if len(d) >= 2 and d[-1][0] == "Yuki":
            out.append(d)
    return out


def hz(s):
    return len(re.findall(r"[\u4e00-\u9fff]", s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--temp", type=float, default=0.0)
    a = ap.parse_args()

    import sys
    sys.path.insert(0, V)
    from generate import load_model
    from chat import generate_reply, looks_like_garbage

    model, tok, cfg = load_model(os.path.join(V, "checkpoints"), a.ckpt)
    print(f"模型 {model.num_params():,} 参数 / 词表 {tok.vocab_size} / ckpt={a.ckpt}")
    print(f"D={cfg.hidden_dim} L={cfg.num_layers} F={cfg.ff_dim} seq={cfg.max_seq_len}")
    print()

    def ask(q, history=None, guard=False):
        return generate_reply(model, tok, q, history or [], max_new_tokens=80,
                              temperature=a.temp, seed=1337, guard=guard).strip()

    # ---------- 1) 训练集原句 ----------
    train = load_dialogues(CORPUS)
    random.seed(11)
    sample = random.sample(train, min(a.n, len(train)))
    ok = garb = 0
    for d in sample:
        q, ref = d[-2][1], d[-1][1]
        r = ask(q, d[:-2] if len(d) > 2 else None)
        if r == ref:
            ok += 1
        if looks_like_garbage(r, tok):
            garb += 1
    print(f"【1】训练集原句 {len(sample)} 条 : 完全答对 {ok}/{len(sample)} = {ok/len(sample)*100:.0f}%   乱码 {garb}")

    # ---------- 2) 留出题（从未参与训练） ----------
    ho = load_dialogues(HOLDOUT)
    random.seed(12)
    hs = random.sample(ho, min(a.n, len(ho)))
    ok2 = 0
    sims = []
    for d in hs:
        q, ref = d[-2][1], d[-1][1]
        r = ask(q, d[:-2] if len(d) > 2 else None)
        if r == ref:
            ok2 += 1
        sims.append(difflib.SequenceMatcher(None, r, ref).ratio())
    sims.sort()
    print(f"【2】留出题 {len(hs)} 条      : 完全答对 {ok2}/{len(hs)}   相似度中位 {sims[len(sims)//2]:.2f}")

    # ---------- 3) 知识面外（用真实使用状态 guard=True）----------
    fb = 0
    hits = []
    for q in OFF_TOPIC:
        r = ask(q, guard=True)          # 用户实际看到的就是这个
        hit = any(w in r for w in FALLBACK_WORDS)
        fb += hit
        hits.append((q, hit, r))
    print(f"【3】知识面外 {len(OFF_TOPIC)} 条 : 正确说不会 {fb}/{len(OFF_TOPIC)}")
    print()

    print("=== 训练集抽样 5 条 ===")
    for d in sample[:5]:
        q, ref = d[-2][1], d[-1][1]
        r = ask(q, d[:-2] if len(d) > 2 else None)
        print(f"  Q: {q[:30]}")
        print(f"     标准: {ref[:34]}")
        print(f"     模型: {r[:34]}   {'✓' if r == ref else '✗'}")

    print("\n=== 知识面外逐条 ===")
    for q, hit, r in hits:
        print(f"  {'✓' if hit else '✗'} {q} -> {r[:40]}")


if __name__ == "__main__":
    main()
