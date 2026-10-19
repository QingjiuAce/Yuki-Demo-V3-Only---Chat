"""把 gen_final/ 下所有批次合并成一份语料，并做跨批去重与统计。"""

import glob
import os
import re
from collections import Counter

GEN = r"E:\AI\Yuki\Yuki Mini Preview\data\gen_final"
OUT = r"E:\AI\Yuki\Yuki Mini Preview\data\闲聊_500.txt"

USER_RE = re.compile(r"^用户：(.+)$")
YUKI_RE = re.compile(r"^Yuki：(.+)$")


def norm(s):
    """去掉标点空格，用于近似判重。"""
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", s)


def parse(path):
    segs, cur = [], []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                if len(cur) == 2:
                    segs.append(tuple(cur))
                cur = []
                continue
            mu, my = USER_RE.match(line), YUKI_RE.match(line)
            if mu:
                cur.append(mu.group(1).strip())
            elif my:
                cur.append(my.group(1).strip())
    if len(cur) == 2:
        segs.append(tuple(cur))
    return segs


def main():
    files = sorted(p for p in glob.glob(os.path.join(GEN, "*.txt"))
                   if not p.endswith(".raw.txt"))
    print(f"[读取] {len(files)} 个批次文件")

    all_pairs = []
    for p in files:
        ps = parse(p)
        all_pairs += ps
        print(f"  {os.path.basename(p):<24} {len(ps):>3} 条")

    print(f"\n[合计] 合并后 {len(all_pairs)} 条")

    # 精确去重 + 近似去重（去标点后相同）
    seen_exact, seen_norm = set(), set()
    kept, dup_exact, dup_norm = [], 0, 0
    for u, y in all_pairs:
        key = (u, y)
        if key in seen_exact:
            dup_exact += 1
            continue
        seen_exact.add(key)
        nk = norm(u)
        if nk in seen_norm:
            dup_norm += 1
            continue
        seen_norm.add(nk)
        kept.append((u, y))

    print(f"[去重] 完全重复 {dup_exact} 条，近似重复（去标点相同）{dup_norm} 条")
    print(f"[保留] {len(kept)} 条")

    with open(OUT, "w", encoding="utf-8") as f:
        for u, y in kept:
            f.write(f"用户：{u}\nYuki：{y}\n\n")
    print(f"[落盘] {OUT}")

    # 统计
    hz = lambda s: len(re.sub(r"[^\u4e00-\u9fff]", "", s))
    yl = [hz(y) for _, y in kept]
    ul = [hz(u) for u, _ in kept]
    starts = Counter(y[:2] for _, y in kept)

    print("\n=== 长度分布 ===")
    print(f"  Yuki  : 最短 {min(yl)}  中位 {sorted(yl)[len(yl)//2]}  最长 {max(yl)}  均 {sum(yl)/len(yl):.1f}")
    print(f"  用户  : 最短 {min(ul)}  中位 {sorted(ul)[len(ul)//2]}  最长 {max(ul)}  均 {sum(ul)/len(ul):.1f}")
    print(f"  Yuki >=35 字: {sum(1 for x in yl if x >= 35)} 条")
    print(f"  用户 >25 字 : {sum(1 for x in ul if x > 25)} 条")

    print("\n=== 开头两字 top10 ===")
    for k, v in starts.most_common(10):
        print(f"  {k}: {v}")

    bad = [(k, v) for k, v in starts.items() if v > 1]
    print(f"\n重复开头: {len(bad)} 种" + (f" -> {bad[:6]}" if bad else ""))

    # 语气词
    part = {p: sum(1 for _, y in kept if p in y) for p in "呗吧呀呢"}
    print(f"语气词分布: {part}")


if __name__ == "__main__":
    main()
