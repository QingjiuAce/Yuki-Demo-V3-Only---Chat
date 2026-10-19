"""对 data_build 下各类别的 var.txt 做质检，并统计最终语料分布。

用法：python qc_corpus.py [--src 目录]
"""
import argparse
import collections
import io
import os
import re

ROOT = r"E:\AI\Yuki\Yuki Mini Preview\data_build"
ORDER = ["chats", "greet", "emotion", "multiturn", "novel", "joy", "safety"]
NAMES = {
    "chats": "日常闲聊与轻松互动", "greet": "基础对话入口", "emotion": "情绪理解与陪伴",
    "multiturn": "多轮上下文连续性", "novel": "新说法与口语变体",
    "joy": "开心分享与正向反馈", "safety": "建议、拒答与安全边界",
}
TARGET_PCT = {"chats": 25, "greet": 0, "emotion": 25, "multiturn": 15,
              "novel": 15, "joy": 10, "safety": 10}

JUNK = re.compile(
    r"[a-zA-Z]{5,}|[^\u4e00-\u9fff\sa-zA-Z0-9，。！？、；：…—～·「」“”‘’（）《》\?\!,\.%\-]")
BAD = ("我理解你", "听起来你", "能感受到", "抱抱", "我一直都在",
       "别往心里去", "别太在意", "没什么大不了", "想开点", "正合你意")


def hz(s):
    return len(re.findall(r"[\u4e00-\u9fff]", s))


def load(path):
    if not os.path.exists(path):
        return []
    out = []
    for b in io.open(path, encoding="utf-8").read().split("###")[1:]:
        d = []
        for line in b.splitlines():
            m = re.match(r"^(用户|Yuki)[：:]\s*(.+)$", line.strip())
            if m:
                d.append((m.group(1), m.group(2).strip()))
        if len(d) >= 2 and d[-1][0] == "Yuki":
            out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=ROOT)
    a = ap.parse_args()

    total = 0
    per = {}
    all_q = []
    print("=" * 74)
    print(f"{'类别':<20}{'条数':>8}{'占比':>8}{'目标':>7}{'偏差':>8}{'知识点':>8}")
    print("=" * 74)
    for cat in ORDER:
        ds = load(os.path.join(a.src, cat, "var.txt"))
        per[cat] = ds
        total += len(ds)
        qs = [d[-2][1] for d in ds]
        all_q += qs

    for cat in ORDER:
        n = len(per[cat])
        pct = 100 * n / max(total, 1)
        tgt = TARGET_PCT[cat]
        print(f"{NAMES[cat]:<20}{n:>8}{pct:>7.1f}%{tgt:>6}%{pct-tgt:>+8.1f}"
              f"{len(per[cat]):>8}")
    print("-" * 74)
    print(f"{'合计':<20}{total:>8}{100.0:>7.1f}%")

    dup = len(all_q) - len(set(all_q))
    print(f"\n唯一问句 {len(set(all_q))} / {len(all_q)}    重复 {dup} 条")

    # 质量指标
    bad_chars = bad_phrase = too_short = too_long = 0
    ylen, ulen, multi = [], [], 0
    for cat in ORDER:
        for d in per[cat]:
            if len(d) > 2:
                multi += 1
            for role, t in d:
                if JUNK.search(t):
                    bad_chars += 1
                if any(p in t for p in BAD):
                    bad_phrase += 1
                if role == "Yuki":
                    L = hz(t)
                    ylen.append(L)
                    if L < 12:
                        too_short += 1
                    if L > 60:
                        too_long += 1
                else:
                    ulen.append(hz(t))

    ylen.sort(); ulen.sort()

    def pct(v):
        return f"中位 {v[len(v)//2]}  最短 {v[0]}  最长 {v[-1]}"

    print(f"\nYuki 回复长度 : {pct(ylen)}   过短(<12) {too_short}  过长(>60) {too_long}")
    print(f"用户消息长度  : {pct(ulen)}")
    print(f"多轮样本      : {multi} 条 ({100*multi/max(total,1):.1f}%)")
    print(f"异常字符      : {bad_chars} 条")
    print(f"禁用语        : {bad_phrase} 条")


if __name__ == "__main__":
    main()
