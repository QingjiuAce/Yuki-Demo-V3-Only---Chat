"""给指定的知识点生成同义问法变体（答案保持不变）。

用途：验证「同一知识点配多种问法」到底能不能带来泛化。
输出：data_variants/pilot.txt（每个知识点 10 条 = 原句 + 9 变体）
"""
import io
import json
import os
import random
import re
import sys
import time
import urllib.request

BASE = os.environ.get("YUKI_API_BASE", "https://api-inference.modelscope.cn/v1")
KEY = os.environ.get("YUKI_API_KEY", "")
MODEL = os.environ.get("YUKI_MODEL", "deepseek-ai/DeepSeek-V4.1-Flash")

N_POINTS = int(os.environ.get("N_POINTS", "40"))
N_VARIANTS = int(os.environ.get("N_VARIANTS", "9"))
PER_CALL = int(os.environ.get("PER_CALL", "5"))

SRC = r"E:\AI\Yuki\Yuki Mini Preview\data\闲聊_500.txt"
OUT_DIR = r"E:\AI\Yuki\Yuki Mini Preview\data_variants"


def load(p):
    segs, cur = [], []
    for line in io.open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
            if len(cur) == 2:
                segs.append(tuple(cur))
            cur = []
            continue
        if line.startswith(("用户：", "Yuki：")):
            cur.append(line.split("：", 1)[1])
    if len(cur) == 2:
        segs.append(tuple(cur))
    return segs


def hz(s):
    return len(re.findall(r"[\u4e00-\u9fff]", s))


def chat(messages, temperature=0.9, max_tokens=3000):
    body = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "reasoning_effort": "none",
        "stream": False,
    }
    req = urllib.request.Request(
        BASE + "/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read().decode("utf-8"))
    return d["choices"][0]["message"].get("content") or ""


SYS = """你是中文口语语料设计师。任务：把一个用户消息改写成多种**意思完全相同、但说法不同**的口语表达。

要求：
1. 保持原来的意思、场景和情绪强度，不增加也不减少信息。
2. 每条变体必须像真实用户随手打出的话：可以省略主语、换用同义词、调换语序、加口语语气词、用「咋/啥/呗」这类口语词。
3. 不要只是原句加一个语气词就交差——要真的换一种说法。
4. 不要写成书面语，不要超过原句长度太多。
5. 不要出现引号、编号、括号说明。
6. 严格按指定格式输出，一行一条。"""

TPL = """下面是 {k} 个用户消息，请为每一个生成 {n} 条意思相同的不同说法。

{items}

输出格式（严格照做，不要任何别的文字）：
[1]
第一条第1种说法
第一条第2种说法
...
[2]
第二条第1种说法
...

每个编号下面恰好 {n} 行。"""


def build_prompt(items):
    lines = []
    for i, q in enumerate(items, 1):
        lines.append(f"{i}. {q}")
    return TPL.format(k=len(items), n=N_VARIANTS, items="\n".join(lines))


def parse(text, k):
    """按 [n] 分组解析，返回 list[list[str]]"""
    groups = {}
    cur = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.fullmatch(r"\[(\d+)\]", line)
        if m:
            cur = int(m.group(1))
            groups[cur] = []
            continue
        if cur is None:
            continue
        line = re.sub(r"^[\-\*\d\.、）\)]\s*", "", line)
        line = line.strip().strip("「」\"'“”")
        if line:
            groups[cur].append(line)
    return [groups.get(i, []) for i in range(1, k + 1)]


def main():
    if not KEY:
        raise SystemExit("先设置 YUKI_API_KEY")

    allp = load(SRC)
    # 挑条件合适的知识点：问句 8~30 汉字、答案 18~45 汉字
    cand = [(q, a) for q, a in allp if 8 <= hz(q) <= 30 and 18 <= hz(a) <= 45]
    print(f"候选知识点 {len(cand)} 个（总 {len(allp)}）")
    random.seed(20261019)
    picked = random.sample(cand, min(N_POINTS, len(cand)))
    print(f"抽取 {len(picked)} 个\n")

    os.makedirs(OUT_DIR, exist_ok=True)
    result = []       # (q_orig, answer, [variants])
    for start in range(0, len(picked), PER_CALL):
        chunk = picked[start:start + PER_CALL]
        t0 = time.time()
        try:
            text = chat([
                {"role": "system", "content": SYS},
                {"role": "user", "content": build_prompt([q for q, _ in chunk])},
            ])
        except Exception as e:
            print(f"  批次 {start//PER_CALL+1} 失败: {e}")
            continue
        groups = parse(text, len(chunk))
        got = 0
        for (q, a), vs in zip(chunk, groups):
            vs = [v for v in vs if v != q and 4 <= hz(v) <= 60]
            seen, uniq = set(), []
            for v in vs:
                if v not in seen:
                    seen.add(v)
                    uniq.append(v)
            if len(uniq) >= 5:
                result.append((q, a, uniq[:N_VARIANTS]))
                got += 1
        print(f"  批次 {start//PER_CALL+1}: 拿到 {got}/{len(chunk)} 个知识点，"
              f"{int(time.time()-t0)}s")

    print(f"\n可用知识点 {len(result)} 个")
    total = sum(1 + len(v) for _, _, v in result)
    print(f"总条数 {total}（原句 + 变体）")

    # 落盘：每个知识点 1 原句 + N 变体，用 ### 分隔便于后面切分
    p = os.path.join(OUT_DIR, "pilot_all.txt")
    with io.open(p, "w", encoding="utf-8") as f:
        for qi, (q, a, vs) in enumerate(result):
            f.write(f"###{qi}\n")
            f.write(f"用户：{q}\nYuki：{a}\n\n")
            for v in vs:
                f.write(f"用户：{v}\nYuki：{a}\n\n")
    print(f"[落盘] {p}")

    with io.open(os.path.join(OUT_DIR, "pilot_meta.tsv"), "w", encoding="utf-8") as f:
        for qi, (q, a, vs) in enumerate(result):
            f.write(f"{qi}\t{len(vs)}\t{q}\t{a}\n")
    print("[落盘] pilot_meta.tsv")


if __name__ == "__main__":
    main()
