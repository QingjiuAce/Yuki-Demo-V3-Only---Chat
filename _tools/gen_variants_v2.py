"""变体生成 v2：加了「核心名词必须保留」约束。

用法：从已有的 pilot_all.txt 里读出 40 个原句，用新提示词重新生成变体，
其余流程（切分 / 训练 / 评测）与 v1 完全一致，保证两次可直接对比。
"""
import io
import json
import os
import re
import time
import urllib.request

BASE = os.environ.get("YUKI_API_BASE", "https://api-inference.modelscope.cn/v1")
KEY = os.environ.get("YUKI_API_KEY", "")
MODEL = os.environ.get("YUKI_MODEL", "deepseek-ai/DeepSeek-V4.1-Flash")
N_VARIANTS = int(os.environ.get("N_VARIANTS", "9"))
PER_CALL = int(os.environ.get("PER_CALL", "5"))

BASE_DIR = r"E:\AI\Yuki\Yuki Mini Preview\data_variants"
SRC = os.path.join(BASE_DIR, "pilot_all.txt")
OUT = os.path.join(BASE_DIR, "pilot_v2.txt")


def load_blocks(p):
    """读 pilot_all.txt，每个 ### 块取「原句 + 答案」"""
    blocks = io.open(p, encoding="utf-8").read().split("###")[1:]
    out = []
    for b in blocks:
        lines = [l.strip() for l in b.splitlines() if l.strip()]
        lines = [l for l in lines if not l.isdigit()]
        if len(lines) < 2:
            continue
        if lines[0].startswith("用户：") and lines[1].startswith("Yuki："):
            out.append((lines[0][3:], lines[1][5:]))
    return out


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
2. **必须保留原句里的核心名词**——就是能定位这个场景的那个词（例如「手机」「地铁」「闹钟」「食堂」「游戏」「瑜伽课」「外卖」「快递」）。可以换动词、换语序、加省略、换语气，但核心名词不能换掉、不能删掉、不能用别的词代替。
3. 每条变体必须像真实用户随手打出的话：可以省略主语、换用同义词、调换语序、加口语语气词、用「咋/啥/呗」这类口语词。
4. 不要只是原句加一个语气词就交差——要真的换一种说法。
5. 不要写成书面语，不要超过原句长度太多。
6. 不要出现引号、编号、括号说明。
7. 严格按指定格式输出，一行一条。"""

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
    return TPL.format(k=len(items), n=N_VARIANTS,
                      items="\n".join(f"{i}. {q}" for i, q in enumerate(items, 1)))


def parse(text, k):
    groups, cur = {}, None
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
        line = re.sub(r"^[\-\*\d\.、）\)]\s*", "", line).strip().strip("「」\"'“”")
        if line:
            groups[cur].append(line)
    return [groups.get(i, []) for i in range(1, k + 1)]


def key_nouns(q):
    """粗取核心名词候选：2~4 字的连续汉字块（去停用词）"""
    STOP = set("然后 现在 一下 一点 有点 真的 就是 还是 已经 一直 时候 感觉 觉得".split())
    toks = re.findall(r"[\u4e00-\u9fff]{2,4}", q)
    return [t for t in toks if t not in STOP]


def main():
    if not KEY:
        raise SystemExit("先设置 YUKI_API_KEY")

    picked = load_blocks(SRC)
    print(f"原句 {len(picked)} 个")

    result = []
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
        print(f"  批次 {start//PER_CALL+1}: {got}/{len(chunk)} 个，{int(time.time()-t0)}s")

    print(f"\n可用 {len(result)} 个知识点，共 {sum(1+len(v) for _,_,v in result)} 条")

    with io.open(OUT, "w", encoding="utf-8") as f:
        for qi, (q, a, vs) in enumerate(result):
            f.write(f"###{qi}\n")
            f.write(f"用户：{q}\nYuki：{a}\n\n")
            for v in vs:
                f.write(f"用户：{v}\nYuki：{a}\n\n")
    print(f"[落盘] {OUT}")

    # 名词保留率自检
    keep = tot = 0
    for q, a, vs in result:
        nouns = key_nouns(q)
        if not nouns:
            continue
        core = max(nouns, key=len)
        for v in vs:
            tot += 1
            if core in v:
                keep += 1
    print(f"\n[自检] 最长核心名词保留率: {keep}/{tot} = {100*keep/max(tot,1):.0f}%")


if __name__ == "__main__":
    main()
