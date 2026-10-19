# -*- coding: utf-8 -*-
base = r"D:\AI\mini-beta\mini-beta\Yuki - Demo\data"
p = base + r"\_parts2\chat_para.txt"
lines = open(p, encoding="utf-8").read().splitlines()

# (旧问法, 新问法) —— 每个只改"第二次出现"(first-only 会在第一次就生效,
# 因此这里先跳过第一次: 用计数器跳过第 1 次)
skip_first = {"我下楼买菜", "我上班去咯", "我一下子很想你"}
repl = {
    "我下楼买菜": "我去市场买菜",
    "我上班去咯": "我出门去上班",
    "我一下子很想你": "我忽然特别想你",
}
cnt = {}
out = []
for l in lines:
    if l.startswith("用户："):
        q = l[3:]
        if q in repl:
            cnt[q] = cnt.get(q, 0) + 1
            if cnt[q] == 2:  # 只改第二处
                l = "用户：" + repl[q]
    out.append(l)
open(p, "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
print("counts:", cnt)
