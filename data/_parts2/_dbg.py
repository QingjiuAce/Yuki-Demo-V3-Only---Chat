import re
base = r"D:\AI\mini-beta\mini-beta\Yuki - Demo\data"
p = base + r"\_parts2\chat_para.txt"
lines = open(p, encoding="utf-8").read().splitlines()
for rep in ["我睡醒啦", "我睡醒咯", "我一觉睡到天亮", "我醒来了", "我回来咯", "我回来了哦"]:
    hits = [i + 1 for i, l in enumerate(lines) if l == "用户：" + rep]
    print(rep, hits)

# 按组打印与这些组相关的问题
def group_of(line_no):
    return (line_no - 1) // 4 + 1
for i, l in enumerate(lines):
    if l.startswith("用户：") and ("睡" in l or "醒" in l or "回来" in l):
        print(group_of(i + 1), i + 1, l)
