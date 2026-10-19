import re, json

base = r"D:\AI\mini-beta\mini-beta\Yuki - Demo\data"
p = base + r"\_parts2\chat_para.txt"
lines = open(p, encoding="utf-8").read().splitlines()

repl = {
    "我回来咯": "我回来了哦",          # 1) 等于源问法 + corpus撞车
    "我睡醒啦": "我一觉睡到天亮",      # 1) 等于源问法 + corpus撞车 (#124 A)
    "谢谢你一直陪着我": "陪着我这么久真好",  # 1) 等于源问法 + corpus撞车
    "我是不是你的骄傲": "我算不算很厉害",     # corpus撞车
    "我困得眼睛都合上了": "我困得睁不开眼睛", # corpus撞车
    "你心里有我吗": "我心里有你吗",           # 2) 内部重复(第82组) - 只改第2处
    "晚上好呀Yuki": "天黑了，晚上好",         # 2) 内部重复
    "我这就去睡": "我要去睡咯",               # 2) 内部重复 + 与源"我准备睡了"相近
    "我肚子空了": "我饿得肚子疼",             # 2) 内部重复
    "hello呀": "问个好呀",                    # 3) 汉字数<2
}
# careful: "我睡醒啦" appears twice (group 124 A and group 150?) -> check
cnt = {}
for l in lines:
    if l.startswith("用户："):
        q = l[3:]
        cnt[q] = cnt.get(q, 0) + 1
print("occurrences:", {k: cnt.get(k) for k in repl})

out = []
seen = {}
for l in lines:
    if l.startswith("用户："):
        q = l[3:]
        if q in repl:
            seen[q] = seen.get(q, 0) + 1
            # 只改第一次出现，除非该词只出现一次
            if cnt[q] == 1 or seen[q] == 1:
                l = "用户：" + repl[q]
    out.append(l)

open(p, "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
print("written")
