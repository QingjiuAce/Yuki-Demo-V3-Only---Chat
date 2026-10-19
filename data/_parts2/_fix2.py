import re
base = r"D:\AI\mini-beta\mini-beta\Yuki - Demo\data"
p = base + r"\_parts2\chat_para.txt"
text = open(p, encoding="utf-8").read()
lines = text.splitlines()
print("total lines before:", len(lines))

# 目标: 用户行文本 -> 新文本, 仅替换第一次出现
first_only = {
    "我回来咯": "我回来了哦",              # 149组#37 A: 等于源问法+corpus撞车
    "谢谢你一直陪着我": "陪着我这么久真好", # 等于源问法+corpus撞车
    "我是不是你的骄傲": "我算不算很厉害",   # corpus撞车
    "我困得眼睛都合上了": "我困得睁不开眼睛", # corpus撞车
    "你心里有我吗": "我心里有你吗",         # 内部重复(第82组)
    "晚上好呀Yuki": "天黑了，晚上好",       # 内部重复
    "我这就去睡": "我要去睡咯",             # 内部重复(第112组)
    "我肚子空了": "我饿得肚子疼",           # 内部重复
    "hello呀": "问个好呀",                  # 汉字数<2
    "我睡醒啦": "我睡醒咯",                 # 第124组 等于源问法(在#150用到)
}
# 逐行替换，维护 each 的 first-only 集合
done = set()
out = []
for l in lines:
    if l.startswith("用户："):
        q = l[3:]
        if q in first_only and q not in done:
            done.add(q)
            l = "用户：" + first_only[q]
    out.append(l)

# 逐行处理第二个 "我睡醒啦"? 已不存在（第一次已换）
new = "\n".join(out) + "\n"
open(p, "w", encoding="utf-8", newline="\n").write(new)
print("replaced first-only:", sorted(done))

# 再定位剩下的 "我醒啦"/"我醒来啦"/"我先去睡咯" 等冲突项
newlines = new.splitlines()
for i, l in enumerate(newlines):
    if l.startswith("用户：") and l[3:] in ("我醒啦", "我醒来啦", "我回来咯", "我睡醒啦", "我这就去睡"):
        print("still:", i + 1, l)
print("total lines after:", len(newlines))
