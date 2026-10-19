import re
base = r"D:\AI\mini-beta\mini-beta\Yuki - Demo\data"
out = open(base + r"\_parts2\chat_para.txt", encoding="utf-8").read().splitlines()
corpus = open(base + r"\Demo_chat_v2.txt", encoding="utf-8-sig").read().splitlines()

newqs = [l[3:] for l in out if l.startswith("用户：")]
cq = [l[3:].strip() for l in corpus if l.startswith("用户：")]
cset = set(cq)

def han(s):
    return re.sub(r"[^\u4e00-\u9fff]", "", s)

# 近撞车: 新问法去掉标点后与某条corpus问法高度重叠(Jaccard>=0.7) 或互为子串
near = []
for q in newqs:
    h = han(q)
    for c in cq:
        hc = han(c)
        if not hc:
            continue
        if h == hc:
            near.append((q, c, "exact-nopunct"))
            break
        if (h in hc or hc in h) and min(len(h), len(hc)) >= 4:
            near.append((q, c, "substring"))
            break
        # bigram jaccard
        a = {h[i:i+2] for i in range(len(h)-1)}
        b = {hc[i:i+2] for i in range(len(hc)-1)}
        if a and b:
            j = len(a & b) / len(a | b)
            if j >= 0.6:
                near.append((q, c, "jaccard=%.2f" % j))
                break
print("near-collision count:", len(near))
for q, c, why in near:
    print(" ", why, "|", q, "||", c)

# 格式检查
print("---\n格式检查:")
print("总行数:", len(out), "(期望 449*3 + 3 = 1350? ->", 450*2 + 450 - 1, ")")
bad_colon = [l for l in out if l.strip() and not (l.startswith("用户：") or l.startswith("Yuki："))]
print("非法行:", bad_colon[:5])
# 空行结构: 每3行一组, 组间空行 => 期望 pattern
pattern_ok = True
idx = 0
groups = 0
while idx < len(out):
    if not out[idx].startswith("用户："): pattern_ok = False; print("bad at", idx+1, out[idx]); break
    if not out[idx+1].startswith("Yuki："): pattern_ok = False; print("bad at", idx+2); break
    idx += 2
    groups += 1
    if idx < len(out):
        if out[idx] != "": pattern_ok = False; print("missing blank at", idx+1); break
        idx += 1
print("结构合规:", pattern_ok, "组数:", groups)
print("末尾空行结尾:", out[-1] == "")
# 答案唯一集合与源对比
src = open(base + r"\_parts\chat_extra.txt", encoding="utf-8-sig").read().splitlines()
sa = [src[i+1][5:] for i in range(0, len(src)-1, 3) if src[i].startswith("用户：")]
answers = [out[i+1][5:] for i in range(0, len(out)-1, 2) if out[i].startswith("用户：")]
print("源答案数:", len(sa), "输出答案数:", len(answers))
print("答案= 源答案各3次依序:", answers == [a for a in sa for _ in range(3)])
