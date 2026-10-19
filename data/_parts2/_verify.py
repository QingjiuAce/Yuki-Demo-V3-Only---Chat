# -*- coding: utf-8 -*-
import re
base = r"D:\AI\mini-beta\mini-beta\Yuki - Demo\data"
p = base + r"\_parts2\chat_para.txt"
raw = open(p, "rb").read()
text = raw.decode("utf-8")
print("BOM:", raw[:3] == b"\xef\xbb\xbf", "| CRLF count:", raw.count(b"\r\n"), "| LF count:", raw.count(b"\n"))
print("ends with newline:", text.endswith("\n"))
lines = text.split("\n")
if lines and lines[-1] == "":
    lines = lines[:-1]  # 末尾换行产生的空元素
print("logical lines:", len(lines))
print("用户行:", sum(1 for l in lines if l.startswith("用户：")))
print("Yuki行:", sum(1 for l in lines if l.startswith("Yuki：")))
print("空行:", sum(1 for l in lines if l == ""))
print("半角冒号行:", [l[:12] for l in lines if ":" in l and re.match(r"^(用户|Yuki):", l)][:3])

# 结构
qi = [i for i, l in enumerate(lines) if l.startswith("用户：")]
ai = [i for i, l in enumerate(lines) if l.startswith("Yuki：")]
struct = all(b == a + 1 for a, b in zip(qi, ai)) and len(qi) == len(ai) == 450
print("每对相邻:", struct)
print("组间空行:", all(lines[qi[i] - 1] == "" for i in range(1, 450)))

src = open(base + r"\_parts\chat_extra.txt", encoding="utf-8-sig").read().split("\n")
sa = [l[5:] for l in src if l.startswith("Yuki：")]
oa = [l[5:] for l in lines if l.startswith("Yuki：")]
exp = [a for a in sa for _ in range(3)]
print("源答案:", len(sa), "输出答案:", len(oa))
print("答案逐字一致且每组重复3次依序:", oa == exp)
if oa != exp:
    for i, (x, y) in enumerate(zip(exp, oa)):
        if x != y:
            print("mismatch", i, repr(x), repr(y)); break

# 每组3个答案与源对应
print("第1对答案==源第1答案:", oa[0] == sa[0], "| 第450对答案==源第150答案:", oa[-1] == sa[-1])
print("抽查第2、75、150组:", oa[3:6] == sa[1:2] * 3, oa[222:225] == sa[74:75] * 3, oa[447:450] == sa[149:150] * 3)

# 问法不重复 & 不等于源问法
q = [l[3:] for l in lines if l.startswith("用户：")]
sq = set(l[3:] for l in src if l.startswith("用户："))
print("问法数:", len(q), "唯一:", len(set(q)), "不等于源问法:", not (set(q) & sq))
hl = sorted(set(len(re.findall(r"[\u4e00-\u9fff]", x)) for x in q))
print("汉字数范围:", hl[0], "-", hl[-1], "| 越界:", [x for x in q if not 2 <= len(re.findall(r"[\u4e00-\u9fff]", x)) <= 14])
