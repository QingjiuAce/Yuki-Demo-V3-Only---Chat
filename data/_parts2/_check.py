import re, io, sys

base = r"D:\AI\mini-beta\mini-beta\Yuki - Demo\data"
src = open(base + r"\_parts\chat_extra.txt", encoding="utf-8-sig").read().splitlines()
out = open(base + r"\_parts2\chat_para.txt", encoding="utf-8").read().splitlines()

def pairs(lines):
    ps = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("用户："):
            ps.append((lines[i][3:], lines[i+1][5:]))
            i += 2
        else:
            i += 1
    return ps

sp = pairs(src)
op = pairs(out)

print("src pairs:", len(sp))
print("out pairs:", len(op))
print("out 用户 lines:", sum(1 for l in out if l.startswith("用户：")))
print("out Yuki lines:", sum(1 for l in out if l.startswith("Yuki：")))

# answer order check
expected = []
for q, a in sp:
    expected += [a, a, a]
actual = [a for q, a in op]
ok = expected == actual
print("answers逐字一致(顺序+重复3次):", ok)
if not ok:
    for i, (e, a) in enumerate(zip(expected, actual)):
        if e != a:
            print("first mismatch idx", i, repr(e), repr(a)); break

# uniqueness of questions
qs = [q for q, a in op]
print("questions:", len(qs), "unique:", len(set(qs)))
dups = [q for q in set(qs) if qs.count(q) > 1]
print("dups:", dups)

# equals source questions?
sq = set(q for q, a in sp)
same = sorted(set(qs) & sq)
print("等于源问法的:", same)

# hanzi length check
bad = [(q, len(re.findall(r"[\u4e00-\u9fff]", q))) for q in qs if not (2 <= len(re.findall(r"[\u4e00-\u9fff]", q)) <= 14)]
print("汉字数越界:", bad)

# corpus collision
corpus_path = base + r"\Demo_chat_v2.txt"
corpus_lines = open(corpus_path, encoding="utf-8-sig").read().splitlines()
corpus_q = set()
for l in corpus_lines:
    if l.startswith("用户："):
        corpus_q.add(l[3:].strip())
print("corpus 用户 lines:", sum(1 for l in corpus_lines if l.startswith("用户：")))
collide = sorted(set(qs) & corpus_q)
print("与 corpus 撞车数:", len(collide))
print("撞车列表:", collide)
