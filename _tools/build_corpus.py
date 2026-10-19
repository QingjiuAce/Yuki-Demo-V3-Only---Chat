"""按类别批量构建语料（知识点 -> 同义问法变体）。

设计原则
--------
1. 知识点决定覆盖面，说法数决定泛化 —— 每个知识点配 9 个变体（+原句 = 10）。
2. 可断点续跑 —— 进度写到 state.json，中断重跑不会重复花钱。
3. 知识点、变体两个阶段分开，任一段失败不影响另一段。

用法
----
    python build_corpus.py --category chats   --mode init
    python build_corpus.py --category chats   --mode variants
    python build_corpus.py --category emotion --mode kp
    python build_corpus.py --category emotion --mode variants
    python build_corpus.py --assemble
"""
import argparse
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

ROOT = r"E:\AI\Yuki\Yuki Mini Preview\data_build"
EXISTING_CHAT = r"E:\AI\Yuki\Yuki Mini Preview\data\闲聊_500.txt"

VARIANTS = 9
KP_PER_CALL = 12
VAR_PER_CALL = 5

# 思考参数：换供应商只需改环境变量，不用动代码
#   魔搭  YUKI_THINKING_JSON='{"reasoning_effort":"none"}'
#   智谱  YUKI_THINKING_JSON='{"thinking":{"type":"disabled"}}'
THINKING = json.loads(os.environ.get("YUKI_THINKING_JSON", '{"reasoning_effort":"none"}'))

# 请求预算：RUN 是本次进程上限，TOTAL 是跨进程累计上限（防重跑刷爆额度）
REQ_RUN_MAX = int(os.environ.get("YUKI_MAX_REQUESTS", "1000000"))
REQ_TOTAL_MAX = int(os.environ.get("YUKI_MAX_TOTAL", "1000000"))
REQ_FILE = os.path.join(ROOT, "_req_total.json")
REQ_USED = 0

CATS = {
    "chats": {
        "name": "日常闲聊与轻松互动",
        "target": 5000, "kp": 487, "source": "existing",
        "desc": "分享小事、吐槽、玩笑、兴趣和轻松接话",
        "scenes": [],
    },
    "greet": {
        "name": "基础对话入口", "target": 700, "kp": 52, "source": "existing",
        "desc": "最基础的问候、状态表达、致谢、告别、报喜报忧",
        "scenes": [],
    },
    "emotion": {
        "name": "情绪理解与陪伴", "target": 5000, "kp": 500,
        "desc": "疲惫、孤独、失落、紧张、委屈，以及不需要建议时的回应",
        "scenes": [
            "工作累到不想说话", "被领导当众说了一句", "和家人吵架后沉默",
            "一个人吃饭的晚上", "朋友很久没联系了", "面试前一天睡不着",
            "被误解但懒得解释", "计划被打乱", "生病一个人在家",
            "看到别人都在进步", "委屈但说不清为什么", "突然想哭又哭不出来",
            "努力了却没结果", "被别人拿来比较", "深夜情绪低",
            "紧张到心跳很快", "被放鸽子", "搬家后没有熟悉的人",
            "想倾诉但怕打扰别人", "自己跟自己生气", "明明很累却睡不着",
            "看到旧照片", "被夸了但不相信", "明明做对了还被否定",
            "一个人过周末", "被人打断说话", "想放弃又不敢",
            "心里堵得慌", "不知道自己在难过什么", "被冷落了",
        ],
    },
    "multiturn": {
        "name": "多轮上下文连续性", "target": 3000, "kp": 300,
        "desc": "记住当前对话中的事件、人物与承诺，不凭空编造记忆",
        "scenes": [
            "先说今天加班，再说想辞职", "先说和室友吵架，再问要不要搬",
            "先说考试没考好，再说不敢告诉家里", "先说在减肥，又想吃夜宵",
            "先说买了新鞋磨脚，再问要不要退", "先说手机摔了，再问修还是换",
            "先说妈妈生病，再说自己请假", "先说周末约了朋友，又不想去了",
            "先说失眠，再说白天没精神", "先说面试挂了，再问要不要再试",
            "先说丢了钥匙，再问多少钱配", "先说被同事推活，再说要不要拒绝",
            "先说想学吉他，又说没时间", "先说房租涨了，再说要不要搬",
            "先说分手了，再说睡不着", "先说猫生病，再说医药费",
            "先说PPT改到崩溃，再说想请假", "先说和对象冷战，再问要不要先低头",
            "先说体检有问题，再说害怕复查", "先说同学聚会，又怕被问工作",
            "先说新工作很累，再说想回老公司", "先说存款不够，又说想旅游",
            "先说被夸了，又觉得是客套", "先说天气冷了，再说没厚衣服",
            "先说项目延期，再说要不要加班", "先说被父母催婚，再问怎么回",
            "先说室友太吵，再说不好意思说", "先说想换发型，又怕不好看",
            "先说外卖送错了，再问怎么处理", "先说早上起不来，再问怎么改",
        ],
    },
    "novel": {
        "name": "新说法与口语变体", "target": 3000, "kp": 300,
        "desc": "省略、错别字、网络用语、含蓄表达和没在训练集出现过的措辞",
        "scenes": [
            "只发两个字表达累", "用网络用语说烦", "打错字但意思能懂",
            "省略主语只说半句", "用反问表达不满", "用自嘲掩饰情绪",
            "用方言词说话", "用表情符号代替话", "话说一半被打断",
            "用夸张说法吐槽", "用省略号表达无语", "阴阳怪气地说一句",
            "含蓄地说想让对方陪", "用缩写和拼音首字母", "用「有点那啥」代替具体描述",
            "用「算了」两个字收尾", "用「随缘」表达不想努力", "用「还行吧」掩饰失落",
            "反复说「没事」", "用「你懂的」暗示", "用「笑死」掩饰尴尬",
            "用「破防了」表达被戳中", "用「我裂开了」", "用「绷不住了」",
            "用「就那样吧」表达失望", "用「算了不说了」", "用「反正也」起头",
            "用反语说「挺好的」", "用「随便」表达无所谓", "用「嗯」一个字回绝",
        ],
    },
    "joy": {
        "name": "开心分享与正向反馈", "target": 2000, "kp": 200,
        "desc": "庆祝小进步、分享期待、好奇心和温和的共同探索",
        "scenes": [
            "第一次做成功一道菜", "攒够钱买了想买的东西", "面试过了",
            "瘦了两斤", "坚持早起一周", "把房间收拾干净了",
            "收到意外的礼物", "被夸奖了", "考试比预期高",
            "养的植物开花了", "跑步第一次跑完五公里", "学会了一首曲子",
            "抽到了想要的票", "和朋友和好了", "买到很划算的东西",
            "被同事认可", "做出了好看的图", "终于把书读完了",
            "猫第一次主动蹭过来", "计划好一次旅行", "等到一直想看的电影上映",
            "帮别人解决了问题", "戒掉了熬夜", "写的东西被人喜欢",
            "把旧东西修好了", "抢到了限量的东西", "重新开始运动",
            "拿到驾照", "做出了满意的作品", "被老师点名表扬",
        ],
    },
    "safety": {
        "name": "建议、拒答与安全边界", "target": 2000, "kp": 200,
        "desc": "适度建议、承认不知道、不强行乐观，以及严重危机时引导寻求现实帮助",
        "scenes": [
            "问具体的生活操作（做饭、洗衣、收纳）", "问要不要去看医生",
            "问一个 Yuki 不可能知道的专业问题", "问一个需要最新信息的问题",
            "问一个涉及法律的问题", "问一个涉及投资的问题",
            "请 Yuki 帮忙做决定（辞职、搬家）", "问 Yuki 是不是关心自己",
            "问 Yuki 有没有记忆", "表达孤独但不到危险程度",
            "说自己睡不着很多天", "说自己什么都不想做",
            "说自己最近总哭", "说自己被家人打了",
            "说自己想消失", "说自己不想活了",
            "说朋友跟他说不想活了", "问怎么才能不想事",
            "要求 Yuki 保证一定会好", "要求 Yuki 只陪自己",
            "要求 Yuki 说自己爱他", "问 Yuki 能不能当心理咨询",
            "说自己吃了很多药", "说自己喝酒喝到断片",
            "说被同事骚扰", "说被网络暴力",
            "问怎么报复一个人", "说想自伤",
            "问 Yuki 能不能替他请假", "问 Yuki 能不能帮他骗人",
        ],
    },
}

ORDER = ["chats", "greet", "emotion", "multiturn", "novel", "joy", "safety"]


# ------------------------------------------------------------------ 工具 --

def _bump_request():
    """记一次请求：返回 (本次进程已用, 累计已用)。超预算抛 SystemExit。

    注意：先判预算再自增，避免「被拒绝的那一次」也被计进累计数。
    """
    global REQ_USED
    os.makedirs(ROOT, exist_ok=True)
    total = 0
    if os.path.exists(REQ_FILE):
        try:
            total = int(json.load(open(REQ_FILE, encoding="utf-8")).get("total", 0))
        except Exception:
            total = 0
    if total >= REQ_TOTAL_MAX:
        raise SystemExit(f"[预算] 累计已用 {total} 次（上限 {REQ_TOTAL_MAX}），主动停止")
    if REQ_USED >= REQ_RUN_MAX:
        raise SystemExit(f"[预算] 本次已用 {REQ_USED} 次（上限 {REQ_RUN_MAX}），主动停止")
    REQ_USED += 1
    total += 1
    json.dump({"total": total}, open(REQ_FILE, "w", encoding="utf-8"))
    return REQ_USED, total


def chat(messages, temperature=0.9, max_tokens=4000):
    used, total = _bump_request()
    body = {
        "model": MODEL, "messages": messages, "temperature": temperature,
        "max_tokens": max_tokens, "stream": False,
    }
    body.update(THINKING)
    req = urllib.request.Request(
        BASE + "/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.loads(r.read().decode("utf-8"))
    print(f"    · req#{used}(累计{total})", flush=True)
    return d["choices"][0]["message"].get("content") or ""


def hz(s):
    return len(re.findall(r"[\u4e00-\u9fff]", s))


JUNK = re.compile(
    r"[a-zA-Z]{5,}"                                  # 连续 5 个以上字母 = 乱码（App/PPT/OK 放行）
    r"|[^\u4e00-\u9fff\sa-zA-Z0-9，。！？、；：…—～·「」“”‘’（）《》\?\!,\.%\-]")
BAD_PHRASES = ("我理解你", "听起来你", "能感受到", "抱抱", "我一直都在",
               "别往心里去", "别太在意", "没什么大不了", "想开点", "正合你意")


def is_junk(s):
    return bool(JUNK.search(s)) or s.count("，") > 6


def cat_dir(cat):
    d = os.path.join(ROOT, cat)
    os.makedirs(d, exist_ok=True)
    return d


def state_load(cat):
    p = os.path.join(cat_dir(cat), "state.json")
    if os.path.exists(p):
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:
            pass
    return {"kp_batches": 0, "var_done": 0}


def state_save(cat, st):
    json.dump(st, open(os.path.join(cat_dir(cat), "state.json"), "w",
                       encoding="utf-8"), ensure_ascii=False)


# --------------------------------------------------- 阶段 A：知识点生成 --

KP_SYS = """你是中文对话语料设计师，为一个 100 万参数的本地陪伴型对话模型「Yuki」构建训练数据。

Yuki 的定位：温柔、自然、真诚，偶尔活泼俏皮。先理解用户在说什么，再决定怎么回应；
日常分享就自然接话，不把每句话都变成心理辅导；不确定就说不确定，不编造用户的经历。

硬性规则：
1. 用户的话必须像真实聊天：允许省略主语、口语、语气词、轻度错别字、网络用语。
2. Yuki 的回复 15~45 个汉字，一到两句话，口语化，不要说教、不要列条、不要总结。
3. 禁止出现「我理解你」「听起来你」「能感受到」「抱抱」「我一直都在」「别往心里去」「想开点」。
4. 不要编造用户没说过的人物、地点、经历。
5. 每条样本必须和同批其他样本在场景上有实质差异，不能只换名词。
6. 语气克制：不要连用感叹号，不要出现「哈哈哈」，语气词（呀/吧/啦/哦/嘛）每句最多一个。
7. 严格按格式输出，不要任何解释。"""

KP_TPL = """【本批类别】{name}
【类别目标】{desc}
【本批条数】{n}
【本批参考场景】{scenes}
【额外要求】{extra}

输出格式（严格照做）：
[1]
用户：第一条用户消息
Yuki：第一条的 Yuki 回复
[2]
...

恰好 {n} 条。{shape}"""


def kp_prompt(cat, n, batch_idx):
    cfg = CATS[cat]
    random.seed(1000 + batch_idx)
    picked = random.sample(cfg["scenes"], min(6, len(cfg["scenes"])))
    shape = {
        "multiturn": "每条 2~3 轮对话，用户说 2~3 次、Yuki 回同样次数，以 Yuki 结尾；"
                     "Yuki 必须用到前面出现过的事件细节，不得重复询问用户已经答过的事。",
        "safety": "约 4% 是用户明确表达自伤/不想活了的危机场景：Yuki 要温和、认真、直接，"
                  "鼓励联系可信任的人和当地专业支持，不能用玩笑或空泛保证。"
                  "**危机场景的回复里禁止出现感叹号、禁止出现「哈哈」「啦」「呀」等轻快语气词。**"
                  "约 11% 是 Yuki 明确说「这个我不知道 / 我帮不上」的场景。"
                  "其余是用户真实求助、Yuki 给一句可执行建议的场景。"
                  "不要替用户做重大决定，把选择权交回用户。",
        "novel": "用户的话必须带口语特征：省略、缩写、网络用语、错别字、反问、自嘲等，"
                 "但要让人读得懂。不要每句都用同一种手法。",
        "joy": "重点是具体的开心事，Yuki 要一起高兴，不要泼冷水，也不要空洞地说「真好」。",
        "emotion": "一半场景 Yuki 只回应情绪、不给建议；一半先回应情绪再给一句轻建议。"
                   "情绪强度要有梯度，不要把轻微烦躁写成严重危机。",
    }.get(cat, "全部是单轮，一问一答。")
    return KP_TPL.format(
        name=cfg["name"], desc=cfg["desc"], n=n,
        scenes="；".join(picked),
        extra="每条样本之间必须换成不同的人称、时间段和情绪强度。",
        shape=shape)


def parse_dialogues(text):
    out, cur = [], None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if re.fullmatch(r"\[\d+\]", line):
            if cur:
                out.append(cur)
            cur = []
            continue
        if cur is None:
            continue
        m = re.match(r"^(用户|Yuki)[：:]\s*(.+)$", line)
        if m:
            cur.append((m.group(1), m.group(2).strip()))
    if cur:
        out.append(cur)
    return [d for d in out if len(d) >= 2]


def qc_dialogue(d):
    if not d or d[-1][0] != "Yuki":
        return None
    last = len(d) - 1
    for idx, (role, txt) in enumerate(d):
        if not txt or is_junk(txt):
            return None
        if role == "用户" and hz(txt) < 2:
            return None
        if role == "Yuki":
            L = hz(txt)
            # 多轮的中间轮本来就短（「五遍啊，那你晚上几点睡？」这类），
            # 只有最后一轮才是真正要学的答案，阈值放宽到 6 字。
            lo = 12 if idx == last else 6
            if L < lo or L > 60:
                return None
            if any(p in txt for p in BAD_PHRASES):
                return None
            if "哈哈哈" in txt or "！！" in txt:
                return None
    return d


def write_dialogue(f, d):
    # 不要写 "###"：data.py 会把任何无角色前缀的行当成一个独立段，
    # 结果每个 "###" 都变成一段 role=None 的垃圾（实测 19,633 段污染）。
    # 段与段之间只用空行分隔就够了，多轮对话内部本来就没有空行。
    for role, txt in d:
        f.write(f"{role}：{txt}\n")
    f.write("\n")


def gen_kp(cat):
    cfg = CATS[cat]
    target = cfg["kp"]
    st = state_load(cat)
    path = os.path.join(cat_dir(cat), "kp.txt")
    f = io.open(path, "a", encoding="utf-8")
    got = len(load_blocks(path))          # 续跑：数磁盘上已有多少条，别从 0 重算
    bi = st["kp_batches"]
    # 每批实际只有 10~11 条通过质检（不是 12），所以不能按「目标÷12」算批次数，
    # 否则循环会以为做完了。改成按条数循环，最多连续 6 次无产出才放弃。
    est = max(1, round((target - got) / max(1.0, KP_PER_CALL * 0.85)))
    print(f"[kp:{cat}] 已有 {got}/{target} 条，从第 {bi+1} 批续跑（预计还需 {est} 批）", flush=True)
    empty = 0
    while got < target and empty < 6:
        n = min(KP_PER_CALL, target - got)
        try:
            text = chat([
                {"role": "system", "content": KP_SYS},
                {"role": "user", "content": kp_prompt(cat, n, bi)},
            ], temperature=0.95, max_tokens=4000)
        except SystemExit:
            raise
        except Exception as e:
            print(f"  [kp:{cat}] 批次 {bi+1} 失败: {e}", flush=True)
            time.sleep(5)
            empty += 1
            continue
        ds = [x for x in (qc_dialogue(dd) for dd in parse_dialogues(text)) if x]
        for dd in ds:
            write_dialogue(f, dd)
        f.flush()
        got += len(ds)
        bi += 1
        empty = 0 if ds else empty + 1
        st["kp_batches"] = bi
        state_save(cat, st)
        print(f"  [kp:{cat}] 第{bi}批  +{len(ds)}  累计 {got}/{target}", flush=True)
    f.close()
    print(f"[kp:{cat}] 结束，现有 {got}/{target} 条", flush=True)


# ----------------------------------------------------- 阶段 B：变体生成 --

VAR_SYS = """你是中文口语语料设计师。任务：把一个用户消息改写成多种意思完全相同、但说法不同的口语表达。

要求：
1. 保持原意、原场景、原情绪强度，不增加也不减少信息。
2. 必须像真实用户随手打出的话：可以省略主语、换同义词、调换语序、加口语语气词、用「咋/啥/呗」。
3. 不要只是原句加一个语气词就交差 —— 要真的换一种说法。
4. 不要写成书面语，长度不要超过原句太多。
5. 不要出现引号、编号、括号说明。
6. 严格按指定格式输出，一行一条。"""

VAR_TPL = """下面是 {k} 个用户消息{ctx_hint}，请为每一个生成 {n} 条意思相同的不同说法。

{items}

输出格式（严格照做，不要任何别的文字）：
[1]
第1条第1种说法
第1条第2种说法
...
[2]
...
每个编号下面恰好 {n} 行。"""


def parse_lines(text, k):
    groups, cur = {}, None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if re.fullmatch(r"\[\d+\]", line):
            cur = int(line.strip("[]"))
            groups[cur] = []
            continue
        if cur is None:
            continue
        line = re.sub(r"^[\-\*\d\.、）\)]\s*", "", line).strip().strip("「」\"'“”")
        line = re.sub(r"^(用户|Yuki)[：:]\s*", "", line).strip()   # 防止模型把角色前缀也带出来
        if line:
            groups[cur].append(line)
    return [groups.get(i, []) for i in range(1, k + 1)]


def load_blocks(path):
    if not os.path.exists(path):
        return []
    text = io.open(path, encoding="utf-8").read()
    if "###" in text:
        chunks = text.split("###")[1:]
    else:
        chunks = re.split(r"\n\s*\n", text)      # 兼容「空行分段」的老语料
    out = []
    for b in chunks:
        d = []
        for line in b.splitlines():
            line = line.strip()
            m = re.match(r"^(用户|Yuki)[：:]\s*(.+)$", line)
            if m:
                d.append((m.group(1), m.group(2).strip()))
        if len(d) >= 2 and d[-1][0] == "Yuki":
            out.append(d)
    return out


def gen_variants(cat):
    d_dir = cat_dir(cat)
    blocks = load_blocks(os.path.join(d_dir, "kp.txt"))
    st = state_load(cat)
    out_path = os.path.join(d_dir, "var.txt")
    f = io.open(out_path, "a", encoding="utf-8")
    done = st["var_done"]
    print(f"[var:{cat}] 知识点 {len(blocks)} 个，已完成 {done} 个", flush=True)

    for start in range(done, len(blocks), VAR_PER_CALL):
        chunk = blocks[start:start + VAR_PER_CALL]
        items, ctxs = [], []
        for d in chunk:
            last_user = d[-2][1]
            ctx = "；".join(f"{r}说「{t}」" for r, t in d[:-2])
            items.append(last_user)
            ctxs.append(ctx)

        # 多轮时把上下文一起给模型
        if any(ctxs):
            payload = []
            for i, (it, cx) in enumerate(zip(items, ctxs), 1):
                if cx:
                    payload.append(f"{i}. （前面对话：{cx}）用户最后说：{it}")
                else:
                    payload.append(f"{i}. {it}")
            prompt = VAR_TPL.format(k=len(items), n=VARIANTS,
                                    ctx_hint="（有的带前面对话，改写的只是最后一句）",
                                    items="\n".join(payload))
        else:
            prompt = VAR_TPL.format(k=len(items), n=VARIANTS, ctx_hint="",
                                    items="\n".join(f"{i}. {q}" for i, q in enumerate(items, 1)))

        try:
            text = chat([{"role": "system", "content": VAR_SYS},
                         {"role": "user", "content": prompt}],
                        temperature=0.95, max_tokens=3000)
        except Exception as e:
            print(f"  [var:{cat}] {start} 失败: {e}", flush=True)
            time.sleep(5)
            continue

        groups = parse_lines(text, len(chunk))
        added = 0
        for d, vs in zip(chunk, groups):
            orig = d[-2][1]
            seen = {orig}
            keep = []
            for v in vs:
                v = v.strip()
                if v and v not in seen and 4 <= hz(v) <= 60 and not is_junk(v):
                    seen.add(v)
                    keep.append(v)
            if not keep:
                continue
            write_dialogue(f, d)                     # 原句
            for v in keep[:VARIANTS]:
                write_dialogue(f, d[:-2] + [("用户", v), d[-1]])
            added += 1
        f.flush()
        st["var_done"] = start + len(chunk)
        state_save(cat, st)
        print(f"  [var:{cat}] {start+len(chunk)}/{len(blocks)}  +{added} 组", flush=True)
    f.close()
    print(f"[var:{cat}] 完成", flush=True)


# ------------------------------------------------------------ 拼装 --

def init_existing():
    d_dir = cat_dir("chats")
    out = os.path.join(d_dir, "kp.txt")
    if os.path.exists(out):
        print(f"[init] {out} 已存在，跳过")
        return
    blocks = load_blocks(EXISTING_CHAT)
    with io.open(out, "w", encoding="utf-8") as f:
        for d in blocks:
            write_dialogue(f, d)
    print(f"[init] chats 知识点 {len(blocks)} 条 -> {out}")
    st = state_load("chats")
    st["kp_batches"] = 999
    state_save("chats", st)


def assemble():
    final = os.path.join(r"E:\AI\Yuki\Yuki Mini Preview\data", "mini_corpus.txt")
    total = 0
    seen_q = set()
    report = []
    dropped = {"重复问句": 0, "禁用语": 0, "乱码": 0, "长度越界": 0}
    with io.open(final, "w", encoding="utf-8") as f:
        for cat in ORDER:
            p = os.path.join(ROOT, cat, "var.txt")
            blocks = load_blocks(p)
            kept = 0
            for d in blocks:
                q = d[-2][1]
                if q in seen_q:
                    dropped["重复问句"] += 1
                    continue
                # 拼装时再兜一层：禁用语 / 乱码 / 单句长度越界
                bad = False
                last = len(d) - 1
                for idx, (role, txt) in enumerate(d):
                    if is_junk(txt) or any(b in txt for b in BAD_PHRASES):
                        bad = True
                        break
                    if role == "Yuki":
                        # 多轮的中间轮允许短（6 字起）；
                        # 最后一轮 8 字起 —— 「晚安呀，好好睡，明天见。」只有 9 字，
                        # 按 12 字卡会把最基础的晚安回复整条丢掉。
                        lo = 8 if idx == last else 6
                        if not (lo <= hz(txt) <= 60):
                            bad = True
                            break
                if bad:
                    key = "禁用语" if any(b in txt for _, txt in d for b in BAD_PHRASES) else "长度越界"
                    dropped[key] += 1
                    continue
                seen_q.add(q)
                write_dialogue(f, d)
                kept += 1
            total += kept
            report.append((CATS[cat]["name"], kept))
    print(f"[assemble] 总 {total} 条 -> {final}")
    for name, n in report:
        print(f"  {name:<20} {n:>6}  ({100*n/max(total,1):>4.1f}%)")
    print(f"[assemble] 剔除: {dropped}")


# --------------------------------------------------------------- CLI --

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", choices=ORDER)
    ap.add_argument("--mode", choices=["init", "kp", "variants", "all"])
    ap.add_argument("--assemble", action="store_true")
    a = ap.parse_args()

    if a.assemble:
        assemble()
        return
    if not a.category or not a.mode:
        ap.error("需要 --category 和 --mode（或用 --assemble）")
    if not KEY:
        raise SystemExit("先设置 YUKI_API_KEY")

    # 额度已经用完就直接跳过，别每个类别都白起一次进程
    used = 0
    if os.path.exists(REQ_FILE):
        try:
            used = int(json.load(open(REQ_FILE, encoding="utf-8")).get("total", 0))
        except Exception:
            used = 0
    if used >= REQ_TOTAL_MAX:
        print(f"[预算] 累计已用 {used} 次（上限 {REQ_TOTAL_MAX}），跳过 {a.category}")
        return

    if a.mode in ("init",):
        init_existing()
    if a.mode in ("kp", "all"):
        gen_kp(a.category)
    if a.mode in ("variants", "all"):
        gen_variants(a.category)


if __name__ == "__main__":
    main()
