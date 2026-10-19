"""验伪测试：确认这个中转站的 DeepSeek-V4.1-Flash 是不是真的。

只发一个极小的请求，看响应结构（model 字段 / usage / 是否有 reasoning_content）。
"""

import json
import os
import urllib.request

BASE = "https://api.hcnsec.cn"
KEY = os.environ.get("YUKI_API_KEY", "")


def chat(model, messages, **kw):
    body = {"model": model, "messages": messages, **kw}
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {KEY}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode("utf-8"))


def show(tag, resp):
    print("=" * 66)
    print(f"  {tag}")
    print("=" * 66)
    print("  model 回显:", resp.get("model"))
    usage = resp.get("usage", {})
    print("  usage:", json.dumps(usage, ensure_ascii=False))
    msg = resp["choices"][0]["message"]
    print("  字段:", sorted(msg.keys()))
    content = msg.get("content") or ""
    print("  content 长度:", len(content))
    print("  content 前 200 字:")
    print("   ", content[:200].replace("\n", "\\n"))
    if "reasoning_content" in msg:
        rc = msg["reasoning_content"] or ""
        print(f"  reasoning_content 长度: {len(rc)}")
    print()


TESTS = [
    ("最小对话", "DeepSeek-V4.1-Flash",
     [{"role": "user", "content": "用一句话回答：1+1等于几"}],
     {"temperature": 0.3, "max_tokens": 100}),
    ("带 reasoning_effort=none", "DeepSeek-V4.1-Flash",
     [{"role": "user", "content": "写一句话描述下雨天"}],
     {"temperature": 0.8, "max_tokens": 100, "reasoning_effort": "none"}),
]

if not KEY:
    raise SystemExit("先设置环境变量 YUKI_API_KEY")

for tag, model, msgs, kw in TESTS:
    try:
        show(tag, chat(model, msgs, **kw))
    except Exception as e:
        print(f"[{tag}] 失败: {type(e).__name__}: {e}")
        try:
            print("  详情:", e.read().decode()[:500])
        except Exception:
            pass
        print()
