# ============================================================
# Yuki - Demo 对话脚本
# ------------------------------------------------------------
# 需要一份用「用户：/Yuki：」对话语料训练出来的 checkpoint
# （也就是 train.py 自动识别为 dialogue 模式、带 <|user|>/<|yuki|> 角色 token 的那种）
# 用法:
#   python chat.py                        # 交互多轮对话
#   python chat.py --once "今天好累啊"      # 单轮，方便脚本化测试
# 交互里输入 quit/exit 退出，reset 清空上下文
# ============================================================

import argparse

import numpy as np

from generate import load_model
from model import softmax
from tokenizer import BOS_ID, EOS_ID, USER_ID, YUKI_ID


def build_prompt_ids(tokenizer, history, user_text, max_seq_len):
    """把多轮历史 + 当前输入拼成 token 序列。

    从最近的一轮往前塞，塞不下就丢掉更早的轮次（Demo 简化版上下文管理）。
    """
    used = [USER_ID] + tokenizer.encode(user_text) + [YUKI_ID]
    for u, a in reversed(history):
        turn = [USER_ID] + tokenizer.encode(u) + [YUKI_ID] + tokenizer.encode(a)
        if len(turn) + len(used) + 1 > max_seq_len:     # +1 是开头的 <bos>
            break
        used = turn + used
    return [BOS_ID] + used


def generate_reply(model, tokenizer, user_text, history=None, max_new_tokens=None,
                   temperature=None, top_k=None, seed=None) -> str:
    """生成 Yuki 的回复：遇到 <eos> 或下一个 <|user|> 就停下来。"""
    cfg = model.cfg
    history = list(history or [])
    max_new_tokens = cfg.max_new_tokens if max_new_tokens is None else max_new_tokens
    temperature = cfg.temperature if temperature is None else temperature
    top_k = cfg.top_k if top_k is None else top_k
    rng = np.random.default_rng(seed)

    # 词表外的字符（例如 "~"）会被编码成 <unk>，而训练时模型从没见过 <unk>，
    # 一见到就会把回复带崩 —— 这里直接过滤掉，并提示一下
    known = tokenizer.stoi
    unknown = [ch for ch in user_text if ch not in known]
    if unknown:
        print(f"[提示] 已忽略词表外的字符: {''.join(unknown)!r}")
        user_text = "".join(ch for ch in user_text if ch in known)
    if not user_text:
        return ""

    ids = build_prompt_ids(tokenizer, history, user_text, cfg.max_seq_len)
    gen = []
    for _ in range(int(max_new_tokens)):
        ctx = np.asarray(ids[-cfg.max_seq_len:], dtype=np.int64)[None, :]
        logits, _ = model.forward(ctx)
        z = logits[0, -1].astype(np.float64)

        if temperature is None or temperature <= 0:
            nxt = int(np.argmax(z))
        else:
            z = z / float(temperature)
            if top_k and 0 < top_k < z.size:
                z = np.where(z < np.sort(z)[-int(top_k)], -np.inf, z)
            nxt = int(rng.choice(z.size, p=softmax(z)))

        if nxt in (EOS_ID, USER_ID):                    # 说完了，或者轮到用户了
            break
        ids.append(nxt)
        gen.append(nxt)

    return tokenizer.decode(gen)


def main():
    ap = argparse.ArgumentParser(description="Yuki - Demo 对话")
    ap.add_argument("--out-dir", type=str, default="checkpoints")
    ap.add_argument("--checkpoint", type=str, default="latest.npz")
    ap.add_argument("--once", type=str, default=None, help="只回答一句就退出")
    ap.add_argument("--temperature", type=float, default=None, help="<=0 表示贪心")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-turns", type=int, default=10, help="最多保留多少轮上下文")
    args = ap.parse_args()

    model, tokenizer, cfg = load_model(args.out_dir, args.checkpoint)

    if args.once is not None:
        reply = generate_reply(model, tokenizer, args.once,
                               max_new_tokens=args.max_new_tokens,
                               temperature=args.temperature,
                               top_k=args.top_k, seed=args.seed)
        print(reply)
        return

    print("=" * 52)
    print(" Yuki - Demo 对话模式")
    print(" quit / exit 退出, reset 清空上下文")
    print("=" * 52)

    history = []
    while True:
        try:
            user = input("\n清久酱: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见喵~")
            break

        if user.lower() in ("quit", "exit", "退出"):
            print("再见喵~")
            break
        if user.lower() == "reset":
            history = []
            print("(上下文已清空)")
            continue
        if not user:
            continue

        reply = generate_reply(model, tokenizer, user, history,
                               max_new_tokens=args.max_new_tokens,
                               temperature=args.temperature,
                               top_k=args.top_k, seed=args.seed)
        print(f"Yuki: {reply}")

        history.append((user, reply))
        if len(history) > args.max_turns:
            history = history[-args.max_turns:]


if __name__ == "__main__":
    main()
