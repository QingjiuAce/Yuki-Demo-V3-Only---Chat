# ============================================================
# Yuki - Demo 生成 / 推理
# 最基础的自回归采样：每步把整段上下文重新前向一次（暂无 KV-Cache）
# 支持 temperature 与 top-k；temperature <= 0 时退化为贪心解码
# ============================================================

import argparse
import json
import os

import numpy as np

from config import YukiConfig, resolve
from model import YukiModel, softmax
from tokenizer import BOS_ID, EOS_ID, CharTokenizer


def generate_text(model: YukiModel, tokenizer: CharTokenizer, prompt: str = "",
                  max_new_tokens: int = None, temperature: float = None,
                  top_k: int = None, seed: int = None) -> str:
    """从 prompt 续写，只返回新生成的那部分文本。"""
    cfg = model.cfg
    max_new_tokens = cfg.max_new_tokens if max_new_tokens is None else max_new_tokens
    temperature = cfg.temperature if temperature is None else temperature
    top_k = cfg.top_k if top_k is None else top_k
    rng = np.random.default_rng(seed)

    ids = tokenizer.encode(prompt)
    if not ids:
        ids = [BOS_ID]
    start = len(ids)

    for _ in range(int(max_new_tokens)):
        ctx = np.asarray(ids[-cfg.max_seq_len:], dtype=np.int64)[None, :]
        logits, _ = model.forward(ctx)
        z = logits[0, -1].astype(np.float64)

        if temperature is None or temperature <= 0:
            nxt = int(np.argmax(z))
        else:
            z = z / float(temperature)
            if top_k and 0 < top_k < z.size:
                thresh = np.sort(z)[-int(top_k)]
                z = np.where(z < thresh, -np.inf, z)
            probs = softmax(z, axis=-1)
            nxt = int(rng.choice(probs.size, p=probs))

        ids.append(nxt)
        if nxt == EOS_ID:
            break

    return tokenizer.decode(ids[start:])


def load_model(out_dir: str, checkpoint: str = "latest.npz", model_file: str = None):
    """从 checkpoint 目录加载模型与分词器。"""
    out_dir = resolve(out_dir)
    model_path = model_file if model_file else os.path.join(out_dir, checkpoint)
    meta_path = os.path.splitext(model_path)[0] + ".json"
    tok_path = os.path.join(out_dir, "tokenizer.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型文件: {model_path}（先跑 train.py 训练）")
    if not os.path.exists(tok_path):
        raise FileNotFoundError(f"找不到分词器: {tok_path}")

    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        cfg = YukiConfig.from_dict(meta.get("config", {}))
    else:
        cfg = YukiConfig()

    tokenizer = CharTokenizer.load(tok_path)
    cfg.vocab_size = tokenizer.vocab_size
    model = YukiModel.from_checkpoint(model_path, cfg)
    return model, tokenizer, cfg


def main():
    ap = argparse.ArgumentParser(description="Yuki - Demo 文本生成")
    ap.add_argument("--prompt", type=str, default="")
    ap.add_argument("--out-dir", type=str, default="checkpoints")
    ap.add_argument("--checkpoint", type=str, default="latest.npz", help="latest.npz / best.npz")
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None, help="<=0 表示贪心解码")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    model, tokenizer, cfg = load_model(args.out_dir, args.checkpoint)
    text = generate_text(model, tokenizer, args.prompt,
                         max_new_tokens=args.max_new_tokens,
                         temperature=args.temperature,
                         top_k=args.top_k,
                         seed=args.seed)

    print("=" * 50)
    print(f"提示词: {args.prompt!r}")
    print("=" * 50)
    print(args.prompt + text)
    print("=" * 50)


if __name__ == "__main__":
    main()
