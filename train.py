# ============================================================
# Yuki - Demo 训练脚本（numpy / CPU，最基础的训练算法）
# ------------------------------------------------------------
# 流程：读语料 -> 清洗 -> 字符分词 -> 切块 -> 前向 -> 交叉熵 -> 手写反向 -> Adam 更新
# 语料形态自动识别：带「用户：/Yuki：」角色前缀的按对话处理（带角色 token），
#                   否则按普通文本处理（每个非空行一篇文档）
# 只保留最基础的部分：没有预热 / 余弦退火 / 混合精度 / 梯度累积 / 阶梯训练
# ============================================================

import argparse
import json
import math
import os
import time

import numpy as np

from chat import generate_reply
from config import YukiConfig, resolve
from data import (batch_iterator, build_corpus, corpus_texts, encode_corpus,
                  epoch_batches, load_texts, make_chunks)
from generate import generate_text
from model import YukiModel
from tokenizer import CharTokenizer


# ============================================================
# 优化器：最基础的 Adam
# ============================================================

class Adam:
    def __init__(self, lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, beta1, beta2, eps, weight_decay
        self.m, self.v, self.t = {}, {}, 0

    def step(self, params: dict, grads: dict) -> None:
        self.t += 1
        b1, b2 = self.b1, self.b2
        c1 = 1.0 - b1 ** self.t
        c2 = 1.0 - b2 ** self.t
        for k, g in grads.items():
            if k not in self.m:
                self.m[k] = np.zeros_like(g)
                self.v[k] = np.zeros_like(g)
            m, v = self.m[k], self.v[k]
            m *= b1
            m += (1.0 - b1) * g
            v *= b2
            v += (1.0 - b2) * (g * g)
            update = (m / c1) / (np.sqrt(v / c2) + self.eps)
            if self.wd:
                update += self.wd * params[k]
            params[k] -= self.lr * update


def global_norm(grads: dict) -> float:
    return float(np.sqrt(sum(float(np.sum(g * g)) for g in grads.values())))


def clip_grads(grads: dict, max_norm: float) -> float:
    """按全局范数裁剪（max_norm <= 0 表示不裁剪），返回裁剪前的范数。"""
    total = global_norm(grads)
    if max_norm and max_norm > 0 and total > max_norm:
        s = max_norm / (total + 1e-12)
        for k in grads:
            grads[k] *= s
    return total


# ============================================================
# 存取 / 继承上一代权重
# ============================================================

def _sha256_file(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_meta(ckpt_path: str) -> dict:
    """读取与 .npz 同名的 .json 元信息；旧 checkpoint 没有就返回空 dict。"""
    meta_path = os.path.splitext(ckpt_path)[0] + ".json"
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_checkpoint(model: YukiModel, cfg: YukiConfig, out_dir: str, name: str,
                    step: int, loss: float, opt: "Adam" = None, rng=None,
                    vocab_sha: str = None) -> str:
    """保存 checkpoint。

    - 权重写进 <name>（.npz），同名的 .json 写元信息（step / loss / config / 分词器指纹）
    - 传入 opt 时，额外把优化器状态写进 <name 去掉后缀>.opt.npz，并记录 RNG 状态，
      这样下次可以用 --resume 真正接着练。只有 latest 需要，best 不写以省磁盘。
    """
    path = os.path.join(out_dir, name)
    model.save(path)

    meta = {"step": step, "loss": loss, "params": model.num_params(),
            "config": cfg.to_dict()}
    if vocab_sha:
        meta["vocab_sha256"] = vocab_sha
    if opt is not None:
        base = os.path.splitext(path)[0]
        state = {}
        for k, v in opt.m.items():
            state["m." + k] = v
        for k, v in opt.v.items():
            state["v." + k] = v
        np.savez_compressed(base + ".opt.npz", **state)
        meta["opt_t"] = opt.t
        if rng is not None:
            meta["rng_state"] = rng.bit_generator.state

    with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return path


def load_into(model: YukiModel, ckpt_path: str, allow_vocab_grow: bool = False) -> int:
    """把 checkpoint 的权重灌进模型；键名或形状不一致就报错（说清原因）。

    allow_vocab_grow=True 时允许 checkpoint 的词表比当前模型小（词表「只追加」的情况）：
    旧行原样复制到新矩阵的前若干行，新增行保持初始化值（lm_head 与它绑定，自动跟随）。
    返回新增的行数（0 表示没扩展）。
    """
    added = 0
    with np.load(ckpt_path) as data:
        missing = [k for k in model.params if k not in data.files]
        extra = [k for k in data.files if k not in model.params]
        if missing or extra:
            raise ValueError(
                "checkpoint 与当前模型结构不一致，无法继承权重：\n"
                f"  缺少 {missing[:3]}  多出 {extra[:3]}\n"
                "  模型结构（层数 / 宽度 / MLP 类型）变了就只能从零训练。")
        for k in model.params:
            old, new = data[k], model.params[k]
            if old.shape == new.shape:
                model.params[k] = old.astype(model.dtype)
            elif (allow_vocab_grow and k == "tok_emb" and old.ndim == 2
                  and new.ndim == 2 and old.shape[1] == new.shape[1]
                  and old.shape[0] < new.shape[0]):
                merged = new.copy()
                merged[:old.shape[0]] = old
                model.params[k] = merged
                added = int(new.shape[0] - old.shape[0])
            else:
                raise ValueError(
                    "checkpoint 与当前模型的形状不一致，无法继承权重：\n"
                    f"  例如 {k}: checkpoint {old.shape} vs 现在 {new.shape}\n"
                    "  结构变了就只能从零训练；"
                    "如果只是词表变大，请让脚本用同一份分词器的「只追加」模式。")
    return added


def restore_optimizer(opt: "Adam", ckpt_path: str, meta: dict, params: dict = None) -> bool:
    """恢复优化器状态。旧 checkpoint 没存过就返回 False（退化成热启动）。"""
    opt_path = os.path.splitext(ckpt_path)[0] + ".opt.npz"
    if not os.path.exists(opt_path) or "opt_t" not in meta:
        return False
    with np.load(opt_path) as data:
        for key in data.files:
            kind, _, pname = key.partition(".")
            target = opt.m if kind == "m" else (opt.v if kind == "v" else None)
            if target is None:
                continue
            arr = data[key]
            # 词表扩展过的话，嵌入对应的动量要补零到新形状（新字符从零动量开始）
            if params is not None and pname in params:
                want = params[pname].shape
                if arr.shape != want:
                    if (arr.ndim == len(want) and arr.shape[1:] == tuple(want[1:])
                            and arr.shape[0] < want[0]):
                        padded = np.zeros(want, dtype=arr.dtype)
                        padded[:arr.shape[0]] = arr
                        arr = padded
                    else:
                        raise ValueError(
                            f"优化器状态 {key} 形状 {arr.shape} 与模型 {tuple(want)} 不符")
            target[pname] = arr
    opt.t = int(meta["opt_t"])
    return True


def load_loss_history(out_dir: str):
    """读回已有的 loss 曲线，续训时接着往下记。"""
    hist_path = os.path.join(out_dir, "loss_history.csv")
    history = []
    if os.path.exists(hist_path):
        with open(hist_path, "r", encoding="utf-8") as f:
            for line in list(f)[1:]:
                s, _, l = line.strip().partition(",")
                try:
                    history.append((int(s), float(l)))
                except ValueError:
                    pass
    return history


def sample_text(model, tokenizer, mode: str, seed: int,
                prompt: str = "今天好累啊") -> str:
    """对话语料就抽一句回复看看，普通语料就纯续写。"""
    if mode == "dialogue":
        reply = generate_reply(model, tokenizer, prompt, max_new_tokens=40,
                               temperature=0.6, seed=seed)
        return f"用户: {prompt}  ->  Yuki: {reply}"
    return generate_text(model, tokenizer, "", max_new_tokens=80, seed=seed)


# ============================================================
# 主流程
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Yuki - Demo 训练（numpy / CPU）")
    ap.add_argument("--data-dir", type=str, default=None, help="语料目录（默认 data/）")
    ap.add_argument("--out-dir", type=str, default=None, help="输出目录（默认 checkpoints/）")
    ap.add_argument("--tokenizer-path", type=str, default=None)
    ap.add_argument("--steps", type=int, default=None, help="总训练步数")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--max-vocab", type=int, default=None, help="词表上限（默认 4200）")
    ap.add_argument("--log-every", type=int, default=None)
    ap.add_argument("--save-every", type=int, default=None)
    ap.add_argument("--sample-every", type=int, default=None, help="0 表示关闭样例生成")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--corpus-mode", choices=["auto", "dialogue", "plain"],
                    default="auto", help="语料形态：auto 自动检测（默认）")
    ap.add_argument("--keep-tokenizer", action="store_true",
                    help="复用已有的 tokenizer.json（默认每次按当前语料重建）")
    ap.add_argument("--keep-line-numbers", action="store_true",
                    help='保留行首序号（默认去掉，例如 "1 句子" -> "句子"）')
    ap.add_argument("--whole-file-docs", action="store_true",
                    help="整个文件作为一篇文档（默认每个非空行一篇）")
    ap.add_argument("--no-shuffle-docs", action="store_true",
                    help="关闭每轮重排文档顺序（默认开启；关闭后模型容易靠语料顺序作弊）")
    ap.add_argument("--init-from", type=str, default=None,
                    help="用已有 checkpoint 的权重做热启动/微调（优化器从零开始）")
    ap.add_argument("--resume", action="store_true",
                    help="从 out-dir/latest.npz 断点续训（连优化器状态、步数、loss 曲线一起恢复）")
    ap.add_argument("--resume-from", type=str, default=None,
                    help="指定要续训的 checkpoint（默认 out-dir/latest.npz）")
    args = ap.parse_args()

    # ---------- 配置 ----------
    cfg = YukiConfig()
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.seq_len: cfg.max_seq_len = args.seq_len
    if args.lr: cfg.learning_rate = args.lr
    if args.steps: cfg.max_steps = args.steps
    if args.max_vocab: cfg.vocab_size = args.max_vocab
    if args.log_every is not None: cfg.log_every = args.log_every
    if args.save_every is not None: cfg.save_every = args.save_every
    if args.sample_every is not None: cfg.sample_every = args.sample_every
    if args.seed is not None: cfg.seed = args.seed
    if args.data_dir: cfg.data_dir = args.data_dir
    if args.out_dir: cfg.out_dir = args.out_dir
    if args.no_shuffle_docs: cfg.shuffle_docs = False

    data_dir = resolve(cfg.data_dir)
    out_dir = resolve(cfg.out_dir)
    tok_path = resolve(args.tokenizer_path) if args.tokenizer_path else os.path.join(out_dir, "tokenizer.json")
    os.makedirs(out_dir, exist_ok=True)

    # ---------- 是否继承上一代权重 ----------
    init_ckpt, resume_mode, init_meta = None, False, {}
    if args.resume or args.resume_from:
        resume_mode = True
        init_ckpt = resolve(args.resume_from) if args.resume_from else os.path.join(out_dir, "latest.npz")
        if not os.path.exists(init_ckpt):
            raise FileNotFoundError(f"找不到要续训的 checkpoint: {init_ckpt}")
        init_meta = _load_meta(init_ckpt)
    elif args.init_from:
        init_ckpt = resolve(args.init_from)
        if not os.path.exists(init_ckpt):
            raise FileNotFoundError(f"找不到要热启动的 checkpoint: {init_ckpt}")
        init_meta = _load_meta(init_ckpt)

    if init_ckpt is not None:
        saved = init_meta.get("config", {})
        if not saved:
            print(f"[继承] 警告：{init_ckpt} 旁边没有 .json 元信息，无法校验结构与分词器")
        else:
            # 形状相关的结构参数跟随 checkpoint，避免莫名的形状报错
            for key in ("hidden_dim", "num_layers", "num_heads", "ff_dim"):
                if key in saved and getattr(cfg, key) != saved[key]:
                    print(f"[继承] 结构参数 {key}: {getattr(cfg, key)} -> {saved[key]}（跟随 checkpoint）")
                    setattr(cfg, key, saved[key])

    print("=" * 62)
    print(" Yuki - Demo 训练 (numpy / CPU)")
    print("=" * 62)

    # ---------- 语料 ----------
    files, texts = load_texts(data_dir)
    docs, dstats = build_corpus(
        texts,
        strip_line_numbers=cfg.strip_line_numbers and not args.keep_line_numbers,
        doc_per_line=cfg.doc_per_line and not args.whole_file_docs,
        mode=args.corpus_mode,
    )
    doc_strings = corpus_texts(docs)
    n_chars = sum(len(t) for t in texts)
    n_doc_chars = sum(len(s) for s in doc_strings)
    print(f"[语料] {len(files)} 个文件, {n_chars:,} 字符")
    for f in files:
        print(f"       - {os.path.basename(f)}")
    print(f"[清洗] 模式={dstats['mode']}, 文档 {dstats['docs']:,} 段 / "
          f"轮次 {dstats['turns']:,} / {dstats['lines']:,} 行, "
          f"去掉行首序号 {dstats['stripped']:,} 行, 有效字符 {n_doc_chars:,}")

    # ---------- 分词器 ----------
    vocab_sha, vocab_grew = None, 0
    if init_ckpt is not None:
        # 继承权重时必须用 checkpoint 配套的分词器。
        # 语料扩充了也**不重建**，而是「只追加」：旧字符 id 一个都不动，新字符加到末尾，
        # 这样旧嵌入行不会错位，加语料就能热启动/续训，而不必从头训练。
        candidates = [tok_path, os.path.join(os.path.dirname(init_ckpt), "tokenizer.json")]
        src_tok = next((p for p in candidates if os.path.exists(p)), None)
        if src_tok is None:
            raise FileNotFoundError(
                "继承权重时必须提供与 checkpoint 配套的 tokenizer.json\n"
                f"  依次找过: {candidates}\n"
                "  用 --tokenizer-path 指定，或把它放到 --out-dir 下")
        base_tok = CharTokenizer.load(src_tok)
        n_old = base_tok.vocab_size

        # checkpoint 自己记录的词表大小，必须与这份 tokenizer 对得上
        saved_v = init_meta.get("config", {}).get("vocab_size")
        if saved_v and saved_v != n_old:
            raise ValueError(
                f"checkpoint 记录词表 {saved_v}，但旁边的 tokenizer.json 是 {n_old} —— 两者不配套\n"
                "  请用 checkpoint 归档目录里的那份 tokenizer.json")

        tokenizer = CharTokenizer.build(doc_strings, max_vocab_size=cfg.vocab_size,
                                        base=base_tok)
        if tokenizer.itos[:n_old] != base_tok.itos:
            raise ValueError("词表扩展异常：旧字符的 id 发生了变化（这会破坏旧权重）")
        vocab_grew = tokenizer.vocab_size - n_old
        print(f"[分词器] 只追加扩展 {os.path.basename(src_tok)}: "
              f"{n_old} -> {tokenizer.vocab_size} (+{vocab_grew} 个新字符，旧 id 全部不变)")
        if getattr(tokenizer, "oov_candidates", 0):
            print(f"[分词器] 警告：有 {tokenizer.oov_candidates} 个新字符超出词表上限，"
                  f"会被当成 <unk>（可调大 --max-vocab）")
        tokenizer.save(tok_path)
        vocab_sha = _sha256_file(tok_path)

        # 兜底覆盖率检查
        known = set(tokenizer.itos)
        total_ch = sum(len(s) for s in doc_strings)
        oov_ch = sum(1 for s in doc_strings for ch in s if ch not in known)
        if oov_ch:
            print(f"[分词器] 警告：语料里仍有 {oov_ch:,}/{total_ch:,} 个字符不在词表内")
    elif args.keep_tokenizer and os.path.exists(tok_path):
        tokenizer = CharTokenizer.load(tok_path)
        vocab_sha = _sha256_file(tok_path)
        print(f"[分词器] 复用已有: {tok_path}")
    else:
        tokenizer = CharTokenizer.build(doc_strings, max_vocab_size=cfg.vocab_size)
        tokenizer.save(tok_path)
        vocab_sha = _sha256_file(tok_path)
        print(f"[分词器] 按当前语料训练并保存: {tok_path}")
    cfg.vocab_size = tokenizer.vocab_size
    print(f"[分词器] 词表大小: {cfg.vocab_size} "
          f"(语料唯一字符 {getattr(tokenizer, 'unique_chars', '?')})")

    # ---------- 数据 ----------
    ids = encode_corpus(tokenizer, docs)
    x, y = make_chunks(ids, cfg.max_seq_len)
    if len(x) < cfg.batch_size:
        print(f"[警告] 样本数 {len(x)} < batch_size {cfg.batch_size}，自动调整 batch_size")
        cfg.batch_size = max(1, len(x))
    print(f"[数据] token 总数 {len(ids):,} -> 训练样本 {len(x):,} 条 "
          f"(每条 {cfg.max_seq_len} token), batch_size={cfg.batch_size}")

    # ---------- 模型 ----------
    rng = np.random.default_rng(cfg.seed)
    model = YukiModel(cfg, rng)
    if init_ckpt is not None:
        added_rows = load_into(model, init_ckpt, allow_vocab_grow=True)
        print(f"[继承] {'断点续训' if resume_mode else '热启动'}：已载入权重 {init_ckpt}")
        if added_rows:
            print(f"[继承] 嵌入矩阵扩展 +{added_rows} 行"
                  f"（旧行原样保留，新字符的行随机初始化）")
        if resume_mode and "rng_state" in init_meta:
            try:
                rng.bit_generator.state = init_meta["rng_state"]
                print("[继承] 已恢复随机数状态")
            except Exception as e:
                print(f"[继承] 随机数状态恢复失败（忽略）: {e}")
    n_param = model.num_params()
    D, F, L, V = cfg.hidden_dim, cfg.ff_dim, cfg.num_layers, cfg.vocab_size
    fixed = L * (4 * D * D + cfg.mlp_mats * D * F + 2 * D) + D   # 与词表无关的结构参数
    print(f"[模型] D={D} L={L} H={cfg.num_heads} F={F} V={V} "
          f"head_dim={cfg.head_dim} mlp={cfg.mlp} tie_weights=True")
    print(f"[模型] 参数量: {n_param:,} ({n_param / 1e4:.1f}W) "
          f"= 词表部分 {V * D:,} + 结构部分 {fixed:,}")
    print(f"[模型] 词表 {cfg.vocab_size} 时约 {fixed + cfg.vocab_size * D:,} "
          f"({(fixed + cfg.vocab_size * D) / 1e4:.1f}W)")

    # ---------- 训练 ----------
    opt = Adam(cfg.learning_rate, cfg.beta1, cfg.beta2, cfg.adam_eps, cfg.weight_decay)
    if cfg.shuffle_docs and dstats["docs"] > 1:
        batches = epoch_batches(docs, tokenizer, cfg.max_seq_len, cfg.batch_size, rng)
        print("[数据] 每个 epoch 重排文档顺序（防止模型靠语料顺序作弊）")
    else:
        batches = batch_iterator(x, y, cfg.batch_size, rng)
    history, window = [], []
    best_loss, step = float("inf"), 0
    if init_ckpt is not None and resume_mode:
        history = load_loss_history(out_dir)
        step = int(init_meta.get("step", 0))
        if "loss" in init_meta:
            best_loss = float(init_meta["loss"])
        if restore_optimizer(opt, init_ckpt, init_meta, params=model.params):
            print(f"[继承] 已恢复优化器状态（Adam step={opt.t}，含一阶/二阶动量）")
        else:
            print("[继承] checkpoint 里没有优化器状态（旧格式）-> 退化成热启动")
        if history:
            print(f"[继承] 已接上 loss 曲线（{len(history)} 个记录点）")
        print(f"[继承] 从第 {step} 步继续 -> 目标 {cfg.max_steps} 步"
              f"（本次再练 {max(0, cfg.max_steps - step)} 步）")
        if step >= cfg.max_steps:
            print("[继承] 已经达到目标步数，不会继续训练；请把 --steps 调大")
    t0 = time.time()
    step_start = step

    print(f"[训练] 总步数 {cfg.max_steps}, lr={cfg.learning_rate}, "
          f"grad_clip={cfg.grad_clip}, 开始...")
    print("-" * 62)

    while step < cfg.max_steps:
        xb, yb = next(batches)
        logits, cache = model.forward(xb)
        loss, dlogits = YukiModel.loss_and_grad(logits, yb)
        grads = model.backward(dlogits, cache)
        gnorm = clip_grads(grads, cfg.grad_clip)
        opt.step(model.params, grads)
        step += 1
        window.append(loss)

        if step % cfg.log_every == 0 or step == 1:
            avg = sum(window) / len(window)
            window.clear()
            history.append((step, avg))
            elapsed = time.time() - t0
            print(f"step {step:>6}/{cfg.max_steps} | loss {avg:.4f} "
                  f"| ppl {math.exp(min(avg, 20.0)):8.1f} | |g| {gnorm:6.3f} "
                  f"| {elapsed:6.1f}s | {(step - step_start) / max(elapsed, 1e-9):5.1f} step/s",
                  flush=True)
            if avg < best_loss:
                best_loss = avg
                save_checkpoint(model, cfg, out_dir, "best.npz", step, avg,
                                vocab_sha=vocab_sha)

        if cfg.save_every and step % cfg.save_every == 0:
            save_checkpoint(model, cfg, out_dir, "latest.npz", step, loss,
                            opt=opt, rng=rng, vocab_sha=vocab_sha)

        if cfg.sample_every and step % cfg.sample_every == 0:
            print(f"  [样例 @{step}] {sample_text(model, tokenizer, dstats['mode'], step)}",
                  flush=True)

    # ---------- 收尾 ----------
    total = time.time() - t0
    save_checkpoint(model, cfg, out_dir, "latest.npz", step,
                    history[-1][1] if history else float("nan"),
                    opt=opt, rng=rng, vocab_sha=vocab_sha)

    with open(os.path.join(out_dir, "loss_history.csv"), "w", encoding="utf-8") as f:
        f.write("step,loss\n")
        for s, l in history:
            f.write(f"{s},{l:.6f}\n")

    print("-" * 62)
    print(f"[完成] 总步数 {step}, 用时 {total:.1f}s, 最佳 loss {best_loss:.4f}")
    print(f"[输出] {out_dir}")
    print(f"[最终样例] {sample_text(model, tokenizer, dstats['mode'], cfg.seed)}")


if __name__ == "__main__":
    main()
