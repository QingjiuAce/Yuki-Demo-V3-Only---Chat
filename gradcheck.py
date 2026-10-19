# ============================================================
# Yuki - Demo 数值梯度校验
# ------------------------------------------------------------
# 手写反向传播最容易出错，这里用中心差分（数值梯度）逐个参数和
# 解析梯度对比。校验通过就说明前向/反向的数学推导是对的。
# 用 float64 跑；判据：绝对误差 <= atol 或 相对误差 <= rtol
# （梯度本身很小时，中心差分噪声会放大相对误差，所以两个判据取其一）
# 用法: python gradcheck.py
# ============================================================

import argparse
import sys

import numpy as np

from config import YukiConfig
from model import YukiModel


def loss_fn(model, ids, targets) -> float:
    logits, _ = model.forward(ids)
    loss, _ = YukiModel.loss_and_grad(logits, targets)
    return loss


def main():
    ap = argparse.ArgumentParser(description="Yuki - Demo 梯度校验")
    ap.add_argument("--per-key", type=int, default=8, help="每个参数张量抽查多少个元素")
    ap.add_argument("--eps", type=float, default=1e-5, help="中心差分步长")
    ap.add_argument("--atol", type=float, default=1e-7, help="绝对误差容差")
    ap.add_argument("--rtol", type=float, default=1e-5, help="相对误差容差")
    ap.add_argument("--mlp", choices=["gelu", "swiglu"], default=None,
                    help="指定 MLP 类型；默认跟随 config.py")
    args = ap.parse_args()

    cfg = YukiConfig()
    if args.mlp:
        cfg.mlp = args.mlp
    cfg.vocab_size = 32
    cfg.hidden_dim = 16
    cfg.num_layers = 2
    cfg.num_heads = 2
    cfg.ff_dim = 32
    cfg.max_seq_len = 8

    rng = np.random.default_rng(0)
    model = YukiModel(cfg, np.random.default_rng(1), dtype=np.float64)

    B, T = 2, cfg.max_seq_len
    ids = rng.integers(0, cfg.vocab_size, (B, T))
    targets = rng.integers(0, cfg.vocab_size, (B, T))

    logits, cache = model.forward(ids)
    loss, dlogits = YukiModel.loss_and_grad(logits, targets)
    grads = model.backward(dlogits, cache)

    print("=" * 70)
    print(" Yuki - Demo 数值梯度校验 (float64, 中心差分)")
    print("=" * 70)
    print(f" 配置: V={cfg.vocab_size} D={cfg.hidden_dim} L={cfg.num_layers} "
          f"H={cfg.num_heads} F={cfg.ff_dim} B={B} T={T}")
    print(f" loss={loss:.10f}  eps={args.eps:g}  atol={args.atol:g}  rtol={args.rtol:g}")
    print("-" * 70)

    missing = set(model.params) ^ set(grads)
    assert not missing, f"参数与梯度不匹配: {missing}"

    worst_key, worst_score = None, -1.0
    max_abs_all, max_rel_all = 0.0, 0.0
    n_checked, n_bad = 0, 0

    for key in sorted(model.params):
        p = model.params[key]
        flat = p.reshape(-1)
        n = min(args.per_key, flat.size)
        idxs = rng.choice(flat.size, size=n, replace=False)
        key_abs, key_rel, key_bad = 0.0, 0.0, 0
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + args.eps
            lp = loss_fn(model, ids, targets)
            flat[i] = orig - args.eps
            lm = loss_fn(model, ids, targets)
            flat[i] = orig
            num = (lp - lm) / (2.0 * args.eps)
            ana = float(grads[key].reshape(-1)[i])
            abs_err = abs(num - ana)
            rel_err = abs_err / max(1e-12, abs(num) + abs(ana))
            key_abs = max(key_abs, abs_err)
            key_rel = max(key_rel, rel_err)
            n_checked += 1
            if not (abs_err <= args.atol or rel_err <= args.rtol):
                key_bad += 1
                n_bad += 1
        flag = "OK " if key_bad == 0 else "BAD"
        print(f"  [{flag}] {key:<18} shape={str(tuple(p.shape)):<12} "
              f"abs_err={key_abs:.2e}  rel_err={key_rel:.2e}")
        max_abs_all = max(max_abs_all, key_abs)
        max_rel_all = max(max_rel_all, key_rel)
        score = min(key_abs / args.atol, key_rel / args.rtol)
        if score > worst_score:
            worst_key, worst_score = key, score

    print("-" * 70)
    print(f" 共校验 {n_checked} 个元素, 超差 {n_bad} 个")
    print(f" 最大绝对误差 {max_abs_all:.3e} (容差 {args.atol:g})")
    print(f" 最大相对误差 {max_rel_all:.3e} (容差 {args.rtol:g})")
    if n_bad == 0:
        print(" [PASS] 解析梯度与数值梯度一致，反向传播推导正确")
        return 0
    print(f" [FAIL] 有 {n_bad} 个元素不满足容差，最可疑的参数: {worst_key}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
