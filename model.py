# ============================================================
# Yuki - Demo 模型：numpy 手写 GPT-like Decoder（前向 + 反向）
# ------------------------------------------------------------
# 结构（Pre-Norm）：
#   Token Embedding
#   -> N x [ RMSNorm -> 多头因果自注意力 -> 残差
#            RMSNorm -> MLP(GELU)          -> 残差 ]
#   -> RMSNorm
#   -> 输出投影（与词嵌入共享权重，tie weights）
# 全部 float32，只依赖 numpy，CPU 上运行；反向传播完全手写。
# ============================================================

import numpy as np


# ============================================================
# 基础算子（前向 + 反向）
# ============================================================

def softmax(x, axis=-1):
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def softmax_backward(dy, y):
    """y 是 softmax 的输出，dy 是上游梯度。"""
    return y * (dy - np.sum(dy * y, axis=-1, keepdims=True))


_GELU_C = 0.7978845608028654  # sqrt(2 / pi)


def gelu(x):
    """tanh 近似的 GELU（numpy 没有 erf，用这个等价近似）。"""
    inner = _GELU_C * (x + 0.044715 * x ** 3)
    return 0.5 * x * (1.0 + np.tanh(inner))


def gelu_backward(x):
    inner = _GELU_C * (x + 0.044715 * x ** 3)
    t = np.tanh(inner)
    d_inner = _GELU_C * (1.0 + 3.0 * 0.044715 * x ** 2)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t ** 2) * d_inner


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def silu(x):
    """SiLU / Swish：x * sigmoid(x)"""
    return x * sigmoid(x)


def silu_backward(x):
    s = sigmoid(x)
    return s * (1.0 + x * (1.0 - s))


def rmsnorm_forward(x, w, eps):
    """RMSNorm：只用均方根缩放，不减均值。返回 (y, cache)。"""
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    xn = x / rms
    return xn * w, (xn, rms)


def rmsnorm_backward(dy, cache, w):
    """RMSNorm 反向：返回 (dx, dw)。"""
    xn, rms = cache
    dxn = dy * w
    dx = (dxn - xn * np.mean(dxn * xn, axis=-1, keepdims=True)) / rms
    axes = tuple(range(dy.ndim - 1))
    dw = np.sum(dy * xn, axis=axes)
    return dx, dw


# ---------------- RoPE（旋转位置编码） ----------------
# 把每个头的 Dh 维拆成 Dh/2 个二维子空间，按位置 t 旋转 t*theta_i 角度。
# 这样 Q/K 点积只依赖「相对距离」，模型就能分清哪一轮对话在前、哪一轮在后。

def precompute_rope_freqs(dim: int, max_len: int, theta: float = 10000.0,
                          dtype=np.float32):
    """预计算 cos/sin 表，形状 [max_len, dim/2]。"""
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    angles = np.arange(max_len, dtype=np.float64)[:, None] * freqs[None, :]
    return np.cos(angles).astype(dtype), np.sin(angles).astype(dtype)


def apply_rope(x, cos, sin):
    """x: [B, H, T, Dh]，cos/sin: [T, Dh/2] -> 旋转后的 x。"""
    c = cos[None, None, :, :]
    s = sin[None, None, :, :]
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1)


def apply_rope_backward(dy, cos, sin):
    """RoPE 的反向：旋转矩阵是正交阵，逆变换就是转置。"""
    c = cos[None, None, :, :]
    s = sin[None, None, :, :]
    half = dy.shape[-1] // 2
    y1, y2 = dy[..., :half], dy[..., half:]
    return np.concatenate([y1 * c + y2 * s, -y1 * s + y2 * c], axis=-1)


# ============================================================
# 模型
# ============================================================

class YukiModel:
    """52W 级 GPT-like Decoder，参数全部放在 self.params 字典里。"""

    def __init__(self, config, rng=None, dtype=np.float32):
        self.cfg = config
        self.dtype = dtype
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.scale = 1.0 / float(np.sqrt(config.head_dim))
        self.swiglu = (getattr(config, "mlp", "gelu") == "swiglu")
        self.params = self._init_params()
        self._mask_cache = None
        self._rope_cache = None

    # ---------------- 参数 ----------------
    def _init_params(self) -> dict:
        cfg = self.cfg
        D, F, L, V = cfg.hidden_dim, cfg.ff_dim, cfg.num_layers, cfg.vocab_size
        assert D % cfg.num_heads == 0, "hidden_dim 必须能被 num_heads 整除"
        rng, dt = self.rng, self.dtype
        std = 0.02
        res_std = std / np.sqrt(2.0 * L)   # 残差分支的输出层缩小初始化（GPT-2 做法）

        p = {"tok_emb": rng.normal(0.0, std, (V, D)).astype(dt)}
        for i in range(L):
            k = f"h{i}"
            p[f"{k}.norm1.w"] = np.ones(D, dtype=dt)
            p[f"{k}.norm2.w"] = np.ones(D, dtype=dt)
            p[f"{k}.attn.Wq"] = rng.normal(0.0, std, (D, D)).astype(dt)
            p[f"{k}.attn.Wk"] = rng.normal(0.0, std, (D, D)).astype(dt)
            p[f"{k}.attn.Wv"] = rng.normal(0.0, std, (D, D)).astype(dt)
            p[f"{k}.attn.Wo"] = rng.normal(0.0, res_std, (D, D)).astype(dt)
            p[f"{k}.mlp.W1"] = rng.normal(0.0, std, (D, F)).astype(dt)
            p[f"{k}.mlp.W2"] = rng.normal(0.0, res_std, (F, D)).astype(dt)
            if self.swiglu:                    # SwiGLU 的第三个矩阵（门控的另一半）
                p[f"{k}.mlp.W3"] = rng.normal(0.0, std, (D, F)).astype(dt)
        p["norm_f.w"] = np.ones(D, dtype=dt)
        return p

    def num_params(self) -> int:
        return int(sum(v.size for v in self.params.values()))

    # ---------------- 因果掩码 ----------------
    def _causal_mask(self, seq_len: int):
        """上三角为 -1e9（屏蔽未来），对角线及以下为 0；按需扩容并缓存。"""
        m = self._mask_cache
        if m is None or m.shape[-1] < seq_len:
            mask = np.triu(np.full((seq_len, seq_len), -1e9, dtype=self.dtype), k=1)
            m = mask.reshape(1, 1, seq_len, seq_len)
            self._mask_cache = m
        return m[:, :, :seq_len, :seq_len]

    # ---------------- 前向 ----------------
    def _rope_tables(self, seq_len: int):
        """按需扩容并缓存 RoPE 的 cos/sin 表。"""
        cached = self._rope_cache
        if cached is None or cached[0].shape[0] < seq_len:
            need = max(seq_len, self.cfg.max_seq_len)
            cached = precompute_rope_freqs(self.cfg.head_dim, need,
                                           self.cfg.rope_theta, self.dtype)
            self._rope_cache = cached
        cos, sin = cached
        return cos[:seq_len], sin[:seq_len]

    def _attention_forward(self, h, key, mask):
        """多头因果自注意力 + RoPE。h: [B, T, D] -> out: [B, T, D]"""
        cfg = self.cfg
        p = self.params
        B, T, D = h.shape
        H, Dh = cfg.num_heads, cfg.head_dim

        q = (h @ p[f"{key}.attn.Wq"]).reshape(B, T, H, Dh).transpose(0, 2, 1, 3)
        k = (h @ p[f"{key}.attn.Wk"]).reshape(B, T, H, Dh).transpose(0, 2, 1, 3)
        v = (h @ p[f"{key}.attn.Wv"]).reshape(B, T, H, Dh).transpose(0, 2, 1, 3)

        # RoPE：只旋转 Q 和 K，V 不动
        cos, sin = self._rope_tables(T)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale + mask   # [B, H, T, T]
        attn = softmax(scores, axis=-1)
        ctx = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, D)      # [B, T, D]
        out = ctx @ p[f"{key}.attn.Wo"]
        cache = {"q": q, "k": k, "v": v, "attn": attn, "ctx": ctx,
                 "h": h, "cos": cos, "sin": sin}
        return out, cache

    def forward(self, ids):
        """ids: [B, T] int -> (logits [B, T, V], cache)"""
        cfg = self.cfg
        p = self.params
        B, T = ids.shape
        mask = self._causal_mask(T)

        x = p["tok_emb"][ids].astype(self.dtype)
        layers = []
        for i in range(cfg.num_layers):
            k = f"h{i}"
            # --- 注意力分支 ---
            h1, c1 = rmsnorm_forward(x, p[f"{k}.norm1.w"], cfg.eps)
            a, ca = self._attention_forward(h1, k, mask)
            x_mid = x + a
            # --- MLP 分支 ---
            h2, c2 = rmsnorm_forward(x_mid, p[f"{k}.norm2.w"], cfg.eps)
            if self.swiglu:
                # SwiGLU: W2( silu(W1 x) * (W3 x) )
                g1 = h2 @ p[f"{k}.mlp.W1"]
                g3 = h2 @ p[f"{k}.mlp.W3"]
                s1 = silu(g1)
                g = s1 * g3
                mlp_cache = {"kind": "swiglu", "g1": g1, "g3": g3, "s1": s1}
            else:
                # GELU: W2( gelu(W1 x) )
                f1 = h2 @ p[f"{k}.mlp.W1"]
                g = gelu(f1)
                mlp_cache = {"kind": "gelu", "pre": f1}
            f2 = g @ p[f"{k}.mlp.W2"]
            x = x_mid + f2
            layers.append({"k": k, "c1": c1, "ca": ca, "c2": c2,
                           "g": g, "h1": h1, "h2": h2, "mlp": mlp_cache})

        hf, cf = rmsnorm_forward(x, p["norm_f.w"], cfg.eps)
        logits = hf @ p["tok_emb"].T
        cache = {"ids": ids, "hf": hf, "cf": cf, "layers": layers, "shape": (B, T)}
        return logits, cache

    # ---------------- 反向 ----------------
    def backward(self, dlogits, cache) -> dict:
        """手写反向传播：返回与 self.params 同结构的梯度字典。"""
        cfg = self.cfg
        p = self.params
        B, T = cache["shape"]
        D, F, H, Dh = cfg.hidden_dim, cfg.ff_dim, cfg.num_heads, cfg.head_dim
        BT = B * T

        grads = {}
        # --- 输出投影（与词嵌入绑定）---
        dlogits2 = dlogits.reshape(BT, -1)
        hf2 = cache["hf"].reshape(BT, D)
        grads["tok_emb"] = dlogits2.T @ hf2                    # [V, D]
        dhf = (dlogits2 @ p["tok_emb"]).reshape(B, T, D)

        dx, grads["norm_f.w"] = rmsnorm_backward(dhf, cache["cf"], p["norm_f.w"])

        # --- 逐层反向 ---
        for i in reversed(range(cfg.num_layers)):
            L_ = cache["layers"][i]
            k = L_["k"]

            # MLP：x_out = x_mid + f2
            df2 = dx
            grads[f"{k}.mlp.W2"] = L_["g"].reshape(BT, F).T @ df2.reshape(BT, D)
            dact = df2 @ p[f"{k}.mlp.W2"].T
            mc = L_["mlp"]
            h2f = L_["h2"].reshape(BT, D)
            if mc["kind"] == "swiglu":
                dg1 = (dact * mc["g3"]) * silu_backward(mc["g1"])
                dg3 = dact * mc["s1"]
                grads[f"{k}.mlp.W1"] = h2f.T @ dg1.reshape(BT, F)
                grads[f"{k}.mlp.W3"] = h2f.T @ dg3.reshape(BT, F)
                dh2 = dg1 @ p[f"{k}.mlp.W1"].T + dg3 @ p[f"{k}.mlp.W3"].T
            else:
                dpre = dact * gelu_backward(mc["pre"])
                grads[f"{k}.mlp.W1"] = h2f.T @ dpre.reshape(BT, F)
                dh2 = dpre @ p[f"{k}.mlp.W1"].T
            dx_mid, grads[f"{k}.norm2.w"] = rmsnorm_backward(dh2, L_["c2"], p[f"{k}.norm2.w"])
            dx_mid = dx_mid + dx                               # 残差

            # 注意力：x_mid = x_in + a
            da = dx_mid
            ca = L_["ca"]
            grads[f"{k}.attn.Wo"] = ca["ctx"].reshape(BT, D).T @ da.reshape(BT, D)
            dctx = (da @ p[f"{k}.attn.Wo"].T).reshape(B, T, H, Dh).transpose(0, 2, 1, 3)

            dv = ca["attn"].transpose(0, 1, 3, 2) @ dctx
            dattn = dctx @ ca["v"].transpose(0, 1, 3, 2)
            dscores = softmax_backward(dattn, ca["attn"]) * self.scale

            dq = dscores @ ca["k"]                             # [B, H, T, Dh]
            dk = dscores.transpose(0, 1, 3, 2) @ ca["q"]
            # RoPE 反向：旋转是正交变换，逆变换就是转置
            dq = apply_rope_backward(dq, ca["cos"], ca["sin"])
            dk = apply_rope_backward(dk, ca["cos"], ca["sin"])
            dq = dq.transpose(0, 2, 1, 3).reshape(BT, D)
            dk = dk.transpose(0, 2, 1, 3).reshape(BT, D)
            dv = dv.transpose(0, 2, 1, 3).reshape(BT, D)

            h1f = L_["h1"].reshape(BT, D)
            grads[f"{k}.attn.Wq"] = h1f.T @ dq
            grads[f"{k}.attn.Wk"] = h1f.T @ dk
            grads[f"{k}.attn.Wv"] = h1f.T @ dv
            dh1 = (dq @ p[f"{k}.attn.Wq"].T
                   + dk @ p[f"{k}.attn.Wk"].T
                   + dv @ p[f"{k}.attn.Wv"].T).reshape(B, T, D)

            dx_in, grads[f"{k}.norm1.w"] = rmsnorm_backward(dh1, L_["c1"], p[f"{k}.norm1.w"])
            dx = dx_mid + dx_in

        # --- 词嵌入：把输入位置的梯度累加回去 ---
        np.add.at(grads["tok_emb"], cache["ids"], dx)
        return grads

    # ---------------- 损失 ----------------
    @staticmethod
    def loss_and_grad(logits, targets):
        """交叉熵（softmax + NLL）合并实现，返回 (loss, dlogits)。"""
        B, T, V = logits.shape
        x = logits.reshape(-1, V)
        y = targets.reshape(-1)
        z = x - np.max(x, axis=-1, keepdims=True)
        expz = np.exp(z)
        s = np.sum(expz, axis=-1, keepdims=True)
        logp = z - np.log(s)
        n = y.shape[0]
        rows = np.arange(n)
        loss = float(-np.mean(logp[rows, y]))
        dlogits = expz / s
        dlogits[rows, y] -= 1.0
        dlogits /= n
        return loss, dlogits.reshape(B, T, V)

    # ---------------- 持久化 ----------------
    def save(self, path: str) -> None:
        np.savez_compressed(path, **self.params)

    @classmethod
    def from_checkpoint(cls, path: str, config, dtype=np.float32) -> "YukiModel":
        """从 npz 载入权重。键名或形状对不上就直接报错，绝不静默产出乱码。"""
        model = cls(config, rng=np.random.default_rng(0), dtype=dtype)
        with np.load(path) as data:
            missing = [k for k in model.params if k not in data.files]
            if missing:
                raise ValueError(
                    f"checkpoint 缺少参数 {missing[:3]}：模型结构与 checkpoint 不一致\n"
                    f"  当前结构: mlp={getattr(config, 'mlp', '?')} "
                    f"F={config.ff_dim} L={config.num_layers} V={config.vocab_size}")
            bad = [k for k in model.params if data[k].shape != model.params[k].shape]
            if bad:
                k = bad[0]
                raise ValueError(
                    f"checkpoint 与当前模型形状不一致：{k} "
                    f"{data[k].shape} vs {model.params[k].shape}\n"
                    f"  当前结构: mlp={getattr(config, 'mlp', '?')} "
                    f"F={config.ff_dim} L={config.num_layers} V={config.vocab_size}")
            for k in model.params:
                model.params[k] = data[k].astype(dtype)
        return model
