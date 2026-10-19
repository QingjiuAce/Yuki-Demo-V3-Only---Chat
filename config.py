# ============================================================
# Yuki - Demo 配置
# 52W 级（约 52 万参数）Transformer Decoder（GPT-like，Pre-Norm）
# 训练：纯 numpy + CPU，只保留最基础的训练算法
# ------------------------------------------------------------
# 参数量公式（输出投影与词嵌入权重绑定）：
#     V x D                           词嵌入 / 输出投影（共享同一份权重）
#   + L x (4D^2 + k*D*F + 2D)         每层：注意力 4 个方阵 + MLP（k=3 SwiGLU / k=2 GELU）
#   + D                               末尾 RMSNorm
# 默认 D=64, L=9, F=162, V=1438, SwiGLU  ->  520,640 ≈ 52W  (V2.1)
# ============================================================

import os
from dataclasses import asdict, dataclass, fields

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve(path: str) -> str:
    """相对路径一律相对项目目录解析，避免受当前工作目录影响。"""
    return path if os.path.isabs(path) else os.path.join(PROJECT_DIR, path)


@dataclass
class YukiConfig:
    # ---------- 模型结构 ----------
    vocab_size: int = 4200      # 词表上限；实际值由语料字符集决定（跑训练时会被覆盖）
    hidden_dim: int = 64        # 隐藏维度 D
    num_layers: int = 9         # 层数 L（V2.1：5 -> 9，配合词表 ~1200 约 52W 参数）
    num_heads: int = 4          # 注意力头数 H（head_dim = D / H = 16）
    ff_dim: int = 162           # MLP 中间维度 F（SwiGLU + 词表 1438 下正好 ~52W）
    mlp: str = "swiglu"         # MLP 类型："swiglu"（3 矩阵门控，等价 LLaMA）/ "gelu"（2 矩阵）
    max_seq_len: int = 128      # 训练与生成的最大序列长度
    eps: float = 1e-6           # RMSNorm 的数值稳定项
    rope_theta: float = 10000.0 # RoPE 基础频率

    # ---------- 基础训练参数 ----------
    batch_size: int = 16        # 每个 batch 的序列条数
    learning_rate: float = 3e-3
    weight_decay: float = 0.0   # Adam 的 L2 项，默认关闭
    beta1: float = 0.9
    beta2: float = 0.999
    adam_eps: float = 1e-8
    grad_clip: float = 1.0      # 全局梯度范数裁剪阈值，0 表示不裁剪
    max_steps: int = 3000       # 总训练步数
    log_every: int = 20         # 每多少步打印一次 loss
    save_every: int = 200       # 每多少步存一次 checkpoint
    sample_every: int = 500     # 每多少步生成一次样例（0 表示关闭）
    seed: int = 1337

    # ---------- 语料清洗 ----------
    strip_line_numbers: bool = True   # 去掉行首序号（"1 句子" -> "句子"）
    doc_per_line: bool = True         # 每个非空行作为一篇文档，用 <bos>/<eos> 包住
    shuffle_docs: bool = True         # 每轮重排文档顺序，防止模型靠语料顺序作弊

    # ---------- 路径 ----------
    data_dir: str = "data"                          # 放 *.txt 语料的目录
    out_dir: str = "checkpoints"                    # 模型与分词器的输出目录
    tokenizer_path: str = "checkpoints/tokenizer.json"

    # ---------- 生成参数 ----------
    temperature: float = 0.8
    top_k: int = 40
    max_new_tokens: int = 200

    # ---------- 工具 ----------
    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads

    @property
    def mlp_mats(self) -> int:
        """MLP 的矩阵个数：SwiGLU 是 3 个（gate/up/down），GELU 是 2 个。"""
        return 3 if self.mlp == "swiglu" else 2

    def param_count(self, vocab_size: int | None = None) -> int:
        """按公式估算参数量（tie weights，无 bias）。"""
        V = self.vocab_size if vocab_size is None else vocab_size
        D, F, L = self.hidden_dim, self.ff_dim, self.num_layers
        return int(V * D + L * (4 * D * D + self.mlp_mats * D * F + 2 * D) + D)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "YukiConfig":
        names = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in names}
        # 兼容旧 checkpoint：V2 之前都没有 mlp 字段，那时一律是 GELU 双矩阵 MLP。
        # 不补这个默认值的话，旧权重会被当成 SwiGLU 加载 -> 形状错位 + W3 随机 -> 输出乱码。
        kwargs.setdefault("mlp", "gelu")
        return cls(**kwargs)
