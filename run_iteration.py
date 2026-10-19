# ============================================================
# Yuki - Demo 迭代管线：augment(可选) -> merge(含留出集) -> train -> eval
# ------------------------------------------------------------
# 一条命令跑完一轮迭代，并把「记忆分 / 留出分」记进 runs/iterations.csv，
# 免得每次都手动 merge -> train -> eval 三步走、还得自己抄数字。
#
# 留出集（--holdout，默认 0.1）：
#   * 第一次运行：从合并后的语料里按固定种子抽 10% 的「整段」，写到
#     data/_holdout/holdout.txt；
#   * 之后每次运行：复用这份固定留出集，并把它的段从训练语料里排除。
#   这样各代的「留出分」是同一批题，才可比。（重新抽会破坏可比性，见 --refresh-holdout）
#
# 典型用法:
#   # 看会执行哪些命令，不真的跑
#   python run_iteration.py --base data/Demo_chat_v24.txt --parts data/_parts6 \
#       --out data/Demo_chat_v25.txt --resume --steps 22000 --lr 5e-4 --tag v2.5 --dry-run
#
#   # 阶段 1：加错字剂量 4× 并续训（先在上一代权重上热启动/续训，不重训）
#   python run_iteration.py --base data/Demo_chat_v24.txt \
#       --augment-input data/Demo_chat_v24.txt --augment-dose 4 \
#       --augment-out data/_parts6/perturb3.txt \
#       --out data/Demo_chat_v25.txt \
#       --resume --steps 22000 --lr 5e-4 --tag v2.5
#
#   # 只想重训 + 评测（语料已经合并好了）
#   python run_iteration.py --skip-merge --resume --steps 22000 --lr 5e-4 --tag v2.5
# ============================================================

import argparse
import csv
import datetime as dt
import glob
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def resolve(path):
    """相对路径一律相对项目目录解析。"""
    return path if os.path.isabs(path) else os.path.join(HERE, path)


def rel(path):
    """转成相对项目目录的路径（命令行更好看）。"""
    try:
        return os.path.relpath(path, HERE)
    except ValueError:
        return path


def show(cmd):
    return " ".join(f'"{c}"' if " " in c else c for c in cmd)


def run(cmd, dry):
    print("\n$ " + show(cmd), flush=True)
    if dry:
        return 0
    # 不捕获输出：让 merge / train / eval 的进度实时打在屏幕上
    return subprocess.call(cmd, cwd=HERE)


def top_level_corpora(data_dir, exclude=None):
    """data/ 顶层的 *.txt —— 训练脚本 load_texts() 读的就是这些（子目录不算）。"""
    files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    if exclude:
        ex = os.path.abspath(exclude)
        files = [f for f in files if os.path.abspath(f) != ex]
    return files


def unique_path(path):
    """目标已存在时给个 _archivedN 后缀，绝不覆盖已有文件。"""
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    i = 1
    while os.path.exists(f"{stem}_archived{i}{ext}"):
        i += 1
    return f"{stem}_archived{i}{ext}"


def log_row(runs_dir, tag, out_corpus, out_dir, args, jp):
    """把这次评测的「记忆分 / 留出分」追加到 runs/iterations.csv。"""
    with open(jp, "r", encoding="utf-8") as f:
        data = json.load(f)
    mem, ho = data.get("memory_avg"), data.get("holdout_avg")

    def robust_acc(sub):
        """从逐组结果里挑出 robust（真·没见过的扰动）——这才是字面鲁棒性的成绩单。"""
        for r in data.get("rows", []):
            if r.get("name", "").startswith("robust") and sub in r["name"]:
                return round(100.0 * r["acc"], 2)
        return ""

    mode = ("resume" if (args.resume or args.resume_from)
            else "init-from" if args.init_from else "scratch")
    row = {
        "time": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "tag": tag,
        "corpus": os.path.basename(out_corpus) if out_corpus else "",
        "train_mode": mode,
        "steps": args.steps if args.steps else "",
        "lr": args.lr if args.lr else "",
        "params": data.get("params", ""),
        "vocab": data.get("vocab_size", ""),
        "robust_del": robust_acc("删"),
        "robust_add": robust_acc("多"),
        "memory_avg": "" if mem is None else round(mem, 2),
        "holdout_avg": "" if ho is None else round(ho, 2),
        "gap": "" if (mem is None or ho is None) else round(mem - ho, 2),
        "out_dir": os.path.basename(out_dir),
    }
    csv_path = os.path.join(runs_dir, "iterations.csv")
    fresh = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if fresh:
            w.writeheader()
        w.writerow(row)
    print(f"\n[记录] 已追加到 {rel(csv_path)}")
    bits = []
    if row["robust_del"] != "":
        bits.append(f"robust 删1字 {row['robust_del']}% / 多1字 {row['robust_add']}%")
    if row["holdout_avg"] != "":
        bits.append(f"记忆分 {row['memory_avg']}% / 留出分 {row['holdout_avg']}%"
                    f"（差 {row['gap']} 点）")
    if bits:
        print("[记录] " + "  |  ".join(bits)
              + "   （判断有没有真变强：先看 robust，再看留出分）")


def main():
    ap = argparse.ArgumentParser(description="Yuki - Demo 一轮迭代：merge -> train -> eval")
    # ---- 语料 / 留出集 ----
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--base", default=None,
                    help="基底语料（默认自动取 data/ 顶层唯一的那个 txt）")
    ap.add_argument("--parts", default=None, help="新增片段目录（不给就只是 base -> out）")
    ap.add_argument("--out", default=None, help="训练语料输出文件（merge 时必须给）")
    ap.add_argument("--holdout", type=float, default=0.1,
                    help="留出集比例（默认 0.1；0 表示不切留出集）")
    ap.add_argument("--holdout-mode", choices=["segment", "answer"], default="segment")
    ap.add_argument("--holdout-out", default=None,
                    help="留出集文件（默认 <data>/_holdout/holdout.txt）")
    ap.add_argument("--refresh-holdout", action="store_true",
                    help="重新抽留出集（会破坏跨代可比性，慎用）")
    ap.add_argument("--keep-old", action="store_true",
                    help="不把旧的训练语料归档到 data/_raw/（默认归档，否则训练会同时读两份）")
    # ---- 扰动增强（可选）----
    ap.add_argument("--augment-input", default=None, help="扰动增强的输入语料")
    ap.add_argument("--augment-out", default=None, help="扰动增强的输出片段文件")
    ap.add_argument("--augment-dose", type=int, default=0,
                    help="每个单轮问句生成几个扰动版本（0=不做增强）")
    ap.add_argument("--augment-seed", type=int, default=None)
    # ---- 训练 ----
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--steps", type=int, default=None, help="总训练步数")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--resume", action="store_true",
                    help="从 out-dir/latest.npz 断点续训（连优化器一起恢复）")
    ap.add_argument("--resume-from", default=None, help="指定要续训的 checkpoint")
    ap.add_argument("--init-from", default=None, help="用旧权重热启动（优化器从零）")
    # ---- 其它 ----
    ap.add_argument("--tag", default=None, help="这次迭代的名字（用于 runs/ 记录，默认取语料名）")
    ap.add_argument("--python", default=None, help="跑子脚本的解释器（默认当前解释器）")
    ap.add_argument("--skip-merge", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令，不执行")
    args = ap.parse_args()

    py = args.python or sys.executable
    data_dir = resolve(args.data_dir)
    out_dir = resolve(args.out_dir)
    out_corpus = resolve(args.out) if args.out else None
    runs_dir = os.path.join(HERE, "runs")
    ho_path = resolve(args.holdout_out) if args.holdout_out else \
        os.path.join(data_dir, "_holdout", "holdout.txt")

    print("=" * 70)
    print(f" Yuki - Demo 迭代管线  |  tag={args.tag or '(自动)'}  |  干跑={args.dry_run}")
    print("=" * 70)

    # ---------- 0) 扰动增强（可选）----------
    parts_dir = resolve(args.parts) if args.parts else None
    if args.augment_dose > 0:
        if not (args.augment_input and args.augment_out):
            sys.exit("[augment] 需要同时给 --augment-input 和 --augment-out")
        cmd = [py, "augment_perturb.py",
               "--input", rel(resolve(args.augment_input)),
               "--out", rel(resolve(args.augment_out)),
               "--per-turn", str(args.augment_dose)]
        if args.augment_seed is not None:
            cmd += ["--seed", str(args.augment_seed)]
        if run(cmd, args.dry_run) != 0:
            sys.exit("[augment] 失败")
        # 没显式给 --parts 时，默认把增强输出所在目录当作新增片段目录
        if parts_dir is None:
            parts_dir = os.path.dirname(resolve(args.augment_out))

    # ---------- 1) merge（含留出集）----------
    base_corpus = resolve(args.base) if args.base else None
    if not args.skip_merge:
        if base_corpus is None:
            cand = top_level_corpora(data_dir, out_corpus)
            if len(cand) == 1:
                base_corpus = cand[0]
                print(f"[merge] 自动识别基底语料: {rel(base_corpus)}")
            elif not cand:
                sys.exit("[merge] data/ 顶层没有语料，请用 --base 指定基底")
            else:
                sys.exit("[merge] data/ 顶层有多个语料，请用 --base 指定基底："
                         + ", ".join(os.path.basename(c) for c in cand))
        if out_corpus is None:
            sys.exit("[merge] 请用 --out 指定训练语料输出文件")

        cmd = [py, "merge_corpus_v21.py",
               "--base", rel(base_corpus), "--out", rel(out_corpus)]
        if args.holdout and args.holdout > 0:
            cmd += ["--holdout", str(args.holdout),
                    "--holdout-mode", args.holdout_mode,
                    "--holdout-out", rel(ho_path)]
            if args.refresh_holdout:
                cmd.append("--refresh-holdout")
        if parts_dir:
            cmd += ["--parts", rel(parts_dir)]
        else:
            cmd.append("--no-parts")
        if run(cmd, args.dry_run) != 0:
            sys.exit("[merge] 失败")

        # ---------- 2) 归档旧语料 ----------
        # 训练脚本会读 data/ 顶层所有 *.txt —— 旧的训练语料不挪走就会被一起训，
        # 语料直接翻倍。所以默认把「不是本次输出」的顶层 txt 挪进 data/_raw/。
        olds = top_level_corpora(data_dir, out_corpus)
        if olds and not args.keep_old:
            for f in olds:
                dst = unique_path(os.path.join(data_dir, "_raw", os.path.basename(f)))
                print(f"[归档] {rel(f)}  ->  {rel(dst)}")
                if not args.dry_run:
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.move(f, dst)
        elif olds:
            print(f"[归档] --keep-old：data/ 顶层仍有 {len(olds)} 份语料会一起被训练，注意")

    # ---------- 3) train ----------
    if not args.skip_train:
        cmd = [py, "train.py", "--data-dir", rel(data_dir), "--out-dir", rel(out_dir)]
        if args.steps: cmd += ["--steps", str(args.steps)]
        if args.lr: cmd += ["--lr", str(args.lr)]
        if args.batch_size: cmd += ["--batch-size", str(args.batch_size)]
        if args.seq_len: cmd += ["--seq-len", str(args.seq_len)]
        if args.resume_from:
            cmd += ["--resume-from", rel(resolve(args.resume_from))]
        elif args.resume:
            cmd += ["--resume"]
        elif args.init_from:
            cmd += ["--init-from", rel(resolve(args.init_from))]
        if run(cmd, args.dry_run) != 0:
            sys.exit("[train] 失败")

    # ---------- 4) eval ----------
    if not args.skip_eval:
        tag = args.tag or (os.path.splitext(os.path.basename(out_corpus))[0]
                           if out_corpus else "run")
        jp = os.path.join(runs_dir, f"eval_{tag}.json")
        cmd = [py, "eval_chat.py", "--out-dir", rel(out_dir), "--json", rel(jp)]
        if args.holdout and args.holdout > 0 and os.path.exists(ho_path):
            cmd += ["--holdout-dir", rel(os.path.dirname(ho_path))]
        if run(cmd, args.dry_run) != 0:
            sys.exit("[eval] 失败")
        if not args.dry_run and os.path.exists(jp):
            log_row(runs_dir, tag, out_corpus, out_dir, args, jp)

    print("\n" + "=" * 70)
    print(f" 完成：tag={args.tag or ''}  语料={rel(out_corpus) if out_corpus else '-'}"
          f"  模型={rel(out_dir)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
