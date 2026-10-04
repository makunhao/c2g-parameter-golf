#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_submission.py —— 从实验产物生成 submission.json 与 ablation.md 的数据表

为什么要有这个脚本
------------------
《方案设计》第四节记了一条教训：**数字一旦被手抄进文档，就和产生它的东西脱钩了。**
所以文档里的表格不手写，而是从这个脚本生成，直接读 `out/ablation_result.json`。

⚠️ 生成出来的 submission.json **不是官方榜单成绩**：
   语料是 parameter-golf 仓库全文（非 FineWeb）、模型 4L×128d（非 9L×512d）、
   CPU（非 8×H100）。字段里带 `"track": "local_micro_benchmark"` 作显式标注。

用法：
    python scripts/make_submission.py --result out/ablation_result.json \
        --json ../十三_C2G_submission.json --md ../十三_C2G_ablation.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main() -> int:
    ap = argparse.ArgumentParser(description="生成 submission.json 与消融表")
    ap.add_argument("--result", default=str(HERE.parent / "out" / "ablation_result.json"))
    ap.add_argument("--json", default=str(HERE.parent.parent / "十三_C2G_submission.json"))
    ap.add_argument("--md", default=None, help="可选：把 Markdown 表格写到哪里")
    ap.add_argument("--prepare", default=str(HERE.parent / "data" / "prepare_report.json"))
    args = ap.parse_args()

    rp = Path(args.result)
    if not rp.exists():
        print(f"[submission] 找不到实验结果：{rp}", file=sys.stderr)
        return 2
    doc = json.loads(rp.read_text(encoding="utf-8"))
    s = doc["summary"]
    by = s["by_vocab"]
    v = s["verdict"]

    prep = {}
    if Path(args.prepare).exists():
        prep = json.loads(Path(args.prepare).read_text(encoding="utf-8"))

    sub = {
        "schema": "parameter-golf/submission",
        "track": "local_micro_benchmark",          # ← 显式标注：不是官方 track
        "official_leaderboard": False,
        "note": ("本文件记录的是 **CPU 微型台架**上的实测结果，"
                 "不是官方 8xH100 / FineWeb 评测的成绩。"
                 "语料、模型规模、硬件均与官方不同，BPB 数值不可与榜单互比。"),
        "author": "十三",
        "task": "C2G",
        "variant": "tokenizer_ablation_sp1024_vs_sp4096",
        "config": {
            "dim": s["config"]["dim"], "layers": s["config"]["layers"],
            "heads": s["config"]["heads"], "kv_heads": s["config"]["kv_heads"],
            "mlp_mult": s["config"]["mlp_mult"], "seq_len": s["config"]["seq_len"],
            "batch_seqs": s["config"]["batch_seqs"], "iters": s["config"]["iters"],
            "optimizer": "Muon(2D) + AdamW(rest)",
            "tie_embeddings": s["config"]["tie_embeddings"],
            "hardware": "CPU (no CUDA)",
        },
        "seeds": s["seeds"],
        "results": {},
        "verdict": {
            "criterion": v["hypothesis"],
            "loss_ratio_threshold_measured": v["threshold_measured"],
            "loss_ratio_mean": v["loss_ratio_mean"],
            "bpb_delta_mean": v["bpb_delta_mean"],
            "bpb_delta_std": v["bpb_delta_std"],
            "all_seeds_consistent": v["all_seeds_consistent"],
            "prediction_holds": v["prediction_holds"],
        },
        "not_measured": [
            "official leaderboard BPB (no 8xH100)",
            "16MB artifact size / int8+zlib packing",
            "SP8192 variant",
            "wallclock-constrained step count",
        ],
    }
    for vocab, d in by.items():
        sub["results"][f"sp{vocab}"] = {
            "val_bpb_mean": d["val_bpb_mean"],
            "val_bpb_std": d["val_bpb_std"],
            "val_bpb_values": d["val_bpb_values"],
            "val_loss_mean": d["val_loss_mean"],
            "tokens_per_byte": d["tokens_per_byte"],
            "n_params": d["n_params"],
        }

    Path(args.json).write_text(json.dumps(sub, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[submission] → {args.json}")

    # ── Markdown 表格（供 ablation.md 直接嵌） ────────────────
    lines = []
    lines.append("| 配置 | BPB 均值 | 标准差 | val_loss 均值 | tokens/byte | 参数量 |")
    lines.append("|---|---|---|---|---|---|")
    for vocab in ("1024", "4096"):
        d = by.get(vocab)
        if not d:
            continue
        lines.append(f"| SP{vocab} | **{d['val_bpb_mean']:.5f}** | {d['val_bpb_std']:.5f} | "
                     f"{d['val_loss_mean']:.5f} | {d['tokens_per_byte']:.5f} | {d['n_params']:,} |")
    lines.append("")
    lines.append("| seed | BPB(SP1024) | BPB(SP4096) | ΔBPB | loss 比 |")
    lines.append("|---|---|---|---|---|")
    for p in s["paired"]:
        lines.append(f"| {p['seed']} | {p['bpb_1024']:.5f} | {p['bpb_4096']:.5f} | "
                     f"**{p['delta']:+.5f}** | {p['loss_ratio']:.4f} |")
    lines.append("")
    lines.append(f"- 判据：`{v['hypothesis']}`")
    lines.append(f"- 阈值（本次实测 tokens/byte）：**{v['threshold_measured']}**"
                 f"　（先验值 {v['threshold_apriori']}，来自 prepare 全量 val）")
    lines.append(f"- 实测 loss 比：均值 **{v['loss_ratio_mean']}**，最大 **{v['loss_ratio_max']}**"
                 f"　（均匀分布上界 {v['uniform_ceiling']}）")
    lines.append(f"- **ΔBPB（4096 − 1024）= {v['bpb_delta_mean']:+.5f} "
                 f"± {v['bpb_delta_std']:.5f}**")
    lines.append(f"- 3 个 seed 方向"
                 f"{'全部一致' if v['all_seeds_consistent'] else '**不一致**'}"
                 f"；预测**{'成立' if v['prediction_holds'] else '不成立'}**")
    md = "\n".join(lines)

    if args.md:
        Path(args.md).write_text(md + "\n", encoding="utf-8")
        print(f"[submission] 表格 → {args.md}")
    else:
        print("\n" + md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
