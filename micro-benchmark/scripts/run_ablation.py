#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_ablation.py —— tokenizer 对照实验：SP1024 vs SP4096

实验问题
--------
**只换 tokenizer，不动模型，BPB 会变好吗？变多少？**

先给一个**可被证伪的预测**（这才叫实验设计，不叫"跑跑看"）
------------------------------------------------------------
BPB = (loss/ln2) × (tokens/bytes)。两套 tokenizer 的 tokens/byte 实测为：

    vocab 1024 → 0.4196
    vocab 4096 → 0.3193

若 4096 的 loss 与 1024 **完全相同**，BPB 会变成 0.761 倍。
但 loss 不可能相同——类别数从 1024 涨到 4096，而均匀分布下的交叉熵是 ln(V)：

    ln(4096)/ln(1024) = 8.318/6.931 = 1.200

所以 4096 的 loss 至少要涨到 1.20 倍（最坏情况），涨多少取决于文本的可预测性。

**把两式相除，得到一条判据：**

    BPB₄₀₉₆ < BPB₁₀₂₄  ⟺  L₄₀₉₆ / L₁₀₂₄ < 0.4196 / 0.3193 = **1.3142**

> **预测：只要 4096 的交叉熵涨幅小于 31.4%，换 tokenizer 就是赚的。**
> 均匀分布的上界是 20.0%，所以**只要模型学到任何东西，这条就应该成立**。
> 如果实测不成立，说明我对机制的理解错了——**那也是有价值的结论**。

统计口径
--------
· 每个配置跑 3 个 seed（1337 / 2025 / 42）
· 两配置**步数相同**（= 同样的计算量）——对应官方"同样的 10 分钟"这条硬约束
· 报告均值 ± 标准差，以及配对比较
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np                                       # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bpb as B                                          # noqa: E402
import train_micro as TM                                 # noqa: E402

SEEDS = [1337, 2025, 42]
VOCABS = [1024, 4096]
# 由 prepare.py 实测得到的 tokens/byte（用作预测里的系数）
TPB_HYPOTHESIS = {1024: 0.4196, 4096: 0.3193}


def main() -> int:
    ap = argparse.ArgumentParser(description="SP1024 vs SP4096 对照实验")
    ap.add_argument("--data", default=str(HERE.parent / "data"))
    ap.add_argument("--iters", type=int, default=800)
    ap.add_argument("--out", default=str(HERE.parent / "out"))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    args = ap.parse_args()

    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]

    cfg = dict(dim=128, layers=4, heads=4, kv_heads=2, mlp_mult=2, seq_len=256,
               qk_gain=1.5, softcap=30.0, tie_embeddings=True,
               batch_seqs=32, iters=args.iters, warmup=20,
               warmdown=max(1, args.iters // 5),
               muon_lr=0.02, muon_momentum=0.95, adam_lr=0.003, eval_seqs=64)

    runs: list[dict] = []
    for vocab in VOCABS:
        sp = B.load_sentencepiece(data / "tokenizers" / f"local_bpe_{vocab}.model")
        luts = B.build_sentencepiece_luts(sp, vocab)
        tr = B.read_shard(data / f"local_train_{vocab}.bin").astype(np.int32)
        va = B.read_shard(data / f"local_val_{vocab}.bin").astype(np.int32)
        print(f"\n===== vocab={vocab}  train={tr.size:,} tok  val={va.size:,} tok =====")
        for sd in seeds:
            print(f"  -- seed {sd} --")
            r = TM.run_one(cfg, tr, va, luts, vocab, sd, log_every=max(1, args.iters // 4))
            print(f"  => val_bpb={r['val_bpb']:.5f}  val_loss={r['val_loss_nats']:.5f}  "
                  f"tpb={r['tokens_per_byte']:.5f}  {r['ms_per_step']}ms/step")
            runs.append(r)

    # ── 汇总 ─────────────────────────────────────────────
    summary: dict = {"config": cfg, "seeds": seeds, "iters": args.iters, "by_vocab": {}}
    for vocab in VOCABS:
        rs = [r for r in runs if r["vocab"] == vocab]
        bpbs = [r["val_bpb"] for r in rs]
        losses = [r["val_loss_nats"] for r in rs]
        summary["by_vocab"][str(vocab)] = {
            "n": len(rs),
            "val_bpb_mean": round(st.mean(bpbs), 5),
            "val_bpb_std": round(st.pstdev(bpbs), 5) if len(bpbs) > 1 else 0.0,
            "val_bpb_values": bpbs,
            "val_loss_mean": round(st.mean(losses), 5),
            "tokens_per_byte": round(st.mean([r["tokens_per_byte"] for r in rs]), 5),
            "n_params": rs[0]["n_params"],
            "ms_per_step": rs[0]["ms_per_step"],
        }

    a = summary["by_vocab"]["1024"]
    b = summary["by_vocab"]["4096"]
    # 配对比较：同一 seed 下 4096 vs 1024
    paired = []
    for sd in seeds:
        r1 = next(r for r in runs if r["vocab"] == 1024 and r["seed"] == sd)
        r2 = next(r for r in runs if r["vocab"] == 4096 and r["seed"] == sd)
        paired.append({"seed": sd,
                       "bpb_1024": r1["val_bpb"], "bpb_4096": r2["val_bpb"],
                       "delta": round(r2["val_bpb"] - r1["val_bpb"], 5),
                       "loss_ratio": round(r2["val_loss_nats"] / r1["val_loss_nats"], 4)})

    ratios = [p["loss_ratio"] for p in paired]
    deltas = [p["delta"] for p in paired]

    # ⚠️ 阈值必须用**本次运行实测**的 tokens/byte，不能用 prepare 阶段的全量值。
    #    原因：eval 只用 val 集的前 N 个序列（这里 64×256），与全量 val 的
    #    tokens/byte 不完全相同（实测 0.42238 / 0.31779 vs 0.4196 / 0.3193）。
    #    第一版我用了先验值 1.3142，结果 150 步试点里出现了
    #    "loss 比 1.3201 > 1.3142 判为不成立，但 ΔBPB 明明是负的"这种自相矛盾。
    #    ——**判据和结论用不同的口径，就会得出互相矛盾的结论。**
    tpb_1024 = a["tokens_per_byte"]
    tpb_4096 = b["tokens_per_byte"]
    thr_measured = tpb_1024 / tpb_4096
    thr_apriori = TPB_HYPOTHESIS[1024] / TPB_HYPOTHESIS[4096]

    summary["paired"] = paired
    summary["verdict"] = {
        "hypothesis": "BPB₄₀₉₆ < BPB₁₀₂₄  ⟺  L₄₀₉₆/L₁₀₂₄ < tpb₁₀₂₄ / tpb₄₀₉₆",
        "threshold_apriori": round(thr_apriori, 4),
        "threshold_measured": round(thr_measured, 4),
        "tpb_apriori": TPB_HYPOTHESIS,
        "tpb_measured": {1024: tpb_1024, 4096: tpb_4096},
        "loss_ratio_mean": round(st.mean(ratios), 4),
        "loss_ratio_max": round(max(ratios), 4),
        "uniform_ceiling": round(math.log(4096) / math.log(1024), 4),
        "bpb_delta_mean": round(st.mean(deltas), 5),
        "bpb_delta_std": round(st.pstdev(deltas), 5) if len(deltas) > 1 else 0.0,
        "all_seeds_consistent": all(d < 0 for d in deltas),
        "prediction_holds": max(ratios) < thr_measured,
    }

    (out / "ablation_result.json").write_text(
        json.dumps({"summary": summary, "runs": runs}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    # ── 打印 ─────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("对照实验结果（CPU 微型台架，非官方榜单）")
    print("=" * 72)
    print(f"{'vocab':>8}{'BPB 均值':>14}{'标准差':>10}{'loss 均值':>12}"
          f"{'tokens/byte':>13}{'参数量':>10}")
    for v in VOCABS:
        s = summary["by_vocab"][str(v)]
        print(f"{v:>8}{s['val_bpb_mean']:>14.5f}{s['val_bpb_std']:>10.5f}"
              f"{s['val_loss_mean']:>12.5f}{s['tokens_per_byte']:>13.5f}{s['n_params']:>10,}")
    print("-" * 72)
    print("配对比较（同一 seed 下 4096 vs 1024）：")
    for p in paired:
        print(f"  seed {p['seed']:<6} ΔBPB = {p['delta']:+.5f}   "
              f"loss 比 = {p['loss_ratio']:.4f}")
    v = summary["verdict"]
    print("-" * 72)
    print(f"判据：{v['hypothesis']}")
    print(f"实测 loss 比：均值 {v['loss_ratio_mean']}，最大 {v['loss_ratio_max']}"
          f"（均匀分布上界 {v['uniform_ceiling']}）")
    print(f"阈值：先验 {v['threshold_apriori']}（prepare 全量 val）｜"
          f"实测 {v['threshold_measured']}（本次 eval 子集）")
    print(f"→ 预测{'成立 ✓' if v['prediction_holds'] else '不成立 ✗'}；"
          f"3 个 seed {'全部一致' if v['all_seeds_consistent'] else '不一致'}")
    print(f"→ ΔBPB（4096 − 1024）= {v['bpb_delta_mean']:+.5f} ± {v['bpb_delta_std']:.5f}")
    print("=" * 72)
    print(f"结果 → {out / 'ablation_result.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
