#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare.py —— 建本地语料 + 自训 tokenizer + 跑字节记账自检

语料来源
--------
`openai/parameter-golf` 官方仓库里的全部 .md / .txt / .py
（≈ 3.1 MB，英文，ML/代码域，公开可核验）。

**为什么用它**：官方评测集 FineWeb 在本机不可得（HuggingFace 不通，且规则禁止联网）。
与其随便塞一段文本，不如用**和这个挑战本身直接相关的语料**——
它可复现（任何人 clone 那个仓库就能重跑），且领域明确。

语料划分
--------
按 9:1 切 train / val。**val 只用于算 BPB，不参与训练。**

用法
----
    python scripts/prepare.py --pg-repo /path/to/parameter-golf-main --out data
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np                                       # noqa: E402
import sentencepiece as spm                              # noqa: E402
import bpb as B                                          # noqa: E402

VOCAB_SIZES = [1024, 4096]


def collect_text(pg_repo: Path) -> str:
    """收集官方仓库里的纯文本。按路径排序 → 保证可复现（不依赖文件系统顺序）。"""
    exts = {".md", ".txt", ".py", ".json", ".yaml", ".yml"}
    files = sorted(p for p in pg_repo.rglob("*") if p.is_file() and p.suffix in exts)
    chunks = []
    for p in files:
        try:
            chunks.append(p.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return "\n".join(chunks)


def split_train_val(text: str, val_ratio: float = 0.1, seed: int = 1337) -> tuple[str, str]:
    """**按整篇文档切**而不是随机切字符——避免训练集和验证集共享上下文。"""
    paras = [p for p in text.split("\n\n") if p.strip()]
    rng = random.Random(seed)
    rng.shuffle(paras)
    n_val = max(1, int(len(paras) * val_ratio))
    return "\n\n".join(paras[n_val:]), "\n\n".join(paras[:n_val])


def train_tokenizer(train_text: str, vocab_size: int, out_dir: Path) -> Path:
    """自训 BPE。官方基线用的就是 BPE（`fineweb_1024_bpe.model`）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / f"local_bpe_{vocab_size}"
    tmp_txt = out_dir / f"_train_{vocab_size}.txt"
    tmp_txt.write_text(train_text, encoding="utf-8")
    spm.SentencePieceTrainer.train(
        input=str(tmp_txt),
        model_prefix=str(prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        byte_fallback=False,
        train_extremely_large_corpus=False,
        num_threads=4,
        minloglevel=2,
    )
    tmp_txt.unlink(missing_ok=True)
    return prefix.with_suffix(".model")


def main() -> int:
    ap = argparse.ArgumentParser(description="建语料 + 训 tokenizer + 字节记账自检")
    ap.add_argument("--pg-repo", required=True, help="parameter-golf-main 目录")
    ap.add_argument("--out", default=str(HERE.parent / "data"))
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("[prepare] 收集语料 …")
    text = collect_text(Path(args.pg_repo))
    (out / "corpus_full.txt").write_text(text, encoding="utf-8")
    print(f"[prepare] 语料：{len(text):,} 字符 / {len(text.encode('utf-8')):,} 字节")

    train_text, val_text = split_train_val(text, args.val_ratio, args.seed)
    (out / "corpus_train.txt").write_text(train_text, encoding="utf-8")
    (out / "corpus_val.txt").write_text(val_text, encoding="utf-8")
    print(f"[prepare] train {len(train_text.encode('utf-8')):,} 字节 / "
          f"val {len(val_text.encode('utf-8')):,} 字节")

    report = {"corpus": {}, "tokenizers": {}}
    report["corpus"] = {
        "full_bytes": len(text.encode("utf-8")),
        "train_bytes": len(train_text.encode("utf-8")),
        "val_bytes": len(val_text.encode("utf-8")),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
    }

    for vs in VOCAB_SIZES:
        print(f"\n[prepare] 训练 tokenizer vocab={vs} …")
        model = train_tokenizer(train_text, vs, out / "tokenizers")
        sp = B.load_sentencepiece(model)   # 见 bpb.load_sentencepiece 的非 ASCII 路径说明

        # ── 关键一步：字节记账自检 ──────────────────────────
        chk = B.selfcheck_byte_accounting(sp, val_text)
        print(f"[prepare]   vocab={vs}")
        print(f"[prepare]     官方口径累加 {chk['counted_bytes']:,} 字节 / {chk['tokens']:,} token")
        print(f"[prepare]     tokenizer 规范化后 {chk['decoded_bytes']:,} 字节"
              f"（原始文件 {chk['raw_text_bytes']:,}，折叠掉 "
              f"{chk['raw_text_bytes'] - chk['decoded_bytes']:,}）")
        print(f"[prepare]     boundary token {chk['boundary_tokens']} 个，"
              f"官方记 0 字节，需补回 {chk['boundary_piece_bytes']} 字节")
        print(f"[prepare]     ★ 补齐后残差 = {chk['residual_explained']} 字节  "
              f"{'✓ 对得上' if chk['ok'] else '✗ 对不上，必须解释！'}")
        print(f"[prepare]     tokens/byte = {chk['tokens_per_byte']:.4f}")

        # 落盘 val 分片（供微型训练脚本读）
        ids_tr = sp.encode(train_text, out_type=int)
        ids_va = sp.encode(val_text, out_type=int)
        B.write_shard(out / f"local_train_{vs}.bin", np.asarray(ids_tr, dtype=np.uint16))
        B.write_shard(out / f"local_val_{vs}.bin", np.asarray(ids_va, dtype=np.uint16))

        report["tokenizers"][str(vs)] = {
            "model_file": str(model),
            "sp_vocab_size": int(sp.vocab_size()),
            "train_tokens": len(ids_tr),
            "val_tokens": len(ids_va),
            "tokens_per_byte_val": ids_va.__len__() / max(1, len(val_text.encode("utf-8"))),
            "byte_selfcheck": chk,
        }

    (out / "prepare_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[prepare] 报告 → {out / 'prepare_report.json'}")

    bad = [k for k, v in report["tokenizers"].items() if not v["byte_selfcheck"]["ok"]]
    if bad:
        print(f"[prepare] ⚠ 有 {len(bad)} 个 tokenizer 的字节记账对不上：{bad}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
