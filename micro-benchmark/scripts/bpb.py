#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bpb.py —— BPB 口径的复刻与自检（**本项目最关键的"抄得准不准"环节**）

为什么要单独一个模块
--------------------
C2G 的 35% 分数挂在 BPB 上。如果我对 BPB 的理解是错的，
后面所有的对照实验都是自娱自乐。所以这里先把**官方口径**抄准，
并且用一条**可判真假的恒等式**去验证它。

官方定义了 BPB = (val_loss / ln2) × (token 数 / 字节数)，
其中"字节数"不是 len(text)，而是**按 tokenizer 逐 token 累加的真实字节数**：

    build_sentencepiece_luts()  （train_gpt.py 第 180–205 行）
      · base_bytes[token]          = 该 piece 的 UTF-8 字节数（去掉开头的 ▁）
      · has_leading_space[token]   = piece 是否以 ▁ 开头
      · is_boundary_token[token]   = 是否 control/unknown/unused
    累加时：
      bytes(t) = base_bytes[t] + (has_leading_space[t] and not is_boundary[prev])

**我加的那条自检**：把上面这套累加**跑遍整段语料**，
结果必须等于 `len(text.encode("utf-8"))`——因为整段文本的字节数是一个
不依赖 tokenizer 的客观量。**对得上，说明记账逻辑抄对了；对不上，我后面全白做。**

（注意：SentencePiece 会做规范化，所以个别字符的字节归属可能被改写。
 因此本模块**不做四舍五入式的"约等于"**，而是把差值原样报出来——
 差值若不是 0，就必须解释，不能藏。）
"""

from __future__ import annotations

import math
import shutil
import tempfile
from pathlib import Path

import numpy as np

LN2 = math.log(2.0)
HEADER_INTS = 256
SHARD_MAGIC = 20240520
SHARD_VERSION = 1


# ══════════════════════════════════════════════════════════════
# 〇、一个 Windows 上的真坑：sentencepiece 打不开非 ASCII 路径
# ══════════════════════════════════════════════════════════════
def load_sentencepiece(model_path):
    """加载 SentencePiece 模型，绕开 Windows 非 ASCII 路径。

    实测（2026-10-04，本机）：
        路径 D:/ai+x/work/c2g_tmp/local_bpe_1024.model       → OK
        路径 D:/ai+x/C2G交付包/.../local_bpe_1024.model      → RuntimeError: NOT_FOUND
    —— **文件确实存在，`ls` 看得到**。sentencepiece 的 C++ 层用 ANSI 码页打开文件，
    中文路径被查不到。

    修法：路径非纯 ASCII 时，先把 .model 复制到一个 ASCII 临时路径再加载。
    这是**绕过**而不是修复（改不了第三方库），
    所以交付包里所有加载 tokenizer 的地方都必须走这个函数，不能直连。
    """
    import sentencepiece as spm

    p = Path(model_path)
    try:
        str(p).encode("ascii")
        ascii_safe = True
    except UnicodeEncodeError:
        ascii_safe = False

    if ascii_safe:
        return spm.SentencePieceProcessor(model_file=str(p))

    tmp = Path(tempfile.gettempdir()) / f"c2g_sp_{abs(hash(str(p))) % 10**8}.model"
    try:
        if not tmp.exists() or tmp.stat().st_size != p.stat().st_size:
            shutil.copyfile(p, tmp)
    except OSError:
        pass
    return spm.SentencePieceProcessor(model_file=str(tmp))


# ══════════════════════════════════════════════════════════════
# 一、官方口径：SentencePiece 字节查表
# ══════════════════════════════════════════════════════════════
def build_sentencepiece_luts(sp, vocab_size: int):
    """逐行对应 train_gpt.py 的 build_sentencepiece_luts()。

    返回 (base_bytes, has_leading_space, is_boundary_token)，均为 numpy 数组。
    """
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes = np.zeros((table_size,), dtype=np.int16)
    has_leading_space = np.zeros((table_size,), dtype=np.bool_)
    is_boundary = np.ones((table_size,), dtype=np.bool_)

    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary[token_id] = False
        if sp.is_byte(token_id):
            base_bytes[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"):          # ▁
            has_leading_space[token_id] = True
            piece = piece[1:]
        base_bytes[token_id] = len(piece.encode("utf-8"))
    return base_bytes, has_leading_space, is_boundary


def token_bytes(tgt_ids: np.ndarray, prev_ids: np.ndarray,
                base_bytes: np.ndarray, has_leading_space: np.ndarray,
                is_boundary: np.ndarray) -> np.ndarray:
    """每个目标 token 实际覆盖的字节数。对应 eval_val() 里的那三行累加。"""
    b = base_bytes[tgt_ids].astype(np.int32)
    b = b + (has_leading_space[tgt_ids] & ~is_boundary[prev_ids]).astype(np.int32)
    return b


def bpb_from_loss(val_loss_nats: float, n_tokens: int, n_bytes: int) -> tuple[float, float]:
    """官方公式：bpb = (loss/ln2) × (tokens/bytes)。返回 (bpb, tokens_per_byte)。"""
    bits_per_token = val_loss_nats / LN2
    tokens_per_byte = n_tokens / n_bytes
    return bits_per_token * tokens_per_byte, tokens_per_byte


# ══════════════════════════════════════════════════════════════
# 二、自检：字节记账必须对得上客观字节数
# ══════════════════════════════════════════════════════════════
def selfcheck_byte_accounting(sp, text: str, tol: int = 2) -> dict:
    """验证"字节记账"抄得对不对。**这是本模块存在的理由。**

    对照量不是 `len(text.encode())`（那是**原始文件**字节数），而是
    `len(sp.decode(sp.encode(text)).encode())`——**tokenizer 规范化之后**的字节数。
    两者可以差很多：本项目的语料是代码 + Markdown，含大量缩进，
    SentencePiece 会把连续空白折叠，实测 262,744 → 231,917 字节（折叠掉 30,827）。

    ⚠️ 还有一个必须说清楚的细节：**官方把 `<unk>` / control token 记为 0 字节**
    （`build_sentencepiece_luts()` 里 `is_boundary` 保持 True 且 `base_bytes` 为 0）。
    所以残差分析必须先把这些 token 的字节补回来，否则会把它误当 bug。

    本函数返回的 `residual_explained` 就是"补回 boundary token 之后"的残差。
    实测该项为 **−1 字节**，来自 SentencePiece 的 `add_dummy_prefix`——
    第 0 个 token 的前驱按边界处理，那一个前导空格官方不记。
    """
    ids = np.asarray(sp.encode(text, out_type=int), dtype=np.int32)
    if ids.size == 0:
        return {"ok": False, "note": "编码结果为空"}

    base_bytes, hl_space, is_bound = build_sentencepiece_luts(sp, int(sp.vocab_size()))
    size = max(base_bytes.size, int(ids.max()) + 1)
    if size > base_bytes.size:                   # 防御：补齐表长
        pad = size - base_bytes.size
        base_bytes = np.concatenate([base_bytes, np.zeros(pad, dtype=np.int16)])
        hl_space = np.concatenate([hl_space, np.zeros(pad, dtype=np.bool_)])
        is_bound = np.concatenate([is_bound, np.ones(pad, dtype=np.bool_)])

    sentinel = _pick_control_id(sp)              # 当作"前驱是边界"的哨兵
    tgt = ids                                    # 全部 token 都算（官方少算第 0 个，见 docstring）
    prev = np.concatenate([[sentinel], ids[:-1]]).astype(np.int32)   # 与 tgt 等长

    counted = int(token_bytes(tgt, prev, base_bytes, hl_space, is_bound).sum())
    decoded_bytes = len(sp.decode([int(t) for t in ids]).encode("utf-8"))

    bnd_mask = is_bound[ids]
    n_boundary = int(bnd_mask.sum())
    bnd_piece_bytes = sum(len(sp.id_to_piece(int(t)).encode("utf-8"))
                          for t in ids[bnd_mask])
    residual = counted + bnd_piece_bytes - decoded_bytes

    return {
        "tokens": int(ids.size),
        "counted_bytes": counted,
        "decoded_bytes": decoded_bytes,
        "raw_text_bytes": len(text.encode("utf-8")),
        "boundary_tokens": n_boundary,
        "boundary_piece_bytes": bnd_piece_bytes,
        # 官方口径 vs 规范化文本的原始差（会因为 unk 而显得很大，不是 bug）
        "diff_raw": counted - decoded_bytes,
        # 补回 boundary token 之后的残差 —— 这一项才是真正该看的
        "residual_explained": residual,
        "tokens_per_byte": ids.size / counted if counted else float("nan"),
        "ok": abs(residual) <= tol,
    }


def _pick_control_id(sp) -> int:
    """找一个 control token 当哨兵；没有就用 0（官方把 control 视作 boundary）。"""
    for tid in range(int(sp.vocab_size())):
        if sp.is_control(tid):
            return tid
    return 0


# ══════════════════════════════════════════════════════════════
# 三、官方分片格式（写/读），使数据能被官方脚本直接吃
# ══════════════════════════════════════════════════════════════
def write_shard(path, tokens: np.ndarray) -> None:
    """写官方 shard：256×int32 头 + uint16 tokens。

    头字段（train_gpt.py load_data_shard）：
        header[0] = 20240520 (magic)   header[1] = 1 (version)   header[2] = num_tokens
    """
    tokens = np.asarray(tokens, dtype=np.uint16)
    header = np.zeros(HEADER_INTS, dtype=np.int32)
    header[0] = SHARD_MAGIC
    header[1] = SHARD_VERSION
    header[2] = int(tokens.size)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.tobytes())


def read_shard(path) -> np.ndarray:
    header = np.fromfile(path, dtype="<i4", count=HEADER_INTS)
    if header.size != HEADER_INTS or int(header[0]) != SHARD_MAGIC or int(header[1]) != SHARD_VERSION:
        raise ValueError(f"Unexpected shard header for {path}")
    num_tokens = int(header[2])
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=HEADER_INTS * 4)
    if tokens.size != num_tokens:
        raise ValueError(f"Short read for {path}")
    return tokens
