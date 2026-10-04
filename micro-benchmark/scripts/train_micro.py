#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_micro.py —— CPU 上的微型训练台架（复刻官方 BPB 口径）

它是什么 / 不是什么
-------------------
**是**：一个把官方 `train_gpt.py` 的核心机制按比例缩到 CPU 能跑的程度、
并在**同一套 BPB 口径**下测量的小台架。用途是**在花 $25 算力之前排掉错假设**。

**不是**：官方榜单成绩。语料不是 FineWeb、模型不是 9L/512d、硬件不是 8×H100。
**本台架上的 BPB 数字不能拿去和 1.2244 / 1.0810 比。**

从官方 `train_gpt.py` 保留的机制（不是随手写的 toy）
---------------------------------------------------
· tied embedding（输入输出共享权重）
· RoPE + QK-gain
· logit softcap（tanh 软截断）
· RMSNorm
· **Muon 优化器**（矩阵参数）+ AdamW（其余），与官方一致
· BPB 口径：`(val_loss/ln2) × (tokens/bytes)`，字节数走 SentencePiece LUT
· 分片读取用官方的 256×int32 头格式

刻意简化的地方（如实列出）
--------------------------
· 单进程 CPU，无 DDP / 无 grad_accum / 无 torch.compile
· bf16 autocast 不启用（CPU 上无收益）
· 不做 int8+zlib 的训练后量化（那是 16MB 约束下的事，本台架不测大小）
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np                                       # noqa: E402
import sentencepiece as spm                              # noqa: E402
import torch                                             # noqa: E402
import torch.nn.functional as F                          # noqa: E402
from torch import Tensor, nn                             # noqa: E402

import bpb as B                                          # noqa: E402
from muon import Muon                                    # noqa: E402


# ══════════════════════════════════════════════════════════════
# 模型
# ══════════════════════════════════════════════════════════════
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class Rotary(nn.Module):
    """RoPE，与官方同构（缓存 cos/sin，按 seq_len 生成）。"""

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)
        self._cos = None
        self._sin = None

    def forward(self, seq_len: int, device, dtype):
        if self._cos is None or self._cos.shape[0] < seq_len or self._cos.device != device:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos, self._sin = freqs.cos().to(dtype), freqs.sin().to(dtype)
        return self._cos[:seq_len], self._sin[:seq_len]


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # x: (B, H, T, D)
    x1, x2 = x[..., ::2], x[..., 1::2]
    c = cos[None, None, :, :]
    s = sin[None, None, :, :]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)


class Attention(nn.Module):
    """GQA：num_heads 个 Q 头，num_kv_heads 个 KV 头（官方 baseline 是 8/4）。"""

    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, qk_gain: float):
        super().__init__()
        assert dim % num_heads == 0
        self.h = num_heads
        self.kv = num_kv_heads
        self.hd = dim // num_heads
        self.q = nn.Linear(dim, num_heads * self.hd, bias=False)
        self.k = nn.Linear(dim, num_kv_heads * self.hd, bias=False)
        self.v = nn.Linear(dim, num_kv_heads * self.hd, bias=False)
        self.o = nn.Linear(num_heads * self.hd, dim, bias=False)
        self.qk_gain = nn.Parameter(torch.tensor(float(qk_gain)))

    def forward(self, x: Tensor, rope: Rotary) -> Tensor:
        Bz, T, D = x.shape
        q = self.q(x).view(Bz, T, self.h, self.hd).transpose(1, 2)
        k = self.k(x).view(Bz, T, self.kv, self.hd).transpose(1, 2)
        v = self.v(x).view(Bz, T, self.kv, self.hd).transpose(1, 2)
        cos, sin = rope(T, x.device, q.dtype)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if self.kv != self.h:
            rep = self.h // self.kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q * self.qk_gain, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(Bz, T, D))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int,
                 qk_gain: float):
        super().__init__()
        self.n1 = RMSNorm(dim)
        self.attn = Attention(dim, num_heads, num_kv_heads, qk_gain)
        self.n2 = RMSNorm(dim)
        hid = int(dim * mlp_mult)
        self.mlp = nn.Sequential(nn.Linear(dim, hid, bias=False), nn.GELU(),
                                 nn.Linear(hid, dim, bias=False))

    def forward(self, x: Tensor, rope: Rotary) -> Tensor:
        x = x + self.attn(self.n1(x), rope)
        return x + self.mlp(self.n2(x))


class MicroGPT(nn.Module):
    def __init__(self, vocab: int, dim: int, layers: int, heads: int, kv_heads: int,
                 mlp_mult: int, seq_len: int, qk_gain: float, softcap: float,
                 tie_embeddings: bool = True):
        super().__init__()
        self.seq_len = seq_len
        self.softcap = softcap
        self.tok = nn.Embedding(vocab, dim)
        self.rope = Rotary(dim // heads)
        self.blocks = nn.ModuleList(
            [Block(dim, heads, kv_heads, mlp_mult, qk_gain) for _ in range(layers)])
        self.norm_f = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        if tie_embeddings:
            self.head.weight = self.tok.weight

    def forward(self, x: Tensor, y: Tensor | None = None):
        h = self.tok(x)
        for blk in self.blocks:
            h = blk(h, self.rope)
        h = self.norm_f(h)
        logits = self.head(h)
        if self.softcap > 0:
            logits = self.softcap * torch.tanh(logits / self.softcap)
        if y is None:
            return logits
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))


# ══════════════════════════════════════════════════════════════
# 学习率调度（warmup + cos 式 warmdown），与官方同风格
# ══════════════════════════════════════════════════════════════
def lr_at(step: int, total: int, base_lr: float, warmup: int, warmdown: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    if step >= total - warmdown:
        remain = max(0, total - step)
        return base_lr * remain / max(1, warmdown)
    return base_lr


# ══════════════════════════════════════════════════════════════
# 评估：走官方 BPB 口径
# ══════════════════════════════════════════════════════════════
@torch.inference_mode()
def eval_bpb(model: MicroGPT, val_tokens: np.ndarray, luts, seq_len: int,
             max_seqs: int) -> tuple[float, float, float]:
    base_bytes, hl, ib = luts
    model.eval()
    n = ((val_tokens.size - 1) // seq_len)
    n = min(n, max_seqs)
    loss_sum, tok_cnt, byte_cnt = 0.0, 0, 0
    for i in range(n):
        s = i * seq_len
        chunk = val_tokens[s:s + seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1]).unsqueeze(0)
        y = torch.from_numpy(chunk[1:]).unsqueeze(0)
        loss = model(x, y)
        loss_sum += float(loss) * y.numel()
        tok_cnt += int(y.numel())
        prev = x.reshape(-1).numpy().astype(np.int32)
        tgt = y.reshape(-1).numpy().astype(np.int32)
        byte_cnt += int(B.token_bytes(tgt, prev, base_bytes, hl, ib).sum())
    val_loss = loss_sum / tok_cnt
    bpb, tpb = B.bpb_from_loss(val_loss, tok_cnt, byte_cnt)
    model.train()
    return val_loss, bpb, tpb


# ══════════════════════════════════════════════════════════════
# 单次训练
# ══════════════════════════════════════════════════════════════
def run_one(cfg: dict, train_tokens: np.ndarray, val_tokens: np.ndarray,
            luts, vocab: int, seed: int, log_every: int = 50) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MicroGPT(vocab=vocab, dim=cfg["dim"], layers=cfg["layers"],
                     heads=cfg["heads"], kv_heads=cfg["kv_heads"],
                     mlp_mult=cfg["mlp_mult"], seq_len=cfg["seq_len"],
                     qk_gain=cfg["qk_gain"], softcap=cfg["softcap"],
                     tie_embeddings=cfg["tie_embeddings"])
    n_params = sum(p.numel() for p in model.parameters())

    # 参数分组：矩阵 → Muon，其余 → AdamW（与官方 baseline 的设计一致）
    muon_params, adam_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and "tok" not in name and "head" not in name:
            muon_params.append(p)
        else:
            adam_params.append(p)
    opts = []
    if muon_params:
        opts.append(Muon(muon_params, lr=cfg["muon_lr"], momentum=cfg["muon_momentum"]))
    opts.append(torch.optim.AdamW(adam_params, lr=cfg["adam_lr"],
                                  betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0))

    seq_len, bsz = cfg["seq_len"], cfg["batch_seqs"]
    total, warmup, warmdown = cfg["iters"], cfg["warmup"], cfg["warmdown"]
    tokens_per_step = seq_len * bsz
    max_start = train_tokens.size - seq_len - 1

    t0 = time.time()
    losses: list[float] = []
    for step in range(total):
        idx = np.random.randint(0, max_start, size=bsz)
        batch = np.stack([train_tokens[i:i + seq_len + 1] for i in idx]).astype(np.int64)
        x = torch.from_numpy(batch[:, :-1])
        y = torch.from_numpy(batch[:, 1:])
        for opt in opts:
            opt.zero_grad(set_to_none=True)
        loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in opts:
            if isinstance(opt, Muon):
                for g in opt.param_groups:
                    g["lr"] = cfg["muon_lr"] * lr_at(step, total, 1.0, warmup, warmdown)
            else:
                for g in opt.param_groups:
                    g["lr"] = cfg["adam_lr"] * lr_at(step, total, 1.0, warmup, warmdown)
            opt.step()
        losses.append(float(loss))
        if log_every and (step % log_every == 0 or step == total - 1):
            print(f"      step {step:5d}/{total}  loss {float(loss):.4f}  "
                  f"lr {cfg['adam_lr'] * lr_at(step, total, 1.0, warmup, warmdown):.2e}")
    train_s = time.time() - t0

    val_loss, bpb, tpb = eval_bpb(model, val_tokens, luts, seq_len, cfg["eval_seqs"])
    return {
        "seed": seed, "vocab": vocab, "n_params": n_params,
        "tokens_per_step": tokens_per_step, "steps": total,
        "train_seconds": round(train_s, 1),
        "ms_per_step": round(1000 * train_s / total, 2),
        "final_train_loss": round(losses[-1], 5),
        "val_loss_nats": round(val_loss, 5),
        "val_bpb": round(bpb, 5),
        "tokens_per_byte": round(tpb, 5),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="CPU 微型训练台架（官方 BPB 口径）")
    ap.add_argument("--data", default=str(HERE.parent / "data"))
    ap.add_argument("--vocab", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = Path(args.data)
    model_file = data / "tokenizers" / f"local_bpe_{args.vocab}.model"
    sp = B.load_sentencepiece(model_file)   # 绕开 Windows 非 ASCII 路径
    train_tokens = B.read_shard(data / f"local_train_{args.vocab}.bin").astype(np.int32)
    val_tokens = B.read_shard(data / f"local_val_{args.vocab}.bin").astype(np.int32)
    luts = B.build_sentencepiece_luts(sp, args.vocab)
    print(f"[micro] vocab={args.vocab}  train_tokens={train_tokens.size:,}  "
          f"val_tokens={val_tokens.size:,}")

    cfg = dict(dim=128, layers=4, heads=4, kv_heads=2, mlp_mult=2, seq_len=256,
               qk_gain=1.5, softcap=30.0, tie_embeddings=True,
               batch_seqs=32, iters=args.iters, warmup=20, warmdown=args.iters // 5,
               muon_lr=0.02, muon_momentum=0.95, adam_lr=0.003, eval_seqs=64)
    res = run_one(cfg, train_tokens, val_tokens, luts, args.vocab, args.seed)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
