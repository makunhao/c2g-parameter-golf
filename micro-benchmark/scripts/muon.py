#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
muon.py —— Muon 优化器（逐行对照官方 train_gpt.py 第 96–167 行）

为什么要把官方实现抄一遍而不是写个"差不多的"
------------------------------------------------
Muon 在这个比赛里不是可选项——**官方基线用的就是它**。
如果我的微型台架里 Muon 是另一个东西，那台架上的结论就迁移不到正式跑。
所以这里保持官方的三处关键细节：

1. **Newton–Schulz 正交化**：`zeropower_via_newtonschulz5`，系数 (3.4445, -4.7750, 2.0315)，
   5 次迭代（官方默认 backend_steps=5）。
2. **Nesterov 动量**：`g = g + momentum * buf`。
3. **尺度修正**：`g *= max(1, rows/cols) ** 0.5`。
   —— 这一行最容易被漏掉，而它决定了 Muon 的等效学习率，
   漏了就等于换了个优化器。

相对官方的简化（如实标注）
--------------------------
· 去掉 DDP 相关的 `all_reduce` 与 rank 切分（本台架单进程）
· bf16 缓冲改成 fp32（CPU 上 bf16 反而更慢且无收益）
· 去掉 `updates_flat` 的扁平化搬运，直接原地更新每个参数
"""

from __future__ import annotations

import torch
from torch import Tensor


def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """用快速 Newton–Schulz 迭代把一个二维更新矩阵正交化。

    系数与官方完全一致：a, b, c = (3.4445, -4.7750, 2.0315)。
    官方把中间量转成 bfloat16（GPU 上省显存+加速）；这里保持 fp32（见模块 docstring）。
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    X = X / (X.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    """Muon：对矩阵参数做正交化更新，对非矩阵参数不适用（应配 AdamW）。"""

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 backend_steps: int = 5, nesterov: bool = True):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      backend_steps=backend_steps, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim != 2:
                    raise ValueError("Muon 只接受二维参数；其余参数请交给 AdamW")

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if nesterov:
                    g = g.add(buf, alpha=momentum)

                g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                # ★ 尺度修正（官方原句）——漏掉这一行就等于换了个优化器
                g = g * (max(1, g.size(0) / g.size(1)) ** 0.5)

                p.add_(g.to(dtype=p.dtype), alpha=-lr)
        return loss
