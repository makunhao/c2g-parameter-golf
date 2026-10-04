#!/usr/bin/env bash
# 十三_C2G_运行脚本_官方提交.sh
#
# ⚠️ 本脚本**没有在本机执行过**——本机无 CUDA（官方 train_gpt.py 会直接退出）。
#    它是"拿到 8×H100 后要跑什么"的书面记录，不是我跑过的证据。
#
# 核心事实：**换 tokenizer 不需要改 train_gpt.py 一行代码。**
#          脚本从环境变量读 VOCAB_SIZE 与 TOKENIZER_PATH。
#
# 用法（在租的 8×H100 实例上）：
#   bash 十三_C2G_运行脚本_官方提交.sh baseline   # 复现基线，验证环境
#   bash 十三_C2G_运行脚本_官方提交.sh sp4096 1337
#   bash 十三_C2G_运行脚本_官方提交.sh sp8192 1337
#
set -euo pipefail

REPO=${REPO:-/root/code/parameter-golf}
VARIANT=${1:-baseline}
SEED=${2:-1337}

cd "$REPO"

case "$VARIANT" in
  baseline)
    export VOCAB_SIZE=1024
    export DATA_PATH=$REPO/data/datasets/fineweb10B_sp1024
    export TOKENIZER_PATH=$REPO/data/tokenizers/fineweb_1024_bpe.model
    ;;
  sp4096)
    export VOCAB_SIZE=4096
    export DATA_PATH=$REPO/data/datasets/fineweb10B_sp4096
    export TOKENIZER_PATH=$REPO/data/tokenizers/fineweb_4096_bpe.model
    ;;
  sp8192)
    export VOCAB_SIZE=8192
    export DATA_PATH=$REPO/data/datasets/fineweb10B_sp8192
    export TOKENIZER_PATH=$REPO/data/tokenizers/fineweb_8192_bpe.model
    ;;
  *) echo "unknown variant: $VARIANT (baseline|sp4096|sp8192)"; exit 2 ;;
esac

# ── 官方基线的其余配置（来自 records/2026-03-17_NaiveBaseline/README.md） ──
export NUM_LAYERS=9
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=2
export TIE_EMBEDDINGS=1
export TIED_EMBED_LR=0.05
export TRAIN_BATCH_TOKENS=524288
export TRAIN_SEQ_LEN=1024

# ── 提交硬约束 ──
export MAX_WALLCLOCK_SECONDS=600     # 10 分钟
export SEED=$SEED
export RUN_ID="c2g_${VARIANT}_seed${SEED}"

# ── 日志（必须留，评审要 3 个 seed 的完整日志） ──
mkdir -p "$REPO/logs"
LOG=$REPO/logs/${RUN_ID}.log

echo "[run] variant=$VARIANT seed=$SEED vocab=$VOCAB_SIZE -> $LOG"
NCCL_IB_DISABLE=1 \
torchrun --standalone --nproc_per_node=8 "$REPO/train_gpt.py" 2>&1 | tee "$LOG"

# ── 跑完立刻记下 6 个必看指标（见《方案设计》3.3） ──
# 1 val_bpb   2 val_loss   3 tokens/byte   4 step_avg + 实际步数
# 5 artifact 字节数        6 peak memory
echo "[run] 已记录日志：$LOG"
echo "[run] 请从日志里抄出上面 6 个指标填进 十三_C2G_leaderboard.md"
