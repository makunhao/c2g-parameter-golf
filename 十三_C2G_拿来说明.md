# C2G · 拿来说明

> 作者：十三　｜　挑战：C2G 参数高尔夫
> 依据挑战原文：「**说明你拿了什么、改了什么、为什么改。这三句话就是你的工程能力证明。**」

---

## 一、起点（先把原物看清楚，不靠文档描述）

C2G 的 `CHALLENGE.md` 里贴了一份 `c2g-parameter-golf-starter/` 目录树——
**那份目录树是错的**（它列的是 `scripts/ingest.py`、`parse_problems.py` 之类，
属于 **C4C** 的 starter）。实际仓库完全不是那个样子。

所以我按**真实源码**理解。`parameter-golf-main.zip`，227 个文件，结构是：

```
parameter-golf-main/
├── train_gpt.py            ← 1126 行，唯一的训练脚本（单体）
├── train_gpt_mlx.py        ← Apple MLX 版
├── requirements.txt
├── data/
│   ├── cached_challenge_fineweb.py     ← 清单驱动的数据下载（--variant sp1024/sp4096）
│   ├── download_hf_docs_and_tokenize.py
│   └── tokenizer_specs.json
└── records/track_10min_16mb/           ← 33 份历史提交，**每份都带完整 train_gpt.py + train.log + README**
    ├── 2026-03-17_NaiveBaseline/       ← 1.2244 基线
    ├── …
    └── 2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/   ← 1.0810 榜首
```

> **最有价值的东西不是 `train_gpt.py`，是 `records/`。**
> 33 份提交里每一份都带自己的 `train_gpt.py` 和 `train.log`——
> **这等于给了 33 组"改动 → 结果"的对照实验数据，而且都有原始日志。**

### 1.1 从 `train_gpt.py` 读出的四件事（决定了我的选题）

| 读到的 | 影响 |
|---|---|
| **基线里已经有 Muon**（第 112 行起，含 Newton–Schulz 正交化） | **删掉了"上 Muon"这个候选方向**——想上 Muon 说明没读代码 |
| 已经有 tied embedding / RoPE / QK-gain(1.5) / logit softcap | 这些不是"改进点"，是基线的既有配置 |
| **硬依赖 CUDA**：`if not torch.cuda.is_available(): …`（752–754 行）+ 满篇 `torch.cuda.synchronize()` | **本机做不了任何官方跑**，只能自己复刻口径 |
| BPB 的定义在 `eval_val()`，字节记账在 `build_sentencepiece_luts()` | 这两段是 35% 分数的根，**必须抄准** |

---

## 二、拿了什么（逐条，附行号/出处）

| # | 拿的东西 | 出处 | 我怎么用 |
|---|---|---|---|
| 1 | **Newton–Schulz 正交化** | `train_gpt.py:96` `zeropower_via_newtonschulz5` | **逐行抄**，含系数 `(3.4445, -4.7750, 2.0315)` |
| 2 | **Muon 优化器** | `train_gpt.py:112` `class Muon` | **逐行抄**，含 Nesterov 与**尺度修正 `max(1, rows/cols)**0.5`** |
| 3 | **BPB 口径** | `train_gpt.py:219` `eval_val()` | 复刻成 `bpb.bpb_from_loss()`，公式一致 |
| 4 | **字节记账查表** | `train_gpt.py:180` `build_sentencepiece_luts()` | 复刻成 `bpb.build_sentencepiece_luts()`，**连 `<unk>` 记 0 字节这个细节都保留了** |
| 5 | **分片二进制格式** | `train_gpt.py:429` `load_data_shard()` | 复刻读写：256×int32 头（magic 20240520）+ uint16 tokens |
| 6 | **层结构**（RMSNorm / RoPE / GQA / tied embed / logit softcap） | `train_gpt.py` 的 `Block`/`Attention` 等 | 按机制复刻，尺寸缩小 |
| 7 | **超参组织方式** | `train_gpt.py:39` `class Hyperparameters`（全部走环境变量） | 直接沿用这个设计——**这也是"换 tokenizer 不用改代码"的原因** |
| 8 | **榜单的 33 组对照数据** | `records/*/README.md` + `train.log` | 作为**选题的证据基础**（见下） |
| 9 | **基线运行的确切指标** | `records/2026-03-17_NaiveBaseline/README.md` | 引用 1.2244 / `val_loss:2.0727` / `15,815,847 bytes` / `step_avg:43.54ms` |
| 10 | **数据下载器** | `data/cached_challenge_fineweb.py` | **读了，没用**（本机 HuggingFace 不通，且规则禁止联网） |

### 2.1 从 `records/` 里读到的、直接决定选题的两条

**第一条（正向）**：**BPB < 1.10 的 8 条，每一条都使用 SP4096 或 SP8192**；
vocab=1024 的条最高只到 1.1748。→ **这是选 tokenizer 方向的硬证据。**

**第二条（反向，更让我警醒）**：1.0979 那份（`2026-04-01_Vocab4096_MLPMult4_WD085`）
的 README 里，第一条写的不是"我做了什么改进"，而是：

> ### Fixes
> *Fixed a small bug in the sliding window evaluation causing it to score tokens
> at the end of the val dataset multiple times. This bug didn't significantly affect
> results: it added roughly 2k duplicate contributions to the total loss and byte
> counts over a validation set of about 6M tokens.*

**——上榜级别的提交，也在 BPB 的字节/词数记账上栽过跟头。**

> 这条直接改变了我做事的顺序：**先把口径抄准并自检，再谈实验。**
> 而它的收益也证实了我的判断——我的自检确实抓出了 **30,848 字节**的偏差
> （虽然根因有两层是我的问题，见《AI日志》Round 2）。

---

## 三、改了什么（以及为什么改）

### 3.1 关于模型本身：**一个字没改**

严格说，**本方案对模型的改动是零**。这正是选 tokenizer 方向的理由之一：

```python
# train_gpt.py 从环境变量读，所以换 tokenizer 不需要改一行代码
vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
```

正式提交要做的只是：

```bash
VOCAB_SIZE=4096 TOKENIZER_PATH=.../fineweb_4096_bpe.model torchrun --nproc_per_node=8 train_gpt.py
```

**"不需要改代码"不是偷懒，是降低风险。** 在 10 分钟、16MB、$25 的约束下，
每一次跑都很贵，**改动面越小，失败面越小**。

### 3.2 我真正改的，是"能不能在本机上验证"这件事

| 改动 | 为什么 |
|---|---|
| **把 BPB 口径抽成独立模块并加恒等式自检** | 官方没有这个自检。加了之后，我才能在没 GPU 时仍然对自己的口径有底气 |
| **写一套 CPU 微型台架** | 官方脚本硬依赖 CUDA（752–754 行），本机跑不了。**不搭台架就一个数字都拿不到** |
| **Muon 的 bf16 缓冲改成 fp32** | CPU 上 bf16 更慢且无收益；尺度修正、Nesterov、正交化系数**一字未动** |
| **去掉 DDP / grad_accum / torch.compile** | 单进程 CPU 用不上 |
| **不启用 bf16 autocast** | CPU 上无收益 |
| **不做 int8+zlib 量化与打包** | 本台架不测 16MB 大小，测也测不出 |

### 3.3 我**没有**拿的东西（如实列出）

| 没拿 | 为什么 |
|---|---|
| `train_gpt_mlx.py` | 本机不是 Apple Silicon |
| 数据下载链路（`cached_challenge_fineweb.py`） | HuggingFace 不通 + 规则禁止联网 |
| 3-Layer Depth Recurrence | 需要 GPU 迭代验证；列为兜底方向（见《方案草案》第四节） |
| Parallel Residuals / XSA / GPTQ / Legal TTT / Partial RoPE | 同上——**收益都在 −0.01~−0.03 量级，但没有一个能在本机验证** |
| LZMA 代码包装 | 属 artifact 打包，本台架不测大小 |

> **不拿的理由只有一条：拿了也验证不了。** 与其在文档里罗列一堆抄来的技术名词，
> 不如把力气花在"能真正测出来"的那一件事上。

### 3.4 与榜首方案（1.0810）的差距：摊开说

| | 榜首（`2026-04-09_…LegalTTT`） | 我的方案 |
|---|---|---|
| tokenizer | SP8192 | SP4096/8192 |
| 量化 | GPTQ SDClip（int6 矩阵 + int8 嵌入） | 未做 |
| 架构 | 11L×512d/8H/4KV，MLP **4x**，LeakyReLU(0.5)² | 保持基线 |
| 深度循环 | **3-layer recurrence（11 物理层 → 17 虚拟层）** | 未做 |
| 残差 | parallel residuals（layer 7+） | 未做 |
| QK-gain | **5.25**（从 4.0 单调调到 5.25） | 保持 1.5 |
| 测试时训练 | Legal Score-First TTT | 未做 |
| 3-seed std | **0.0002** | 见《方案设计》第四节 |
| artifact | 15,992,694 字节（贴着 16MB） | 未做打包 |

**差距是 7 个技术点，不是 1 个。** 我的方案只动其中 1 个（tokenizer），
目标定在 Level 2（< 1.18）。**这是算力预算决定的，不是认知决定的。**

---

## 四、为什么这个"拿来"是有效的

挑战原文说：**"严禁从零写 Transformer"**。我核对了一下自己有没有真的做到：

| 组件 | 我是从零写的吗 |
|---|---|
| Newton–Schulz / Muon | ❌ 逐行抄官方（含三处易漏细节） |
| BPB 口径与字节记账 | ❌ 逐行抄官方（含 `<unk>` 记 0 字节这个反直觉细节） |
| 分片格式 | ❌ 按官方 `load_data_shard` 复刻 |
| 层结构（RoPE/GQA/tied/softcap） | ❌ 按官方机制复刻，只缩尺寸 |
| 超参组织（环境变量） | ❌ 沿用官方 `Hyperparameters` 的设计 |
| **选题的依据** | ❌ 来自 `records/` 里 33 组真实对照，不是我的直觉 |
| 微型台架的胶水代码 | ✅ 这部分是我写的 |

> **"拿来"的质量不体现在"我抄了多少行"，而体现在
> "我知道每一行为什么在、漏掉哪一行会怎样"。**
> Muon 的尺度修正那一行，我特意标了注释——**漏了它 Muon 就不是 Muon**。

---

## 五、一句话总结

| | |
|---|---|
| **拿了** | Muon 全套（含尺度修正）、BPB 口径与字节记账、分片格式、层结构与超参组织、**以及 `records/` 里 33 组带日志的对照实验** |
| **改了** | 模型改动为零；改的是"让它在 CPU 上可验证"——抽出口径模块 + 加恒等式自检 + 搭微型台架 |
| **没拿** | 量化、深度循环、parallel residual、TTT、XSA 等——**理由只有一个：拿了也验证不了** |
| **核心判断** | 挑战说"不奖励从零开始"，我补一句：**也不奖励抄了一堆却一个都没验证** |
