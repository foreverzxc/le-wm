# 实验记录：白化层替代 SIGReg 作为 JEPA 防坍缩机制

## 背景

LeWM 使用 SIGReg（Sketched Isotropic Gaussian Regularizer）作为 anti-collapse 正则项，迫使隐空间服从各向同性高斯分布。核心问题是：这个正则项是**事后惩罚**——"先做，不行再罚"。是否可以从**架构层面**让隐空间天然就是各向同性的，从而不需要专门的惩罚项？

## 方案

### 白化层（WhiteningLayer）
在 encoder 的 project 后接一个可微分白化变换（基于 Cholesky/特征分解），**保证**输出协方差 = I。

$$
\text{encoder} \rightarrow \text{projector} \rightarrow \boxed{\text{WhiteningLayer}} \rightarrow \text{embedding}
$$

### 噪声注入（NoiseInjection）
训练时向隐空间注入各向同性高斯噪声 $z = h + \sigma\varepsilon$，推理时去掉。

## 实验设计

### Phase 1: 从零训练（小规模）

4 组对比实验，PushT 数据集，每 epoch 30 batch × batch_size=8：

| 实验 | SIGReg | 白化层 | 噪声 | 结果 |
|---|---|---|---|---|
| A: Baseline | ✅ | ❌ | ❌ | val_loss 0.075→0.005 ✅ |
| B: 白化层 | ❌ | ✅ (unfrozen) | ❌ | **发散** ❌（batch 统计量导致目标不稳定） |
| C: 噪声 | ❌ | ❌ | ✅ | val_loss 0.085→0.004 ✅ |
| D: 白化+噪声 | ❌ | ✅ (unfrozen) | ✅ | 发散 ❌ |

Phase 1 结论：
- 噪声注入可以工作，但与 SIGReg 效果相当（无本质优势）
- 白化层从零训练不稳定（batch 间协方差波动大 + B*T < D）

### Phase 2: 预训练模型 + 白化微调

从论文的预训练模型（100 epoch SIGReg）加载，附加白化层，仅用 MSE 微调。

#### 方案 2a: unfrozen encoder

| Epoch | val_loss |
|---|---|
| 0 (预训练+白化) | 0.767 |
| 1 | 0.003 |
| 3 | 0.0007 |
| 6 | 0.00010 |
| 10 | **0.00008** |

**各向同性**：cond#=3.2M（encoder 漂移导致预计算的白化统计量过时）

#### 方案 2b: frozen encoder

| Epoch | val_loss |
|---|---|
| 0 | 0.773 |
| 5 | 0.223 |
| 10 | **0.141** |

**各向同性**：cond#=32（良好！但预测器欠拟合）

## 最终验证：开环预测对比

200 个样本，10 步开环预测（使用真实帧作为 context）：

| 指标 | 预训练 (SIGReg) | 白化微调 (unfrozen) | 提升 |
|---|---|---|---|
| 单步 MSE | 0.006-0.011 | **0.000073-0.000078** | ~80-110× |
| 10步累积 MSE | 0.084 | **0.00074** | **113×** |
| 误差累积趋势 | 增长 ~1.7× | **持平不变** | **无累积** |

### 关键发现
1. 白化微调模型预测误差低 100 倍以上
2. 误差在 10 步内**不累积**——每一步预测精度一致
3. 白化后的隐空间去相关，预测被"对角化"

## 代码改动

### 新增模块
| 文件 | 改动 |
|---|---|
| `module.py` | 新增 `WhiteningLayer`（eigh 白化 + frozen mode）+ `NoiseInjection` |
| `jepa.py` | `JEPA.encode()` 嵌入白化层调用，兼容旧 checkpoint |
| `train.py` | 支持 4 种配置的 `lejepa_forward` |
| `config/train/lewm.yaml` | 新增 `whitening.enabled`、`noise.enabled` 配置 |

### 实验脚本
| 文件 | 用途 |
|---|---|
| `experiments/finetune_whitening.py` | 预训练模型加载 + 白化微调（支持 freeze-encoder） |
| `experiments/openloop_eval.py` | 开环预测对比评估 |
| `experiments/eval_planning.py` | MPC 规划评估（待跑通） |
| `experiments/compare_isotropy.py` | 各向同性指标计算 |
| `experiments/run_*.sh` | 批量运行辅助脚本 |

## Git 记录

```
bf04d3e — 原始基线
54ea2a3 — baseline: add isotropy analysis scripts
0624ee7 — feat: add WhiteningLayer + NoiseInjection
1dfa00b — experiments: 4-way comparison (Phase 1)
cf96afd — experiments: pre-trained + whitening fine-tune converges
7333b16 — validation: whitening fine-tune achieves 92x lower prediction error
a28af9d — validation: final comparison — 113x lower, zero error accumulation
```

## 下一步计划

### 短期
- [ ] **跑通 MPC 规划评估**：eval_planning.py 用的 CEM solver 单步太慢（需要完整 rollout）。可以：
  - 简化 solver 参数（num_samples=100, n_steps=2, topk=10）
  - 或者直接用预训练模型 + 白化层的组合来跑规划
- [ ] **更彻底的各向同性测试**：白化微调后 encoder 漂移导致 cond#=3.2M。可以试：
  - 周期性重算白化统计量（每 3 epoch 重新收集 embedding → update running stats）
  - 或者在微调的最后几个 epoch 冻结 encoder，只精调 predictor

### 中期
- [ ] **验证解码质量**：用 decoder 将 latent rollout 解码为像素，直观对比预测画面
- [ ] **多环境验证**：在 Reacher、TwoRooms、Cube 上复现实验
- [ ] **消融实验**：
  - 白化层在训练中的更新频率（每 batch vs 每 epoch vs 固定）
  - embedding 维度的选择（dim 64 vs 128 vs 192 vs 256）
  - 不同白化方法（Cholesky vs SVD vs per-dim 归一化）

### 长期
- [ ] **论文级实验**：完整跑 50 episode 的 MPC 规划，报告 success rate
- [ ] **物理量探测（Probing）**：验证白化后的 embedding 是否保留物理结构信息
- [ ] **Surprise Evaluation**：物理异常检测实验
