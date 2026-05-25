# LeWM 实验结论

> 区别于实验日志（logs/），本文档只记录经过验证的结论性发现。

---

## 1. 跨数据集惊讶度（Surprise）分析

**实验日期**: 2026-05-22
**方法**: 用 4 个预训练模型（Pusht, Cube, Reacher, TwoRooms），分别在 4 个数据集上用两种方式计算惊讶度：
- **With-action**: 使用真实 action（仅在 action 维度匹配时有效）
- **Without-action**: 使用零向量 action（全矩阵，纯视觉 domain gap）

### 1.1 Action 维度匹配情况

| | Pusht(10D) | Cube(25D) | Reacher(10D) | TwoRooms(10D) |
|---|---|---|---|---|
| Pusht | ✓ | ✗ | ✓ | ✓ |
| Cube | ✗ | ✓ | ✗ | ✗ |
| Reacher | ✓ | ✗ | ✓ | ✓ |
| TwoRooms | ✓ | ✗ | ✓ | ✓ |

Cube 数据集 action 维度（25D）与其他所有数据集（10D）不同，因此 Cube 在 with-action 表中只能与自身比较。

### 1.2 With-Action 惊讶度矩阵

| Model \ Dataset | PUSHT | CUBE | REACHER | TWOROOMS |
|-----------------|-------|------|---------|----------|
| PUSHT | 0.1612 | · | 0.0737 | **0.0211** |
| CUBE | · | 0.2040 | · | · |
| REACHER | 0.2634 | · | 0.1144 | 0.4661 |
| TWOROOMS | 2.7144 | · | 6.8519 | 0.2229 |

**发现**:
- Pusht 模型在 TwoRooms 数据上的惊讶度（0.02）低于在自己数据集上的惊讶度（0.16），说明 TwoRooms 的视觉场景对 Pusht 模型来说非常容易预测
- Reacher 模型跨数据集泛化最差（off-diag 全部高于自身）

### 1.3 Without-Action 惊讶度矩阵（纯视觉）

| Model \ Dataset | PUSHT | CUBE | REACHER | TWOROOMS | AVG |
|-----------------|-------|------|---------|----------|-----|
| PUSHT | 0.23 | 0.08 | 0.07 | 0.02 | 0.10 |
| CUBE | 0.14 | 0.39 | 0.10 | 0.14 | 0.19 |
| REACHER | 0.23 | 0.18 | 0.45 | 0.23 | 0.27 |
| TWOROOMS | 2.70 | 4.92 | 6.66 | 1.45 | 3.93 |

### 1.4 Action 贡献度（Δ = no-action surprise − with-action surprise）

在对角线上（模型在自己的数据集上），action 贡献越大说明该任务越依赖动作信号：

| 模型 | With-action | No-action | Δ | 解读 |
|------|------------|-----------|-----|------|
| Pusht | 0.1612 | 0.2306 | **+0.07** | 视觉信号主导，action 帮助有限 |
| Cube | 0.2040 | 0.3900 | **+0.19** | action 有一定帮助 |
| Reacher | 0.1144 | 0.4505 | **+0.34** | action 显著改善预测 |
| TwoRooms | 0.2229 | 1.4502 | **+1.23** | **action 至关重要**，导航任务中动作决定方向 |

### 1.5 数据集复杂度排名

按所有模型在该数据集上的平均惊讶度排序（without-action，越高越复杂）:

1. **Reacher** (1.82) — 机械臂运动，视觉变化复杂
2. **Cube** (1.39) — 旋转方块
3. **Pusht** (0.83) — 推物体
4. **TwoRooms** (0.46) — 最简单的视觉场景（灰白方格）

### 1.6 重要注意事项

- **跨模型 surprise 不可直接比较**：TwoRooms 模型的 embedding 空间尺度与其他模型不同（surprise 2-7 vs 0.02-0.45），绝对值没有可比性
- **同一模型跨数据集比较是有效的**：因为模型权重固定，surprise 差异反映数据集的视觉难度
- **With-action 和 without-action 的差异衡量了 action 对预测的贡献**：Δ 越大，说明该任务越依赖动作信息

---

## 2. LIBERO-50 收敛实验

**实验日期**: 2026-05-22
**脚本**: `make libero-small`
**最佳 checkpoint**: `~/.stable_worldmodel/checkpoints/lewm_weights_epoch_6.pt`

### 2.1 有效超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| lr | 2e-5 | 5e-5 会导致梯度 spike |
| SIGReg λ | 0.05 | 低于默认 0.09，允许多任务数据更好拟合 |
| grad_clip | 0.5 | 低于默认 1.0，防止多任务梯度爆炸 |
| batch_size | 4 | 多任务最小稳定 batch（RTX 3050 上限） |

### 2.2 收敛结果

- fit/pred_loss: 2.64e-4 → 1.70e-5（15.5× 下降，6 epochs）
- 零 spike，单调收敛
- 第 1 个 epoch 完成 90% 的收敛

### 2.3 对比：失败的超参数

lr=5e-5, SIGReg λ=0.09, grad_clip=1.0, batch=2:
- Epoch 14 和 17 出现 100× loss spike
- 训练不稳定，无法收敛

---

## 3. PushT 训练时间估算

- 数据集: 18.7K episodes, 467K samples
- batch=2: ~32 分钟/epoch（RTX 3050 6GB 保守配置）
- batch=8: ~8 分钟/epoch
- batch=128（原始默认）: ~2 分钟/epoch，但 RTX 3050 6GB 可能 OOM

---

## 4. 输出目录规范

所有自动生成文件输出到以下目录：
- `output/` — Lightning logs, Hydra runs, 报告, 图表
- `logs/train/` — 训练日志（时间戳文件名）
- `logs/infer/` — 推理/评估日志（时间戳文件名）
- `~/.stable_worldmodel/checkpoints/` — 模型权重

---

## 5. 跨数据集 World Model（实验 1）

**实验日期**: 2026-05-22 ~ 2026-05-25
**方法**: Pusht + Cube + Reacher + TwoRooms 4 数据集联合训练，action 统一 zero-pad 到 25D（Cube 最大）
**代码**: `multidata.py`, `config/train/data/cross4.yaml`, `make cross4`

### 5.1 收敛情况（3 epochs，RTX 3050）

| Epoch | fit/pred_loss | val/pred_loss | 耗时 |
|-------|---------------|---------------|------|
| 0 | 0.000371 | 0.000714 | ~20h |
| 1 | 0.000143 | 0.000748 | ~20h |
| 2 | 0.000087 | 0.001161 | ~44h |

- fit/pred_loss 持续下降（4.3×），跨数据集训练可行
- **瓶颈**：6.35M 样本 × batch=8，RTX 3050 ~20h/epoch，需要服务器 GPU
- Epoch 2 耗时翻倍可能是系统过热降频

### 5.2 Pipeline 验证

- MultiDomainDataset 正确加载 4 个数据集，统一 action 到 25D
- 不同数据集的 key 不一致问题已通过只返回 `pixels` + `action` 解决
- DataLoader collation 正常
- 训练 forward/backward 无误

### 5.3 后续

- 挪到服务器（A100/4090）跑完整 10 epoch
- 可增大 batch_size 减少 epoch 时间
- Epoch 2 val_loss 回升可能需要 weighted sampling 平衡数据集

