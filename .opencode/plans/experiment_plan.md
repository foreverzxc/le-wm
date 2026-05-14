# 4组实验实施计划

## Phase 0: Baseline Commit

1. 更新 `.gitignore`：添加 `data/`、`lewm/`
2. `git add` 所有代码文件 + 分析报告
3. `git commit -m "baseline: isotropy analysis and experiment scripts"`

## Phase 1: 代码修改

### 需要改的文件

#### 1. `module.py` — 新增两个类

```python
class WhiteningLayer(nn.Module):
    """可微分白化层：保证输出协方差 = I
    
    训练时用 batch 统计量做 Cholesky 白化，
    推理时用 running 统计量（与 BatchNorm 一致）。
    """
    def __init__(self, dim, momentum=0.9, eps=1e-6):
        # register_buffer: running_mean (dim,), running_cov (dim, dim)
        # running_cov 初始化为 I，running_mean 初始化为 0
    
    def forward(self, x):
        # x: (B, T, D)
        # 训练: batch mean/cov → Cholesky Σ=LL^T → z = L^{-1}(x-μ)
        # 推理: z = L_run^{-1}(x - μ_run)
```

```python
class NoiseInjection(nn.Module):
    """向隐空间注入各向同性高斯噪声"""
    def __init__(self, std=0.1):
        self.std = std
    
    def forward(self, x):
        if self.training:
            return x + torch.randn_like(x) * self.std
        return x  # 推理时不加噪声
```

#### 2. `jepa.py` — encode 中加白化

```python
class JEPA(nn.Module):
    def __init__(self, encoder, predictor, action_encoder, projector=None, 
                 pred_proj=None, whitening=None):
        super().__init__()
        ...
        self.whitening = whitening or nn.Identity()
    
    def encode(self, info):
        ...
        emb = self.projector(pixels_emb)
        emb = self.whitening(emb)  # 白化（如果启用）
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)
        ...
```

#### 3. `train.py` — 支持4种配置

核心改动在 `lejepa_forward`：

```python
def lejepa_forward(self, batch, stage, cfg):
    ...
    output = self.model.encode(batch)
    emb = output["emb"]
    
    # 噪声注入（在 predictor 之前）
    if cfg.noise.enabled:
        noise_layer = self.noise_injection
        emb = noise_layer(emb)
    
    # SIGReg 损失（基线）
    if cfg.loss.sigreg.enabled:
        output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
        output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]
    else:
        output["loss"] = output["pred_loss"]
    ...
```

构造模型时：

```python
whitening = WhiteningLayer(cfg.wm.embed_dim) if cfg.whitening.enabled else None
noise_injection = NoiseInjection(cfg.noise.std) if cfg.noise.enabled else None

world_model = JEPA(
    ..., 
    whitening=whitening,
)

world_model = spt.Module(
    model=world_model,
    sigreg=SIGReg(...) if cfg.loss.sigreg.enabled else None,
    noise_injection=noise_injection,
    forward=partial(lejepa_forward, cfg=cfg),
)
```

#### 4. `config/train/lewm.yaml` — 新增配置项

```yaml
loss:
  sigreg:
    enabled: true    # 开关 SIGReg
    weight: 0.09
    kwargs:
      knots: 17
      num_proj: 1024

whitening:
  enabled: false     # 开关白化层

noise:
  enabled: false     # 开关噪声注入
  std: 0.1           # 噪声标准差
```

### 4 组实验的配置

| 实验 | loss.sigreg.enabled | whitening.enabled | noise.enabled | 文件名后缀 |
|---|---|---|---|---|
| A: Baseline | true | false | false | baseline |
| B: +白化 | false | true | false | whitening |
| C: +噪声 | false | false | true | noise |
| D: +白化+噪声 | false | true | true | whitening_noise |

## Phase 2: 小规模训练

### 缩规模方式

在 `config/train/data/pusht.yaml` 或命令行覆盖：

```bash
# 小规模：减少 epoch + 用部分数据
python train.py \
    trainer.max_epochs=20 \           # 默认100
    loader.batch_size=64 \            # 默认128
    # 或者用 subset
```

### 运行命令

```bash
# A: Baseline
python train.py output_model_name=lewm_baseline

# B: 白化
python train.py output_model_name=lewm_whitening \
    loss.sigreg.enabled=false \
    whitening.enabled=true

# C: 噪声
python train.py output_model_name=lewm_noise \
    loss.sigreg.enabled=false \
    noise.enabled=true \
    noise.std=0.1

# D: 白化+噪声
python train.py output_model_name=lewm_whitening_noise \
    loss.sigreg.enabled=false \
    whitening.enabled=true \
    noise.enabled=true \
    noise.std=0.1
```

## Phase 3: 评估

每组训练完后：

```bash
# 各向同性测试
python test_isotropy.py --model data/pusht/lewm_*_object.ckpt --output-dir isotropy_results/*/

# 下游任务评估
python eval.py ...  
```

### 对比指标

- 训练损失曲线（MSE 是否收敛，是否有 collapse 信号）
- 特征值谱（有效维度、条件数）
- 预测精度（MSE）
- 下游 planning success rate（可选，小规模可跳过）
