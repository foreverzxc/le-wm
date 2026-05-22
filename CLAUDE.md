# CLAUDE.md

LeWorldModel (LeWM) — JEPA world model learning end-to-end from pixels. ~18M params, single GPU.

## Quick Start

```bash
# All experiments via Makefile
make help           # List all targets
make overfit        # 1-episode sanity check (30s init, 1min/ep)
make libero-small   # 50 episodes, fast convergence test
make libero-10      # Full LIBERO-10 (500 episodes)
make pusht          # PushT benchmark
make info           # GPU + dataset status

# Manual training
PYTHONUNBUFFERED=1 python train.py data=pusht
```

## Project Structure

```
le-wm/
├── train.py              # Training entry point (Hydra)
├── eval.py               # Evaluation entry point
├── jepa.py               # JEPA model (encode, predict, rollout, get_cost)
├── module.py             # Building blocks (SIGReg, ARPredictor, Embedder, MLP)
├── utils.py              # Image preprocessing, ZScoreNormalizer, SaveCkptCallback
├── libero_data.py        # LIBERO HDF5 adapter (file handle caching)
├── Makefile              # Experiment launchers
├── config/
│   └── train/
│       ├── lewm.yaml     # Main training config
│       ├── model/lewm.yaml
│       ├── data/*.yaml   # Per-dataset configs (pusht, libero_10, libero_overfit, …)
│       └── launcher/local.yaml
├── scripts/              # Helper scripts
│   ├── viz_overfit.py    # Embedding visualization
│   └── eval_overfit.py   # Offline MPC evaluation
└── output/               # Generated artifacts (reports, plots, rollouts)
    ├── report_*.html
    └── rollouts/
```

## Key Hyperparameters (LIBERO multi-task)

| Param | Value | Note |
|-------|-------|------|
| lr | 2e-5 | 5e-5 caused gradient spikes on multi-task |
| SIGReg λ | 0.05 | Lower than default 0.09, allows better fitting |
| grad_clip | 0.5 | Tighter than default 1.0 |
| batch_size | 4 | Minimum for stable multi-task gradients |
| img_size | 128 | LIBERO native resolution |

## Known Checkpoints

`~/.stable_worldmodel/checkpoints/README.md` — all checkpoints documented.

- `lewm_weights_epoch_6.pt` — Best LIBERO-50 model (fit=1.70e-5, val=3.27e-5)
- `lewm_weights_epoch_9.pt` — Overfit on 1 episode (fit=7.4e-5)

## Performance Notes

- **RTX 3050 6GB**: batch≤4 at 224×224, batch≤8 at 128×128. Use bf16 AMP always.
- **num_workers=0** for LIBERO (HDF5 not fork-safe), **num_workers=2** for PushT.
- **Prefetch/persistent workers** must be disabled when num_workers=0.
- **PushT**: 467K samples/ep, ~8 min/ep at batch=32, ~2 min/ep at batch=128.
- **LIBERO I/O bottleneck**: HDF5 random access limits speed. File handle caching helps. For full 500-ep training, consider pre-loading to memory or Lance format.

## Output Directory Policy

All generated files go to `output/` or `logs/`. Root directory stays clean:
- `output/` — Lightning logs, environment dumps, checkpoints, Hydra runs, reports
- `logs/` — Tee logs from manual runs (`/tmp/` also acceptable for temp logs)
- Hydra and Lightning `default_root_dir` are configured to use `output/`

## Record Locations

| What | Where | Example |
|------|-------|---------|
| Experiment logs (training) | `logs/train/` | `logs/train/20260522_163000_pusht.log` |
| Experiment logs (inference) | `logs/infer/` | `logs/infer/20260522_170000_surprise_pusht_ep0.log` |
| HTML reports | `output/` | `output/report_libero10.html` |
| Plots & figures | `output/` | `output/surprise_pusht_ep0.png` |
| **Conclusions & findings** | `output/CONCLUSIONS.md` | Verified experimental results |
| Model checkpoints | `~/.stable_worldmodel/checkpoints/` | `lewm_weights_epoch_6.pt` |
| Memory (persistent) | `~/.claude/projects/.../memory/` | Cross-session context |
| Project configs | `config/` | `config/train/lewm.yaml` |

**Key distinction**: `logs/` = raw experiment output. `output/` = processed artifacts + conclusions.
`CONCLUSIONS.md` = curated findings, not raw logs. Update it when experiments yield clear results.
