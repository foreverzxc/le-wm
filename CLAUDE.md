# CLAUDE.md

LeWorldModel (LeWM) — JEPA world model + Planner, end-to-end from pixels. ~18M params, single GPU.

## Quick Start

```bash
make help           # List all targets
make info           # GPU + dataset status
make list-logs      # Show experiment logs
```

## Project Structure

```
le-wm/
├── train.py, train_planner.py     # 训练入口 (WM / Planner)
├── eval.py                        # 评估入口 (MPC planning)
├── jepa.py                        # JEPA WM (encode, predict, rollout, get_cost)
├── module.py                      # SIGReg, ARPredictor, Embedder, MLP, Transformer
├── planner.py                     # PlannerDecoder + PlannerLoss + planner_rollout
├── utils.py                       # Image preprocessing, ZScoreNormalizer, SaveCkptCallback
├── libero_data.py                 # LIBERO HDF5 adapter (file handle caching)
├── multidata.py                   # MultiDomainDataset (cross-dataset training)
├── Makefile                       # All experiment launchers
├── config/train/
│   ├── lewm.yaml                  # Main config (WM + Planner params)
│   ├── model/lewm.yaml            # JEPA model definition
│   └── data/*.yaml                # Per-dataset configs
├── scripts/                       # Analysis & visualization scripts
│   ├── compare_gt_sim_pixels.py    # GT sim vs dataset pixel comparison (standard)
│   ├── surprise.py                 # Single trajectory surprise
│   ├── batch_surprise.py           # Cross-dataset surprise matrix
│   └── gt_rollout_error.py         # WM pred_loss vs rollout cost analysis
└── output/
    ├── CONCLUSIONS.md             # Verified experimental findings
    ├── viz/                       # All generated visualizations
    │   └── INDEX.html             # Visual index page
    └── lightning/                 # PyTorch Lightning logs (auto-generated)
```

## Record Locations

| What | Where |
|------|-------|
| Experiment logs (training) | `logs/train/YYYYMMDD_HHMMSS_<name>.log` |
| Experiment logs (inference) | `logs/infer/YYYYMMDD_HHMMSS_<name>.log` |
| Visualizations & plots | `output/viz/` |
| **Conclusions & findings** | `output/CONCLUSIONS.md` |
| Model checkpoints | `~/.stable_worldmodel/checkpoints/` |
| Checkpoint catalog | `~/.stable_worldmodel/checkpoints/README.md` |
| Pretrained weights | `~/.stable_worldmodel/<name>_weights.pt` |
| Memory (cross-session) | `~/.claude/projects/.../memory/` |
| Config files | `config/train/` |

## Key Design Decisions

### Planner Architecture
- **Perceiver-style decoder**: N learned queries, self-attention + cross-attention to [goal_emb, ctx_emb]
- **Full history context**: planner sees all HS frames of ctx_emb, not just the last frame
- **Best-of-N loss**: only the query with lowest rollout cost gets gradient
- **DETR-style diversity**: cosine similarity penalty pushes non-best queries away from best
- **Action constraint**: `(2*sigmoid(x)-1) * action_range` — smooth normalization to [-range, range]
- **Individual sub-actions**: planner outputs `horizon * action_substeps * action_dim` scalars, reshaped to `(horizon, action_substeps, action_dim)`. Each substep is a unique action (no repeat_interleave)
- **Real action history**: pass dataset actions as `hist_actions` to rollout in `fs*raw_dim` form, NOT averaged
- **Explicit goal**: pass `goal_emb` to `planner_rollout`, NOT derived from context
- **Horizon guard**: train_planner.py raises ValueError if `horizon > num_steps - history_size`

### Planner Rollout Gradient Flow
- WM params have `requires_grad=False` — never updated
- Context encoding uses `torch.no_grad()` — saves memory
- **Rollout loop does NOT use no_grad** — gradient flows through action_encoder → predictor → pred_embs → loss → actions → planner
- **hist_act alignment**: t=0 replaces only `hist_act[:, -1:]` with P[0] (keeps A[0],A[1] aligned with f0,f1). t≥1 does shift+append synchronised with emb truncation

### WM Training Hyperparameters
- lr=2e-5 (5e-5 causes gradient spikes on multi-task data)
- SIGReg λ=0.05 (lower than default 0.09)
- grad_clip=0.5 (tighter than default 1.0)
- batch≥4 for multi-task stability

### GT Action Simulation
- Dataset stores actions as `(num_steps, frameskip * raw_dim)` in chronological order
- GT simulation: use ALL individual actions `(T, frameskip, raw_dim)`, NOT averaged
- `_set_state` works after `env.reset()` — sets agent position, block position, angle
- PushT physics simulation has inherent irreproducibility: same state + actions produce different agent trajectories (~90px divergence after 5 steps)
- Block position stays accurate (Δ<3px) — good enough for task evaluation

### LiberoDataset
- File handle caching (`_file_handles`) avoids repeated HDF5 open/close
- `max_episodes` parameter for quick overfitting tests
- Column pre-caching for small columns (action, proprio, state)
- Sampled normalizer: use `load_episode` for a few episodes instead of full `get_col_data` scan

### Multi-domain Training
- `MultiDomainDataset` wraps multiple HDF5Dataset instances
- Action zero-padding to max dimension (Cube=25D for cross-dataset)
- Returns only `pixels` + `action` keys to avoid DataLoader collation errors

## Known Issues & Limitations

1. **Planner action_head init**: action_head output layer not yet using small-init (scale 0.1 + zero bias). May cause sigmoid saturation early in training
2. **PushT simulation irreproducibility**: same state + actions produce divergent agent trajectories in PyMunk. Verify with `compare_gt_sim_pixels.py`
3. **Cross-model surprise comparison**: TwoRooms model has different embedding scale (surprise 2-7 vs 0.02-0.45). Only compare across datasets for the SAME model
4. **Cube action dimension**: 25D (5×5), all others 10D. Requires zero-padding for cross-dataset training
5. **HDF5 not fork-safe**: use `num_workers=0` for LIBERO datasets, disable `persistent_workers` and `prefetch_factor` when `num_workers=0`
6. **RTX 3050 6GB**: batch≤8 at 224×224, batch≤4 at 128×128 with bf16

## Experiment Results Summary

**NOTE: Previous conclusions (planner > GT, etc.) are unverified pending re-evaluation with the fixes in this commit.**

See `output/CONCLUSIONS.md` for historical context. All findings need re-validation.
