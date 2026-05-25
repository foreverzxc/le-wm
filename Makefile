# LeWM Experiment Makefile
# Usage: make <target>
VENV       := .venv/bin/python
TS         := $(shell date +%Y%m%d_%H%M%S)
LOG_TRAIN  := logs/train/$(TS)
LOG_INFER  := logs/infer/$(TS)

logs/train logs/infer:
	mkdir -p logs/train logs/infer

# ═══════════════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════════════

# ── LIBERO Overfitting (1 episode) ────────────────────────────────────
.PHONY: overfit
overfit: logs/train
	PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 $(VENV) train.py \
		data=libero_overfit img_size=128 \
		trainer.max_epochs=30 \
		loader.batch_size=2 loader.num_workers=0 \
		loader.prefetch_factor=null loader.persistent_workers=false \
		2>&1 | tee $(LOG_TRAIN)_overfit.log

# ── LIBERO-10 Small (50 episodes, convergence test) ───────────────────
.PHONY: libero-small
libero-small: logs/train
	PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 $(VENV) train.py \
		data=libero_10 img_size=128 \
		trainer.max_epochs=100 \
		loader.batch_size=4 loader.num_workers=0 \
		loader.prefetch_factor=null loader.persistent_workers=false \
		optimizer.lr=2e-5 \
		loss.sigreg.weight=0.05 \
		trainer.gradient_clip_val=0.5 \
		data.dataset.max_episodes=50 \
		2>&1 | tee $(LOG_TRAIN)_libero_small.log

# ── Cross-dataset WM (4 datasets: Pusht+Cube+Reacher+TwoRooms) ──────
.PHONY: cross4
cross4: logs/train
	PYTHONUNBUFFERED=1 $(VENV) train.py \
		data=cross4 img_size=224 \
		trainer.max_epochs=10 \
		loader.batch_size=8 loader.num_workers=2 \
		optimizer.lr=2e-5 \
		loss.sigreg.weight=0.05 \
		trainer.gradient_clip_val=0.5 \
		2>&1 | tee $(LOG_TRAIN)_cross4.log

# ── Planner (train on frozen WM) ─────────────────────────────────────
.PHONY: planner
planner: logs/train
	PYTHONUNBUFFERED=1 $(VENV) train_planner.py \
		data=pusht img_size=224 \
		trainer.max_epochs=30 \
		loader.batch_size=4 loader.num_workers=0 \
		loader.prefetch_factor=null loader.persistent_workers=false \
		optimizer.lr=1e-4 \
		planner.ckpt=pusht \
		planner.horizon=5 planner.num_queries=8 \
		planner.num_layers=3 \
		planner.diversity_weight=0.1 \
		2>&1 | tee $(LOG_TRAIN)_planner.log

# ── PushT (benchmark) ─────────────────────────────────────────────────
.PHONY: pusht
pusht: logs/train
	PYTHONUNBUFFERED=1 $(VENV) train.py \
		data=pusht trainer.max_epochs=1 \
		loader.batch_size=2 loader.num_workers=2 \
		2>&1 | tee $(LOG_TRAIN)_pusht.log

# ── PushT full training ───────────────────────────────────────────────
.PHONY: pusht-full
pusht-full: logs/train
	PYTHONUNBUFFERED=1 $(VENV) train.py \
		data=pusht trainer.max_epochs=50 \
		loader.batch_size=2 loader.num_workers=2 \
		2>&1 | tee $(LOG_TRAIN)_pusht_full.log

# ── LIBERO-10 Full (500 episodes) ─────────────────────────────────────
.PHONY: libero-10
libero-10: logs/train
	PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 $(VENV) train.py \
		data=libero_10 img_size=128 \
		trainer.max_epochs=100 \
		loader.batch_size=4 loader.num_workers=2 \
		optimizer.lr=2e-5 loss.sigreg.weight=0.05 \
		trainer.gradient_clip_val=0.5 \
		2>&1 | tee $(LOG_TRAIN)_libero_10.log

# ── TwoRooms ──────────────────────────────────────────────────────────
.PHONY: tworoom
tworoom: logs/train
	PYTHONUNBUFFERED=1 $(VENV) train.py \
		data=tworooms trainer.max_epochs=50 \
		loader.batch_size=32 loader.num_workers=2 \
		2>&1 | tee $(LOG_TRAIN)_tworoom.log

# ── DMC / Reacher ─────────────────────────────────────────────────────
.PHONY: dmc
dmc: logs/train
	PYTHONUNBUFFERED=1 $(VENV) train.py \
		data=dmc trainer.max_epochs=50 \
		loader.batch_size=32 loader.num_workers=2 \
		2>&1 | tee $(LOG_TRAIN)_dmc.log

# ═══════════════════════════════════════════════════════════════════════
#  Inference / Visualization
# ═══════════════════════════════════════════════════════════════════════

# ── Surprise (anomaly detection along a trajectory) ───────────────────
.PHONY: surprise
surprise: logs/infer
	PYTHONUNBUFFERED=1 $(VENV) scripts/surprise.py \
		--dataset $(DATASET) --ep $(EP) --save output/surprise_$(DATASET)_ep$(EP).png \
		2>&1 | tee $(LOG_INFER)_surprise_$(DATASET)_ep$(EP).log

# ── Batch surprise (all 4 pretrained models) ──────────────────────────
.PHONY: batch-surprise
batch-surprise: logs/infer
	PYTHONUNBUFFERED=1 $(VENV) scripts/batch_surprise.py \
		2>&1 | tee $(LOG_INFER)_batch_surprise.log

# ── Overfit visualization ─────────────────────────────────────────────
.PHONY: viz-overfit
viz-overfit: logs/infer
	PYTHONUNBUFFERED=1 $(VENV) scripts/viz_overfit.py \
		2>&1 | tee $(LOG_INFER)_viz_overfit.log

# ── Planner visualisation ─────────────────────────────────────────────
.PHONY: viz-planner
viz-planner:
	PYTHONUNBUFFERED=1 $(VENV) scripts/viz_planner_actions.py

# ── Planner experiments ───────────────────────────────────────────────
.PHONY: planner-t1 planner-t5 planner-sim gt-error

planner-t1: logs/infer
	PYTHONUNBUFFERED=1 $(VENV) scripts/planner_overfit_t1.py \
		2>&1 | tee $(LOG_INFER)_planner_t1.log

planner-t5: logs/infer
	PYTHONUNBUFFERED=1 $(VENV) scripts/planner_overfit_t5.py \
		2>&1 | tee $(LOG_INFER)_planner_t5.log

planner-sim:
	PYTHONUNBUFFERED=1 $(VENV) scripts/sim_planner_action.py

gt-error:
	PYTHONUNBUFFERED=1 $(VENV) scripts/gt_rollout_error.py

# ── Evaluation (MPC planning) ─────────────────────────────────────────
.PHONY: eval-pusht eval-cube eval-reacher eval-tworoom

eval-pusht: logs/infer
	PYTHONUNBUFFERED=1 $(VENV) eval.py --config-name=pusht.yaml policy=pusht/lewm \
		2>&1 | tee $(LOG_INFER)_eval_pusht.log

eval-cube: logs/infer
	PYTHONUNBUFFERED=1 $(VENV) eval.py --config-name=cube.yaml policy=cube/lewm \
		2>&1 | tee $(LOG_INFER)_eval_cube.log

eval-reacher: logs/infer
	PYTHONUNBUFFERED=1 $(VENV) eval.py --config-name=reacher.yaml policy=reacher/lewm \
		2>&1 | tee $(LOG_INFER)_eval_reacher.log

eval-tworoom: logs/infer
	PYTHONUNBUFFERED=1 $(VENV) eval.py --config-name=tworoom.yaml policy=tworooms/lewm \
		2>&1 | tee $(LOG_INFER)_eval_tworoom.log

# ═══════════════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════════════
.PHONY: clean kill-gpu info report list-logs

report:
	@echo "Reports: output/report_*.html"

list-logs:
	@echo "=== Training logs ===" && ls -lt logs/train/ 2>/dev/null || echo "(none)"
	@echo "=== Inference logs ===" && ls -lt logs/infer/ 2>/dev/null || echo "(none)"

clean:
	rm -rf logs/ lightning_logs/ output/hydra/ .pycache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

kill-gpu:
	@nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null; echo "GPU processes killed"

info:
	@echo "=== GPU ===" && nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv 2>/dev/null
	@echo "=== Python ===" && $(VENV) --version
	@echo "=== Datasets ===" && ls ~/.stable_worldmodel/*.h5 2>/dev/null || echo "(none)"
	@echo "=== Checkpoints ===" && ls -lt ~/.stable_worldmodel/checkpoints/*.pt 2>/dev/null | head -5 || echo "(none)"

help:
	@echo "LeWM Experiment Targets:"
	@echo ""
	@echo "  Training:"
	@echo "    make overfit        Overfit on 1 LIBERO episode"
	@echo "    make libero-small   50 episodes, convergence test"
	@echo "    make libero-10      Full LIBERO-10 (500 episodes)"
	@echo "    make planner        Train planner on frozen PushT WM"
	@echo "    make pusht          PushT benchmark (1 epoch timing)"
	@echo "    make pusht-full     PushT full training (50 epochs)"
	@echo "    make tworoom        TwoRooms benchmark"
	@echo "    make dmc            DMC / Reacher benchmark"
	@echo ""
	@echo "  Inference & Eval:"
	@echo "    make batch-surprise                 4×4 cross-dataset surprise"
	@echo "    make planner-t1 / planner-t5         Planner overfit experiments"
	@echo "    make planner-sim                     Simulate planner action in env"
	@echo "    make gt-error                        GT rollout error analysis"
	@echo "    make surprise DATASET=libero EP=0    Surprise along trajectory"
	@echo "    make batch-surprise                  4×4 cross-dataset surprise"
	@echo "    make viz-overfit / viz-planner       Visualizations"
	@echo ""
	@echo "  Utilities:"
	@echo "    make list-logs      Show all experiment logs"
	@echo "    make info           GPU, Python, dataset status"
	@echo "    make clean          Remove logs and caches"
	@echo "    make kill-gpu       Kill all GPU processes"
	@echo ""
	@echo "  Logs saved to: logs/train/ and logs/infer/ (timestamped filenames)"
