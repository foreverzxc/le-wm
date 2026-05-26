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

# ── Planner SFT (action imitation) ────────────────────────────────────
.PHONY: planner-sft
planner-sft: logs/train
	PYTHONUNBUFFERED=1 $(VENV) train_planner.py --config-name=planner_sft \
		2>&1 | tee $(LOG_TRAIN)_planner_sft.log

# ── Planner SFT overfit (1 sample) ─────────────────────────────────────
.PHONY: planner-sft-overfit
planner-sft-overfit: logs/train
	PYTHONUNBUFFERED=1 $(VENV) train_planner.py --config-name=planner_sft_overfit \
		2>&1 | tee $(LOG_TRAIN)_planner_sft_overfit.log

# ── Planner FT (WM rollout, from SFT checkpoint) ───────────────────────
.PHONY: planner-ft
planner-ft: logs/train
	PYTHONUNBUFFERED=1 $(VENV) train_planner.py \
		trainer.max_epochs=30 \
		loader.batch_size=4 \
		optimizer.lr=1e-4 \
		2>&1 | tee $(LOG_TRAIN)_planner_ft.log

# ── Planner FT overfit (1 sample) ──────────────────────────────────────
.PHONY: planner-ft-overfit
planner-ft-overfit: logs/train
	PYTHONUNBUFFERED=1 $(VENV) train_planner.py --config-name=planner_ft_overfit \
		2>&1 | tee $(LOG_TRAIN)_planner_ft_overfit.log

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

# ── GT rollout error analysis ────────────────────────────────────────
.PHONY: gt-error
gt-error:
	PYTHONUNBUFFERED=1 $(VENV) scripts/gt_rollout_error.py

# ── GT sim vs dataset pixel comparison ────────────────────────────────
.PHONY: gt-sim-pixels
gt-sim-pixels:
	PYTHONUNBUFFERED=1 $(VENV) scripts/compare_gt_sim_pixels.py

# ── Planner visualization ──────────────────────────────────────────────
CKPT ?= $$HOME/.stable_worldmodel/checkpoints/planner_sft_pusht_epoch1.ckpt
EPISODE ?= 0
SAMPLE ?= 0
.PHONY: viz
viz:
	PYTHONUNBUFFERED=1 $(VENV) scripts/viz_result.py \
		--ckpt $(CKPT) --sample $(SAMPLE) --episode $(EPISODE)

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
	@echo "    make planner-sft    SFT: action imitation (no WM rollout)"
	@echo "    make planner-ft     FT: WM rollout + best-of-N (from SFT ckpt)"
	@echo "    make planner-sft-overfit / planner-ft-overfit  Overfit variants"
	@echo "    make pusht          PushT benchmark (1 epoch timing)"
	@echo "    make pusht-full     PushT full training (50 epochs)"
	@echo "    make tworoom        TwoRooms benchmark"
	@echo "    make dmc            DMC / Reacher benchmark"
	@echo ""
	@echo "  Inference & Eval:"
	@echo "    make gt-sim-pixels                   GT sim vs dataset pixel comparison"
	@echo "    make viz CKPT=... SAMPLE=0            Planner action visualization"
	@echo "    make surprise DATASET=libero EP=0    Surprise along trajectory"
	@echo "    make batch-surprise                  4×4 cross-dataset surprise"
	@echo "    make gt-error                        GT rollout error analysis"
	@echo ""
	@echo "  Utilities:"
	@echo "    make list-logs      Show all experiment logs"
	@echo "    make info           GPU, Python, dataset status"
	@echo "    make clean          Remove logs and caches"
	@echo "    make kill-gpu       Kill all GPU processes"
	@echo ""
	@echo "  Logs saved to: logs/train/ and logs/infer/ (timestamped filenames)"
