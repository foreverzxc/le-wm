#!/bin/bash
BASE_ARGS="trainer.max_epochs=3 trainer.limit_train_batches=30 trainer.limit_val_batches=10 loader.batch_size=8 loader.num_workers=2 wandb.enabled=false"
VENV=".venv/bin/python3"

echo "============================================================"
echo "Experiment A: Baseline (SIGReg)"
echo "============================================================"
$VENV train.py $BASE_ARGS \
    output_model_name=lewm_baseline \
    subdir=experiments/baseline \
    > experiments/baseline.log 2>&1
echo "A done (exit code: $?)"

echo "============================================================"
echo "Experiment B: Whitening"
echo "============================================================"
$VENV train.py $BASE_ARGS \
    loss.sigreg.enabled=false \
    whitening.enabled=true \
    output_model_name=lewm_whitening \
    subdir=experiments/whitening \
    > experiments/whitening.log 2>&1
echo "B done (exit code: $?)"

echo "============================================================"
echo "Experiment C: Noise"
echo "============================================================"
$VENV train.py $BASE_ARGS \
    loss.sigreg.enabled=false \
    noise.enabled=true \
    noise.std=0.1 \
    output_model_name=lewm_noise \
    subdir=experiments/noise \
    > experiments/noise.log 2>&1
echo "C done (exit code: $?)"

echo "============================================================"
echo "Experiment D: Whitening + Noise"
echo "============================================================"
$VENV train.py $BASE_ARGS \
    loss.sigreg.enabled=false \
    whitening.enabled=true \
    noise.enabled=true \
    noise.std=0.1 \
    output_model_name=lewm_whitening_noise \
    subdir=experiments/whitening_noise \
    > experiments/whitening_noise.log 2>&1
echo "D done (exit code: $?)"

echo "============================================================"
echo "ALL DONE"
echo "============================================================"
