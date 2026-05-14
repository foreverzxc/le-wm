#!/bin/bash
VENV=".venv/bin/python3"
DATA_DIR="$HOME/.stable_worldmodel"

echo "Running isotropy tests on all 4 models..."

for exp in baseline whitening noise whitening_noise; do
    MODEL="$DATA_DIR/experiments/$exp/lewm_${exp}_epoch_3_object.ckpt"
    OUT="experiments/isotropy_$exp"
    mkdir -p "$OUT"
    
    echo "=== $exp ==="
    if [ -f "$MODEL" ]; then
        $VENV test_isotropy.py \
            --model "$MODEL" \
            --n-samples 500 \
            --output-dir "$OUT" \
            > "$OUT/report.txt" 2>&1
        echo "exit: $?"
    else
        echo "MODEL NOT FOUND: $MODEL"
    fi
done

echo "All done!"
