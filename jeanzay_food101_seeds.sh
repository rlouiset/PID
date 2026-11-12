#!/bin/bash
# Usage: bash jeanzay_food101_seeds.sh [seeds...]      (default seeds: 0 1)
#
# For each seed, one CoMM model is trained once and shared by both PID methods:
#   1. LSMI job trains the model and saves it to $MODEL_DIR/food101_seed<S>.pth
#   2. CCS job starts only after the LSMI job succeeds, loads that model, and skips training
# Logs: food101_lsmi_seed<S>.txt and food101_ccs_seed<S>.txt (in the submit directory).
set -euo pipefail

SEEDS=${@:-0 1}
TAG=${TAG:-lrfix}   # distinguishes these models from ones trained with the old single-LR setup
MODEL_DIR=$SCRATCH/food101_models
mkdir -p "$MODEL_DIR"

for S in $SEEDS; do
    MODEL=$MODEL_DIR/food101_${TAG}_seed${S}.pth
    if [ -e "$MODEL" ]; then
        # The LSMI job would load it instead of training: refuse rather than reuse silently.
        echo "ERROR: $MODEL already exists. Delete it or set TAG=<other> to retrain." >&2
        exit 1
    fi

    LSMI_ID=$(sbatch --parsable \
                     --export=ALL,SEED=$S,MODEL=$MODEL \
                     --job-name food101_lsmi_s${S} \
                     --output food101_lsmi_seed${S}.txt \
                     jeanzay_food101_lsmi.sh)

    CCS_ID=$(sbatch --parsable \
                    --dependency=afterok:${LSMI_ID} \
                    --export=ALL,SEED=$S,MODEL=$MODEL \
                    --job-name food101_ccs_s${S} \
                    --output food101_ccs_seed${S}.txt \
                    jeanzay_food101_ccs.sh)

    echo "seed $S: lsmi=$LSMI_ID  ccs=$CCS_ID (after lsmi)  model=$MODEL"
done
