#!/bin/bash
# Usage: bash jeanzay_launch_all_ccs.sh <seed>
# Submits food101, sarcasm, humor and mosei with that seed; logs go to <name>_ccs_seed<seed>.txt
SEED=${1:?usage: bash jeanzay_launch_all_ccs.sh <seed>}
for name in food101 sarcasm humor mosei; do
    sbatch --export=ALL,SEED=$SEED \
           --job-name ${name}_ccs_s${SEED} \
           --output ${name}_ccs_seed${SEED}.txt \
           jeanzay_${name}_ccs.sh
done
