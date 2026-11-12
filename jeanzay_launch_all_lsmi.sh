#!/bin/bash
# Usage: bash jeanzay_launch_all_lsmi.sh <seed> [names...]   (default names: food101 sarcasm humor mosei)
# Logs go to <name>_lsmi_seed<seed>.txt
SEED=${1:?usage: bash jeanzay_launch_all_lsmi.sh <seed> [names...]}; shift
NAMES=${@:-food101 sarcasm humor mosei}
for name in $NAMES; do
    sbatch --export=ALL,SEED=$SEED \
           --job-name ${name}_lsmi_s${SEED} \
           --output ${name}_lsmi_seed${SEED}.txt \
           jeanzay_${name}_lsmi.sh
done
