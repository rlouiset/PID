#!/bin/bash

#SBATCH --job-name mosei_ccs
#SBATCH --time=00-19:59:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --constraint h100
#SBATCH --account haj@h100
#SBATCH --output mosei_ccs_vision_text.txt

module purge # purge modules inherited by default
conda deactivate # deactivate environments inherited by default
module load miniforge/24.9.0
conda activate py39
export WANDB_MODE=offline
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}/lustre/fshomisc/home/rech/genpuc01/uik24xv/PID"

DATA=/lustre/fswork/projects/rech/haj/uik24xv/datasets/mosei_senti_data.pkl   # TODO: check path
SEED=${SEED:-0}
MODEL=${MODEL:-saved_models/mosei_comm_vision-text_seed${SEED}.pt}
mkdir -p saved_models

srun python3 affect/mosei_comm_ccs_source_redundancy.py \
    --data-path "$DATA" \
    --mod0 vision --mod1 text \
    --seed "$SEED" \
    --saved-model "$MODEL" \
    --load-model "$MODEL"
