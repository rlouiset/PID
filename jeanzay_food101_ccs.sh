#!/bin/bash

#SBATCH --job-name food101_ccs
#SBATCH --time=00-19:59:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --constraint h100
#SBATCH --account haj@h100
#SBATCH --output food101_ccs_image_text.txt

module purge # purge modules inherited by default
conda deactivate # deactivate environments inherited by default
module load miniforge/24.9.0
conda activate py39
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1   # compute nodes have no internet: pre-download models on the login node
export HF_HOME=$WORK/hf_cache
export TORCH_HOME=$WORK/torch_cache
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}/lustre/fshomisc/home/rech/genpuc01/uik24xv/PID"

DATASETS=$SCRATCH/datasets   # Food101 is too big for $WORK quota

srun python3 food101/food101_comm_ccs_pid.py \
    --data-root "$DATASETS/Food101" \
    --load-model "$DATASETS/food101_model.pth" \
    --mod0 image --mod1 text
