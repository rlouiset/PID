#!/bin/bash

#SBATCH --job-name mosei_lsmi
#SBATCH --time=00-19:59:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --constraint h100
#SBATCH --account haj@h100
#SBATCH --output mosei_lsmi_vision_text.txt

module purge # purge modules inherited by default
conda deactivate # deactivate environments inherited by default
module load miniforge/24.9.0
conda activate py39
export WANDB_MODE=offline
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}/lustre/fshomisc/home/rech/genpuc01/uik24xv/PID"

# Code lives in $HOME (small quota): write all outputs (lightning_logs,
# checkpoints, saved_models) to $SCRATCH instead.
PID_DIR=/lustre/fshomisc/home/rech/genpuc01/uik24xv/PID
RUN_DIR=$SCRATCH/pid_runs
mkdir -p "$RUN_DIR" && cd "$RUN_DIR"

DATA=/lustre/fswork/projects/rech/haj/uik24xv/datasets/mosei_senti_data.pkl   # TODO: check path
SEED=${SEED:-1}
MODEL=${MODEL:-saved_models/mosei_lsmi_vision-text_seed${SEED}.pt}
mkdir -p saved_models

srun python3 "$PID_DIR"/affect/mosei_comm_lsmi_pid.py \
    --data-path "$DATA" \
    --mod0 vision --mod1 text \
    --seed "$SEED" \
    --saved-model "$MODEL" \
    --load-model "$MODEL"
