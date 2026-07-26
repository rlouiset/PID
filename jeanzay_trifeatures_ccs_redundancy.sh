#!/bin/bash

#SBATCH --job-name trifeat_ccs_r
#SBATCH --time=00-19:59:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --constraint a100
#SBATCH --account haj@a100
#SBATCH --output trifeat_ccs_r.txt

module purge # purge modules inherited by default
conda deactivate # deactivate environments inherited by default
module load miniforge/24.9.0
conda activate py39
export WANDB_MODE=offline
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}/lustre/fshomisc/home/rech/genpuc01/uik24xv/PID"
srun python3 trifeatures/trifeatures_comm_ccs_pid.py --task share --num-classes 10 \
    --data-root /lustre/fswork/projects/rech/haj/uik24xv/datasets/trifeatures_data
