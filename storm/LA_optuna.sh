#!/bin/bash
#SBATCH -J LA_optuna
#SBATCH -c 8
#SBATCH --array=0-N%10
#SBATCH -o ../output_scripts/%A_%a_LA_optuna.out

time srun uv run ../methods/optuna_studies/LA.py \
  --jobs=8 \
  --trials=400 \
  --instance-idx="${SLURM_ARRAY_TASK_ID}"