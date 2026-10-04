#!/bin/bash
#SBATCH -J CNTS_optuna
#SBATCH -c 8
#SBATCH --array=0-N%10
#SBATCH -o ../output_scripts/%A_%a_CNTS_optuna.out

time srun uv run ../methods/optuna_studies/CNTS.py \
  --jobs=8 \
  --trials=400 \
  --instance-idx="${SLURM_ARRAY_TASK_ID}"