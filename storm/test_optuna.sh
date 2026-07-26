#!/bin/bash
#SBATCH -J STORMing_the_Bayestille
#SBATCH -c 81
#SBATCH -o ./output_scripts/%A_real_data_output.out

time srun uv run ../tests/test_optuna_study.py --jobs=80 --trials=160
