#!/bin/bash
#SBATCH -J STORMing_the_Bayestille
#SBATCH -c 10
#SBATCH -o ./output_scripts/%A_real_data_output.out

srun time uv run ./storm/storm_test.py 9

