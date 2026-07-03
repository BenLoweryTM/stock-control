#!/bin/bash
#SBATCH -J STORMing_the_Bayestille
#SBATCH -c 10
#SBATCH -o ./output_scripts/real_data_output.out

time srun uv run ./storm/storm_test.py 9

