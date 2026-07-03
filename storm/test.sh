#!/bin/bash
#SBATCH -J STORMing_the_Bayestille
#SBATCH -c 10
#SBATCH -o ./realdataoutput.out

srun uv run ~/storm/storm_test.py 9

