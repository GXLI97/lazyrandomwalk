#!/bin/bash
#SBATCH -N 1  # 1 node
#SBATCH -n 1  # 1 task per node
#SBATCH -c 4
#SBATCH -t 10:00:00 # time required, here it is 1 min
#SBATCH --mem=200G
#SBATCH -o slurms/logs/preprocess.out # stdout is redirected to that file
#SBATCH -e slurms/logs/preprocess.err # stderr is redirected to that file

module load anaconda3
src/preprocess.sh