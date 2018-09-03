#!/bin/bash
#SBATCH -N 1  # 1 node
#SBATCH -n 1  # 1 task per node
#SBATCH -c 8
#SBATCH -t 10:00:00 # time required, here it is 1 min
#SBATCH --mem=400G
#SBATCH -o logs/lazy.out # stdout is redirected to that file
#SBATCH -e logs/lazy.err # stderr is redirected to that file
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
# replace your_netid by your actual netid
#SBATCH --mail-user=gxli@princeton.edu

module load anaconda3
src/lazy.sh
