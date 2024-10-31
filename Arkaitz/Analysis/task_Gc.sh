#!/bin/bash -l

#SBATCH --job-name=COULOMB_Gc
#SBATCH --output=logs/log_out.o%j
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40


crun -p ~/envs/coulomb_lattice python3 Analysis_Gc.py $1 $2

