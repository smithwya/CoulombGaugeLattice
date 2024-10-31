#!/bin/bash -l

#SBATCH --job-name=COULOMB_Gc
#SBATCH --output=logs/log_out.o%j
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40


crun -p ~/envs/coulomb_lattice python3 Analysis_Gc.py $1 $2

crun -p ~/envs/coulomb_lattice python3 Analysis_Vr.py $1 $2

mv logs/log_out.o$SLURM_JOB_ID logs/log_out_${1:10}.o$SLURM_JOB_ID
