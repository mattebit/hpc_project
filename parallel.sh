#!/bin/bash

#PBS -l select=10:ncpus=10:mem=32gb

# set max execution time
#PBS -l walltime=0:05:00

# imposta la coda di esecuzione
#PBS -q short_cpuQ
module load mpich-3.2
mpirun.actual -n 100 ~/hpc_project/parallel.o