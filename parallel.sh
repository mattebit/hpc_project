#!/bin/bash

#PBS -l select=1:ncpus=64:mem=48gb

# set max execution time
#PBS -l walltime=0:20:00

# imposta la coda di esecuzione
#PBS -q short_cpuQ
module load mpich-3.2
mpirun.actual -n 16 ~/hpc_project/src/main.o 50000 ~/hpc_project/graphs/graph_50000_dense.txt