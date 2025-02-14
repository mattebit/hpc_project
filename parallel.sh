#!/bin/bash

#PBS -l select=2:ncpus=64:mem=256gb
#PBS -l place=scatter
#PBS -l walltime=1:00:00
#PBS -q short_cpuQ

module load mpich-3.2

# Set number of OpenMP threads per MPI process
export OMP_NUM_THREADS=8
# Enable thread binding
export OMP_PROC_BIND=true
export OMP_PLACES=cores

# Run with MPICH-style process mapping
mpirun.actual -n 16 \
    -ppn 8 \
    ~/hpc_project/src/main.o 80000 ~/hpc_project/graphs/graph_80000_dense.txt