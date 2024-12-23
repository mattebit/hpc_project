#!/bin/bash

# set max execution time
#PBS -l walltime=0:10:00

# imposta la coda di esecuzione
#PBS -q short_cpuQ

module load mpich-3.2
mpirun.actual -n $NUM_PROCESS ~/hpc_project/src/main