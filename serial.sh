#!/bin/bash

#PBS -l select=1:ncpus=1:mem=32gb

# set max execution time
#PBS -l walltime=0:20:00

# imposta la coda di esecuzione
#PBS -q short_cpuQ
~/hpc_project/serial.o ~/hpc_project/large_graph.txt