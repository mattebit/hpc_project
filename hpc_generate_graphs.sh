#!/bin/bash


#PBS -l select=4:ncpus=64:mem=128gb

#PBS -l walltime=1:00:00

#PBS -q short_cpuQ

# Compile the graph generator if needed
gcc -O3 -std=c99 -fopenmp ~/hpc_project/graph_generator.c -o ~/hpc_project/graph_generator >/dev/null 2>&1

# Array of vertex counts to test
VERTICES=(80000)

# For each vertex count, generate graphs with different edge densities
for v in "${VERTICES[@]}"; do
    # Calculate different edge counts:
    # 1. Sparse graph (~2*v edges)
    # 2. Medium density (~v*log(v) edges)
    # 3. Dense graph (~v*v/4 edges)
    
    min_edges=$((v - 1))
    sparse_edges=$((v * 2))
    medium_edges=$((v * $(echo "l($v)/l(2)" | bc -l | cut -d'.' -f1)))
    dense_edges=$((v * v / 4))
    
    ~/hpc_project/graph_generator $v $dense_edges
done