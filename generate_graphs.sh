#!/bin/bash

# Compile the graph generator if needed
gcc -fopenmp graph_generator.c -o graph_generator

# Array of vertex counts to test
VERTICES=(1000 1500 2000 5000 10000 20000)

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
    
    echo "Generating graphs with $v vertices..."
    
    # Generate sparse graph
    echo "  Generating sparse graph ($sparse_edges edges)..."
    ./graph_generator $v $sparse_edges
    mv random_graph.txt "graphs/graph_${v}_sparse.txt"
    
    # Generate medium density graph
    echo "  Generating medium graph ($medium_edges edges)..."
    ./graph_generator $v $medium_edges
    mv random_graph.txt "graphs/graph_${v}_medium.txt"
    
    # Generate dense graph
    echo "  Generating dense graph ($dense_edges edges)..."
    ./graph_generator $v $dense_edges
    mv random_graph.txt "graphs/graph_${v}_dense.txt"
done

echo "Graph generation complete!"