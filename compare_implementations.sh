#!/bin/bash

# Compile both implementations with optimization flags
gcc -O3 serial.c -o serial
mpicc -std=c99 -O3 -fopenmp -march=native -mtune=native -ftree-vectorize -funroll-loops -flto -o src/main.o src/main.c

# Create directory for results if it doesn't exist
mkdir -p results

# Function to extract MST weight from output
get_mst_weight() {
    echo "$1" | grep "MST Weight:" | head -n1 | grep -o '[0-9]\+'
}

# Function to extract MST weight from the last non-empty line of output
get_time() {
    echo "$1" | grep "Computation Time:" | head -n1 | grep -o '[0-9.]\+'
}

# Test different graph sizes
for graph in $(ls graphs/graph_*_*.txt | sort -t_ -k2 -n); do
    echo "============================================"
    echo "Testing $graph..."
    
    # Get number of vertices from first line of the graph file
    num_vertices=$(head -n 1 "$graph" | cut -d' ' -f1)
    
    # Run serial implementation
    echo "Running serial implementation..."
    serial_output=$(./serial.o "$graph")
    serial_weight=$(get_mst_weight "$serial_output")
    serial_time=$(get_time "$serial_output")
    
    # Run parallel implementation
    echo "Running parallel implementation..."
    parallel_output=$(mpirun -np 2 --bind-to none -x OMP_NUM_THREADS=4 src/main.o "$num_vertices" "$graph")
    parallel_weight=$(get_mst_weight "$parallel_output")
    parallel_time=$(get_time "$parallel_output")
    
    # Save results to file
    result_file="results/$(basename "$graph" .txt)_result.txt"
    
    {
        echo "Graph: $graph"
        echo "Number of vertices: $num_vertices"
        
        if [ "$serial_weight" = "$parallel_weight" ]; then
            echo "✅ Test passed! MST weights match: $serial_weight"
        else
            echo "❌ Test failed!"
            # Calculate weight difference and percentage
            weight_diff=$((parallel_weight - serial_weight))
            weight_percent=$(echo "scale=2; ($weight_diff / $serial_weight) * 100" | bc)
            echo "Weight difference: $weight_diff"
            echo "Weight difference percentage: ${weight_percent}%"
        fi

        # Always show timing metrics
        echo "Serial MST weight: $serial_weight"
        echo "Parallel MST weight: $parallel_weight"
        echo "Serial time: ${serial_time}s"
        echo "Parallel time: ${parallel_time}s"
        
        # Calculate and show timing metrics
        if [ ! -z "$parallel_time" ] && [ ! -z "$serial_time" ]; then
            # Use bc for all floating point calculations and remove newlines
            time_diff=$(echo "$parallel_time - $serial_time" | bc -l | tr -d '\n')

            if (( $(echo "$parallel_time > $serial_time" | bc -l) )); then
                # Calculate slowdown
                slowdown=$(echo "scale=4; $parallel_time/$serial_time" | bc -l | tr -d '\n')
                printf "Time difference: %.4fs\n" "$time_diff"
                printf "Slowdown: %.4fx\n" "$slowdown"
                echo "Parallel was slower than serial."
            else
                # Calculate speedup
                speedup=$(echo "scale=4; $serial_time/$parallel_time" | bc -l | tr -d '\n')
                efficiency=$(echo "scale=4; $speedup/2" | bc -l | tr -d '\n')
                time_percent=$(echo "scale=2; ($time_diff / $serial_time) * 100" | bc -l | tr -d '\n')
                
                printf "Time difference: %.4fs\n" "$time_diff"
                printf "Speedup: %.4fx\n" "$speedup" 
                printf "Parallel efficiency: %.4f\n" "$efficiency"
                printf "Time improvement: %.2f%%\n" "$time_percent"
            fi
        fi
        echo "----------------------------------------"
    } | tee "$result_file"
    
done

# Print summary
echo "============================================"
echo "Testing completed. Results saved in 'results' directory"