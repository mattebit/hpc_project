#!/bin/bash

# Compile both MPI implementations with optimization flags
mpicc -std=c99 -O3 -fopenmp -march=native -mtune=native -ftree-vectorize -funroll-loops -flto -o src/main.o src/main.c
mpicc -std=c99 -O3 -fopenmp -march=native -mtune=native -ftree-vectorize -funroll-loops -flto -o src/new_main.o src/new_main.c

# Create directory for results if it doesn't exist
mkdir -p results

# Function to extract MST weight from output
get_mst_weight() {
    echo "$1" | grep "MST Weight:" | head -n1 | grep -o '[0-9]\+'
}

# Function to extract computation time from output
get_time() {
    echo "$1" | grep "Computation Time:" | head -n1 | grep -o '[0-9.]\+'
}

# Test different graph sizes
for graph in $(ls graphs/graph_*_*.txt | sort -t_ -k2 -n); do
    echo "============================================"
    echo "Testing $graph..."
    
    # Get number of vertices from first line of the graph file
    num_vertices=$(head -n 1 "$graph" | cut -d' ' -f1)
    
    # Run first implementation (main.c)
    echo "Running main.c implementation..."
    main_output=$(mpirun -n 8 src/main.o "$num_vertices" "$graph")
    main_weight=$(get_mst_weight "$main_output")
    main_time=$(get_time "$main_output")
    
    # Run second implementation (new_main.c)
    echo "Running new_main.c implementation..."
    new_main_output=$(mpirun -n 8 src/new_main.o "$num_vertices" "$graph")
    new_main_weight=$(get_mst_weight "$new_main_output")
    new_main_time=$(get_time "$new_main_output")
    
    # Save results to file
    result_file="results/$(basename "$graph" .txt)_mpi_comparison.txt"
    
    {
        echo "Graph: $graph"
        echo "Number of vertices: $num_vertices"
        
        if [ "$main_weight" = "$new_main_weight" ]; then
            echo "✅ Test passed! MST weights match: $main_weight"
        else
            echo "❌ Test failed!"
            # Calculate weight difference and percentage
            weight_diff=$((new_main_weight - main_weight))
            weight_percent=$(echo "scale=2; ($weight_diff / $main_weight) * 100" | bc)
            echo "Weight difference: $weight_diff"
            echo "Weight difference percentage: ${weight_percent}%"
        fi

        # Always show timing metrics
        echo "Main MST weight: $main_weight"
        echo "New Main MST weight: $new_main_weight"
        echo "Main time: ${main_time}s"
        echo "New Main time: ${new_main_time}s"
        
        # Calculate and show timing metrics
        if [ ! -z "$new_main_time" ] && [ ! -z "$main_time" ]; then
            # Use bc for all floating point calculations and remove newlines
            time_diff=$(echo "$new_main_time - $main_time" | bc -l | tr -d '\n')

            if (( $(echo "$new_main_time > $main_time" | bc -l) )); then
                # Calculate slowdown
                slowdown=$(echo "scale=4; $new_main_time/$main_time" | bc -l | tr -d '\n')
                printf "Time difference: %.4fs\n" "$time_diff"
                printf "Slowdown: %.4fx\n" "$slowdown"
                echo "New implementation was slower."
            else
                # Calculate speedup
                speedup=$(echo "scale=4; $main_time/$new_main_time" | bc -l | tr -d '\n')
                efficiency=$(echo "scale=4; $speedup/2" | bc -l | tr -d '\n')
                time_percent=$(echo "scale=2; ($time_diff / $main_time) * 100" | bc -l | tr -d '\n')
                
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