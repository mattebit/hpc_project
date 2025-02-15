#!/bin/bash

# Clean up old files
rm -f hpc_cmp_impl_*.sh
rm -f bench_*.log

sub_job() {
    # Args:
    # 1: MPI_CORES - number of MPI processes
    # 2: OMP_THREADS - OpenMP threads per process
    # 3: MEM_GB - total memory in GB
    local job_name="mpi${1}_omp${2}_${3}gb"
    local ncpus=$((${1} * ${2}))
    echo "Creating job with ${1} MPI processes, ${2} OpenMP threads (${ncpus} cores), ${3}GB RAM"
    
    cat << 'EOF' > hpc_cmp_impl_${job_name}.sh
#!/bin/bash
#PBS -l select=1:ncpus=NCPUS:mem=MEMGB
#PBS -l walltime=1:10:00
#PBS -q short_cpuQ

module load mpich-3.2
export OMP_NUM_THREADS=OMPTH

# Compile both implementations
# gcc -std=c99 -O3 ~/hpc_project/serial.c -o ~/hpc_project/serial.o
# mpicc -std=c99 -O3 -fopenmp -march=native -mtune=native -ftree-vectorize -funroll-loops -flto -o ~/hpc_project/src/main.o ~/hpc_project/src/main.c

for graph in ~/hpc_project/graphs/graph_*_*.txt; do
    echo "============================================"
    echo "Testing $graph..."
    
    # Get number of vertices from first line of the graph file
    num_vertices=$(head -n 1 "$graph" | cut -d' ' -f1)
    
    # Run serial implementation
    echo "Running serial implementation..."
    serial_output=$(~/hpc_project/serial.o "$num_vertices" "$graph")
    serial_weight=$(echo "$serial_output" | grep "MST Weight:" | head -n1 | grep -o '[0-9]\+')
    serial_time=$(echo "$serial_output" | grep "Computation Time:" | head -n1 | grep -o '[0-9.]\+')
    
    # Run parallel implementation
    echo "Running parallel implementation..."
    parallel_output=$(mpirun.actual -n MPINUM ~/hpc_project/src/main.o "$num_vertices" "$graph")
    parallel_weight=$(echo "$parallel_output" | grep "MST Weight:" | head -n1 | grep -o '[0-9]\+')
    parallel_time=$(echo "$parallel_output" | grep "Computation Time:" | head -n1 | grep -o '[0-9.]\+')
    
    # Output results
    echo "Graph: $graph"
    echo "Number of vertices: $num_vertices"
    echo "Serial MST weight: $serial_weight"
    echo "Parallel MST weight: $parallel_weight"
    echo "Serial time: ${serial_time}s"
    echo "Parallel time: ${parallel_time}s"
    
    # Compare weights
    if [ "$serial_weight" = "$parallel_weight" ]; then
        echo "✅ MST weights match"
    else
        echo "❌ MST weights differ!"
    fi
    
    # Calculate speedup/slowdown
    if [ -n "$parallel_time" ] && [ -n "$serial_time" ]; then
        time_diff=$(echo "$parallel_time - $serial_time" | bc -l | tr -d '\n')
        speedup=$(echo "scale=4; $serial_time/$parallel_time" | bc)
        efficiency=$(echo "scale=4; $speedup/MPINUM" | bc -l | tr -d '\n')
        time_percent=$(echo "scale=2; ($time_diff / $serial_time) * 100" | bc -l | tr -d '\n')
        
        printf "Time difference: %.4fs\n" "$time_diff"
        printf "Speedup: %.4fx\n" "$speedup"
        printf "Parallel efficiency: %.4f\n" "$efficiency"
        printf "Time improvement: %.2f%%\n" "$time_percent"
    fi
    echo "----------------------------------------"
done

echo "============================================"
echo "Testing completed"
EOF

    # Replace placeholders with actual values
    sed -i "s/NCPUS/${ncpus}/g" hpc_cmp_impl_${job_name}.sh
    sed -i "s/MEMGB/${3}gb/g" hpc_cmp_impl_${job_name}.sh
    sed -i "s/OMPTH/${2}/g" hpc_cmp_impl_${job_name}.sh
    sed -i "s/MPINUM/${1}/g" hpc_cmp_impl_${job_name}.sh
    
    chmod +x hpc_cmp_impl_${job_name}.sh
    qsub -o bench_${job_name}.log hpc_cmp_impl_${job_name}.sh
}

# Launch configurations with 2 OpenMP threads per MPI process
echo "Launching jobs with 2 OpenMP threads per MPI process..."
sub_job 2 2 128   # 2 MPI cores, 2 OpenMP threads, 128GB RAM
sub_job 4 2 128   # 4 MPI cores, 2 OpenMP threads, 128GB RAM
sub_job 8 2 256   # 8 MPI cores, 2 OpenMP threads, 256GB RAM
sub_job 16 2 256  # 16 MPI cores, 2 OpenMP threads, 256GB RAM
sub_job 32 2 512  # 32 MPI cores, 2 OpenMP threads, 512GB RAM

# Monitor jobs
watch "qstat -u $USER"