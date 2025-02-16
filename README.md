# HPC Project: Parallel MST Implementation 

## A high-performance hybrid MPI+OpenMP implementation of Minimum Spanning Tree algorithms

This project delivers a scalable parallel implementation of MST algorithms designed for high-performance computing clusters, leveraging both MPI for distributed memory and OpenMP for shared memory parallelization.

## Table of Contents

- [Key Features](#key-features-)
- [Project Structure](#project-structure-)
- [Build & Run](#build--run-%EF%B8%8F)
- [Usage Guide](#usage-guide-)
- [Dependencies](#dependencies-)
- [Cluster Configuration](#cluster-configuration-)
- [Performance Analysis](#performance-analysis-)
- [Scripts](#scripts-%EF%B8%8F)
- [Future Improvements](#future-improvements-)

## Key Features 

- **Hybrid Parallelization**: Combined MPI+OpenMP implementation
- **Reference Implementation**: Serial version for comparison
- **Automated Testing**: Test graph generation suite
- **Performance Analysis**: Comprehensive benchmarking tools
- **Visualization**: Performance metrics plotting

## Project Structure

```bash
├── src/
│   └── main.c                      # Parallel MPI+OpenMP implementation
├── serial.c                        # Serial implementation
├── graph_generator.sh              # Test graph generator
├── utils/                          # Analysis tools
├── Makefile                        # Build, test and monitor automation (HPC settings)  
├── compare_mpi_implementations.sh  # Test different MPI implementations (local)
└── compare_implementations.sh      # Compare parallel vs serial (local)
```

## Build & Run

1. Compile both implementations:
```bash
make compile
```

2. Generate test graphs:
```bash
./generate_graphs.sh
```

3. Launch benchmarks:
```bash
make submit
```

4. Monitor jobs:
```bash
make monitor
```

## Usage Guide

### Local Testing
```bash
mpirun -np <processes> --bind-to none src/main.o <vertices> <graph_file>
```

### Cluster Deployment
```bash
make submit         # Submit jobs
make monitor       # Check status
make clean         # Cleanup
make cancel        # Stop jobs
make watch-output  # View results
```

## Dependencies

![MPI](https://img.shields.io/badge/MPI-MPICH%203.2-blue?style=flat-square)
![OpenMP](https://img.shields.io/badge/OpenMP-Enabled-green?style=flat-square)
![GCC](https://img.shields.io/badge/GCC-Required-red?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.x-yellow?style=flat-square)
![PBS](https://img.shields.io/badge/PBS-Scheduler-orange?style=flat-square)

## Cluster Configuration ⚡

| Resource | Specification |
|----------|---------------|
| Nodes | 1-32 compute nodes |
| Memory | 16GB-512GB per node |
| Queue | short_cpuQ |
| Wall Time | 1:00 hour max |
| MPI Processes | 2-32 per node |
| OpenMP Threads | 2-16 per process |

## Performance Analysis

The project includes comprehensive performance analysis tools:
- Speedup measurements
- Efficiency calculations
- Scalability analysis
- Resource utilization metrics

## Scripts

| Script | Description |
|--------|-------------|
| `benchmark.sh` | Runs performance tests |
| `compare_implementations.sh` | Compares parallel vs serial |
| `generate_graphs.sh` | Creates test graphs |
| `hpc_generate_graphs.sh` | Generates large-scale graphs |

## Future Improvements

- Enhance visualization capabilities
- Optimize communication overhead
- Optimize memory management