# Compiler and flags
CC=gcc
MPICC=mpicc
PARALLEL_CFLAGS=-std=c99 -O3 -fopenmp -march=native -mtune=native -ftree-vectorize -funroll-loops -flto
SERIAL_CFLAGS=-std=c99 -O3

# Paths
PROJECT_DIR=$(HOME)/hpc_project
SRC_DIR=$(PROJECT_DIR)/src
SERIAL_SRC=$(PROJECT_DIR)/serial.c
PARALLEL_SRC=$(SRC_DIR)/main.c

# Executables
SERIAL_EXE=$(PROJECT_DIR)/serial.o
PARALLEL_EXE=$(SRC_DIR)/main.o

# Default target
all: clean compile submit

# Compile both implementations
compile:
	@echo "Compiling serial implementation..."
	$(CC) $(SERIAL_CFLAGS) $(SERIAL_SRC) -o $(SERIAL_EXE)
	@chmod +x $(SERIAL_EXE)
	@echo "Compiling parallel implementation..."
	@. /etc/profile.d/modules.sh && module load mpich-3.2 && \
	$(MPICC) $(PARALLEL_CFLAGS) $(PARALLEL_SRC) -o $(PARALLEL_EXE)
	@chmod +x $(PARALLEL_EXE)
	@echo "Setting permissions..."
	@chmod 755 $(SERIAL_EXE) $(PARALLEL_EXE)

# Clean old files
clean:
	@echo "Cleaning up old files..."
	rm -f hpc_cmp_impl_*.sh
	rm -f hpc_cmp_impl_mpi*
	rm -f bench_*.log
	rm -f parallel.sh.*

# Submit all benchmark jobs
submit:
	@echo "Submitting benchmark jobs..."
	./benchmark.sh

# Monitor jobs
monitor:
	@watch "qstat -u $(USER)"

# Show job outputs as they arrive
watch-outputs:
	@watch "cat bench_mpi*_*.log 2>/dev/null"

# Cancel all jobs
cancel:
	@echo "Canceling all jobs..."
	@for job in `qstat -u $(USER) | grep "hpc-hea" | awk '{print $$1}' | cut -d'.' -f1`; do \
		echo "Canceling job $$job"; \
		qdel $$job; \
	done

# Help target
help:
	@echo "Available targets:"
	@echo "  make all         - Clean, compile and submit jobs"
	@echo "  make compile    - Compile serial and parallel implementations"
	@echo "  make clean      - Remove compiled files and old outputs"
	@echo "  make submit     - Submit benchmark jobs"
	@echo "  make monitor    - Monitor job status"
	@echo "  make watch-outputs - Watch job outputs in real time"
	@echo "  make cancel     - Cancel all running jobs"

.PHONY: all compile clean submit monitor watch-outputs cancel help