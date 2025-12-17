module load mpich-3.2
mpicc -std=c99 -O3 -fopenmp -march=native -mtune=native -ftree-vectorize -funroll-loops -flto -o src/main.o src/main.c