mpicc -std=c99 -O3 -march=native -mtune=native -ftree-vectorize -funroll-loops -flto -o src/main src/main.c