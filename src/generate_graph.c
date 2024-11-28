#include <omp.h>
#include <stdlib.h>
#include <stdio.h>

// # define DEBUG
#define LOGGING

int NODE_COUNT = 40000;
int MAX_EDGE_VALUE = __INT_MAX__;

int MY_NODES_FROM = 0;
int MY_NODES_TO = 100;

char* FILE_NAME = "graph.txt";

double last_time = 0.0;
double start_time = 0.0;

#define NULL ((void *) 0)

int **allocate_and_init_matrix() {
    int **min_graph = (int **) malloc(NODE_COUNT * sizeof(int *));
    for (int i = 0; i < NODE_COUNT; i++) {
        min_graph[i] = (int *) calloc(NODE_COUNT, sizeof(int));
    }
    return min_graph;
}

int **fill_graph() {
    srand(127);
    int **matrix = allocate_and_init_matrix();

    int *used_values = (int *) calloc(NODE_COUNT * NODE_COUNT * 4, sizeof(int));

#pragma omp parallel for
    for (int i = 0; i < NODE_COUNT; i++) {
        for (int j = 0; j < NODE_COUNT; j++) {
            if (i == j) {
                matrix[i][j] = -1;
            } else {
                int gen;
                do {
                    gen = rand() % (NODE_COUNT * NODE_COUNT * 4);
                } while (used_values[gen]);
                used_values[gen] = 1;
                matrix[i][j] = gen;
                matrix[j][i] = gen;
            }
        }
    }
    free(used_values);

    return matrix;
}

void print_matrix(int** matrix) {
    // Print the adjacency matrix
    for (int i = 0; i < NODE_COUNT; i++)
    {
        for (int j = 0; j < NODE_COUNT; j++)
        {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}

void save_matrix(int** matrix) {
    FILE *fp = fopen(FILE_NAME, "w");
    for (int i = 0; i < NODE_COUNT; i++)
    {
        for (int j = 0; j < NODE_COUNT; j++)
        {
            fprintf(fp,"%d ", matrix[i][j]);
        }
        fprintf(fp,"\n");
    }
    fclose(fp);
}

int main(int argc, char **argv) {
    int **graph = fill_graph();
    save_matrix(graph);
    free(graph);
}
