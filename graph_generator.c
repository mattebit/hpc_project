#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include <stdint.h>

// Struct to hold edge information
typedef struct {
    uint64_t start_vertex;
    uint64_t end_vertex;
    double weight;
} Edge;

// Parallel Kronecker Graph Generator
Edge* generate_kronecker_graph(int scale, int edgefactor, uint64_t* num_edges) {
    // Set number of vertices and edges
    uint64_t N = 1ULL << scale;
    uint64_t M = edgefactor * N;
    *num_edges = M;

    // Initiator probabilities
    double A = 0.57, B = 0.19, C = 0.19;
    double ab = A + B;
    double c_norm = C / (1 - (A + B));
    double a_norm = A / (A + B);

    // Allocate memory for edges
    Edge* edges = (Edge*)malloc(M * sizeof(Edge));
    if (!edges) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    // Seed random number generator
    srand(time(NULL));

    // Parallel generation of edges
    #pragma omp parallel
    {
        // Use thread-local random state to avoid contention
        unsigned int seed = omp_get_thread_num() + time(NULL);

        // Parallel loop over edges
        #pragma omp for
        for (uint64_t k = 0; k < M; k++) {
            uint64_t ii = 0, jj = 0;

            // Kronecker graph generation for each edge
            for (int ib = 0; ib < scale; ib++) {
                // Generate random values
                double r1 = (double)rand_r(&seed) / RAND_MAX;
                double r2 = (double)rand_r(&seed) / RAND_MAX;

                // Set bits based on probabilities
                int ii_bit = (r1 > ab) ? 1 : 0;
                int jj_bit = (r2 > (c_norm * ii_bit + a_norm * (1 - ii_bit))) ? 1 : 0;

                // Update vertex indices
                ii |= (ii_bit << ib);
                jj |= (jj_bit << ib);
            }

            // Generate weight
            double weight = (double)rand_r(&seed) / RAND_MAX;

            // Store edge
            edges[k].start_vertex = ii;
            edges[k].end_vertex = jj;
            edges[k].weight = weight;
        }
    }

    // Permute vertex labels
    #pragma omp parallel for
    for (uint64_t k = 0; k < M; k++) {
        edges[k].start_vertex = (edges[k].start_vertex * N) % N;
        edges[k].end_vertex = (edges[k].end_vertex * N) % N;
    }

    return edges;
}

// Function to write edges to a file
void write_edges_to_file(const char* filename, Edge* edges, uint64_t num_edges) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Cannot open file for writing\n");
        return;
    }

    for (uint64_t i = 0; i < num_edges; i++) {
        fprintf(f, "%lu %lu %f\n", 
                edges[i].start_vertex, 
                edges[i].end_vertex, 
                edges[i].weight);
    }

    fclose(f);
}

int main(int argc, char* argv[]) {
    // Check command-line arguments
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <SCALE> <EDGEFACTOR>\n", argv[0]);
        return 1;
    }

    // Parse arguments
    int scale = atoi(argv[1]);
    int edgefactor = atoi(argv[2]);

    // Generate graph
    uint64_t num_edges;
    Edge* edges = generate_kronecker_graph(scale, edgefactor, &num_edges);

    // Write edges to file
    write_edges_to_file("kronecker_graph.txt", edges, num_edges);

    // Cleanup
    free(edges);

    return 0;
}