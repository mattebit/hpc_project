#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include <stdint.h>

typedef struct {
    uint64_t start_vertex;
    uint64_t end_vertex;
    int weight;
} Edge;

// Function to ensure graph connectivity
void ensure_connectivity(Edge* edges, uint64_t num_vertices, uint64_t base_edges) {
    #pragma omp parallel for
    for (uint64_t i = 0; i < num_vertices - 1; i++) {
        edges[i].start_vertex = i;
        edges[i].end_vertex = i + 1;
        edges[i].weight = (rand() % 1000) + 1; // Random integer between 1-1000
    }
}

Edge* generate_random_graph(uint64_t num_vertices, uint64_t num_edges, uint64_t* actual_edges) {
    if (num_edges < num_vertices - 1) {
        num_edges = num_vertices - 1; // Ensure minimum edges for connectivity
    }

    // Allocate memory for edges
    Edge* edges = (Edge*)malloc(num_edges * sizeof(Edge));
    if (!edges) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    // Ensure basic connectivity first
    ensure_connectivity(edges, num_vertices, num_vertices - 1);

    // Generate remaining random edges
    #pragma omp parallel
    {
        unsigned int seed = omp_get_thread_num() + time(NULL);
        
        #pragma omp for
        for (uint64_t i = num_vertices - 1; i < num_edges; i++) {
            do {
                edges[i].start_vertex = rand_r(&seed) % num_vertices;
                edges[i].end_vertex = rand_r(&seed) % num_vertices;
            } while (edges[i].start_vertex == edges[i].end_vertex);
            
            edges[i].weight = (rand_r(&seed) % 1000) + 1; // Random integer between 1-1000
            
            if (edges[i].start_vertex > edges[i].end_vertex) {
                uint64_t temp = edges[i].start_vertex;
                edges[i].start_vertex = edges[i].end_vertex;
                edges[i].end_vertex = temp;
            }
        }
    }

    *actual_edges = num_edges;
    return edges;
}

void write_edges_to_file(const char* filename, Edge* edges, uint64_t num_edges) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Cannot open file for writing\n");
        return;
    }

    uint64_t max_vertex = 0;
    for (uint64_t i = 0; i < num_edges; i++) {
        if (edges[i].start_vertex > max_vertex) max_vertex = edges[i].start_vertex;
        if (edges[i].end_vertex > max_vertex) max_vertex = edges[i].end_vertex;
    }
    fprintf(f, "%lu %lu\n", max_vertex + 1, num_edges);

    for (uint64_t i = 0; i < num_edges; i++) {
        fprintf(f, "%lu %lu %d\n",  // Changed %.6f to %d for integer weights
                edges[i].start_vertex, 
                edges[i].end_vertex, 
                edges[i].weight);
    }

    fclose(f);
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <NUM_VERTICES> <NUM_EDGES>\n", argv[0]);
        return 1;
    }

    uint64_t num_vertices = strtoull(argv[1], NULL, 10);
    uint64_t num_edges = strtoull(argv[2], NULL, 10);
    
    if (num_edges < num_vertices - 1) {
        fprintf(stderr, "Error: Number of edges must be at least n-1 (where n is the number of vertices)\n");
        return 1;
    }

    // Initialize random seed
    srand(time(NULL));

    // Generate graph
    uint64_t actual_edges;
    Edge* edges = generate_random_graph(num_vertices, num_edges, &actual_edges);

    // Write edges to file
    write_edges_to_file("random_graph.txt", edges, actual_edges);

    // Cleanup
    free(edges);

    return 0;
}