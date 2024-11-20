#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <stdbool.h>

// Structure to represent an edge
typedef struct {
    int src;
    int dest;
    int weight;
} Edge;

// Structure to represent a component
typedef struct {
    int parent;
    int rank;
} Component;

// Function to create MPI Edge datatype
MPI_Datatype create_mpi_edge_type() {
    MPI_Datatype MPI_EDGE;
    int blocklengths[3] = {1, 1, 1};
    MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_INT};
    MPI_Aint offsets[3];
    offsets[0] = offsetof(Edge, src);
    offsets[1] = offsetof(Edge, dest);
    offsets[2] = offsetof(Edge, weight);
    MPI_Type_create_struct(3, blocklengths, offsets, types, &MPI_EDGE);
    MPI_Type_commit(&MPI_EDGE);
    return MPI_EDGE;
}

// Function to find root of a component (path compression)
int find_root(Component* components, int vertex) {
    if (components[vertex].parent != vertex) {
        components[vertex].parent = find_root(components, components[vertex].parent);
    }
    return components[vertex].parent;
}

// Function to unite two components (union by rank)
void union_components(Component* components, int x, int y) {
    int root_x = find_root(components, x);
    int root_y = find_root(components, y);

    if (root_x != root_y) {
        if (components[root_x].rank < components[root_y].rank) {
            components[root_x].parent = root_y;
        } else if (components[root_x].rank > components[root_y].rank) {
            components[root_y].parent = root_x;
        } else {
            components[root_y].parent = root_x;
            components[root_x].rank++;
        }
    }
}

// Function to find minimum weight edge for a vertex range
void find_min_edges(Edge* edges, int num_edges, Component* components, 
                   int start_vertex, int end_vertex, Edge* min_edges) {
    
    for (int v = start_vertex; v < end_vertex; v++) {
        min_edges[v].weight = INT_MAX;
        int root_v = find_root(components, v);

        for (int e = 0; e < num_edges; e++) {
            if (edges[e].src == v || edges[e].dest == v) {
                int other = (edges[e].src == v) ? edges[e].dest : edges[e].src;
                int root_other = find_root(components, other);

                if (root_v != root_other && edges[e].weight < min_edges[v].weight) {
                    min_edges[v] = edges[e];
                }
            }
        }
    }
}

// Main Boruvka algorithm
void parallel_boruvka(Edge* edges, int num_edges, int num_vertices, Edge* mst, int* mst_size, MPI_Datatype MPI_EDGE) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Initialize components (each vertex in its own component)
    Component* components = (Component*)malloc(num_vertices * sizeof(Component));
    for (int i = 0; i < num_vertices; i++) {
        components[i].parent = i;
        components[i].rank = 0;
    }

    // Calculate vertices per process
    int vertices_per_proc = num_vertices / size;
    int start_vertex = rank * vertices_per_proc;
    int end_vertex = (rank == size - 1) ? num_vertices : start_vertex + vertices_per_proc;

    Edge* min_edges = (Edge*)malloc(num_vertices * sizeof(Edge));
    Edge* global_min_edges = (Edge*)malloc(num_vertices * sizeof(Edge));
    *mst_size = 0;
    bool components_merged = true;

    // Initialize min_edges with INT_MAX weight
    for (int i = 0; i < num_vertices; i++) {
        min_edges[i].weight = INT_MAX;
    }

    while (components_merged) {
        components_merged = false;

        // Find minimum weight edges for assigned vertices
        find_min_edges(edges, num_edges, components, start_vertex, end_vertex, min_edges);

        // Gather all minimum edges
        MPI_Allgather(min_edges + start_vertex, vertices_per_proc, MPI_EDGE,
                     global_min_edges, vertices_per_proc, MPI_EDGE, MPI_COMM_WORLD);

        // All processes process edges in the same order
        for (int v = 0; v < num_vertices; v++) {
            if (global_min_edges[v].weight != INT_MAX) {
                int root1 = find_root(components, global_min_edges[v].src);
                int root2 = find_root(components, global_min_edges[v].dest);

                if (root1 != root2) {
                    union_components(components, root1, root2);
                    if (rank == 0) {  // Only rank 0 builds the MST
                        mst[*mst_size] = global_min_edges[v];
                        (*mst_size)++;
                    }
                    components_merged = true;
                }
            }
        }

        // Broadcast whether components were merged
        MPI_Bcast(&components_merged, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    }

    // Cleanup
    free(components);
    free(min_edges);
    free(global_min_edges);
}

// Function to read graph from file
void readGraphFromFile(const char *filename, int* V, int* E, Edge** edges) {
    printf("Reading graph from file...\n");
    
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Read vertex and edge count
    if (fscanf(file, "%d %d", V, E) != 2) {
        perror("Error reading graph metadata");
        fclose(file);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Allocate memory for edges
    *edges = (Edge *)malloc(*E * sizeof(Edge));
    if (!*edges) {
        perror("Memory allocation failed");
        fclose(file);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Read edges
    for (int i = 0; i < *E; i++) {
        if (fscanf(file, "%d %d %d", &(*edges)[i].src, &(*edges)[i].dest, &(*edges)[i].weight) != 3) {
            perror("Error reading edges");
            free(*edges);
            fclose(file);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    printf("Finished reading\n");
    fclose(file);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Create MPI Edge datatype
    MPI_Datatype MPI_EDGE = create_mpi_edge_type();

    int num_vertices, num_edges;
    Edge* edges;
    double start = 0, end = 0;

    // Only rank 0 reads the file
    if (rank == 0) {
        readGraphFromFile(argv[1], &num_vertices, &num_edges, &edges);
    }

    // Broadcast graph information to all processes
    MPI_Bcast(&num_vertices, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_edges, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate memory for edges in non-root processes
    if (rank != 0) {
        edges = (Edge*)malloc(num_edges * sizeof(Edge));
    }

    // Broadcast the edge list to all processes
    MPI_Bcast(edges, num_edges, MPI_EDGE, 0, MPI_COMM_WORLD);

    Edge* mst = (Edge*)malloc((num_vertices - 1) * sizeof(Edge));
    int mst_size;

    if (rank == 0) start = MPI_Wtime();
    parallel_boruvka(edges, num_edges, num_vertices, mst, &mst_size, MPI_EDGE);
    if (rank == 0) {
        end = MPI_Wtime();
    }

    // Print results from root process
    if (rank == 0) {
        int totalCostMST = 0;
        printf("Minimum Spanning Tree edges:\n");
        for (int i = 0; i < mst_size; i++) {
            totalCostMST += mst[i].weight;
            printf("Edge %d-%d: weight %d\n", 
                   mst[i].src, mst[i].dest, mst[i].weight);
        }
        printf("Total cost MST: %d\n", totalCostMST);
        printf("Time taken: %f\n", end - start);
    }

    // Cleanup
    free(edges);
    free(mst);
    MPI_Type_free(&MPI_EDGE);
    MPI_Finalize();
    return 0;
}