#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>
#include <string.h>

typedef struct {
    int parent, rank;
} Component;

typedef struct {
    int src;
    int dest;
    int weight;
} Edge;

// Path compression
int find_root(Component components[], int i) {
    if (components[i].parent != i) {
        components[i].parent = find_root(components, components[i].parent);
    }
    return components[i].parent;
}

// Union by rank
void Union(int src, int dest, Component components[]) {
    int rootSrc = find_root(components, src);
    int rootDest = find_root(components, dest);

    if (rootSrc == rootDest) 
        return;
    
    if (components[rootSrc].rank < components[rootDest].rank) {
        components[rootSrc].parent = rootDest;
    } else if (components[rootSrc].rank > components[rootDest].rank) {
        components[rootDest].parent = rootSrc;
    } else {
        components[rootDest].parent = rootSrc;
        components[rootSrc].rank++;
    }
}

// Function to read graph from file
void readGraphFromFile(const char *filename, int* V, int* E, int*** graph) {
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
    *graph = (int**)malloc(*V * sizeof(int*));
    for (int i = 0; i < *V; i++) {
        (*graph)[i] = (int*)malloc(*V * sizeof(int));
        for (int j = 0; j < *V; j++) {
            (*graph)[i][j] = INT_MAX;
        }
    }
    printf("allocated and initialized the matrix");
    //Read edges
    for (int i = 0; i < *E; i++) {
        int src, dest, weight;
        if (fscanf(file, "%d %d %d", &src, &dest, &weight) != 3) {
            perror("Error reading edges");
            for (int j = 0; j < *V; j++) {
                free((*graph)[j]);
            }
            free(*graph);
            fclose(file);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        (*graph)[src][dest] = weight;
    }
    printf("Finished reading\n");
    fclose(file);
}

// Custom reduction operation to find the minimum weight edge across components
void findMinimumComponentEdges(void* in, void* inout, int* len, MPI_Datatype* datatype) {
    Edge* inEdges = (Edge*)in;
    Edge* inoutEdges = (Edge*)inout;

    for (int i = 0; i < *len; i++) {
        // Compare and update if the incoming edge has a lower weight
        if (inEdges[i].weight < inoutEdges[i].weight) {
            inoutEdges[i] = inEdges[i];
        }
    }
}

void boruvkaMST(int V, int** graph, int rank, int size) {
    Component* components = (Component*)malloc(V * sizeof(Component));
    int* cheapest = (int*)malloc(V * sizeof(int));
    
    // Initialize components
    for (int v = 0; v < V; v++) {
        components[v].parent = v;
        components[v].rank = 0;
        cheapest[v] = -1;
    }

    int numTrees = V;
    int globalMSTWeight = 0;
    int chunks = V / size;
    int start = rank * chunks;
    int end = (rank == size - 1) ? V : start + chunks;

    // Define the custom MPI datatype for Edge
    MPI_Datatype MPI_Edge;
    MPI_Type_contiguous(3, MPI_INT, &MPI_Edge);
    MPI_Type_commit(&MPI_Edge);

    // Define the custom reduction operation
    MPI_Op minEdgeOp;
    MPI_Op_create(findMinimumComponentEdges, 1, &minEdgeOp);

    // Array to store minimum edges for each vertex
    Edge* localMinEdges = (Edge*)malloc(V * sizeof(Edge));
    for (int i = 0; i < V; i++) {
        localMinEdges[i].src = -1;
        localMinEdges[i].dest = -1;
        localMinEdges[i].weight = INT_MAX;
    }

    while (numTrees > 1) {
        if (rank == 0) printf("Number of components: %d\n", numTrees);
        
        // Reset minimum edges
        for (int i = 0; i < V; i++) {
            localMinEdges[i].src = -1;
            localMinEdges[i].dest = -1;
            localMinEdges[i].weight = INT_MAX;
        }

        // Parallel search for minimum weight edges between components
        for (int v = start; v < end; v++) {
            int root_v = find_root(components, v);

            for (int u = 0; u < V; u++) {
                int root_u = find_root(components, u);

                // Skip if vertices are in the same component or no edge exists
                if (root_v == root_u || graph[v][u] == INT_MAX)
                    continue;

                // Update minimum edge for this vertex if a lower weight edge is found
                if (graph[v][u] < localMinEdges[v].weight) {
                    localMinEdges[v].src = v;
                    localMinEdges[v].dest = u;
                    localMinEdges[v].weight = graph[v][u];
                }
            }
        }

        // Gather and reduce minimum edges across all processes
        Edge* globalMinEdges = (Edge*)malloc(V * sizeof(Edge));
        MPI_Allreduce(localMinEdges, globalMinEdges, V, MPI_Edge, minEdgeOp, MPI_COMM_WORLD);

        // Track whether any merges occurred in this iteration
        int mergeCount = 0;
        MPI_Allreduce(&mergeCount, &mergeCount, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        // Add minimum weight edges and merge components
        for (int i = 0; i < V; i++) {
            if (globalMinEdges[i].src == -1 || globalMinEdges[i].dest == -1)
                continue;

            int root_v = find_root(components, globalMinEdges[i].src);
            int root_u = find_root(components, globalMinEdges[i].dest);

            if (root_v != root_u) {
                Union(root_v, root_u, components);
                
                if (rank == 0) { 
                    globalMSTWeight += globalMinEdges[i].weight;
                }
                
                numTrees--;
                mergeCount++;
            }
        }

        // Break if no merges occurred
        if (mergeCount == 0) break;

        // Optional: Add a barrier to synchronize processes
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == 0) {
        printf("Final MST weight is %d\n", globalMSTWeight);
    }

    // Free allocated memory
    free(components);
    free(cheapest);
    free(localMinEdges);

    // Free the custom MPI datatype and operation
    MPI_Type_free(&MPI_Edge);
    MPI_Op_free(&minEdgeOp);
}

int main(int argc, char** argv) {
    
     if (argc != 2) {
        return 1;
    }

    
    int V, E;
    double start = 0, end = 0;
    int** graph;
    readGraphFromFile(argv[1], &V, &E, &graph);
    
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

   

    // Allocate memory for the adjacency matrix in non-root process
    if (rank != 0) {
        graph = (int**)malloc(V * sizeof(int*));
        if (graph == NULL) {
            fprintf(stderr, "Process %d: Memory allocation failed\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int i = 0; i < V; i++) {
            graph[i] = (int*)malloc(V * sizeof(int));
            if (graph[i] == NULL) {
                fprintf(stderr, "Process %d: Memory allocation failed for row %d\n", rank, i);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            // Initialize with a specific value
            for (int j = 0; j < V; j++) {
                graph[i][j] = INT_MAX;
            }
        }
    }

    // MPI_Barrier(MPI_COMM_WORLD); //do we really need it???

    if (rank == 0) {
        printf("Finished broadcasting\n");
        printf("Starting computation of the MST... \n");
        start = MPI_Wtime();
    }

    boruvkaMST(V, graph, rank, size);

    if (rank == 0) {
        end = MPI_Wtime();
        printf("Time taken: %f\n", end - start);
    }

    //free matrix
    for (int i = 0; i < V; i++) {
        free(graph[i]);
    }
    free(graph);
    MPI_Finalize();
    return 0;
}