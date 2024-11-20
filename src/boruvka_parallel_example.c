#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>

typedef struct {
    int src, dest, weight;
} Edge;

typedef struct {
    int parent, rank;
} Subset;

// Path compression
int find(Subset subsets[], int i) {
    if (subsets[i].parent != i) {
        subsets[i].parent = find(subsets, subsets[i].parent);
    }
    return subsets[i].parent;
}

// Union by rank
void Union(int src, int dest, Subset subsets[]) {
    int rootSrc = find(subsets, src);
    int rootDest = find(subsets, dest);

    if (rootSrc == rootDest) 
        return;
    
    if (subsets[rootSrc].rank < subsets[rootDest].rank) {
        subsets[rootSrc].parent = rootDest;
    } else if (subsets[rootSrc].rank > subsets[rootDest].rank) {
        subsets[rootDest].parent = rootSrc;
    } else {
        subsets[rootDest].parent = rootSrc;
        subsets[rootSrc].rank++;
    }
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

void boruvkaMST(int V, int E, Edge edges[], int rank, int size) {
    Subset* subsets = (Subset*)malloc(V * sizeof(Subset));
    int* cheapest = (int*)malloc(V * sizeof(int));
    
    // Initialize subsets
    for (int v = 0; v < V; v++) {
        subsets[v].parent = v;
        subsets[v].rank = 0;
        cheapest[v] = -1;
    }

    int numTrees = V;
    int localMSTWeight = 0;
    int globalMSTWeight = 0;
    int chunks = E / size;
    int start = rank * chunks;
    int end = (rank == size - 1) ? E : start + chunks;

    while (numTrees > 1) {
        // Reset cheapest edges
        for (int v = 0; v < V; ++v) {
            cheapest[v] = -1;
        }

        // Each process works on its chunk of edges
        // Find the local minimum edge
        int localMinEdgeIndex = -1;
        int localMinEdgeWeight = INT_MAX;

        for (int i = start; i < end; i++) {
            int set1 = find(subsets, edges[i].src);
            int set2 = find(subsets, edges[i].dest);

            if (set1 == set2)
                continue;

            if (cheapest[set1] == -1 || edges[cheapest[set1]].weight > edges[i].weight) {
                cheapest[set1] = i;
            }
            if (cheapest[set2] == -1 || edges[cheapest[set2]].weight > edges[i].weight) {
                cheapest[set2] = i;
            }
        }

        // Find the local minimum edge
        for (int v = 0; v < V; ++v) {
            if (cheapest[v] != -1 && edges[cheapest[v]].weight < localMinEdgeWeight) {
                localMinEdgeWeight = edges[cheapest[v]].weight;
                localMinEdgeIndex = cheapest[v];
            }
        }

        // Gather all local minimum edges at process 0
        struct {
            int weight;
            int index;
        } localMinEdge = {localMinEdgeWeight, localMinEdgeIndex}, globalMinEdge;

        MPI_Reduce(&localMinEdge, &globalMinEdge, 1, MPI_2INT, MPI_MINLOC, 0, MPI_COMM_WORLD);

        bool anyMerge = false;

        // Only rank 0 performs merging and weight calculation
        if (rank == 0) {
            if (globalMinEdge.index != -1) {
                int set1 = find(subsets, edges[globalMinEdge.index].src);
                int set2 = find(subsets, edges[globalMinEdge.index].dest);

                if (set1 != set2) {
                    printf("adding src: %d dest: %d weight: %d\n", 
                        edges[globalMinEdge.index].src, 
                        edges[globalMinEdge.index].dest,
                        edges[globalMinEdge.index].weight);
                    localMSTWeight += edges[globalMinEdge.index].weight;
                    Union(set1, set2, subsets);
                    numTrees--;
                    anyMerge = true;
                }
            }
        }

        // Synchronize all processes before proceeding
        MPI_Barrier(MPI_COMM_WORLD);

        // Broadcast updated information to all processes
        MPI_Bcast(subsets, V * sizeof(Subset), MPI_BYTE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&numTrees, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&anyMerge, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

        if (!anyMerge) break;
    }

    // Reduce MSTWeight to process 0
    MPI_Reduce(&localMSTWeight, &globalMSTWeight, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Final MST weight is %d\n", globalMSTWeight);
    }

    free(subsets);
    free(cheapest);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2) {
        if (rank == 0) {
            printf("Usage: %s <input_file>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int V, E;
    Edge* edges;
    double start = 0, end = 0;

    // Only rank 0 reads the file and broadcasts the data
    if (rank == 0) {
        readGraphFromFile(argv[1], &V, &E, &edges);
    }

    if (rank == 0) start = MPI_Wtime();

    // Broadcast V and E to all processes
    MPI_Bcast(&V, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&E, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Other processes allocate memory for edges
    if (rank != 0) {
        edges = (Edge*)malloc(E * sizeof(Edge));
    }

    // Define the MPI data type for the Edge structure
    MPI_Datatype MPI_Edge;
    int lengths[3] = {1, 1, 1};
    const MPI_Aint displacements[3] = {offsetof(Edge, src), offsetof(Edge, dest), offsetof(Edge, weight)};
    MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_INT};
    MPI_Type_create_struct(3, lengths, displacements, types, &MPI_Edge);

    // Resize the data type to remove padding
    MPI_Aint lb, extent;
    MPI_Type_get_extent(MPI_Edge, &lb, &extent);
    MPI_Type_create_resized(MPI_Edge, lb, sizeof(Edge), &MPI_Edge);
    MPI_Type_commit(&MPI_Edge);

    // Broadcast edges to all processes using the custom MPI data type
    MPI_Bcast(edges, E, MPI_Edge, 0, MPI_COMM_WORLD);

    boruvkaMST(V, E, edges, rank, size);

    if (rank == 0) {
        end = MPI_Wtime();
        printf("Time taken: %f\n", end - start);
    }

    free(edges);
    MPI_Type_free(&MPI_Edge);
    MPI_Finalize();
    return 0;
}