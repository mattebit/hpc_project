#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>
#include <time.h>

// Structure to represent an edge in the graph
typedef struct {
    int src;
    int dest;
    int weight;
} Edge;

// Disjoint Set data structure for efficient connectivity checks
typedef struct {
    int* parent;
    int* rank;
    int n;
} DisjointSet;

// Initialize Disjoint Set
DisjointSet* createDisjointSet(int n) {
    DisjointSet* ds = (DisjointSet*)malloc(sizeof(DisjointSet));
    ds->parent = (int*)malloc(n * sizeof(int));
    ds->rank = (int*)malloc(n * sizeof(int));
    ds->n = n;

    for (int i = 0; i < n; i++) {
        ds->parent[i] = i;
        ds->rank[i] = 0;
    }
    return ds;
}

// Modified find operation with pointer jumping
int find(DisjointSet* ds, int x) {
    // First pass: Find the root
    int root = x;
    while (ds->parent[root] != root) {
        root = ds->parent[root];
    }
    
    // Second pass: Path compression with pointer jumping
    int current = x;
    int next;
    while (current != root) {
        next = ds->parent[current];
        ds->parent[current] = root;
        current = next;
    }
    
    return root;
}
// Union by rank
void unionSets(DisjointSet* ds, int x, int y) {
    int rootX = find(ds, x);
    int rootY = find(ds, y);

    if (rootX == rootY)
        return;

    if (ds->rank[rootX] < ds->rank[rootY])
        ds->parent[rootX] = rootY;
    else if (ds->rank[rootX] > ds->rank[rootY])
        ds->parent[rootY] = rootX;
    else {
        ds->parent[rootY] = rootX;
        ds->rank[rootX]++;
    }
}

// Borůvka's algorithm for Minimum Spanning Tree using edge list
Edge* boruvkaMST(Edge* edges, int V, int E, int* mstEdgeCount) {
    DisjointSet* ds = createDisjointSet(V);
    Edge* mst = malloc((V-1) * sizeof(Edge));  // MST has V-1 edges
    *mstEdgeCount = 0;

    int componentsCount = V;
    
    while (componentsCount > 1) {
        printf("Number of components: %d\n", componentsCount);
        
        // Array to store cheapest edge for each component
        Edge* cheapestEdge = malloc(V * sizeof(Edge));
        for (int i = 0; i < V; i++) {
            cheapestEdge[i].weight = INT_MAX;
        }

        // Find cheapest edges
        for (int i = 0; i < E; i++) {
            int src = edges[i].src;
            int dest = edges[i].dest;
            int weight = edges[i].weight;

            int rootSrc = find(ds, src);
            int rootDest = find(ds, dest);

            if (rootSrc != rootDest) {
                // Update cheapest edge for both components
                if (weight < cheapestEdge[rootSrc].weight) {
                    cheapestEdge[rootSrc] = edges[i];
                }
                if (weight < cheapestEdge[rootDest].weight) {
                    cheapestEdge[rootDest] = edges[i];
                }
            }
        }

        // Add cheapest edges to MST
        for (int i = 0; i < V; i++) {
            if (cheapestEdge[i].weight != INT_MAX) {
                int rootSrc = find(ds, cheapestEdge[i].src);
                int rootDest = find(ds, cheapestEdge[i].dest);

                if (rootSrc != rootDest) {
                    unionSets(ds, rootSrc, rootDest);
                    mst[(*mstEdgeCount)++] = cheapestEdge[i];
                    componentsCount--;
                }
            }
        }

        free(cheapestEdge);
    }

    free(ds->parent);
    free(ds->rank);
    free(ds);
    return mst;
}

// Function to read graph from file
void readGraphFromFile(const char *filename, int* V, int* E, Edge** edges) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file %s\n", filename);
        exit(1);
    }

    if (fscanf(file, "%d %d", V, E) != 2 || *V <= 0 || *E <= 0) {
        fprintf(stderr, "Invalid graph metadata\n");
        fclose(file);
        exit(1);
    }

    *edges = (Edge*)malloc(*E * sizeof(Edge));
    if (!*edges) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(file);
        exit(1);
    }

    for (int i = 0; i < *E; i++) {
        if (fscanf(file, "%d %d %d", &(*edges)[i].src, &(*edges)[i].dest, &(*edges)[i].weight) != 3 ||
            (*edges)[i].src < 0 || (*edges)[i].src >= *V || (*edges)[i].dest < 0 || (*edges)[i].dest >= *V || (*edges)[i].weight < 0) {
            fprintf(stderr, "Invalid edge data\n");
            free(*edges);
            fclose(file);
            exit(1);
        }
    }
    fclose(file);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <input_file>\n", argv[0]);
        return 1;
    }

    int V, E;
    Edge* edges = NULL;

    // Read graph from file
    readGraphFromFile(argv[1], &V, &E, &edges);

    printf("Computing Minimum Spanning Tree...\n");
    clock_t start = clock();

    int mstEdgeCount = 0;
    Edge* mst = boruvkaMST(edges, V, E, &mstEdgeCount);

    clock_t end = clock();
    double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

    printf("\nPerformance Metrics:\n");
    printf("Vertices: %d\n", V);
    printf("Edges: %d\n", E);
    printf("MST Edges: %d\n", mstEdgeCount);
    printf("Computation Time: %.4f seconds\n", cpu_time_used);

    // Print selected edges and calculate total weight
    // printf("\nSelected MST Edges:\n");
    // printf("From -> To : Weight\n");
    // printf("-------------------\n");
    long long totalWeight = 0;
    for (int i = 0; i < mstEdgeCount; i++) {
        // printf("%4d -> %4d : %d\n", mst[i].src, mst[i].dest, mst[i].weight);
        totalWeight += mst[i].weight;
    }
    printf("\nTotal MST Weight: %lld\n", totalWeight);

    // Cleanup
    free(edges);
    free(mst);

    return 0;
}