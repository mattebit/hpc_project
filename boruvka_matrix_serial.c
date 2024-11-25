#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>
#include <time.h>

#define NODE_COUNT 15000;

typedef struct {
    int* parent;
    int* rank;
    int n;
} DisjointSet;

typedef struct {
    int src;
    int dest;
    int weight;
} Edge;

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

// Find with path compression
int find(DisjointSet* ds, int x) {
    if (ds->parent[x] != x)
        ds->parent[x] = find(ds, ds->parent[x]);
    return ds->parent[x];
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

// Borůvka's algorithm using adjacency matrix
Edge* boruvkaMST(int** graph, int V, int* mstEdgeCount) {
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
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < V; j++) {
                if (graph[i][j] == INT_MAX) continue;
                
                int rootI = find(ds, i);
                int rootJ = find(ds, j);

                if (rootI != rootJ) {
                    // Update cheapest edge for both components
                    if (graph[i][j] < cheapestEdge[rootI].weight) {
                        cheapestEdge[rootI] = (Edge){i, j, graph[i][j]};
                    }
                    if (graph[i][j] < cheapestEdge[rootJ].weight) {
                        cheapestEdge[rootJ] = (Edge){i, j, graph[i][j]};
                    }
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

// Read graph into adjacency matrix
int** readGraphFromFile(const char* filename, int* V, int* E) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        exit(1);
    }

    if (fscanf(file, "%d %d", V, E) != 2) {
        perror("Error reading graph metadata");
        fclose(file);
        exit(1);
    }

    // Allocate and initialize adjacency matrix
    int** graph = malloc(*V * sizeof(int*));
    for (int i = 0; i < *V; i++) {
        graph[i] = malloc(*V * sizeof(int));
        for (int j = 0; j < *V; j++) {
            graph[i][j] = INT_MAX;  // Initialize with infinity
        }
    }

    // Read edges
    for (int i = 0; i < *E; i++) {
        int src, dest, weight;
        if (fscanf(file, "%d %d %d", &src, &dest, &weight) != 3) {
            break;
        }
        graph[src][dest] = weight;
    }

    fclose(file);
    return graph;
}

//init matrix from code
int** allocate_and_init_matrix(int V) {
    int ** min_graph = (int**)malloc(V * sizeof(int*));
    for (int i=0; i < V; i++) {
        min_graph[i] = (int*)calloc(V, sizeof(int));
    }
    return min_graph;
}
//fill matrix to ensure no arch has the same value
void fill_graph(int*** graph, int V) {
    srand(127);
    int* used_values = calloc(V * V * 4, sizeof(int));
    
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            if (i == j) {
                (*graph)[i][j] = -1;
            } else {
                int gen;
                do {
                    gen = rand() % (V * V * 4);
                } while (used_values[gen]);
                used_values[gen] = 1;
                (*graph)[i][j] = gen;
                (*graph)[j][i] = gen;
            }
        }
    }
    free(used_values);
}

int main(int argc, char* argv[]) {
    // const char* GRAPH_FILENAME = argc > 1 ? argv[1] : "large_graph.txt";

    // printf("Reading graph from file: %s...\n", GRAPH_FILENAME);
    int V = NODE_COUNT;
    int** graph = allocate_and_init_matrix(V);
    fill_graph(&graph, V);

    printf("Computing Minimum Spanning Tree...\n");
    clock_t start = clock();

    int mstEdgeCount = 0;
    Edge* mst = boruvkaMST(graph, V, &mstEdgeCount);

    clock_t end = clock();
    double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

    printf("\nPerformance Metrics:\n");
    printf("Vertices: %d\n", V);
    // printf("Edges: %d\n", E);
    printf("MST Edges: %d\n", mstEdgeCount);
    printf("Computation Time: %.4f seconds\n", cpu_time_used);

    long long totalWeight = 0;
    for (int i = 0; i < mstEdgeCount; i++) {
        totalWeight += mst[i].weight;
    }
    printf("Total MST Weight: %lld\n", totalWeight);

    // Cleanup
    for (int i = 0; i < V; i++) {
        free(graph[i]);
    }
    free(graph);
    free(mst);

    return 0;
}