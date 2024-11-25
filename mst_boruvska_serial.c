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

// Comparison function for sorting edges
int compareEdges(const void* a, const void* b) {
    return ((Edge*)a)->weight - ((Edge*)b)->weight;
}

// Borůvka's algorithm for Minimum Spanning Tree
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
                if (graph[i][j] == -1) continue;  // Skip self-loops
                
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

// Initialize matrix from code
int** allocate_and_init_matrix(int V) {
    int** matrix = (int**)malloc(V * sizeof(int*));
    for (int i = 0; i < V; i++) {
        matrix[i] = (int*)calloc(V, sizeof(int));
    }
    return matrix;
}

// Fill matrix to ensure no edge has the same value
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

// Convert adjacency matrix to edge list
Edge* convertMatrixToEdgeList(int** graph, int V, int* edgeCount) {
    int maxEdges = V * (V - 1) / 2;
    Edge* edges = malloc(maxEdges * sizeof(Edge));
    *edgeCount = 0;

    for (int i = 0; i < V; i++) {
        for (int j = i + 1; j < V; j++) {
            if (graph[i][j] != -1) {
                edges[*edgeCount].src = i;
                edges[*edgeCount].dest = j;
                edges[*edgeCount].weight = graph[i][j];
                (*edgeCount)++;
            }
        }
    }

    return edges;
}

int main(int argc, char* argv[]) {
    int V = 20000;  // Example vertex count
    int** graph = allocate_and_init_matrix(V);
    fill_graph(&graph, V);

    int edgeCount;
    Edge* edges = convertMatrixToEdgeList(graph, V, &edgeCount);

    printf("Computing Minimum Spanning Tree...\n");
    clock_t start = clock();

    int mstEdgeCount = 0;
    Edge* mst = boruvkaMST(graph, V, &mstEdgeCount);

    clock_t end = clock();
    double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

    printf("\nPerformance Metrics:\n");
    printf("Vertices: %d\n", V);
    printf("Edges: %d\n", edgeCount);
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
    free(edges);
    free(mst);

    return 0;
}