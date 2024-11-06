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
Edge* boruvkaMST(Edge* edges, int edgeCount, int vertexCount, int* mstEdgeCount) {
    DisjointSet* ds = createDisjointSet(vertexCount);
    Edge* mst = malloc(vertexCount * sizeof(Edge));
    *mstEdgeCount = 0;

    // Sort edges by weight
    qsort(edges, edgeCount, sizeof(Edge), compareEdges);

    // Track cheapest edges for each component
    Edge* cheapestEdge = malloc(vertexCount * sizeof(Edge));
    for (int i = 0; i < vertexCount; i++) {
        cheapestEdge[i].weight = INT_MAX;
    }

    int componentsCount = vertexCount;
    while (componentsCount > 1) {
        printf("Number of components: %d\n", componentsCount);
        // Reset cheapest edges
        for (int i = 0; i < vertexCount; i++) {
            cheapestEdge[i].weight = INT_MAX;
        }

        // Find the cheapest edge for each component
        for (int i = 0; i < edgeCount; i++) {
            int rootSrc = find(ds, edges[i].src);
            int rootDest = find(ds, edges[i].dest);

            if (rootSrc == rootDest)
                continue;

            // Update cheapest edge for components
            if (edges[i].weight < cheapestEdge[rootSrc].weight) {
                cheapestEdge[rootSrc] = edges[i];
            }
            if (edges[i].weight < cheapestEdge[rootDest].weight) {
                cheapestEdge[rootDest] = edges[i];
            }
        }

        // Add cheapest edges to MST
        for (int i = 0; i < vertexCount; i++) {
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
    }

    // Free temporary memory
    free(cheapestEdge);
    free(ds->parent);
    free(ds->rank);
    free(ds);

    return mst;
}

// Function to read graph from file
Edge* readGraphFromFile(const char* filename, int* vertexCount, int* edgeCount) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        exit(1);
    }

    // Read vertex and edge count
    if (fscanf(file, "%d %lld", vertexCount, (long long*)edgeCount) != 2) {
        perror("Error reading graph metadata");
        fclose(file);
        exit(1);
    }

    // Allocate memory for edges
    Edge* edges = malloc(*edgeCount * sizeof(Edge));
    if (!edges) {
        perror("Memory allocation failed");
        fclose(file);
        exit(1);
    }

    // Read edges
    int edgeRead = 0;
    while (edgeRead < *edgeCount) {
        if (fscanf(file, "%d %d %d", 
            &edges[edgeRead].src, 
            &edges[edgeRead].dest, 
            &edges[edgeRead].weight) != 3) {
            break;
        }
        edgeRead++;
    }

    *edgeCount = edgeRead;  // Update actual number of edges read
    fclose(file);
    return edges;
}

int main(int argc, char* argv[]) {
    // Default filename
    const char* GRAPH_FILENAME = "large_graph.txt";

    // Allow filename to be passed as command-line argument
    if (argc > 1) {
        GRAPH_FILENAME = argv[1];
    }

    // Read graph from file
    int vertexCount, edgeCount;
    printf("Reading graph from file: %s...\n", GRAPH_FILENAME);
    Edge* edges = readGraphFromFile(GRAPH_FILENAME, &vertexCount, &edgeCount);

    // Perform MST computation
    printf("Computing Minimum Spanning Tree...\n");
    
    // Start timing
    clock_t start = clock();

    int mstEdgeCount = 0;
    Edge* mst = boruvkaMST(edges, edgeCount, vertexCount, &mstEdgeCount);

    // End timing
    clock_t end = clock();
    double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    // Print results
    printf("\nPerformance Metrics:\n");
    printf("Vertices: %d\n", vertexCount);
    printf("Edges: %d\n", edgeCount);
    printf("MST Edges: %d\n", mstEdgeCount);
    printf("Computation Time: %.4f seconds\n", cpu_time_used);

    // Calculate total MST weight
    long long totalWeight = 0;
    for (int i = 0; i < mstEdgeCount; i++) {
        totalWeight += mst[i].weight;
    }
    printf("Total MST Weight: %lld\n", totalWeight);

    // Free memory
    free(edges);
    free(mst);

    return 0;
}