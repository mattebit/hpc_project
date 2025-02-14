#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stddef.h>

//# define DEBUG
# define LOGGING
# define DEBUG
# define NULL ((void *)0)

int NODE_COUNT = 20;
char* GRAPH_PATH; // path of the graph to import
uint16_t MAX_EDGE_VALUE = UINT16_MAX;

clock_t last_time;
clock_t start_time;

typedef struct {
    uint16_t weight;
    int from;
    int to;
} Edge;

typedef enum {
    PRINT_NORMAL,
    PRINT_FULL,
    PRINT_MST_EDGES
} PrintMode;

typedef struct {
    uint16_t** graph;
    uint16_t** min_graph;
    int* parent;
    int* rank;
    int vertex_per_process;
} MSTContext;





/**
 * Used to log a message
 */
void log_message(int process_id, char* message) {
    time_t now = time(NULL);
    struct tm* t = localtime(&now);

    printf("[%d-%02d-%02d %02d:%02d:%02d] %s\n",
        t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
        t->tm_hour, t->tm_min, t->tm_sec, message);
}

void init_union_find(int* parent, int* rank, int n) {
    for (int i = 0; i < n; i++) {
        parent[i] = i;
        rank[i] = 0;
    }
}

int find(int* parent, int node) {
    if (parent[node] != node) {
        parent[node] = find(parent, parent[node]);
    }
    return parent[node];
}

void union_sets(int* parent, int* rank, int node1, int node2) {
    int root1 = find(parent, node1);
    int root2 = find(parent, node2);
    if (root1 != root2) {
        if (rank[root1] > rank[root2]) {
            parent[root2] = root1;
        } else if (rank[root1] < rank[root2]) {
            parent[root1] = root2;
        } else {
            parent[root2] = root1;
            rank[root1]++;
        }
    }
}

uint16_t** allocate_and_init_matrix(int default_value) {
    // Allocate array of pointers
    uint16_t** matrix = (uint16_t**)malloc(NODE_COUNT * sizeof(uint16_t*));
    if (!matrix) {
        fprintf(stderr, "Failed to allocate matrix pointers\n");
        exit(EXIT_FAILURE);
    }
    
    // Allocate each row separately
    for (int i = 0; i < NODE_COUNT; i++) {
        // Each row i needs (i+1) elements
        matrix[i] = (uint16_t*)calloc(i + 1, sizeof(uint16_t));
        if (!matrix[i]) {
            // Cleanup previously allocated rows
            for (int j = 0; j < i; j++) {
                free(matrix[j]);
            }
            free(matrix);
            fprintf(stderr, "Failed to allocate row %d\n", i);
            exit(EXIT_FAILURE);
        }

        // init matrix with default value
        for (int j =0; j<i; j++){
            matrix[i][j] = default_value;
        }
    }
    
    return matrix;
}


// Add this cleanup function
void free_matrix(uint16_t** matrix) {
    if (matrix) {
        free(matrix[0]); // Free the contiguous data block
        free(matrix);    // Free the array of pointers
    }
}

uint16_t get_from_matrix(uint16_t** matrix, int row, int col) {
    return (row < col) ? matrix[col][row] : matrix[row][col];
}

void set_to_matrix(uint16_t** matrix, int row, int col, uint16_t value) {
    if (row < col)
        matrix[col][row] = value;
    else
        matrix[row][col] = value;
}

void print_matrix_data(uint16_t** matrix, uint16_t** weight_matrix, PrintMode mode) {
    switch (mode) {
        case PRINT_NORMAL:
            for (int i = 0; i < NODE_COUNT; i++) {
                for (int j = 0; j < i; j++) {
                    printf("%hd ", get_from_matrix(matrix, i, j));
                }
                printf("\n");
            }
            break;
            
        case PRINT_FULL:
            for (int i = 0; i < NODE_COUNT; i++) {
                for (int j = 0; j < NODE_COUNT; j++) {
                    printf("%hd ", get_from_matrix(matrix, i, j));
                }
                printf("\n");
            }
            break;
            
        case PRINT_MST_EDGES:
            printf("\nSelected MST Edges:\n");
            printf("From -> To : Weight\n");
            printf("-------------------\n");
            for (int i = 0; i < NODE_COUNT; i++) {
                for (int j = 0; j < i; j++) {
                    if (get_from_matrix(matrix, i, j) == 1) {
                        printf("%4d -> %4d : %hd\n", i, j, 
                               get_from_matrix(weight_matrix, i, j));
                    }
                }
            }
            break;
    }
}


uint16_t** readGraphFromFile(const char* filename) {
    // Open file with a large buffer for efficient reading
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        exit(1);
    }

    // Set up a large buffer for file I/O (16MB buffer)
    char* buffer = (char*)malloc(16 * 1024 * 1024);
    if (buffer == NULL) {
        perror("Failed to allocate file buffer");
        fclose(file);
        exit(1);
    }
    setvbuf(file, buffer, _IOFBF, 16 * 1024 * 1024);

    // Read header
    int V, E;
    if (fscanf(file, "%d %d\n", &V, &E) != 2) {
        perror("Error reading graph metadata");
        free(buffer);
        fclose(file);
        exit(1);
    }
    NODE_COUNT = V;

    // Allocate graph matrix
    uint16_t** graph = allocate_and_init_matrix(MAX_EDGE_VALUE);
    
    // Read edges sequentially with buffered I/O
    int src, dest;
    uint16_t weight;
    for (int i = 0; i < E; i++) {
        if (fscanf(file, "%d %d %hd\n", &src, &dest, &weight) != 3) {
            fprintf(stderr, "Error reading edge %d\n", i);
            free(buffer);
            fclose(file);
            free_matrix(graph);
            exit(1);
        }
        
        // Update matrix only if new edge is lighter
        uint16_t val = get_from_matrix(graph, src, dest);
        if (val == 0 || val > weight) {
            set_to_matrix(graph, src, dest, weight);
        }
    }

    // Cleanup
    free(buffer);
    fclose(file);
    return graph;
}

void update_min_graph_first_iter(int* parent, int * rank, int* min_ids, uint16_t** min_graph) {
    for (int i = 0; i < NODE_COUNT; i++) {
        int min_id_to = min_ids[i];
        if (min_id_to != -1) {
            union_sets(parent, rank, i, min_id_to);
            set_to_matrix(min_graph, i, min_id_to, 1);
        }
    }
}

void 
update_min_graph_subsequent_iter(int* parent, int* rank, Edge* root_mins, uint16_t** graph, uint16_t** min_graph) {
    // Process edges in a consistent order
    for (int i = 0; i < NODE_COUNT; i++) {
        if (root_mins[i].weight != MAX_EDGE_VALUE && root_mins[i].from != -1) {
            int root1 = find(parent, root_mins[i].from);
            int root2 = find(parent, root_mins[i].to);

            if (root1 != root2) {
                union_sets(parent, rank, root_mins[i].from, root_mins[i].to);
                set_to_matrix(min_graph, root_mins[i].from, root_mins[i].to, 1);
            }
        }
    }
}

void minus_array(int* array, int size) {
    for (int i = 0; i < size; i++) {
        array[i] = -1;
    }
}

void print_array(int* array, char* name, int size) {
    printf("%s: ", name);
    for (int i = 0; i < size; i++) {
        printf("%d ", array[i]);
    }
    printf("\n");
}

int find_num_components(int* parent) {
    bool* unique_roots = (bool*)calloc(NODE_COUNT, sizeof(bool));
    for (int i = 0; i < NODE_COUNT; i++) {
        find(parent,i);
        unique_roots[parent[i]] = true;
    }

    int num_components = 0;
    for (int i = 0; i < NODE_COUNT; i++) {
        if (unique_roots[i]) {
            num_components++;
        }
    }
    free(unique_roots);
    return num_components;
}

void time_print(char* desc) {
#ifdef LOGGING
    clock_t actual_time = clock();
    double cpu_time_used = ((double)(actual_time - last_time)) / CLOCKS_PER_SEC;
    printf("Time taken: %f sec (%s)\n", cpu_time_used, desc);
    last_time = actual_time;
#endif
}

bool find_min_components(int* parent, uint16_t** graph, Edge* min_edges) {
    for (int i = 0; i < NODE_COUNT; i++) {
        int root_i = find(parent, i);
        Edge min_edge = {MAX_EDGE_VALUE, -1, -1};
        
        // Find minimum weight edge connecting to different component
        for (int j = 0; j < NODE_COUNT; j++) {
            uint16_t weight = get_from_matrix(graph, i, j);
            if (weight == MAX_EDGE_VALUE) continue; // Skip invalid edges
            
            int root_j = find(parent, j);
            if (root_i != root_j && weight < min_edge.weight) {
                min_edge.weight = weight;
                min_edge.from = i;
                min_edge.to = j;
            }
        }
        
        min_edges[i] = min_edge;
    }
}

// New helper functions for initialization
static void parse_command_line_args(int argc, char** argv) {
    if (argc > 1) {
        if (argc < 2) {
            printf("Invalid number of parameters, expected \"%s NODE_COUNT GRAPH_PATH\" or nothing\n", argv[0]);
            exit(1);
        }
#ifdef LOGGING
        printf("Received arguments:\n");
        printf("argv[0] %s\n", argv[0]);
        printf("argv[1] (NODE_COUNT) %s\n", argv[1]);
        printf("argv[2] (GRAPH_PATH) %s\n", argv[2]);
#endif
        NODE_COUNT = atoi(argv[1]);
        GRAPH_PATH = argv[2];  // Fixed: removed atoi here since GRAPH_PATH is char*
    }
}

static long calculate_mst_weight(uint16_t** min_graph, uint16_t** graph) {
    long final_mst_weight = 0;
    for (int i = 0; i < NODE_COUNT; i++) {
        for (int j = 0; j < i; j++) {  // Only check lower triangular part
            if (get_from_matrix(min_graph, i, j) == 1) {
                final_mst_weight += get_from_matrix(graph, i, j);
            }
        }
    }
    return final_mst_weight;
}

// Helper functions for MST iterations
static Edge* initialize_edge_array(int size) {
    Edge* edges = malloc(size * sizeof(Edge));
    for (int i = 0; i < size; i++) {
        edges[i] = (Edge){MAX_EDGE_VALUE, -1, -1};
    }
    return edges;
}

static void find_local_minimum_edges(uint16_t** graph, int* lightest_edges) {
    for (int i = 0; i < NODE_COUNT; i++) {
        uint16_t local_min_weight = MAX_EDGE_VALUE;
        int local_min_id = -1;
        
        // First find minimum weight without SIMD
        for (int j = 0; j < NODE_COUNT; j++) {
            if (i == j) continue;
            uint16_t weight = get_from_matrix(graph, i, j);
            if (weight != MAX_EDGE_VALUE && weight < local_min_weight) {
                local_min_weight = weight;
                local_min_id = j;
            }
        }
        
        lightest_edges[i] = local_min_id;
    }
}

static void gather_root_counts(int* parent, int* roots) {
    for (int i = 0; i < NODE_COUNT; i++) {
        roots[parent[i]]++;
    }
}

static Edge find_minimum_edge_for_root(Edge* min_edges, int* parent, int root_i) {
    Edge local_pair = {MAX_EDGE_VALUE, 0, 0};
    
    for (int j = 0; j < NODE_COUNT; j++) {
        if (min_edges[j].weight == MAX_EDGE_VALUE) continue;
        
        if (find(parent, min_edges[j].from) == root_i && 
            min_edges[j].weight < local_pair.weight) {
            local_pair = min_edges[j];
        }
        
    }
    return local_pair;
}

void first_iteration_mst(uint16_t** graph, int* parent, int* rank, uint16_t** min_graph) {
    // Allocate and find local minimum edges
    int* ids_lightest_edges = calloc(NODE_COUNT, sizeof(int));
    find_local_minimum_edges(graph, ids_lightest_edges);
    
    time_print("1 find_local_minimum_edges");

    // Update MST with gathered results
    update_min_graph_first_iter(parent, rank, ids_lightest_edges, min_graph);
    
    time_print("1 update graph");

    // Cleanup
    free(ids_lightest_edges);

    time_print("1 cleanup");

}

// Replace individual MPI_Allreduce calls with batch communication
static void batch_process_components(Edge* min_edges, Edge* recv_buff, int* roots, int* parent) {
    int count = 0;

    for (int i = 0; i < NODE_COUNT; i++) {
        if (roots[i] == 0) continue;
        recv_buff[count++] = find_minimum_edge_for_root(min_edges, parent, i);
    }
}

void subsequent_iterations_mst(int* parent, uint16_t** graph, int* rank, uint16_t** min_graph) {
    // Initialize edge arrays
    Edge* min_edges = initialize_edge_array(NODE_COUNT);
    Edge* recv_buff = initialize_edge_array(NODE_COUNT);

    time_print("2 init data structures");

    // Find minimum components
    find_min_components(parent, graph, min_edges);

    time_print("2 find_min_components");

    // Count roots
    int* roots = calloc(NODE_COUNT, sizeof(int));
    gather_root_counts(parent, roots);

    time_print("2 alloc and gather_roots");

    // Process each component
    batch_process_components(min_edges, recv_buff, roots, parent);

    time_print("2 finish batch_processing and communication");

    // Update MST with gathered results
    update_min_graph_subsequent_iter(parent, rank, recv_buff, graph, min_graph);

    time_print("2 finish update_min_graph");
    
    // Cleanup
    free(min_edges);
    free(recv_buff);
    free(roots);

    time_print("2 cleanup");
}

void compute_mst(MSTContext* ctx) {

    init_union_find(ctx->parent, ctx->rank, NODE_COUNT);

    // First iteration
    first_iteration_mst(ctx->graph, ctx->parent, ctx->rank, ctx->min_graph);
    
    int num_components = NODE_COUNT;
    num_components = find_num_components(ctx->parent); // NOT remove, 
    printf("Components: %d\n", num_components);

    //if (ctx->mpi_ctx->world_rank == 0) print_matrix_data(ctx->min_graph, ctx->graph, PRINT_NORMAL);

    // Subsequent iterations
    while(num_components > 1) {
        subsequent_iterations_mst(ctx->parent, ctx->graph,
                                ctx->rank, ctx->min_graph);
        num_components = find_num_components(ctx->parent);
        printf("Components: %d\n", num_components);
    }
}

// Refactored main function
int main(int argc, char** argv) {
    // Parse command line arguments
    parse_command_line_args(argc, argv);

    uint16_t** graph = readGraphFromFile(GRAPH_PATH);
    
    last_time = clock();
    start_time = clock();
    
    MSTContext mst_ctx = {
        .graph = graph,
        .min_graph = allocate_and_init_matrix(0),
        .parent = malloc(NODE_COUNT * sizeof(int)),
        .rank = malloc(NODE_COUNT * sizeof(int)),
    };

    if (mst_ctx.parent == NULL || mst_ctx.rank == NULL) {
        fprintf(stderr, "Failed to allocate memory for union-find structures\n");
        exit(EXIT_FAILURE);
    }

    time_print("Allocated stuff");

    #ifdef LOGGING
    printf("[START] NODE_COUNT:%d\n", NODE_COUNT);
    #endif

    
    compute_mst(&mst_ctx);
    time_print("Finished computing MST");
    
    // Print results if root process
    double cpu_time_used = ((double)(clock() - start_time)) / CLOCKS_PER_SEC;
    printf("Computation Time: %.4f\n", cpu_time_used);
    printf("Total MST Weight: %ld\n", calculate_mst_weight(mst_ctx.min_graph, mst_ctx.graph));

    // Cleanup
    free_matrix(mst_ctx.min_graph);
    free_matrix(mst_ctx.graph);
    free(mst_ctx.parent);
    free(mst_ctx.rank);
    return 0;
}