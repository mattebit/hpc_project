#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <stdbool.h>
#include <omp.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

//# define DEBUG
# define LOGGING
# define DEBUG
# define NULL ((void *)0)

int NODE_COUNT = 20;
char* GRAPH_PATH; // path of the graph to import
int MAX_EDGE_VALUE = __INT_MAX__;

int MY_NODES_FROM = 0;
int MY_NODES_TO = 100;

double last_time = 0.0;
double start_time = 0.0;

typedef struct {
    int weight;
    int from;
    int to;
} Edge;

typedef enum {
    PRINT_NORMAL,
    PRINT_FULL,
    PRINT_MST_EDGES
} PrintMode;

typedef struct {
    int process_count;
    int world_rank;
    MPI_Datatype edge_type;
    MPI_Op min_edge_op;
} MPIContext;

typedef struct {
    int** graph;
    int** min_graph;
    int* parent;
    int* rank;
    int vertex_per_process;
    MPIContext* mpi_ctx;
} MSTContext;


void print_matrix_data(int** matrix, int** weight_matrix, PrintMode mode) {
    switch (mode) {
        case PRINT_NORMAL:
            for (int i = 0; i < NODE_COUNT; i++) {
                for (int j = 0; j < i; j++) {
                    printf("%d ", get_from_matrix(matrix, i, j));
                }
                printf("\n");
            }
            break;
            
        case PRINT_FULL:
            for (int i = 0; i < NODE_COUNT; i++) {
                for (int j = 0; j < NODE_COUNT; j++) {
                    printf("%d ", get_from_matrix(matrix, i, j));
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
                        printf("%4d -> %4d : %d\n", i, j, 
                               get_from_matrix(weight_matrix, i, j));
                    }
                }
            }
            break;
    }
}


/**
 * Used to log a message
 */
void log_message(int process_id, char* message) {
    time_t now = time(NULL);
    struct tm* t = localtime(&now);

    printf("[%d-%02d-%02d %02d:%02d:%02d] [ID:%d] %s\n",
        t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
        t->tm_hour, t->tm_min, t->tm_sec, process_id, message);
}

void init_union_find(int* parent, int* rank, int n) {
    #pragma omp simd
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

int** allocate_and_init_matrix() {
    // Allocate contiguous block for all matrix data
    int total_size = 0;
    for (int i = 0; i < NODE_COUNT; i++) {
        total_size += i + 1;
    }
    
    int* data = (int*)calloc(total_size, sizeof(int));
    if (!data) {
        fprintf(stderr, "Failed to allocate matrix data\n");
        exit(EXIT_FAILURE);
    }
    
    // Allocate array of pointers
    int** min_graph = (int**)malloc(NODE_COUNT * sizeof(int*));
    if (!min_graph) {
        free(data);
        fprintf(stderr, "Failed to allocate matrix pointers\n");
        exit(EXIT_FAILURE);
    }
    
    // Setup row pointers
    int offset = 0;
    for (int i = 0; i < NODE_COUNT; i++) {
        min_graph[i] = &data[offset];
        offset += i + 1;
    }
    
    return min_graph;
}

// Add this cleanup function
void free_matrix(int** matrix) {
    if (matrix) {
        free(matrix[0]); // Free the contiguous data block
        free(matrix);    // Free the array of pointers
    }
}

int get_from_matrix(int** matrix, int row, int col) {
    return (row < col) ? matrix[col][row] : matrix[row][col];
}

void set_to_matrix(int** matrix, int row, int col, int value) {
    if (row < col)
        matrix[col][row] = value;
    else
        matrix[row][col] = value;
}



int** readGraphFromFile(const char* filename) {
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
    int** graph = allocate_and_init_matrix();
    
    // Read edges sequentially with buffered I/O
    int src, dest, weight;
    for (int i = 0; i < E; i++) {
        if (fscanf(file, "%d %d %d\n", &src, &dest, &weight) != 3) {
            fprintf(stderr, "Error reading edge %d\n", i);
            free(buffer);
            fclose(file);
            free_matrix(graph);
            exit(1);
        }
        
        // Update matrix only if new edge is lighter
        int val = get_from_matrix(graph, src, dest);
        if (val == 0 || val > weight) {
            set_to_matrix(graph, src, dest, weight);
        }
    }

    // Cleanup
    free(buffer);
    fclose(file);
    return graph;
}

void update_min_graph_first_iter(int* parent, int * rank, int* min_values, int** min_graph) {
    // #pragma omp parallel for
    for (int i = 0; i < NODE_COUNT; i++) {
        int weight = min_values[i];
        if (weight > 0) {
            union_sets(parent, rank, i, weight);
            set_to_matrix(min_graph, i, weight, 1);
        }
    }
}

void update_min_graph_subsequent_iter(int* parent, int* rank, Edge* root_mins, int** graph, int** min_graph) {
    // Process edges in a consistent order
    for (int i = 0; i < NODE_COUNT; i++) {
        if (root_mins[i].weight > 0 && root_mins[i].from != -1) {
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
    // #pragma omp simd
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

void time_print(char* desc, int world_rank) {
#ifdef LOGGING
    if (world_rank != 0)
        return;
    double actual_time = MPI_Wtime();
    double diff = actual_time - last_time;
    printf("[ID:%d] Time taken: %f sec (%s)\n", world_rank, diff, desc);
    last_time = actual_time;
#endif
}

bool find_min_components(int* parent, int** graph, Edge* min_edges) {
    bool can_connect = false;

    // Use atomic to handle the shared can_connect variable
    #pragma omp parallel
    {
        bool thread_can_connect = false;
        
        // Parallelize the outer loop since each iteration is independent
        #pragma omp for schedule(static)
        for (int i = MY_NODES_FROM; i < MY_NODES_TO; i++) {
            int root_i = find(parent, i);
            Edge min_edge = {MAX_EDGE_VALUE, -1, -1};
            
            // Find minimum weight edge connecting to different component
            for (int j = 0; j < NODE_COUNT; j++) {
                int weight = get_from_matrix(graph, i, j);
                if (weight <= 0) continue; // Skip invalid edges
                
                int root_j = find(parent, j);
                if (root_i != root_j && weight < min_edge.weight) {
                    min_edge.weight = weight;
                    min_edge.from = i;
                    min_edge.to = j;
                    thread_can_connect = true;
                }
            }
            
            min_edges[i - MY_NODES_FROM] = min_edge;
        }

        // Combine thread results atomically
        if (thread_can_connect) {
            #pragma omp atomic write
            can_connect = true;
        }
    }

    return can_connect;
}

// Define custom reduction operation for Edge
void min_edge_reduce(void* in, void* inout, int* len, MPI_Datatype* datatype) {
    Edge* in_edges = (Edge*)in;
    Edge* inout_edges = (Edge*)inout;
    for (int i = 0; i < *len; i++) {
        if ((in_edges[i].weight != -1 || in_edges[i].weight != MAX_EDGE_VALUE) && 
        (inout_edges[i].weight != -1 || inout_edges[i].weight != MAX_EDGE_VALUE) &&
        in_edges[i].weight < inout_edges[i].weight) {
            inout_edges[i] = in_edges[i];
        }
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

static void init_mpi(int* process_count, int* world_rank) {
    MPI_Init(NULL, NULL);
    last_time = MPI_Wtime();
    start_time = MPI_Wtime();
    MPI_Comm_size(MPI_COMM_WORLD, process_count);
    MPI_Comm_rank(MPI_COMM_WORLD, world_rank);
}

static void setup_vertex_distribution(int process_count, int world_rank, int* vertex_per_process) {
    *vertex_per_process = NODE_COUNT / process_count;
    int remainder = NODE_COUNT % process_count;
    
    if (world_rank < remainder) {
        (*vertex_per_process)++;
        MY_NODES_FROM = world_rank * (*vertex_per_process);
    } else {
        MY_NODES_FROM = world_rank * (*vertex_per_process) + remainder;
    }
    MY_NODES_TO = MY_NODES_FROM + *vertex_per_process;
}

static long calculate_mst_weight(int** min_graph, int** graph) {
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
    #pragma omp simd
    for (int i = 0; i < size; i++) {
        edges[i] = (Edge){-1, -1, -1};
    }
    return edges;
}

static void find_local_minimum_edges(int vertex_per_process, int** graph, int* lightest_edges) {
    #pragma omp parallel for schedule(dynamic)
    for (int i = MY_NODES_FROM; i < MY_NODES_TO; i++) {
        int local_min_weight = MAX_EDGE_VALUE;
        int local_min_id = -1;
        
        // Enable vectorization for weight comparison
        #pragma omp simd reduction(min:local_min_weight)
        for (int j = 0; j < NODE_COUNT; j++) {
            if (i == j) continue;
            int weight = get_from_matrix(graph, i, j);
            if (weight > 0 && weight < local_min_weight) {
                local_min_weight = weight;
                local_min_id = j;
            }
        }
        
        lightest_edges[i - MY_NODES_FROM] = local_min_id;
    }
}

static void gather_root_counts(int* parent, int* roots) {
    #pragma omp parallel for
    for (int i = 0; i < NODE_COUNT; i++) {
        #pragma omp atomic
        roots[parent[i]]++;
    }
}

static Edge find_minimum_edge_for_root(Edge* min_edges, int* parent, int root_i, int vertex_per_process) {
    Edge local_pair = {MAX_EDGE_VALUE, 0, 0};
    
    for (int j = 0; j < vertex_per_process; j++) {
        if (min_edges[j].weight == -1 || min_edges[j].weight == MAX_EDGE_VALUE) continue;
        
        if (find(parent, min_edges[j].from) == root_i && 
            min_edges[j].weight < local_pair.weight) {
            local_pair = min_edges[j];
        }
    }
    return local_pair;
}

void first_iteration_mst(int vertex_per_process, int** graph, int world_rank, 
                        int* parent, int* rank, int** min_graph) {
    // Allocate and find local minimum edges
    int* lightest_edges = calloc(vertex_per_process, sizeof(int));
    find_local_minimum_edges(vertex_per_process, graph, lightest_edges);
    
    time_print("1 find_local_minimum_edges", world_rank);

    // Gather results from all processes
    int* recv_values = malloc(NODE_COUNT * sizeof(int));
    minus_array(recv_values, NODE_COUNT);

    time_print("1 before allgather", world_rank);

    MPI_Allgather(lightest_edges, vertex_per_process, MPI_INT,
                  recv_values, vertex_per_process, MPI_INT, MPI_COMM_WORLD);

    time_print("1 after allgather", world_rank);

    // Update MST with gathered results
    update_min_graph_first_iter(parent, rank, recv_values, min_graph);
    
    time_print("1 update graph", world_rank);

    // Cleanup
    free(lightest_edges);
    free(recv_values);

    time_print("1 cleanup", world_rank);

}

// Replace individual MPI_Allreduce calls with batch communication
static void batch_process_components(Edge* min_edges, Edge* recv_buff, int* roots, int vertex_per_process, 
                                  int* parent, MPI_Datatype MPI_EDGE, MPI_Op MPI_MIN_EDGE) {
    // Prepare batch of edges for all components
    Edge* batch_edges = malloc(NODE_COUNT * sizeof(Edge));
    int batch_size = 0;
    
    for (int i = 0; i < NODE_COUNT; i++) {
        if (roots[i] == 0) continue;
        int root_i = find(parent, i);
        batch_edges[batch_size++] = find_minimum_edge_for_root(min_edges, parent, root_i, vertex_per_process);
    }
    
    // Single collective communication for all edges
    MPI_Allreduce(batch_edges, recv_buff, batch_size, MPI_EDGE, MPI_MIN_EDGE, MPI_COMM_WORLD);
    
    free(batch_edges);
}

void subsequent_iterations_mst(int vertex_per_process, int* parent, int** graph,
                             int world_rank, MPI_Datatype MPI_EDGE, MPI_Op MPI_MIN_EDGE,
                             int* rank, int** min_graph) {
    // Initialize edge arrays
    Edge* min_edges = initialize_edge_array(vertex_per_process);
    Edge* recv_buff = initialize_edge_array(NODE_COUNT);

    time_print("2 init data structures", world_rank);

    // Find minimum components
    bool can_connect = find_min_components(parent, graph, min_edges);


    time_print("2 find_min_components", world_rank);

    // Count roots
    int* roots = calloc(NODE_COUNT, sizeof(int));
    gather_root_counts(parent, roots);

    time_print("2 alloc and gather_roots", world_rank);

    // Process each component
    if (can_connect) {
        batch_process_components(min_edges, recv_buff, roots, vertex_per_process, 
                               parent, MPI_EDGE, MPI_MIN_EDGE);
    }

    time_print("2 finish batch_processing and communication", world_rank);

    // Update MST with gathered results
    update_min_graph_subsequent_iter(parent, rank, recv_buff, graph, min_graph);

    time_print("2 finish update_min_graph", world_rank);
    
    // Cleanup
    free(min_edges);
    free(recv_buff);
    free(roots);

    time_print("2 cleanup", world_rank);
}

void compute_mst(MSTContext* ctx) {

    init_union_find(ctx->parent, ctx->rank, NODE_COUNT);

    // First iteration
    first_iteration_mst(ctx->vertex_per_process, ctx->graph, ctx->mpi_ctx->world_rank,
                       ctx->parent, ctx->rank, ctx->min_graph);
    
    // Subsequent iterations
    while(find_num_components(ctx->parent) > 1) {
        subsequent_iterations_mst(ctx->vertex_per_process, ctx->parent, ctx->graph,
                                ctx->mpi_ctx->world_rank, ctx->mpi_ctx->edge_type,
                                ctx->mpi_ctx->min_edge_op, ctx->rank, ctx->min_graph);
    }
}

MPIContext init_mpi_context(int argc, char** argv) {
    MPIContext ctx;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &ctx.process_count);
    MPI_Comm_rank(MPI_COMM_WORLD, &ctx.world_rank);
    
    // Setup custom types
    MPI_Type_contiguous(3, MPI_INT, &ctx.edge_type);
    MPI_Type_commit(&ctx.edge_type);
    MPI_Op_create((MPI_User_function*)min_edge_reduce, 1, &ctx.min_edge_op);
    start_time = MPI_Wtime();
    last_time = MPI_Wtime();
    
    return ctx;
}

void cleanup_mpi_context(MPIContext* ctx) {
    MPI_Type_free(&ctx->edge_type);
    MPI_Op_free(&ctx->min_edge_op);
    MPI_Finalize();
}

// Refactored main function
int main(int argc, char** argv) {
    // Parse command line arguments
    parse_command_line_args(argc, argv);

    int** graph = readGraphFromFile(GRAPH_PATH);

    MPIContext mpi_ctx = init_mpi_context(argc, argv);
    MSTContext mst_ctx = {
        .graph = graph,
        .min_graph = allocate_and_init_matrix(),
        .parent = malloc(NODE_COUNT * sizeof(int)),
        .rank = malloc(NODE_COUNT * sizeof(int)),
        .mpi_ctx = &mpi_ctx
    };

    if (mst_ctx.parent == NULL || mst_ctx.rank == NULL) {
        fprintf(stderr, "Failed to allocate memory for union-find structures\n");
        exit(EXIT_FAILURE);
    }

    time_print("Allocated stuff", mpi_ctx.world_rank);

    #ifdef LOGGING
    if (mpi_ctx.world_rank == 0) {
        printf("[START] NODE_COUNT:%d,process_count:%d\n", NODE_COUNT, mpi_ctx.process_count);
    }
    #endif

    // Setup vertex distribution
    setup_vertex_distribution(mpi_ctx.process_count, mpi_ctx.world_rank, 
        &mst_ctx.vertex_per_process);
    printf("[ID:%d] MY_NODES_FROM = %d, MYN_NODES_TO= %d, VERTEX_PER_PROCESS = %d\n",
           mpi_ctx.world_rank, MY_NODES_FROM, MY_NODES_TO, mst_ctx.vertex_per_process);

    
    compute_mst(&mst_ctx);
    time_print("Finished computing MST", mpi_ctx.world_rank);
    
    // Print results if root process
    if (mpi_ctx.world_rank == 0) {
        printf("Computation Time: %f\n", MPI_Wtime() - start_time);
        printf("Total MST Weight: %ld\n", calculate_mst_weight(mst_ctx.min_graph, mst_ctx.graph));
    }

    // Cleanup
    free_matrix(mst_ctx.min_graph);
    free_matrix(mst_ctx.graph);
    free(mst_ctx.parent);
    free(mst_ctx.rank);
    cleanup_mpi_context(&mpi_ctx);
    return 0;
}