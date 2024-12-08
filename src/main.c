#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <stdbool.h>
#include <omp.h>

//# define DEBUG
# define LOGGING

int NODE_COUNT = 20000;
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

#define NULL ((void *)0)

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
    int** min_graph = (int**)malloc(NODE_COUNT * sizeof(int*));
    for (int i = 0; i < NODE_COUNT; i++)
    {
        min_graph[i] = (int*)calloc(i+1, sizeof(int));
    }
    return min_graph;
}

int get_from_matrix(int** matrix, int row, int col) {
    if (row < col) {
        return matrix[col][row];
    } else {
        return matrix[row][col];
    }
}

void set_to_matrix(int** matrix, int row, int col, int value) {
    if (row < col) {
        matrix[col][row] = value;
    } else {
        matrix[row][col] = value;
    }
}

int** fill_graph() {
    srand(127);
    int** matrix = allocate_and_init_matrix();
    int* used_values = (int*)calloc(NODE_COUNT * NODE_COUNT * 4, sizeof(int));

    for (int i = 0; i < NODE_COUNT; i++) {
        for (int j = 0; j < i; j++) {
            if (i == j) {
                set_to_matrix(matrix, i, j, -1);
            } else {
                int gen;
                do {
                    gen = rand() % (NODE_COUNT * NODE_COUNT * 4);
                } while (used_values[gen]);
                used_values[gen] = 1;
                set_to_matrix(matrix, i, j, gen);
            }
        }
    }
    free(used_values);

    return matrix;
}

int** read_graph() {
    int** matrix = allocate_and_init_matrix();

    FILE* fp = fopen("graph.txt", "r");

    char* line = malloc(10000 * sizeof(char));

    int col_count = 0;
    int row_count = 0;
    int done = 0;
    while (fgets(line, 10000, fp) != NULL && done == 0) {
        char* token = strtok(line, " ");
        while (token != NULL) {
            set_to_matrix(matrix, row_count, col_count++, atoi(token));
            if (col_count == NODE_COUNT) {
                col_count = 0;
                row_count++;
            }
            if (row_count == NODE_COUNT) {
                done = 1;
                break;
            }
            token = strtok(NULL, " ");
        }
    }
    free(line);
    fclose(fp);
    return matrix;
}

/**
 * Finds the lightest edge among the edges of the specified node and returns the id of the node connected by that edge
 */
int find_lightest_edge(int** graph, int node_id) {
    int min = MAX_EDGE_VALUE;
    int min_id_to = -1;
    for (int i = 0; i < NODE_COUNT; i++) {
        if (i == node_id) {
            continue;
        }
        int act = get_from_matrix(graph, node_id, i);
        if (act != -1 && act < min) {
            min = act;
            min_id_to = i;
        }
    }
    return min_id_to;
}

/**
 * Update the min graph with the min values received by the nodes.
 * The min_values array should be of length NODE_COUNT, and contain one value for each index
 */
void update_min_graph(int* min_values, int** min_graph) {
    for (int i = 0; i < NODE_COUNT; i++) {
        int act = min_values[i];
        if (act != -1) {
            set_to_matrix(min_graph, i, act, 1);
        }
    }
}

void update_min_graph_union_find(int* parent, int * rank, int* min_values, int** min_graph) {
    #pragma omp parallel for
    for (int i = 0; i < NODE_COUNT; i++) {
        int act = min_values[i];
        if (act != -1) {
            #pragma omp critical  
            {
                union_sets(parent, rank, i, act);
            }
            set_to_matrix(min_graph, i, act, 1);
        }
    }
}


// int find_root(int node_id, int** min_graph, int* visited, int min_reported, int* roots) {
//     if (visited[node_id] == 1) {
//         return -1;
//     }

//     visited[node_id] = 1;
//     int min_root = min_reported;
//     for (int i = 0; i < NODE_COUNT; i++) {
//         if (get_from_matrix(min_graph, node_id, i) == 1) {
//             int act_root = find_root(i, min_graph, visited, min_root, roots);
//             if (act_root != -1 && act_root < min_root) {
//                 min_root = act_root;
//                 roots[node_id] = min_root;
//             }
//         }
//     }
//     roots[node_id] = min_root;
//     return min_root;
// }

/**
 * Prunes the given graph from the edges between the same component
 */
void prune_graph(int* parent, int** graph) {
    #pragma omp parallel for
    for (int i = 0; i < NODE_COUNT; i++) {
        for (int j = 0; j < NODE_COUNT; j++) {
            if (get_from_matrix(graph, i, j) != -1 && parent[i] == parent[j]) {
                set_to_matrix(graph, i, j, -1);
            }
        }
    }
}

void print_matrix(int** matrix) {
    // Print the adjacency matrix
    for (int i = 0; i < NODE_COUNT; i++) {
        for (int j = 0; j < i; j++) {
            printf("%d ", get_from_matrix(matrix, i, j));
        }
        printf("\n");
    }
}

void zero_array(int* array, int size) {
    #pragma omp simd
    for (int i = 0; i < size; i++) {
        array[i] = 0;
    }
}

void minus_array(int* array, int size) {
    #pragma omp simd
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

int find_min_in_array(int* arr, int size) {
    int min = MAX_EDGE_VALUE;

    #pragma omp parallel for reduction(min:min)
    for (int i = 0; i < size; i++) {
        if (arr[i] < min) {
            min = arr[i];
        }
    }

    return min;
}

int find_num_components(int* parent) {
    bool* unique_roots = (bool*)calloc(NODE_COUNT, sizeof(bool));
    for (int i = 0; i < NODE_COUNT; i++) {
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
    bool can_connect = 0;
    int edge_index = 0;
    int min = MAX_EDGE_VALUE;
    int min_id = -1;
    
    #pragma omp parallel for private(min, min_id) reduction(||:can_connect)
    for (int i = MY_NODES_FROM; i < MY_NODES_TO; i++) {
        min = MAX_EDGE_VALUE;
        min_id = -1;
        int root_i = find(parent, i);
        for (int j = 0; j < NODE_COUNT; j++) {
            int act = get_from_matrix(graph, i, j);
            if (act != -1 && act < min) {
                if (root_i != find(parent, j)) {
                    min = act;
                    min_id = j;
                }
            }
        }
        if (min != MAX_EDGE_VALUE) {
            can_connect = true;
        }
        Edge min_edge = {min, i, min_id};
        min_edges[edge_index++] = min_edge;
    }

    return can_connect;
}

void update_min_graph_from_roots_not_id(int* parent, int* rank, Edge* root_mins, int** graph, int** min_graph) {
    #pragma omp parallel for
    for (int i = 0; i < NODE_COUNT; i++) {
        if (root_mins[i].weight != -1) {
            #pragma omp critical
            {
                union_sets(parent, rank, root_mins[i].from, root_mins[i].to);
            }
            set_to_matrix(min_graph, root_mins[i].from, root_mins[i].to, 1);
            set_to_matrix(graph, root_mins[i].from, root_mins[i].to, -1);
        }
    }
}

// Define custom reduction operation for Edge
void min_edge_reduce(void* in, void* inout, int* len, MPI_Datatype* datatype) {
    Edge* in_edges = (Edge*)in;
    Edge* inout_edges = (Edge*)inout;
    for (int i = 0; i < *len; i++) {
        if (in_edges[i].weight < inout_edges[i].weight) {
            inout_edges[i] = in_edges[i];
        }
    }
}

int main(int argc, char** argv) {
    int** graph = fill_graph(); // generate graph before forking
    // int** graph = read_graph();
    // print_matrix(graph);

    

    MPI_Init(NULL, NULL);
    last_time = MPI_Wtime();
    start_time = MPI_Wtime();
    int process_count;
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    time_print("Start", world_rank);
    // print_matrix(graph);

#ifdef LOGGING
    if (world_rank == 0) printf("[START] NODE_COUNT:%d,process_count:%d\n", NODE_COUNT, process_count);
#endif

    // Find indexes of which vertex this node is responsible for
    int vertex_per_process = NODE_COUNT / process_count;
    int remainder = NODE_COUNT % process_count;
    if (world_rank < remainder) {
        vertex_per_process++;
        MY_NODES_FROM = world_rank * vertex_per_process;
    } else {
        MY_NODES_FROM = world_rank * vertex_per_process + remainder;
    }
    MY_NODES_TO = MY_NODES_FROM + vertex_per_process;

    int** min_graph = allocate_and_init_matrix();

    int* parent = (int*)malloc(NODE_COUNT * sizeof(int));
    int* rank = (int*)malloc(NODE_COUNT * sizeof(int));

    if (parent == NULL || rank == NULL) {
        fprintf(stderr, "Failed to allocate memory for union-find structures\n");
        exit(EXIT_FAILURE);
    }

    init_union_find(parent, rank, NODE_COUNT);

    time_print("Allocated stuff", world_rank);

    // Define custom MPI datatype for Edge
    MPI_Datatype MPI_EDGE;
    MPI_Type_contiguous(3, MPI_INT, &MPI_EDGE);
    MPI_Type_commit(&MPI_EDGE);

    // Create the reduction operation
    MPI_Op MPI_MIN_EDGE;
    MPI_Op_create((MPI_User_function*)min_edge_reduce, 1, &MPI_MIN_EDGE);

    int num_components = 0;
    int count = 0;
    while (1) {
        if (count == 0) {
            int* ligthest_edges = (int*)malloc(vertex_per_process * sizeof(int));
            zero_array(ligthest_edges, vertex_per_process);
            for (int i = MY_NODES_FROM; i < MY_NODES_TO; i++) {
                int min_id_to = find_lightest_edge(graph, i);
                ligthest_edges[i - MY_NODES_FROM] = min_id_to;
            }

            int* recv_values = malloc(NODE_COUNT * sizeof(int));
            minus_array(recv_values, NODE_COUNT);

            time_print("1 min compute", world_rank);

            MPI_Allgather(
                ligthest_edges,
                vertex_per_process,
                MPI_INT,
                recv_values,
                vertex_per_process,
                MPI_INT,
                MPI_COMM_WORLD
            );

            time_print("1 allgather", world_rank);

            free(ligthest_edges);
            update_min_graph_union_find(parent, rank, recv_values, min_graph);
            free(recv_values);

            time_print("1 update graph", world_rank);
        } else {
            int* result = malloc(NODE_COUNT * sizeof(int));
            minus_array(result, NODE_COUNT);

            Edge min_edges[vertex_per_process];
            for (int i = 0; i < vertex_per_process; i++) {
                min_edges[i].weight = -1;
                min_edges[i].from = -1;
                min_edges[i].to = -1;
            }

            bool can_connect = find_min_components(parent, graph, min_edges);

            Edge recv_buff[NODE_COUNT];
            for (int i = 0; i < NODE_COUNT; i++) {
                recv_buff[i].weight = -1;
                recv_buff[i].from = -1;
                recv_buff[i].to = -1;
            }

            time_print("2 before all gather", world_rank);
            for (int i = 0; i < NODE_COUNT && can_connect; i++) {
                int root_i = find(parent, i);
                Edge local_pair = {MAX_EDGE_VALUE, 0, 0};   
                for (int j = 0; j < vertex_per_process; j++) {
                    //there cannot exist a minimum weight
                    if (min_edges[j].weight == -1)
                        continue;
                    
                    //if the node currently looking at is part of the current component
                    //save if the min_edge found for that node is the current 
                    //minimum for the actual component
                    if (find(parent, min_edges[j].from) == root_i) {
                        if (min_edges[j].weight < local_pair.weight) {
                            local_pair = min_edges[j];
                        }
                    }
                }

                // printf("lightest edge from [PID: %d]-> w(%d), u(%d), v(%d)\n", world_rank, local_pair.weight, local_pair.from, local_pair.to);
                
                Edge recv_val;
                MPI_Allreduce(
                    &local_pair,
                    &recv_val,
                    1,
                    MPI_EDGE,
                    MPI_MIN_EDGE,
                    MPI_COMM_WORLD
                );

                recv_buff[root_i] = recv_val;
            }

            time_print("2 All gather", world_rank);

            update_min_graph_from_roots_not_id(parent, rank, recv_buff, graph, min_graph);
            time_print("2 updated min graph", world_rank);
        }

        for (int i = 0; i < NODE_COUNT; i++) {
            find(parent, i);
        }

        time_print("Find root", world_rank);

        if (world_rank == 0) {
            printf("Num components %d\n", find_num_components(parent));
        }

        num_components = find_num_components(parent);
        if (num_components == 1) {
            if (world_rank == 0) {
                // print_matrix(min_graph);
            }
            break;
        }

        prune_graph(parent, graph);
        time_print("Prune graph", world_rank);
        count++;
    }
    time_print("Finished computing MST", world_rank);
    if (world_rank == 0) printf("[ID:%d] Total time: %f\n", world_rank, MPI_Wtime() - start_time);

    free(parent);
    free(rank);

    // Cleanup custom MPI datatype and operation
    MPI_Type_free(&MPI_EDGE);
    MPI_Op_free(&MPI_MIN_EDGE);

    MPI_Finalize();
}