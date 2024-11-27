#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <omp.h>

//# define DEBUG

int NODE_COUNT = 100;
int MAX_EDGE_VALUE = __INT_MAX__;

int MY_NODES_FROM = 0;
int MY_NODES_TO = 100;

double last_time = 0.0;

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

int** allocate_and_init_matrix() {
    int** min_graph = (int**)malloc(NODE_COUNT * sizeof(int*));
    for (int i = 0; i < NODE_COUNT; i++)
    {
        min_graph[i] = (int*)calloc(NODE_COUNT, sizeof(int));
    }
    return min_graph;
}

int** fill_graph() {
    srand(127);
    int** matrix = allocate_and_init_matrix();

    int* used_values = (int*)calloc( NODE_COUNT * NODE_COUNT * 4, sizeof(int));

    #pragma omp parallel for
    for (int i = 0; i < NODE_COUNT; i++) {
        for (int j = 0; j < NODE_COUNT; j++) {
            if (i == j) {
                matrix[i][j] = -1;
            } else {
                int gen;
                do {
                    gen = rand() % (NODE_COUNT * NODE_COUNT * 4);
                } while (used_values[gen]);
                used_values[gen] = 1;
                matrix[i][j] = gen;
                matrix[j][i] = gen;
            }
        }
    }
    free(used_values);

    return matrix;
}

/**
 * Finds the lightest edge among the edges of the specified node and returns the id of the node connected by that edge
 */
int find_lightest_edge(int** graph, int node_id) {
    int i = 0;
    int min = MAX_EDGE_VALUE;
    int min_id = -1;
    for (i; i < NODE_COUNT; i++) {
        int act = graph[node_id][i];
        if (act != -1 && act < min) {
            min = act;
            min_id = i;
        }
    }

    return min_id;
}

/**
 * Update the min graph with the min values received by the nodes.
 * The min_values array should be of length NODE_COUNT, and contain one value for each index
 */
void update_min_graph(int* min_values, int** min_graph) {
    int i = 0;
    for (i; i < NODE_COUNT; i++) {
        int act = min_values[i];
        if (act == -1) {
            //do nothing
        }
        else {
            // Setting to 1 means that that edge is included
            min_graph[i][act] = 1;
            min_graph[act][i] = 1;
        }
    }
}

int find_root(int node_id, int** min_graph, int* visited, int min_reported, int* roots) {
    if (visited[node_id] == 1) {
        return -1;
    }

    visited[node_id] = 1;
    int min_root = min_reported;
    int i = 0;
    for (i; i < NODE_COUNT; i++) {
        if (min_graph[node_id][i] == 1) {
            int act_root = find_root(i, min_graph, visited, min_root, roots);
            if (act_root != -1 && act_root < min_root) {
                min_root = act_root;
                roots[node_id] = min_root;
            }
        }
    }
    roots[node_id] = min_root;
    return min_root;
}

/**
 * Prunes the given graph from the edges between the same component
 */
void prune_graph(int* roots, int** graph) {
    int i = 0;
    for (i; i < NODE_COUNT; i++)
    {
        int j = 0;
        for (j; j < NODE_COUNT; j++)
        {
            if (graph[i][j] != -1 && roots[i] == roots[j]) {
                graph[i][j] = -1;
                graph[j][i] = -1;
            }
        }
    }

}

void print_matrix(int** matrix) {
    // Print the adjacency matrix
    for (int i = 0; i < NODE_COUNT; i++)
    {
        for (int j = 0; j < NODE_COUNT; j++)
        {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}

void zero_array(int* array, int size) {
    int i = 0;
    for (i; i < size; i++) {
        array[i] = 0;
    }
}

void minus_array(int* array, int size) {
    int i = 0;
    for (i; i < size; i++) {
        array[i] = -1;
    }
}

void print_array(int* array, char* name, int size) {
    printf("%s: ", name);
    int i = 0;
    for (i; i < size; i++) {
        printf("%d ", array[i]);
    }
    printf("\n");
}

int find_min_in_array(int* arr, int size) {
    int min = MAX_EDGE_VALUE;
    for (int i = 0; i < size; i++) {
        if (arr[i] < min) {
            min = arr[i];
        }
    }
    return min;
}

int find_num_components(int* roots) {
    int* buff = malloc(NODE_COUNT * sizeof(int));
    zero_array(buff, NODE_COUNT);
    int i = 0;
    for (i; i< NODE_COUNT; i++) {
        buff[roots[i]] += 1;
    }

    int c = 0;
    i = 0;
    for (i; i< NODE_COUNT; i++) {
        if (buff[i] > 0) {
            c += 1;
        }
    }
    free(buff);
    return c;
}

void time_print(char* desc, int world_rank) {
    if (world_rank != 0)
        return;
    double actual_time = MPI_Wtime();
    double diff = actual_time - last_time;
    printf("[ID:%d] Time taken: %f sec (%s) -> \n", world_rank, diff, desc);
    last_time = actual_time;
}

int find_min_components(int*roots, int** graph, int* result) {
    int i = MY_NODES_FROM;
    int can_connect = 0;
    for (i; i< MY_NODES_TO; i++) {
        int act_root = roots[i];

        int min = MAX_EDGE_VALUE;
        int min_id = -1;
        int j = 0;
        for (j; j < NODE_COUNT; j++) {
            if (graph[i][j] != -1 && graph[i][j] < min) {
                min = graph[i][j];
                min_id = j;
            }
        }
        if (min != MAX_EDGE_VALUE) {
            can_connect = 1;
        }
        result[i] = min_id;
    }
    return can_connect;
}

void update_min_graph_from_roots(int* roots, int** graph, int** min_graph, int* result, int* unique_roots) {
    int i = 0;
    for (i; i < NODE_COUNT; i++) {
        if (unique_roots[i] != -1) {
            int j = 0;
            int min_id_from = -1;
            int min_id_to = -1;
            int min = MAX_EDGE_VALUE;
            for (j; j < NODE_COUNT; j++) {
                // fin the min inside the values of that root
                if (roots[j] == i) {
                    if (graph[j][result[j]] < min) {
                        min = graph[j][result[j]];
                        min_id_from = j;
                        min_id_to = result[j];
                    }
                }
            }
            min_graph[min_id_from][min_id_to] = 1;
            min_graph[min_id_to][min_id_from] = 1;
        }
    }
}

void update_min_graph_from_roots_not_id(int* root_mins, int**graph, int** min_graph) {
    int i = 0;
    for (i; i < NODE_COUNT; i++) {
        if (root_mins[i] != -1) {
            int root_min_edge_value = root_mins[i];
            //printf("%d\n", root_min_edge_value);
            int j = 0;
            for (j; j < NODE_COUNT; j++) {
                int k = 0;
                for (k; k < NODE_COUNT; k++) {
                    if (graph[j][k] != -1 && graph[j][k] == root_min_edge_value) {
                        //printf("root_min_edge_value= %d\n", root_min_edge_value);
                        printf("Connect %d with %d. Weight = %d\n", j, k, graph[j][k]);
                        min_graph[j][k] = 1;
                        min_graph[k][j] = 1;
                        graph[j][k] = -1;
                        graph[k][j] = -1;
                    }
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    int** graph = fill_graph(); // generate graph before forking

    MPI_Init(NULL, NULL);
    last_time = MPI_Wtime();
    int process_count;
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    time_print("generated graph", world_rank);

    // Find indexes of which vertex this node is responsible for
    int vertex_per_process = NODE_COUNT / process_count; // TODO: fix funziona solo con pari
    MY_NODES_FROM = world_rank*vertex_per_process;
    MY_NODES_TO = MY_NODES_FROM + vertex_per_process;

    //log_message(world_rank, "Started");
    //printf("ID: %d, indexes [%d,%d]\n", world_rank, MY_NODES_FROM, MY_NODES_TO);

    int** min_graph = allocate_and_init_matrix();

    // time_print("Allocated min matrix", world_rank);

    int* roots = (int*)malloc(NODE_COUNT * sizeof(int));
    zero_array(roots, NODE_COUNT);

    time_print("Allocated stuff", world_rank);

    int num_components = 0;
    int count = 0;
    while (1)
    {
        if (count == 0) {
            int* ligthest_edges = (int*)malloc(vertex_per_process * sizeof(int));
            zero_array(ligthest_edges, vertex_per_process);
            int i = MY_NODES_FROM;
            for (i; i < MY_NODES_TO; i++) {
                int val = find_lightest_edge(graph, i);
                ligthest_edges[i - MY_NODES_FROM] = val;
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
            update_min_graph(recv_values, min_graph);
            free(recv_values);

            time_print("1 update graph", world_rank);

            // time_print("update min graph", world_rank);
        } else {
            int* result = malloc(NODE_COUNT * sizeof(int));
            minus_array(result, NODE_COUNT);

            find_min_components(roots, graph, result);

            int* recv_buff = malloc(NODE_COUNT * sizeof(int));
            minus_array(recv_buff, NODE_COUNT);

            time_print("2 before all gather", world_rank);

            int* unique_roots = malloc(NODE_COUNT * sizeof(int));
            minus_array(unique_roots, NODE_COUNT);

            int i = 0;
            for (i; i< NODE_COUNT; i++) {
                unique_roots[roots[i]] += 1;
            }

            printf("[ID:%d][%d] Num components: %d\n", world_rank, count, num_components);
            i = 0;
            for (i; i < NODE_COUNT; i++) {
                if (unique_roots[i] == -1)
                    continue;

                int root = i;
                int min = MAX_EDGE_VALUE;
                int j = 0;
                for (j; j < NODE_COUNT; j++) {
                    if (result[j] == -1)
                        continue;

                    if (roots[j] == root) {
                        if (graph[j][result[j]] != -1) {
                            if(graph[j][result[j]] < min) {
                                min = graph[j][result[j]];
                            }
                        }
                    }
                }

                printf("[ID:%d][%d] Done\n", world_rank, count);
                //if (world_rank == 1) printf("min = %d\n", min);

                int* recv_val = malloc(1*sizeof(int));
                //if (world_rank==0) printf("Root: %d: val: %d\n", root, min);
                //if (world_rank==0) print_array(roots,"roots", NODE_COUNT);
                MPI_Allreduce(
                    &min,
                    recv_val,
                    1,
                    MPI_INT,
                    MPI_MIN,
                    MPI_COMM_WORLD
                );

                recv_buff[root] = *recv_val;
                free(recv_val);
            }

            print_array(recv_buff, "recv_buff", NODE_COUNT);

            free(unique_roots);
            time_print("2 All gather", world_rank);

            //update_min_graph_from_roots(roots, graph, min_graph, recv_buff, unique_roots);
            update_min_graph_from_roots_not_id(recv_buff, graph, min_graph);
            time_print("2 updated min graph", world_rank);
            free(recv_buff);
        }

        // this array tells which nodes are part of the component of this node
        // find nodes that are part of this node component
        int* visited = malloc(NODE_COUNT * sizeof(int));
        zero_array(visited, NODE_COUNT);

        // compute roots of each node
        int i = 0;
        for (i; i < NODE_COUNT; i++) {
            find_root(i, min_graph, visited, i, roots);
        }
        free(visited);

        //if (world_rank==1) printf("d0ne\n");
        time_print("Find root", world_rank);

        if (world_rank == 0) {
            printf("Num components %d\n", find_num_components(roots));
        }

        if ( world_rank == 0 ) {
            //print_matrix(min_graph);
        }

        num_components = find_num_components(roots);
        if (num_components == 1) {
            if ( world_rank == 0 ) {
                print_matrix(min_graph);
            }
            break;
        }


        prune_graph(roots, graph);
        time_print("Prune graph", world_rank);
        count++;
    }
    time_print("Finished computing MST", world_rank);
    // Finalize the MPI environment. No more MPI calls can be made after this
    MPI_Finalize();
}
