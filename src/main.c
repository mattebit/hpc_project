#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <unistd.h>

// #define NODE_COUNT 10

int NODE_COUNT = 0;

/**
 * Used to log a message
 */
void log_message(int process_id, char *message)
{
    time_t now = time(NULL);
    struct tm *t = localtime(&now);

    printf("[%d-%02d-%02d %02d:%02d:%02d] [ID:%d] %s\n",
           t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
           t->tm_hour, t->tm_min, t->tm_sec, process_id, message);
}

int **allocate_and_init_matrix()
{
    int **min_graph = (int **)malloc(NODE_COUNT * sizeof(int *));
    for (int i = 0; i < NODE_COUNT; i++)
    {
        min_graph[i] = (int *)malloc(NODE_COUNT * sizeof(int));
    }

    int i = 0;
    for (i; i < NODE_COUNT; i++)
    {
        int j = 0;
        for (j; j < NODE_COUNT; j++)
        {
            min_graph[i][j] = 0;
        }
    }
    return min_graph;
}

int **fill_graph()
{
    int **matrix = allocate_and_init_matrix();

    int used_values[NODE_COUNT * NODE_COUNT * 2];
    int k = 0;
    for (k; k < NODE_COUNT * 2; k++)
    {
        used_values[k] = false;
    }

    int i = 0;
    for (i; i < NODE_COUNT; i++)
    {
        int j = 0;
        for (j; j < NODE_COUNT; j++)
        {
            if (i == j)
            {
                matrix[i][j] = -1;
            }
            else
            {
                int gen = 0;
                while (true)
                {
                    gen = rand() % (NODE_COUNT * NODE_COUNT * 2);
                    if (!used_values[gen])
                    {
                        used_values[gen] = true;
                        break;
                    }
                }
                if (matrix[i][j] == 0)
                {
                    matrix[i][j] = gen;
                }
                if (matrix[j][i] == 0)
                {
                    matrix[j][i] = gen;
                }
            }
        }
    }

    return matrix;
}

/**
 * Finds the lightest edge among the edges of the specified node and returns the id of the node connected by that edge
 */
int find_lightest_edge(int **graph, int node_id)
{
    int i = 0;
    int min = NODE_COUNT * NODE_COUNT * 2 + 100; // graph[node_id][0];
    int min_id = 0;
    for (i; i < NODE_COUNT; i++)
    {
        int act = graph[node_id][i];
        if (act != -1 && act < min)
        {
            min = act;
            min_id = i;
        }
    }

    if (min == -1)
    {
        return -1;
    }

    return min_id;
}

int find_min_indx(int *nodes_id, int **graph)
{
    int i = 0;
    int min = NODE_COUNT * NODE_COUNT * 2 + 100; // graph[0][nodes_id[0]];
    int min_id = 0;
    for (i; i < NODE_COUNT; i++)
    {
        int act = graph[i][nodes_id[i]];
        if (act != -1 && act < min)
        {
            min = act;
            min_id = i;
        }
    }

    if (min == -1)
    {
        return -1;
    }

    return min_id;
}

/**
 * Send current node value to all node, and receive values from all the other nodes.
 * This function returns an array containing the results of the nodes, having the result of
 * node i in index i
 */
void update_all_nodes(int value, int node_id, int *recvbuff)
{
    int sendbuff[NODE_COUNT];
    // the send buff contains in every index, the data to be sent to the process with that index
    // i.e. sendbuff[2] will be sent to process 2

    // Set the value to be sent to all the nodes
    int i = 0;
    for (i; i < NODE_COUNT; i++)
    {
        sendbuff[i] = value;
    }

    // recvbuff will contain in every index the value that has been received by the process with that index

    // printf("Nodeid: %d my values: %d,%d,%d,%d\n", node_id, sendbuff[0], sendbuff[1], sendbuff[2], sendbuff[3]);
    MPI_Alltoall(sendbuff, 1, MPI_INT, recvbuff, 1, MPI_INT, MPI_COMM_WORLD);
    // printf("Nodeid: %d received values: %d,%d,%d,%d\n", node_id, recvbuff[0], recvbuff[1], recvbuff[2], recvbuff[3]);
}

/**
 * Update the min graph with the min values received by the nodes.
 * The min_values array should be of length NODE_COUNT, and contain one value for each index
 */
void update_min_graph(int *min_values, int **min_graph)
{
    int i = 0;
    for (i; i < NODE_COUNT; i++)
    {
        int act = min_values[i];
        if (act == -1)
        {
            printf("something wrong\n");
        }
        else
        {
            // Setting to true means that that edge is included
            min_graph[i][act] = 1;
        }
    }
}

/**
 * Searches for the nodes that are part of the same component of the given node.
 * NOTE: this function exploits the fact that the min graph has nodes connected at most 1 time, so there are no cycles
 */
void find_component(bool *nodes_in_component, int node_id, int **min_graph)
{
    if (nodes_in_component[node_id])
    {
        return; // already analyzed
    }

    nodes_in_component[node_id] = true;
    int i = 0;

    for (i; i < NODE_COUNT; i++)
    {
        if (min_graph[node_id][i] == 1)
        {
            if (!nodes_in_component[i])
            {
                find_component(nodes_in_component, i, min_graph);
            }
            else
            {
                int k = 0;
                for (k; k < NODE_COUNT; k++)
                {
                    if (nodes_in_component[k])
                    {
                        int j = 0;
                        for (j; j < NODE_COUNT; j++)
                        {
                            if (min_graph[j][k] == 1)
                            {
                                find_component(nodes_in_component, j, min_graph);
                            }
                        }
                    }
                }
            }
            return;
        }
    }
}

/**
 * Prunes the given graph from the edges between the same component
 */
void prune_graph(bool *nodes_in_component, int **graph)
{
    int i = 0;
    for (i; i < NODE_COUNT; i++)
    {
        if (nodes_in_component[i])
        {
            int j = 0;
            for (j; j < NODE_COUNT; j++)
            {
                if (nodes_in_component[j])
                {
                    graph[i][j] = -1;
                }
            }
        }
    }
}

/**
 * Check wheter the algorithm finished
 */
bool is_connected(bool *component_nodes)
{
    int i = 0;
    for (i; i < NODE_COUNT; i++)
    {
        if (!component_nodes[i])
        {
            return false;
        }
    }
    return true;
}

void print_matrix(int **matrix)
{
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

int main(int argc, char **argv)
{
    MPI_Init(NULL, NULL);

    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &NODE_COUNT);

    // printf("world: %d\n", NODE_COUNT);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // log_message(world_rank, "Started");

    int **graph = fill_graph();
    int **min_graph = allocate_and_init_matrix();

    /*
    if (world_rank == 0)
    {
        int i = 0;
        for (i; i < NODE_COUNT; i++)
        {
            int j = 0;
            int min = 2000;
            for (j; j < NODE_COUNT; j++)
            {
                if (graph[i][j] != -1 && graph[i][j] < min)
                {
                    min = graph[i][j];
                }
            }
            j = 0;
            for (j; j < NODE_COUNT; j++)
            {
                if (graph[i][j] == min)
                {
                    graph[i][j] = min;
                }
                else
                {
                    graph[i][j] = 0;
                }
            }
        }

        print_matrix(graph);
    }
    exit(0);
    */

    // print_matrix(graph);
    // print_matrix(min_graph);

    int count = 0;

    while (true)
    {
        int lightest = find_lightest_edge(graph, world_rank);
        // printf("lightest %d: %d\n", world_rank, lightest);

        int *recv_values = malloc(NODE_COUNT * sizeof(int));

        update_all_nodes(lightest, world_rank, recv_values);

        // not all nodes need to receive at step >0

        // TODO: if second iteration choose the lightest among the one selected by the nodes
        if (count != 0)
        {
            int min_id = find_min_indx(recv_values, graph);
            int min_dest = recv_values[min_id];
            min_graph[min_id][min_dest] = 1;
        }
        else
        {
            update_min_graph(recv_values, min_graph);
        }

        free(recv_values);

        // this array tells which nodes are part of the component of this node
        bool component_nodes[NODE_COUNT];
        int i = 0;
        for (i; i < NODE_COUNT; i++)
        {
            component_nodes[i] = false;
        }

        // find nodes that are part of this node component
        find_component(component_nodes, world_rank, min_graph);

        // printf("component: %d,%d,%d,%d\n", component_nodes[0], component_nodes[1], component_nodes[2], component_nodes[3]);

        // check if there is a single component (algorithm has finished)

        if (0 == 1)
        {
            int i = 0;
            for (i; i < NODE_COUNT; i++)
            {
                printf("%d,", component_nodes[i]);
            }
            printf("\n");
            print_matrix(min_graph);
            printf("\n");
        }

        if (is_connected(component_nodes))
        {
            if (true)
            {
                printf("%d is connected\n", world_rank);
            }
            break;
        }

        // prune the graph by removing edges between nodes of the same component (needed for finding lightest edge)
        prune_graph(component_nodes, graph);

        count++;
    }

    // Finalize the MPI environment. No more MPI calls can be made after this
    MPI_Finalize();
}