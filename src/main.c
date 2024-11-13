#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>

#define NODE_COUNT 100

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

int **fill_graph()
{
    int **matrix = (int **)malloc(NODE_COUNT * sizeof(int *));
    for (int i = 0; i < NODE_COUNT; i++)
    {
        matrix[i] = (int *)malloc(NODE_COUNT * sizeof(int));
    }

    int i = 0;
    for (i; i < NODE_COUNT; i++)
    {
        int j = 0;
        for (j; j < NODE_COUNT; j++)
        {
            if (i == j)
            {
                matrix[i][j] = 101;
            }
            else
            {
                matrix[i][j] = rand() % 100;
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
    int i = 1;
    int min = graph[node_id][0];
    int min_id = 0;
    for (i; i < NODE_COUNT; i++)
    {
        int act = graph[node_id][i];
        if (act < min)
        {
            min = act;
            min_id = i;
        }
    }

    return min_id;
}

int find_min_indx(int *nodes_id, int **graph)
{
    int i = 1;
    int min = graph[0][nodes_id[0]];
    int min_id = 0;
    for (i; i < NODE_COUNT; i++)
    {
        int act = graph[i][nodes_id[i]];
        if (act < min)
        {
            min = act;
            min_id = i;
        }
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

    printf("Nodeid: %d my values: %d,%d,%d,%d\n", node_id, sendbuff[0], sendbuff[1], sendbuff[2], sendbuff[3]);
    MPI_Alltoall(sendbuff, 1, MPI_INT, recvbuff, 1, MPI_INT, MPI_COMM_WORLD);
    printf("Nodeid: %d received values: %d,%d,%d,%d\n", node_id, recvbuff[0], recvbuff[1], recvbuff[2], recvbuff[3]);
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
        // Setting to true means that that edge is included
        min_graph[i][act] = 1;
    }
}

/**
 * Searches for the nodes that are part of the same component of the given node.
 * NOTE: this function exploits the fact that the min graph has nodes connected at most 1 time, so there are no cycles
 */
void find_component(bool *nodes_in_component, int node_id, int **min_graph)
{
    nodes_in_component[node_id] = true;
    // printf("component: %d,%d,%d,%d\n", nodes_in_component[0], nodes_in_component[1], nodes_in_component[2], nodes_in_component[3]);
    //  TODO: test
    int i = 0;

    for (i; i < NODE_COUNT; i++)
    {
        if (min_graph[node_id][i] == 1)
        {
            // printf("found\n");
            if (!nodes_in_component[i])
            {
                find_component(nodes_in_component, i, min_graph);
            }
            else
            {
                int j = 0;
                for (j; j < NODE_COUNT; j++)
                {
                    if (!nodes_in_component[j])
                    {
                        find_component(nodes_in_component, j, min_graph);
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
        bool act = nodes_in_component[i];
        if (act)
        {
            int j = 0;
            for (j; j < NODE_COUNT; j++)
            {
                if (nodes_in_component[j])
                {
                    graph[act][j] = 0;
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
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    log_message(world_rank, "Started");

    int **graph = fill_graph();
    int **min_graph = allocate_and_init_matrix();

    print_matrix(graph);

    print_matrix(min_graph); // something rotto

    int count = 0;

    while (true)
    {
        int lightest = find_lightest_edge(graph, world_rank);
        printf("lightest %d: %d\n", world_rank, lightest);

        int *recv_values = malloc(NODE_COUNT * sizeof(int));

        update_all_nodes(lightest, world_rank, recv_values);

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

        print_matrix(min_graph);
        free(recv_values);

        // this array tells which nodes are part of the component of this node
        bool component_nodes[NODE_COUNT];

        // find nodes that are part of this node component
        find_component(component_nodes, world_rank, min_graph);

        // printf("component: %d,%d,%d,%d\n", component_nodes[0], component_nodes[1], component_nodes[2], component_nodes[3]);

        // check if there is a single component (algorithm has finished)
        if (is_connected(component_nodes))
        {
            printf("%d is connected\n", world_rank);
            break;
        }

        // prune the graph by removing edges between nodes of the same component (needed for finding lightest edge)
        prune_graph(component_nodes, graph);

        count++;
    }

    // Finalize the MPI environment. No more MPI calls can be made after this
    MPI_Finalize();
}