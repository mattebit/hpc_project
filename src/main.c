#include <mpi.h>
#include <stdio.h>
#include <time.h>
#include <stdbool.h>

#define NODE_COUNT 3

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
    // TODO fill graph
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
            act = min;
            min_id = i;
        }
    }

    return min_id;
}

int *find_min(int *values)
{
    int i = 1;
    int min = values[0];
    int min_id = 0;
    for (i; i < NODE_COUNT; i++)
    {
        int act = values[i];
        if (min == NULL | act < min)
        {
            act = min;
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
int *update_all_nodes(int value, int node_id)
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
    int recvbuff[NODE_COUNT];

    printf("Nodeid: %d my values: %d,%d,%d\n", node_id, sendbuff[0], sendbuff[1], sendbuff[2]);
    MPI_Alltoall(sendbuff, 1, MPI_INT, recvbuff, 1, MPI_INT, MPI_COMM_WORLD);
    printf("Nodeid: %d received values: %d,%d,%d\n", node_id, recvbuff[0], recvbuff[1], recvbuff[2]);

    return recvbuff;
}

/**
 * Update the min graph with the min values received by the nodes.
 * The min_values array should be of length NODE_COUNT, and contain one value for each index
 */
void update_min_graph(int *min_values, bool **min_graph)
{
    int i = 0;
    for (i; i < NODE_COUNT; i++)
    {
        int act = min_values[i];
        // Setting to true means that that edge is included
        min_graph[i][act] = true;
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

    // int** graph = fill_graph();
    bool min_graph[NODE_COUNT][NODE_COUNT];

    // int lightest = find_lightest_edge(graph, world_rank);

    int *recv_values[NODE_COUNT] = update_all_nodes(world_rank, world_rank);
    update_min_graph(recv_values, min_graph);

    // Finalize the MPI environment. No more MPI calls can be made after this
    MPI_Finalize();
}