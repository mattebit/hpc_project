#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <unistd.h>

// #define NODE_COUNT 10

int NODE_COUNT = 20;
int MAX_EDGE_VALUE = __INT_MAX__;

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
        min_graph[i] = (int*)malloc(NODE_COUNT * sizeof(int));
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

int** fill_graph() {
    srand(17);
    int** matrix = allocate_and_init_matrix();

    int used_values[NODE_COUNT * NODE_COUNT * 4];
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
                    gen = rand() % (NODE_COUNT * NODE_COUNT * 4);
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
int find_lightest_edge(int** graph, int node_id) {
    int i = 0;
    int min = MAX_EDGE_VALUE;
    int min_id = -1;
    for (i; i < NODE_COUNT; i++)
    {
        int act = graph[node_id][i];
        if (act != -1 && act < min)
        {
            min = act;
            min_id = i;
        }
    }

    return min_id;
}

int find_min_indx(int* nodes_id, int** graph) {
    int i = 0;
    int min = MAX_EDGE_VALUE;
    int min_id = -1;
    for (i; i < NODE_COUNT; i++)
    {
        if (nodes_id[i] != -1)
        {
            int act = graph[i][nodes_id[i]];
            if (act != -1 && act < min)
            {
                min = act;
                min_id = i;
            }
        }
    }

    return min_id;
}

/**
 * Send current node value to all node, and receive values from all the other nodes.
 * This function returns an array containing the results of the nodes, having the result of
 * node i in index i
 */
void update_all_nodes(int value, int node_id, int* recvbuff, MPI_Group mygroup, int size, bool* group_members) {
    int sendbuff[size];
    // the send buff contains in every index, the data to be sent to the process with that index
    // i.e. sendbuff[2] will be sent to process 2

    // Set the value to be sent to all the nodes
    int i = 0;
    for (i; i < size; i++) {
        sendbuff[i] = value;
    }

    // recvbuff will contain in every index the value that has been received by the process with that index

    if (mygroup != NULL) {
        MPI_Comm new_comm;
        MPI_Comm_create(MPI_COMM_WORLD, mygroup, &new_comm);

        int* fakebuff = malloc(sizeof(int) * size);

        // printf("Nodeid: %d my values: %d,%d,%d,%d\n", node_id, sendbuff[0], sendbuff[1], sendbuff[2], sendbuff[3]);
        MPI_Alltoall(sendbuff, 1, MPI_INT, fakebuff, 1, MPI_INT, new_comm);
        //printf("Nodeid: %d received values:", node_id);

        int i = 0;
        int c = 0;
        for (i; i < NODE_COUNT; i++)
        {
            if (group_members[i])
            {
                recvbuff[i] = fakebuff[c++];
            }
        }
        free(fakebuff);
    } else {
        MPI_Alltoall(sendbuff, 1, MPI_INT, recvbuff, 1, MPI_INT, MPI_COMM_WORLD);
    }
}

/**
 * Update the min graph with the min values received by the nodes.
 * The min_values array should be of length NODE_COUNT, and contain one value for each index
 */
void update_min_graph(int* min_values, int** min_graph) {
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
            min_graph[act][i] = 1;
        }
    }
}

void add_recursive_references(int** min_graph) {
    int i = 0;
    for (i; i < NODE_COUNT; i++)
    {
        int j = 0;
        for (j; j < NODE_COUNT; j++)
        {
            if (min_graph[i][j] == 1 || min_graph[j][i] == 1)
            {
                if (min_graph[i][j]==0 || min_graph[j][i] == 0) {
                    printf("not symmetric\n");
                }
                min_graph[j][i] = 1;
                min_graph[i][j] = 1;
            }
        }
    }
}

/**
 * Searches for the nodes that are part of the same component of the given node.
 * NOTE: this function exploits the fact that the min graph has nodes connected at most 1 time, so there are no cycles
 */
void find_component(bool* nodes_in_component, int node_id, int** min_graph, bool* visited) {
    if (visited[node_id]) {
        return;
    }

    visited[node_id] = true;
    nodes_in_component[node_id] = true;

    int i = 0;
    for (i; i < NODE_COUNT; i++) {
        if (min_graph[node_id][i] == 1) {
            find_component(nodes_in_component, i, min_graph, visited);
        }
    }
}

/**
 * Prunes the given graph from the edges between the same component
 */
void prune_graph(bool* nodes_in_component, int** graph) {
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
bool is_connected(bool* component_nodes) {
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

void update_mpi_group(bool* component_nodes, MPI_Group* new_group, int* group_size) {
    int ranks[NODE_COUNT];

    int i = 0;
    int c = 0;
    for (i; i < NODE_COUNT; i++)
    {
        if (component_nodes[i])
        {
            ranks[c++] = i;
        }
    }
    
    //printf("Rank: ");
    int ranks_ok[c];
    i = 0;
    for (i; i < c; i++)
    {
    //    printf("%d,", ranks[i]);
        ranks_ok[i] = ranks[i];
    }
    //printf("\n");

    /*
    i = 0;
    for (i; i < NODE_COUNT; i++)
    {
        printf("%d,", component_nodes[i]);
    }
    printf("\n\n");
    */

    *group_size = c;

    MPI_Group group;
    MPI_Comm_group(MPI_COMM_WORLD, &group);
    int size = 0;
    MPI_Group_size(group, &size);
    printf("group world size: %d; new size: %d; \n", size, c);
    MPI_Group_incl(group, c, ranks_ok, new_group);
}

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &NODE_COUNT);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    //log_message(world_rank, "Started");

    int** graph = fill_graph();
    int** min_graph = allocate_and_init_matrix();

    MPI_Group mygroup;
    int group_size = 0;

    bool component_nodes[NODE_COUNT];
    int i = 0;
    for (i; i < NODE_COUNT; i++)
    {
        component_nodes[i] = false;
    }

    int count = 0;
    while (true)
    {
        int lightest = find_lightest_edge(graph, world_rank);
        // printf("lightest %d: %d\n", world_rank, lightest);

        int* recv_values = malloc(NODE_COUNT * sizeof(int));
        int l = 0;
        for (l; l < NODE_COUNT; l++)
        {
            recv_values[l] = -1;
        }

        // not all nodes need to receive at step >0
        // TODO: if second iteration choose the lightest among the one selected by the nodes
        if (count != 0)
        {
            update_all_nodes(lightest, world_rank, recv_values, mygroup, group_size, component_nodes);
            int min_id = find_min_indx(recv_values, graph);
            int min_dest = recv_values[min_id];
            // printf("min_dest: %d\n", min_dest);
            //printf("min_graph_value: %d\n", min_graph[min_id][min_dest]);
            min_graph[min_id][min_dest] = 1;
            min_graph[min_dest][min_id] = 1;
        }
        else
        {
            update_all_nodes(lightest, world_rank, recv_values, NULL, NODE_COUNT, NULL);
            update_min_graph(recv_values, min_graph);
        }

        free(recv_values);

        // this array tells which nodes are part of the component of this node
        // find nodes that are part of this node component
        bool visited[NODE_COUNT];
        int i = 0;
        for (i; i < NODE_COUNT; i++)
        {
            visited[i] = false;
        }
        find_component(component_nodes, world_rank, min_graph, visited);

        if (world_rank == 0)
        {
            int i = 0;
            for (i; i < NODE_COUNT; i++)
            {
                printf("%d,", component_nodes[i]);
            }
            printf("\n\n");
            print_matrix(min_graph);
            printf("\n");
            //print_matrix(graph);
        }

        //add_recursive_references(min_graph);

        // check if there is a single component (algorithm has finished)
        if (is_connected(component_nodes))
        {
            if (1 == 1)
            {
                printf("%d is connected\n", world_rank);
            }
            break;
        }

        // prune the graph by removing edges between nodes of the same component (needed for finding lightest edge)
        prune_graph(component_nodes, graph);
        update_mpi_group(component_nodes, &mygroup, &group_size);

        printf("[ID:%d][cycl:%d] Component#: %d;\tcomponents:", world_rank, count, group_size);
        i = 0;
        for (i; i<NODE_COUNT; i++) {
            printf("%d,", component_nodes[i]);
        }
        printf("\n");

        count++;
    }

    // Finalize the MPI environment. No more MPI calls can be made after this
    MPI_Finalize();
}