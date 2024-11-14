import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sys

# Define the adjacency matrix
adj_matrix = []

adj_matrix_str = ""


arg1 = sys.stdin.readlines()
#arg1 = sys.argv[1]
print(arg1)

if len(arg1) > 0:
    for l in arg1:
        adj_matrix_str += l



def parse_matrix():
    rows = adj_matrix_str.split("\n")

    for r in rows:
        elements = r.split(" ")
        res_r = []
        for e in elements:
            if len(e)>0:
                res_r.append(int(e))

        adj_matrix.append(res_r)

parse_matrix()
print(adj_matrix)

adj_matrix = np.array(adj_matrix)

def show_graph_with_labels(adjacency_matrix):
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)

    #edge_labels = nx.get_edge_attributes(gr, 'weight')

    #pos = nx.spring_layout(gr) # kamada_kawai_layout or spring_layout
    pos = nx.kamada_kawai_layout(gr)

    nx.draw(gr, pos=pos, node_size=500, with_labels=True)
    plt.show()

show_graph_with_labels(adj_matrix)

exit()

# Create a new figure and plot the nodes
plt.figure(figsize=(8, 8))
plt.plot(np.arange(len(adj_matrix)), np.arange(len(adj_matrix)), 'o', markersize=15, markerfacecolor='lightblue')

# Add edges between nodes based on the adjacency matrix
for i in range(len(adj_matrix)):
    for j in range(len(adj_matrix)):
        if adj_matrix[i][j] != 1:
            plt.plot([i, j], [i, j], 'b-', alpha=0.7)

# Set labels for nodes
plt.xticks(np.arange(len(adj_matrix)), np.arange(1, len(adj_matrix) + 1))
plt.yticks(np.arange(len(adj_matrix)), np.arange(1, len(adj_matrix) + 1))

# Remove axis spines
plt.axis('off')

# Show the graph
plt.title("Adjacency Matrix Visualization")
plt.show()