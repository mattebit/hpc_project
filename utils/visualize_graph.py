import networkx as nx
import matplotlib.pyplot as plt
import sys

def read_graph(filename):
    with open(filename, 'r') as f:
        # Read first line for number of vertices and edges
        n, m = map(int, f.readline().split())
        
        # Create empty graph
        G = nx.Graph()
        
        # Add all vertices
        G.add_nodes_from(range(n))
        
        # Read edges
        for line in f:
            src, dest, weight = map(int, line.split())
            # Add edge with weight (if multiple edges exist, keep minimum weight)
            if G.has_edge(src, dest):
                current_weight = G[src][dest]['weight']
                if weight < current_weight:
                    G[src][dest]['weight'] = weight
            else:
                G.add_edge(src, dest, weight=weight)
    
    return G

def visualize_graph(G, output_file=None):
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Get positions for nodes using spring layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=500, alpha=0.8)
    
    # Draw edges with weights
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    
    # Normalize weights for edge thickness
    max_weight = max(weights)
    edge_width = [2 * w / max_weight for w in weights]
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=edge_width, alpha=0.5)
    
    # Add edge labels
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels)
    
    # Add node labels
    nx.draw_networkx_labels(G, pos)
    
    plt.title("Graph Visualization")
    plt.axis('off')
    
    if output_file:
        plt.savefig(output_file, format='png', dpi=300, bbox_inches='tight')
    else:
        plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_graph.py <input_graph_file> [output_image_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Read and visualize graph
    G = read_graph(input_file)
    visualize_graph(G, output_file)