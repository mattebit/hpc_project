#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <stdint.h>
#include <string.h>

// Improved random number generation
typedef struct {
    uint64_t state;
} xorshift64_state;

// Fast XorShift random number generator
uint64_t xorshift64(xorshift64_state *state) {
    state->state ^= state->state << 13;
    state->state ^= state->state >> 7;
    state->state ^= state->state << 17;
    return state->state;
}

// Graph generation parameters
typedef struct {
    int64_t vertex_count;     // Total number of vertices
    int edge_density;         // Percentage of possible edges to include (1-100)
    int min_weight;           // Minimum edge weight
    int max_weight;           // Maximum edge weight
    const char* filename;     // Output filename
} GraphConfig;

// Bit vector for edge tracking to reduce memory usage
typedef struct {
    uint64_t* bits;
    int64_t vertices;
} EdgeBitmap;

// Initialize bit vector
EdgeBitmap* create_edge_bitmap(int64_t vertices) {
    EdgeBitmap* bitmap = malloc(sizeof(EdgeBitmap));
    if (!bitmap) {
        perror("Memory allocation failed for bitmap");
        exit(1);
    }
    
    // Calculate number of 64-bit words needed
    int64_t bit_words = (vertices * vertices + 63) / 64;
    bitmap->bits = calloc(bit_words, sizeof(uint64_t));
    
    if (!bitmap->bits) {
        free(bitmap);
        perror("Memory allocation failed for bitmap bits");
        exit(1);
    }
    
    bitmap->vertices = vertices;
    return bitmap;
}

// Check if an edge exists using bit manipulation
bool edge_exists(EdgeBitmap* bitmap, int64_t src, int64_t dest) {
    if (src == dest) return true;
    
    int64_t index = src * bitmap->vertices + dest;
    int64_t word_index = index / 64;
    int64_t bit_index = index % 64;
    
    return (bitmap->bits[word_index] & (1ULL << bit_index)) != 0;
}

// Mark an edge in the bit vector
void mark_edge(EdgeBitmap* bitmap, int64_t src, int64_t dest) {
    int64_t index = src * bitmap->vertices + dest;
    int64_t word_index = index / 64;
    int64_t bit_index = index % 64;
    
    bitmap->bits[word_index] |= (1ULL << bit_index);
}

// Free bitmap memory
void free_edge_bitmap(EdgeBitmap* bitmap) {
    if (bitmap) {
        free(bitmap->bits);
        free(bitmap);
    }
}

// Improved random range function
int64_t random_range(xorshift64_state* rng, int64_t min, int64_t max) {
    uint64_t range = max - min + 1;
    return min + (xorshift64(rng) % range);
}

void generate_large_graph(GraphConfig config) {
    // Open output file with larger buffer
    FILE* file = fopen(config.filename, "w");
    if (!file) {
        perror("Error opening output file");
        exit(1);
    }
    setvbuf(file, NULL, _IOFBF, 1024 * 1024);  // 1 MB buffer

    // Calculate max possible edges
    int64_t max_possible_edges = (config.vertex_count * (config.vertex_count - 1)) / 2;
    int64_t target_edge_count = (max_possible_edges * config.edge_density) / 100;

    // Create edge bitmap for tracking
    EdgeBitmap* edge_bitmap = create_edge_bitmap(config.vertex_count);

    // Initialize random number generator
    xorshift64_state rng = {.state = time(NULL)};

    // Write header
    fprintf(file, "%lld %lld\n", (long long)config.vertex_count, (long long)target_edge_count);

    // Generate edges
    int64_t edge_count = 0;
    while (edge_count < target_edge_count) {
        // Ensure different source and destination
        int64_t src = random_range(&rng, 0, config.vertex_count - 2);
        int64_t dest = random_range(&rng, src + 1, config.vertex_count - 1);

        // Check if edge already exists
        if (!edge_exists(edge_bitmap, src, dest)) {
            int weight = random_range(&rng, config.min_weight, config.max_weight);

            // Write edge in both directions for undirected graph
            fprintf(file, "%lld %lld %d\n", (long long)src, (long long)dest, weight);
            fprintf(file, "%lld %lld %d\n", (long long)dest, (long long)src, weight);

            // Mark edges to prevent duplicates
            mark_edge(edge_bitmap, src, dest);
            mark_edge(edge_bitmap, dest, src);

            edge_count++;
        }
    }

    // Clean up
    free_edge_bitmap(edge_bitmap);
    fclose(file);

    printf("Graph Generation Complete:\n");
    printf("- Vertices: %lld\n", (long long)config.vertex_count);
    printf("- Target Edges: %lld\n", (long long)target_edge_count);
    printf("- File: %s\n", config.filename);
}

int main(int argc, char* argv[]) {
    // Configurable graph generation parameters
    GraphConfig config = {
        .vertex_count = 50000,    // Increased to 100k
        .edge_density = 20,         // 25% of possible edges
        .min_weight = 1,            // Minimum edge weight
        .max_weight = 2000,         // Maximum edge weight
        .filename = "large_graph.txt"  // Output filename
    };

    // Optional command-line parameter overrides
    if (argc > 1) config.vertex_count = atoll(argv[1]);
    if (argc > 2) config.edge_density = atoi(argv[2]);
    if (argc > 3) config.filename = argv[3];

    // Generate graph
    printf("Generating large graph...\n");
    generate_large_graph(config);

    return 0;
}