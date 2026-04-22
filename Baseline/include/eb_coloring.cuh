#pragma once
#include <cstdint>
#include <cstdio>
#include <cassert>
#include <climits>

// Constants
#define BITS_PER_CS   32         
#define NO_COLOR       0         
#define TENTATIVE_FLAG (1u << 31)
#define MAX_ROUNDS     200       

// CSR graph on device
struct GraphCSR {
    int n_vertices;
    long n_edges; // directed (each undirected edge stored twice)

    int *row_ptr;
    int *col_idx;
};

// Per-vertex coloring state
struct ColorState {
    uint32_t *color;
    uint32_t *vforbidden; // bitset of forbidden colors in current CS
    uint32_t *tvforbidden; // bitset of tentatively forbidden colors
    int *cs;
};

//  Worklist (conflict list)
struct Worklist {
    int *vertices;
    int *count;
    int capacity;
};

// Edge worklist
struct EdgeWorklist {
    int *edges;
    int *count;
    int capacity;
};

// Statistics collected each round
struct RoundStats {
    int round;
    int n_conflicted;
    int n_colored_this_round;
    float time_ms;
};

// Overall result
struct ColoringResult {
    int n_colors;
    int n_rounds;
    float total_time_ms;
    RoundStats round_stats[MAX_ROUNDS];

    int *class_sizes; // host array, size = n_colors+1
};

//  Error checking macro
#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)
