#include "eb_coloring.cuh"

// Helpers
__device__ __forceinline__
uint32_t first_available_bit(uint32_t forbidden) {
    // Returns a uint32 with exactly the lowest 0-bit of forbidden set.
    uint32_t avail = ~forbidden;
    return avail & (-avail);
}

__device__ __forceinline__
bool is_tentative(uint32_t c) {
    return (c & TENTATIVE_FLAG) != 0;
}

__device__ __forceinline__
uint32_t strip_tentative(uint32_t c) {
    return c & ~TENTATIVE_FLAG;
}

// Kernel: FORBIDCOLORS
__global__
void k_forbid_colors(
        const int *__restrict__ edge_list,  // active edge indices
        int n_active_edges,
        const int *__restrict__ col_idx, // graph adjacency
        const uint32_t *__restrict__ color,
        const int *__restrict__ cs,
        uint32_t *vforbidden    // output: atomically updated
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_active_edges) return;

    int eidx = edge_list[tid];
    (void)eidx; (void)col_idx; (void)color; (void)cs; (void)vforbidden;
}

__global__
void k_forbid_colors_full(
        const int *__restrict__ active_edges_src,  // source vertex of edge
        const int *__restrict__ active_edges_dst,  // dest vertex of edge
        int n_active_edges,
        const uint32_t *__restrict__ color,
        const int *__restrict__ cs,
        uint32_t *vforbidden
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_active_edges) return;

    int u = active_edges_src[tid];
    int v = active_edges_dst[tid];

    bool u_colored = (color[u] != NO_COLOR) && !is_tentative(color[u]);
    bool v_colored = (color[v] != NO_COLOR) && !is_tentative(color[v]);

    // Process only when CS values match (same color band)
    if (cs[u] != cs[v]) return;

    if (u_colored && !v_colored) {
        atomicOr(&vforbidden[v], color[u]);
    }
    if (v_colored && !u_colored) {
        atomicOr(&vforbidden[u], color[v]);
    }
}

// Kernel: TENTATIVECOLOR
__global__
void k_tentative_color(
        const int *__restrict__ active_edges_src,
        const int *__restrict__ active_edges_dst,
        int n_active_edges,
        uint32_t *color,        // read/write (tentative writes)
        const int *__restrict__ cs,
        const uint32_t *__restrict__ vforbidden,
        uint32_t *tvforbidden   // read/write
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_active_edges) return;

    int u = active_edges_src[tid];
    int v = active_edges_dst[tid];

    if (cs[u] != cs[v]) return;

    bool u_colored   = (color[u] != NO_COLOR) && !is_tentative(color[u]);
    bool v_colored   = (color[v] != NO_COLOR) && !is_tentative(color[v]);
    bool u_tentative = is_tentative(color[u]);
    bool v_tentative = is_tentative(color[v]);
    bool u_any       = u_colored || u_tentative;
    bool v_any       = v_colored || v_tentative;

    // Case 1: neither colored  give tentative color to higher index vertex
    if (!u_any && !v_any) {
        if (u > v) {
            uint32_t allforbid = vforbidden[u] | tvforbidden[u];
            uint32_t tc = first_available_bit(allforbid);
            if (tc != 0) {
                // Try to set tentative color with CAS
                uint32_t expected = NO_COLOR;
                atomicCAS(&color[u], expected, tc | TENTATIVE_FLAG);
                // Tell v about u's tentative color
                if (tc != 0)
                    atomicOr(&tvforbidden[v], tc);
            }
        }
        return;
    }

    // Case 2: one is tentatively colored, other not
    if (u_tentative && !v_any) {
        uint32_t tc_u = strip_tentative(color[u]);
        atomicOr(&tvforbidden[v], tc_u);
    }
    if (v_tentative && !u_any) {
        uint32_t tc_v = strip_tentative(color[v]);
        atomicOr(&tvforbidden[u], tc_v);
    }

    // Case 3: both tentatively colored and conflict
    if (u_tentative && v_tentative) {
        uint32_t tc_u = strip_tentative(color[u]);
        uint32_t tc_v = strip_tentative(color[v]);
        if (tc_u == tc_v) {
            // Recolor the lower index vertex
            int loser = (u < v) ? u : v;
            int winner = (u < v) ? v : u;
            uint32_t allforbid = vforbidden[loser] | tvforbidden[loser];
            uint32_t new_tc = first_available_bit(allforbid | tc_u);
            if (new_tc != 0) {
                atomicExch(&color[loser], new_tc | TENTATIVE_FLAG);
                atomicOr(&tvforbidden[winner], new_tc);
            }
        }
    }
}

// Kernel: ASSIGNCOLORS with tentative
__global__
void k_assign_colors(
        const int *__restrict__ worklist,   // uncolored vertex indices
        int n_worklist,
        uint32_t *color,
        int *cs,
        const uint32_t *__restrict__ vforbidden,
        const uint32_t *__restrict__ tvforbidden
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_worklist) return;

    int v = worklist[tid];

    // If already tentatively colored confirm it
    if (is_tentative(color[v])) {
        color[v] = strip_tentative(color[v]);
        return;
    }

    uint32_t allforbid = vforbidden[v] | tvforbidden[v];

    uint32_t tc = first_available_bit(allforbid);
    if (tc != 0) {
        color[v] = tc;   // assign final color (within current CS)
    } else {
        if (~vforbidden[v] == 0u) {
            cs[v]++;
        }
        color[v] = NO_COLOR;
    }
}

// Kernel: DETECTCONFLICTS : It Uses ATOMIC worklist update variant.
__global__
void k_detect_conflicts_atomic(
        const int *__restrict__ active_edges_src,
        const int *__restrict__ active_edges_dst,
        int n_active_edges,
        uint32_t color,
        const int __restrict__ cs,
        int new_worklist,
        int new_count          // atomic counter
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_active_edges) return;

    int u = active_edges_src[tid];
    int v = active_edges_dst[tid];

    uint32_t cu = color[u];
    uint32_t cv = color[v];

    // Both must be finally colored
    if (cu == NO_COLOR || cv == NO_COLOR) return;
    if (is_tentative(cu) || is_tentative(cv)) return;
    if (cs[u] != cs[v]) return;   // different bands -> no conflict in same band
    // Actual same color check
    if (cu == cv) {
        // Lower index vertex loses its color
        int loser = (u < v) ? u : v;
        uint32_t expected = cu;
        if (atomicCAS(&color[loser], expected, NO_COLOR) == expected) {
            // Add loser to new conflict worklist
            int pos = atomicAdd(new_count, 1);
            new_worklist[pos] = loser;
        }
    }
}

// Kernel: CREATENEWEDGELIST
__global__
void k_mark_inactive_edges(
        const int    *__restrict__ active_edges_src,
        const int    *__restrict__ active_edges_dst,
        int          n_active_edges,
        const uint32_t *__restrict__ color,
        const int    *__restrict__ cs,
        int          *keep_flag   // 1 = keep, 0 = prune
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_active_edges) return;

    int u = active_edges_src[tid];
    int v = active_edges_dst[tid];

    bool u_final = (color[u] != NO_COLOR) && !is_tentative(color[u]);
    bool v_final = (color[v] != NO_COLOR) && !is_tentative(color[v]);

    // Rule 1: both colored
    if (u_final && v_final) { keep_flag[tid] = 0; return; }

    // Rule 2: one colored, already processed by FORBIDCOLORS (same CS)
    if ((u_final && !v_final && cs[u] == cs[v]) ||
        (v_final && !u_final && cs[u] == cs[v])) {
        keep_flag[tid] = 0; return;
    }

    // Rule 3: colored vertex has lower CS than uncolored
    if (u_final && !v_final && cs[u] < cs[v]) { keep_flag[tid] = 0; return; }
    if (v_final && !u_final && cs[v] < cs[u]) { keep_flag[tid] = 0; return; }

    keep_flag[tid] = 1;
}

// Kernel: PPS-based compact (Parallel Prefix Sum worklist)
__global__
void k_compact_edges(
        const int *__restrict__ src_in,
        const int *__restrict__ dst_in,
        int n_in,
        const int *__restrict__ keep_flag,
        const int *__restrict__ prefix_sum,  // exclusive scan of keep_flag
        int *src_out,
        int *dst_out
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_in) return;
    if (keep_flag[tid]) {
        int out_pos = prefix_sum[tid];
        src_out[out_pos] = src_in[tid];
        dst_out[out_pos] = dst_in[tid];
    }
}

// Kernel: Reset per-vertex forbidden arrays between rounds
__global__
void k_reset_forbidden(
        const int *__restrict__ worklist,
        int n_worklist,
        uint32_t *vforbidden,
        uint32_t *tvforbidden
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_worklist) return;
    int v = worklist[tid];
    vforbidden[v]  = 0u;
    tvforbidden[v] = 0u;
}

// Kernel: Build initial edge arrays (src/dst) from CSR
__global__
void k_build_edge_arrays(
        const int *__restrict__ row_ptr,
        const int *__restrict__ col_idx,
        int        n_vertices,
        int       *edge_src,
        int       *edge_dst
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n_vertices) return;
    for (int i = row_ptr[v]; i < row_ptr[v+1]; i++) {
        edge_src[i] = v;
        edge_dst[i] = col_idx[i];
    }
}

// Kernel: Count uncolored vertices (for convergence check)
__global__
void k_count_uncolored(
        const int *__restrict__ worklist,
        int n_worklist,
        const uint32_t *__restrict__ color,
        int *uncolored_count
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_worklist) return;
    int v = worklist[tid];
    if (color[v] == NO_COLOR) {
        atomicAdd(uncolored_count, 1);
    }
}

// Kernel: Collect final colored vertices into worklist
__global__
void k_rebuild_vertex_worklist(
        int n_vertices,
        const uint32_t *__restrict__ color,
        int *worklist,
        int *count
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n_vertices) return;
    if (color[v] == NO_COLOR) {
        int pos = atomicAdd(count, 1);
        worklist[pos] = v;
    }
}
