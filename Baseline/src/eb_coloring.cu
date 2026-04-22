#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <string>
#include <fstream>
#include <sstream>
#include <unordered_set>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include "eb_coloring.cuh"

// Forward declare kernels from eb_kernels.cu
extern __global__ void k_forbid_colors_full(
    const int*, const int*, int, const uint32_t*, const int*, uint32_t*);
extern __global__ void k_tentative_color(
    const int*, const int*, int, uint32_t*, const int*, const uint32_t*, uint32_t*);
extern __global__ void k_assign_colors(
    const int*, int, uint32_t*, int*, const uint32_t*, const uint32_t*);
extern __global__ void k_detect_conflicts_atomic(
    const int*, const int*, int, uint32_t*, const int*, int*, int*);
extern __global__ void k_mark_inactive_edges(
    const int*, const int*, int, const uint32_t*, const int*, int*);
extern __global__ void k_compact_edges(
    const int*, const int*, int, const int*, const int*, int*, int*);
extern __global__ void k_reset_forbidden(const int*, int, uint32_t*, uint32_t*);
extern __global__ void k_build_edge_arrays(const int*, const int*, int, int*, int*);
extern __global__ void k_rebuild_vertex_worklist(int, const uint32_t*, int*, int*);

// Graph Input Output
struct HostGraph {
    int n;
    long m;   // number of directed edges (each undirected edge counted twice)
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
};

// Read Matrix Market format (symmetric)
HostGraph read_mtx(const char *filename) {
    std::ifstream f(filename);
    if (!f.is_open()) {
        fprintf(stderr, "Cannot open %s\n", filename);
        exit(1);
    }

    std::string line;
    // Skip comments
    while (std::getline(f, line)) {
        if (line[0] != '%') break;
    }

    int n, m_raw;
    {
        std::istringstream ss(line);
        int dummy;
        ss >> n >> dummy >> m_raw;
    }

    std::vector<std::pair<int,int>> edges;
    edges.reserve(m_raw * 2);

    for (int i = 0; i < m_raw; i++) {
        int u, v;
        f >> u >> v;
        u--; v--;   // 1 indexed to 0 indexed
        if (u == v) continue;   // skip self loops
        edges.push_back({u, v});
        if (u != v) edges.push_back({v, u}); // symmetrize
    }

    // Sort and deduplicate
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

    HostGraph g;
    g.n = n;
    g.m = edges.size();
    g.row_ptr.resize(n + 1, 0);

    for (auto &e : edges) g.row_ptr[e.first + 1]++;
    for (int i = 0; i < n; i++) g.row_ptr[i+1] += g.row_ptr[i];

    g.col_idx.resize(g.m);
    std::vector<int> cur(g.row_ptr.begin(), g.row_ptr.end());
    for (auto &e : edges) {
        g.col_idx[cur[e.first]++] = e.second;
    }

    printf("Graph: n=%d  m=%ld (directed)\n", g.n, g.m);
    return g;
}

// Read edge list format: first line "N M", then "u v" lines (0 indexed)
HostGraph read_edgelist(const char *filename) {
    std::ifstream f(filename);
    if (!f.is_open()) { fprintf(stderr, "Cannot open %s\n", filename); exit(1); }

    int n, m_raw;
    f >> n >> m_raw;

    std::vector<std::pair<int,int>> edges;
    edges.reserve(m_raw * 2);
    for (int i = 0; i < m_raw; i++) {
        int u, v; f >> u >> v;
        if (u == v) continue;
        edges.push_back({u, v});
        edges.push_back({v, u});
    }
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

    HostGraph g;
    g.n = n; g.m = edges.size();
    g.row_ptr.resize(n + 1, 0);
    for (auto &e : edges) g.row_ptr[e.first + 1]++;
    for (int i = 0; i < n; i++) g.row_ptr[i+1] += g.row_ptr[i];
    g.col_idx.resize(g.m);
    std::vector<int> cur(g.row_ptr.begin(), g.row_ptr.end());
    for (auto &e : edges) g.col_idx[cur[e.first]++] = e.second;

    printf("Graph: n=%d  m=%ld (directed)\n", g.n, g.m);
    return g;
}

// Decode final color: cs * 32 + bit_position
inline int decode_color(uint32_t c, int cs_val) {
    if (c == 0) return -1;  // uncolored
    int bit = __builtin_ctz(c) + 1;  // +1 for 1 indexed
    return cs_val * BITS_PER_CS + bit;
}

// Verify coloring on host
bool verify_coloring(
        const HostGraph &g,
        const std::vector<uint32_t> &color_h,
        const std::vector<int> &cs_h
) {
    bool ok = true;
    int uncolored = 0;
    for (int v = 0; v < g.n; v++) {
        if (color_h[v] == NO_COLOR) { uncolored++; continue; }
        int cv = decode_color(color_h[v], cs_h[v]);
        for (int i = g.row_ptr[v]; i < g.row_ptr[v+1]; i++) {
            int u = g.col_idx[i];
            if (color_h[u] == NO_COLOR) continue;
            int cu = decode_color(color_h[u], cs_h[u]);
            if (cv == cu) {
                fprintf(stderr, "CONFLICT: v=%d u=%d both color %d\n", v, u, cv);
                ok = false;
                if (!ok) return false;
            }
        }
    }
    if (uncolored > 0) {
        fprintf(stderr, "WARNING: %d vertices still uncolored\n", uncolored);
        ok = false;
    }
    return ok;
}

// Color class size distribution
std::vector<int> compute_class_sizes(
        const std::vector<uint32_t> &color_h,
        const std::vector<int> &cs_h,
        int n_vertices,
        int &n_colors_out
) {
    std::vector<int> counts;
    for (int v = 0; v < n_vertices; v++) {
        int c = decode_color(color_h[v], cs_h[v]);
        if (c < 0) continue;
        if (c >= (int)counts.size()) counts.resize(c + 1, 0);
        counts[c]++;
    }
    n_colors_out = (int)counts.size();
    return counts;
}

// Print statistics
void print_stats(
        const std::string &graph_name,
        const ColoringResult &res,
        const std::vector<int> &class_sizes,
        bool pps_mode
) {
    printf("\n=== EB Coloring Results ===\n");
    printf("Graph          : %s\n", graph_name.c_str());
    printf("Worklist mode  : %s\n", pps_mode ? "PPS" : "ATOMIC");
    printf("Total rounds   : %d\n", res.n_rounds);
    printf("Total colors   : %d\n", res.n_colors);
    printf("Total time     : %.2f ms\n", res.total_time_ms);
    printf("\nPer round breakdown:\n");
    printf("  %-6s  %-12s  %-12s  %-10s\n",
           "Round", "Conflicted", "Colored", "Time(ms)");
    for (int r = 0; r < res.n_rounds; r++) {
        printf("  %-6d  %-12d  %-12d  %-10.2f\n",
               res.round_stats[r].round,
               res.round_stats[r].n_conflicted,
               res.round_stats[r].n_colored_this_round,
               res.round_stats[r].time_ms);
    }

    // Color class distribution analysis
    int n_colors = (int)class_sizes.size();
    if (n_colors > 0) {
        int max_size = *std::max_element(class_sizes.begin(), class_sizes.end());
        int min_size = *std::min_element(class_sizes.begin(), class_sizes.end());
        double mean = std::accumulate(class_sizes.begin(), class_sizes.end(), 0.0) / n_colors;
        double sq_sum = 0;
        for (int s : class_sizes) sq_sum += (s - mean) * (s - mean);
        double std_dev = std::sqrt(sq_sum / n_colors);

        printf("\nColor class distribution:\n");
        printf("  Min size   : %d\n", min_size);
        printf("  Max size   : %d\n", max_size);
        printf("  Mean size  : %.1f\n", mean);
        printf("  Std dev    : %.1f\n", std_dev);
        printf("  Imbalance  : %.2fx  (max/mean)\n", max_size / mean);

        // Print top 10 and bottom 10 class sizes to show first fit skew
        printf("\n  Top 10 largest color classes:\n");
        std::vector<std::pair<int,int>> sorted_classes;
        for (int i = 0; i < n_colors; i++)
            sorted_classes.push_back({class_sizes[i], i});
        std::sort(sorted_classes.rbegin(), sorted_classes.rend());
        for (int i = 0; i < std::min(10, (int)sorted_classes.size()); i++)
            printf("    Color %4d : %d vertices\n",
                   sorted_classes[i].second, sorted_classes[i].first);

        printf("  Bottom 10 smallest color classes:\n");
        for (int i = std::max(0, (int)sorted_classes.size()-10);
             i < (int)sorted_classes.size(); i++)
            printf("    Color %4d : %d vertices\n",
                   sorted_classes[i].second, sorted_classes[i].first);
    }
}

// Save color class sizes to CSV
void save_distribution_csv(
        const std::string &outfile,
        const std::vector<int> &class_sizes
) {
    FILE *f = fopen(outfile.c_str(), "w");
    if (!f) { fprintf(stderr, "Cannot write %s\n", outfile.c_str()); return; }
    fprintf(f, "color_id,size\n");
    for (int i = 0; i < (int)class_sizes.size(); i++)
        fprintf(f, "%d,%d\n", i, class_sizes[i]);
    fclose(f);
    printf("Color distribution saved to: %s\n", outfile.c_str());
}

// MAIN EB coloring function
ColoringResult run_eb_coloring(
        const HostGraph &g,
        bool use_pps   // true = PPS worklist, false = ATOMIC
) {
    int N = g.n;
    long M = g.m;
    int BLOCK = 256;

    ColoringResult result{};

    // Allocate device graph
    int  *d_row_ptr, *d_col_idx;
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (N+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, M*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_row_ptr, g.row_ptr.data(), (N+1)*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, g.col_idx.data(), M*sizeof(int), cudaMemcpyHostToDevice));

    // Build flat edge src/dst arrays
    int *d_edge_src, *d_edge_dst;
    CUDA_CHECK(cudaMalloc(&d_edge_src, M*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_edge_dst, M*sizeof(int)));
    {
        int blocks = (N + BLOCK - 1) / BLOCK;
        k_build_edge_arrays<<<blocks, BLOCK>>>(d_row_ptr, d_col_idx, N, d_edge_src, d_edge_dst);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Allocate coloring state
    uint32_t *d_color, *d_vforbidden, *d_tvforbidden;
    int      *d_cs;
    CUDA_CHECK(cudaMalloc(&d_color,       N*sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_vforbidden,  N*sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_tvforbidden, N*sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_cs,          N*sizeof(int)));
    CUDA_CHECK(cudaMemset(d_color,       0, N*sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_vforbidden,  0, N*sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_tvforbidden, 0, N*sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_cs,          0, N*sizeof(int)));

    // Vertex worklist (conflict list)
    int *d_wl_cur, *d_wl_next, *d_wl_count;
    CUDA_CHECK(cudaMalloc(&d_wl_cur,   N*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_wl_next,  N*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_wl_count, sizeof(int)));

    // Initial worklist = all vertices
    {
        std::vector<int> init(N);
        std::iota(init.begin(), init.end(), 0);
        CUDA_CHECK(cudaMemcpy(d_wl_cur, init.data(), N*sizeof(int), cudaMemcpyHostToDevice));
    }
    int n_wl_cur = N;

    // Active edge arrays (compacted each round)
    int *d_esrc_cur, *d_edst_cur, *d_esrc_next, *d_edst_next;
    int *d_edge_count;
    CUDA_CHECK(cudaMalloc(&d_esrc_cur,   M*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_edst_cur,   M*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_esrc_next,  M*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_edst_next,  M*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_edge_count, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_esrc_cur, d_edge_src, M*sizeof(int), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_edst_cur, d_edge_dst, M*sizeof(int), cudaMemcpyDeviceToDevice));
    long n_active_edges = M;

    // Conflict detection output
    int *d_conflict_wl, *d_conflict_count;
    CUDA_CHECK(cudaMalloc(&d_conflict_wl,    N*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_conflict_count, sizeof(int)));

    // Thrust device vectors for PPS
    thrust::device_vector<int> d_keep_flag, d_prefix;

    // Timing
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    float total_ms = 0.0f;

    int round = 0;
    while (n_wl_cur > 0 && round < MAX_ROUNDS) {
        float round_ms = 0.0f;
        CUDA_CHECK(cudaEventRecord(ev_start));

        int vblocks = (n_wl_cur + BLOCK - 1) / BLOCK;
        int eblocks = (n_active_edges + BLOCK - 1) / BLOCK;

        // Reset forbidden arrays for vertices in worklist
        k_reset_forbidden<<<vblocks, BLOCK>>>(d_wl_cur, n_wl_cur, d_vforbidden, d_tvforbidden);

        // FORBIDCOLORS
        if (eblocks > 0)
            k_forbid_colors_full<<<eblocks, BLOCK>>>(
                d_esrc_cur, d_edst_cur, (int)n_active_edges,
                d_color, d_cs, d_vforbidden);

        // TENTATIVECOLOR
        if (eblocks > 0)
            k_tentative_color<<<eblocks, BLOCK>>>(
                d_esrc_cur, d_edst_cur, (int)n_active_edges,
                d_color, d_cs, d_vforbidden, d_tvforbidden);

        // ASSIGNCOLORS
        k_assign_colors<<<vblocks, BLOCK>>>(
            d_wl_cur, n_wl_cur,
            d_color, d_cs, d_vforbidden, d_tvforbidden);

        // DETECTCONFLICTS
        CUDA_CHECK(cudaMemset(d_conflict_count, 0, sizeof(int)));
        if (eblocks > 0)
            k_detect_conflicts_atomic<<<eblocks, BLOCK>>>(
                d_esrc_cur, d_edst_cur, (int)n_active_edges,
                d_color, d_cs,
                d_conflict_wl, d_conflict_count);

        // CREATENEWEDGELIST (prune edges)
        if (n_active_edges > 0) {
            if (use_pps) {
                // PPS variant
                d_keep_flag.resize(n_active_edges);
                d_prefix.resize(n_active_edges);
                k_mark_inactive_edges<<<eblocks, BLOCK>>>(
                    d_esrc_cur, d_edst_cur, (int)n_active_edges,
                    d_color, d_cs, thrust::raw_pointer_cast(d_keep_flag.data()));
                thrust::exclusive_scan(thrust::device,
                    d_keep_flag.begin(), d_keep_flag.end(), d_prefix.begin());
                // Total kept = last keep_flag + last prefix
                int last_flag, last_prefix;
                CUDA_CHECK(cudaMemcpy(&last_flag,   thrust::raw_pointer_cast(d_keep_flag.data()) + n_active_edges - 1,
                                      sizeof(int), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(&last_prefix, thrust::raw_pointer_cast(d_prefix.data()) + n_active_edges - 1,
                                      sizeof(int), cudaMemcpyDeviceToHost));
                long n_next = last_flag + last_prefix;
                if (n_next < n_active_edges && n_next > 0) {
                    k_compact_edges<<<eblocks, BLOCK>>>(
                        d_esrc_cur, d_edst_cur, (int)n_active_edges,
                        thrust::raw_pointer_cast(d_keep_flag.data()),
                        thrust::raw_pointer_cast(d_prefix.data()),
                        d_esrc_next, d_edst_next);
                    std::swap(d_esrc_cur, d_esrc_next);
                    std::swap(d_edst_cur, d_edst_next);
                    n_active_edges = n_next;
                }
            }
        }

        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        CUDA_CHECK(cudaEventElapsedTime(&round_ms, ev_start, ev_stop));
        total_ms += round_ms;

        // Read back conflict count
        int n_conflicts = 0;
        CUDA_CHECK(cudaMemcpy(&n_conflicts, d_conflict_count, sizeof(int), cudaMemcpyDeviceToHost));

        // Swap worklists
        int prev_wl = n_wl_cur;
        std::swap(d_wl_cur, d_conflict_wl);
        n_wl_cur = n_conflicts;

        result.round_stats[round] = {round+1, n_conflicts, prev_wl - n_conflicts, round_ms};
        round++;

        if (round <= 5 || round % 10 == 0)
            printf("  Round %3d: conflicts=%6d  active_edges=%8ld  time=%.2fms\n",
                   round, n_conflicts, n_active_edges, round_ms);
    }

    result.n_rounds     = round;
    result.total_time_ms = total_ms;

    // Copy results back to host
    std::vector<uint32_t> h_color(N);
    std::vector<int>      h_cs(N);
    CUDA_CHECK(cudaMemcpy(h_color.data(), d_color, N*sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_cs.data(),   d_cs,     N*sizeof(int),      cudaMemcpyDeviceToHost));

    // Compute n_colors and class sizes
    int n_colors = 0;
    std::vector<int> class_sizes = compute_class_sizes(h_color, h_cs, N, n_colors);
    result.n_colors  = n_colors;
    result.class_sizes = new int[n_colors];
    memcpy(result.class_sizes, class_sizes.data(), n_colors * sizeof(int));

    // Verify
    printf("\nVerifying coloring correctness...\n");
    bool valid = verify_coloring(g, h_color, h_cs);
    printf("Coloring %s\n", valid ? "VALID ✓" : "INVALID ✗");

    // Free device memory
    cudaFree(d_row_ptr); cudaFree(d_col_idx);
    cudaFree(d_edge_src); cudaFree(d_edge_dst);
    cudaFree(d_color); cudaFree(d_vforbidden); cudaFree(d_tvforbidden); cudaFree(d_cs);
    cudaFree(d_wl_cur); cudaFree(d_wl_next); cudaFree(d_wl_count);
    cudaFree(d_esrc_cur); cudaFree(d_edst_cur);
    cudaFree(d_esrc_next); cudaFree(d_edst_next); cudaFree(d_edge_count);
    cudaFree(d_conflict_wl); cudaFree(d_conflict_count);
    cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop);

    return result;
}

// main
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <graph.mtx|graph.el> [--pps]\n", argv[0]);
        return 1;
    }

    bool use_pps = false;
    for (int i = 2; i < argc; i++)
        if (strcmp(argv[i], "--pps") == 0) use_pps = true;

    std::string fname(argv[1]);
    HostGraph g;
    if (fname.size() >= 4 && fname.substr(fname.size()-4) == ".mtx")
        g = read_mtx(argv[1]);
    else
        g = read_edgelist(argv[1]);

    printf("Running EB coloring (%s mode)...\n", use_pps ? "PPS" : "ATOMIC");

    ColoringResult res = run_eb_coloring(g, use_pps);

    std::vector<int> class_sizes(res.class_sizes, res.class_sizes + res.n_colors);
    print_stats(fname, res, class_sizes, use_pps);

    // Save distribution CSV for plotting
    std::string csv_out = fname + (use_pps ? "_pps" : "_atomic") + "_distribution.csv";
    save_distribution_csv(csv_out, class_sizes);

    delete[] res.class_sizes;
    return 0;
}
