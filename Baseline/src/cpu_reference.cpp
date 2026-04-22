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
#include <random>
#include <cassert>
#include <climits>

// Graph representation
struct Graph {
    int n;
    long m;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
};

Graph read_graph(const char *fname) {
    std::ifstream f(fname);
    if (!f.is_open()) { fprintf(stderr, "Cannot open %s\n", fname); exit(1); }

    std::string ext(fname);
    ext = ext.substr(ext.rfind('.') + 1);

    std::vector<std::pair<int,int>> edges;
    int n = 0;

    if (ext == "mtx") {
        std::string line;
        while (std::getline(f, line)) if (line[0] != '%') break;
        int m_raw, dummy;
        std::istringstream ss(line);
        ss >> n >> dummy >> m_raw;
        edges.reserve(m_raw * 2);
        for (int i = 0; i < m_raw; i++) {
            int u, v; f >> u >> v; u--; v--;
            if (u != v) { edges.push_back({u,v}); edges.push_back({v,u}); }
        }
    } else {
        int m_raw; f >> n >> m_raw;
        edges.reserve(m_raw * 2);
        for (int i = 0; i < m_raw; i++) {
            int u, v; f >> u >> v;
            if (u != v) { edges.push_back({u,v}); edges.push_back({v,u}); }
        }
    }

    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

    Graph g; g.n = n; g.m = edges.size();
    g.row_ptr.resize(n+1, 0);
    for (auto &e : edges) g.row_ptr[e.first+1]++;
    for (int i = 0; i < n; i++) g.row_ptr[i+1] += g.row_ptr[i];
    g.col_idx.resize(g.m);
    std::vector<int> cur(g.row_ptr.begin(), g.row_ptr.end());
    for (auto &e : edges) g.col_idx[cur[e.first]++] = e.second;

    printf("Graph: n=%d  m=%ld (directed)\n", g.n, g.m);
    return g;
}

// 1. Sequential greedy first fit coloring
std::vector<int> seq_first_fit(const Graph &g) {
    std::vector<int> color(g.n, -1);
    std::vector<int> forbidden(g.n, -1); // forbidden[c] = v means color c forbidden for curr vertex

    for (int v = 0; v < g.n; v++) {
        // Mark neighbor colors as forbidden
        for (int i = g.row_ptr[v]; i < g.row_ptr[v+1]; i++) {
            int u = g.col_idx[i];
            if (color[u] >= 0) forbidden[color[u]] = v;
        }
        // Find first available color
        int c = 0;
        while (forbidden[c] == v) c++;
        color[v] = c;
    }
    return color;
}

// 2. Simulate EB coloring
constexpr uint32_t NO_COLOR      = 0u;
constexpr uint32_t TENTATIVE_BIT = (1u << 31);
constexpr int      BITS           = 32;

inline bool is_tentative(uint32_t c) { return (c & TENTATIVE_BIT) != 0; }
inline uint32_t strip_tent(uint32_t c) { return c & ~TENTATIVE_BIT; }
inline uint32_t first_avail(uint32_t forbidden) {
    uint32_t avail = ~forbidden;
    return avail & (-avail);
}
inline int decode_color(uint32_t c, int cs) {
    if (c == 0) return -1;
    return cs * BITS + __builtin_ctz(c) + 1;
}

struct EBState {
    std::vector<uint32_t> color;
    std::vector<uint32_t> vforbid;
    std::vector<uint32_t> tvforbid;
    std::vector<int>      cs;
};

struct EdgePair { int u, v; };

std::vector<int> eb_coloring_cpu(const Graph &g, int &n_rounds_out) {
    int N = g.n;
    long M = g.m;

    EBState st;
    st.color.assign(N, NO_COLOR);
    st.vforbid.assign(N, 0u);
    st.tvforbid.assign(N, 0u);
    st.cs.assign(N, 0);

    // Build edge list
    std::vector<EdgePair> edges(M);
    for (int v = 0; v < N; v++)
        for (int i = g.row_ptr[v]; i < g.row_ptr[v+1]; i++)
            edges[i] = {v, g.col_idx[i]};

    // Vertex worklist: initially all vertices
    std::vector<int> wl(N);
    std::iota(wl.begin(), wl.end(), 0);

    int round = 0;
    while (!wl.empty() && round < 200) {
        // Reset forbidden for worklist vertices
        for (int v : wl) { st.vforbid[v] = 0u; st.tvforbid[v] = 0u; }

        // FORBIDCOLORS
        for (auto &e : edges) {
            int u = e.u, v = e.v;
            if (st.cs[u] != st.cs[v]) continue;
            bool uf = (st.color[u] != NO_COLOR) && !is_tentative(st.color[u]);
            bool vf = (st.color[v] != NO_COLOR) && !is_tentative(st.color[v]);
            if (uf && !vf) st.vforbid[v] |= st.color[u];
            if (vf && !uf) st.vforbid[u] |= st.color[v];
        }

        // TENTATIVECOLOR
        for (auto &e : edges) {
            int u = e.u, v = e.v;
            if (st.cs[u] != st.cs[v]) continue;
            bool uf = (st.color[u] != NO_COLOR) && !is_tentative(st.color[u]);
            bool vf = (st.color[v] != NO_COLOR) && !is_tentative(st.color[v]);
            bool ut = is_tentative(st.color[u]);
            bool vt = is_tentative(st.color[v]);

            if (!uf && !vf && !ut && !vt) {
                // neither colored: assign tentative to higher-index
                int hi = (u > v) ? u : v;
                int lo = (u > v) ? v : u;
                if (st.color[hi] == NO_COLOR) {
                    uint32_t allforbid = st.vforbid[hi] | st.tvforbid[hi];
                    uint32_t tc = first_avail(allforbid);
                    if (tc) { st.color[hi] = tc | TENTATIVE_BIT; st.tvforbid[lo] |= tc; }
                }
            } else if (ut && !vf && !vt) {
                st.tvforbid[v] |= strip_tent(st.color[u]);
            } else if (vt && !uf && !ut) {
                st.tvforbid[u] |= strip_tent(st.color[v]);
            } else if (ut && vt) {
                uint32_t cu = strip_tent(st.color[u]);
                uint32_t cv = strip_tent(st.color[v]);
                if (cu == cv) {
                    int loser = (u < v) ? u : v;
                    int winner = (u < v) ? v : u;
                    uint32_t allforbid = st.vforbid[loser] | st.tvforbid[loser];
                    uint32_t new_tc = first_avail(allforbid | cu);
                    if (new_tc) {
                        st.color[loser] = new_tc | TENTATIVE_BIT;
                        st.tvforbid[winner] |= new_tc;
                    }
                }
            }
        }

        // ASSIGNCOLORS
        for (int v : wl) {
            if (is_tentative(st.color[v])) {
                st.color[v] = strip_tent(st.color[v]);
                continue;
            }
            uint32_t allforbid = st.vforbid[v] | st.tvforbid[v];
            uint32_t tc = first_avail(allforbid);
            if (tc) {
                st.color[v] = tc;
            } else {
                // All 32 slots in this CS taken → advance to next CS
                // Advance only when all vforbid bits are set (not just tvforbid)
                if (st.vforbid[v] == ~0u) {
                    st.cs[v]++;
                    st.vforbid[v] = 0u;
                    st.tvforbid[v] = 0u;
                }
                st.color[v] = NO_COLOR;
            }
        }

        // DETECTCONFLICTS
        std::unordered_set<int> conflict_set;
        for (auto &e : edges) {
            int u = e.u, v = e.v;
            if (st.color[u] == NO_COLOR || st.color[v] == NO_COLOR) continue;
            if (is_tentative(st.color[u]) || is_tentative(st.color[v])) continue;
            if (st.cs[u] != st.cs[v]) continue;
            if (st.color[u] == st.color[v]) {
                int loser = (u < v) ? u : v;
                conflict_set.insert(loser);
            }
        }

        for (int v : conflict_set) st.color[v] = NO_COLOR;

        // Rebuild worklist: all uncolored vertices
        wl.clear();
        for (int v = 0; v < N; v++)
            if (st.color[v] == NO_COLOR) wl.push_back(v);

        printf("  Round %3d: uncolored=%6d  conflicts=%6d\n",
               round+1, (int)wl.size(), (int)conflict_set.size());
        round++;
        if (conflict_set.empty() && !wl.empty()) {
            for (int v : wl) {
                // Find highest color index used by any neighbor
                int max_nc = -1;
                for (int i = g.row_ptr[v]; i < g.row_ptr[v+1]; i++) {
                    int u = g.col_idx[i];
                    if (st.color[u] != NO_COLOR && !is_tentative(st.color[u]))
                        max_nc = std::max(max_nc, decode_color(st.color[u], st.cs[u]));
                }
                // Advance cs[v] to the band that sits above the highest neighbor color
                int needed_cs = (max_nc + 1) / BITS;
                if (needed_cs > st.cs[v]) {
                    st.cs[v]       = needed_cs;
                    st.vforbid[v]  = 0u;
                    st.tvforbid[v] = 0u;
                } else {
                    st.vforbid[v]  = 0u;
                    st.tvforbid[v] = 0u;
                    for (int i = g.row_ptr[v]; i < g.row_ptr[v+1]; i++) {
                        int u = g.col_idx[i];
                        if (st.color[u] == NO_COLOR || is_tentative(st.color[u])) continue;
                        int cu    = decode_color(st.color[u], st.cs[u]);
                        int cu_cs = cu / BITS;
                        if (cu_cs == st.cs[v])
                            st.vforbid[v] |= (1u << (cu % BITS));
                    }
                }
                st.color[v] = NO_COLOR;
            }
        }
    }
    n_rounds_out = round;

    // Decode colors
    std::vector<int> result(N);
    for (int v = 0; v < N; v++)
        result[v] = decode_color(st.color[v], st.cs[v]);
    return result;
}

// Verify coloring
bool verify(const Graph &g, const std::vector<int> &color) {
    bool ok = true;
    for (int v = 0; v < g.n; v++) {
        if (color[v] < 0) { printf("UNCOLORED: %d\n", v); ok = false; continue; }
        for (int i = g.row_ptr[v]; i < g.row_ptr[v+1]; i++) {
            int u = g.col_idx[i];
            if (color[u] == color[v]) {
                printf("CONFLICT: %d-%d both color %d\n", v, u, color[v]);
                ok = false;
            }
        }
    }
    return ok;
}

// Print color class distribution stats
void print_distribution(const std::vector<int> &color, int n,
                        const std::string &label) {
    int max_c = *std::max_element(color.begin(), color.end());
    std::vector<int> hist(max_c + 1, 0);
    for (int v = 0; v < n; v++) if (color[v] >= 0) hist[color[v]]++;

    int max_sz = *std::max_element(hist.begin(), hist.end());
    int min_sz = *std::min_element(hist.begin(), hist.end());
    double mean = (double)n / (max_c + 1);
    double sq = 0; for (int s : hist) sq += (s-mean)*(s-mean);
    double std_dev = std::sqrt(sq / (max_c+1));

    printf("\n[%s]\n", label.c_str());
    printf("  Colors  : %d\n", max_c + 1);
    printf("  Min size: %d  Max size: %d\n", min_sz, max_sz);
    printf("  Mean    : %.1f  StdDev: %.1f\n", mean, std_dev);
    printf("  Imbalance (max/mean): %.2fx\n", max_sz / mean);

    // Sort by size descending
    std::vector<std::pair<int,int>> sorted_h;
    for (int i = 0; i <= max_c; i++) sorted_h.push_back({hist[i], i});
    std::sort(sorted_h.rbegin(), sorted_h.rend());

    printf("  Top 10 color classes:\n");
    for (int i = 0; i < std::min(10, (int)sorted_h.size()); i++)
        printf("    Color %4d → %6d vertices\n", sorted_h[i].second, sorted_h[i].first);
    printf("  Bottom 10 color classes:\n");
    for (int i = std::max(0,(int)sorted_h.size()-10); i < (int)sorted_h.size(); i++)
        printf("    Color %4d → %6d vertices\n", sorted_h[i].second, sorted_h[i].first);

    // Save CSV
    FILE *f = fopen((label + "_dist.csv").c_str(), "w");
    if (f) {
        fprintf(f, "color_id,size\n");
        for (int i = 0; i <= max_c; i++) fprintf(f, "%d,%d\n", i, hist[i]);
        fclose(f);
        printf("  Saved: %s_dist.csv\n", label.c_str());
    }
}

// main
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <graph.mtx|graph.el>\n", argv[0]);
        return 1;
    }

    Graph g = read_graph(argv[1]);
    std::string base(argv[1]);

    // Sequential first-fit
    printf("\n--- Sequential First-Fit Greedy Coloring ---\n");
    auto t0 = std::chrono::high_resolution_clock::now();
    auto seq_color = seq_first_fit(g);
    auto t1 = std::chrono::high_resolution_clock::now();
    double seq_ms = std::chrono::duration<double,std::milli>(t1-t0).count();
    bool seq_ok = verify(g, seq_color);
    printf("Time: %.2f ms  Valid: %s\n", seq_ms, seq_ok ? "YES" : "NO");
    print_distribution(seq_color, g.n, "SEQ");

    // CPU EB simulation
    printf("\n--- CPU Simulation of Edge-Based EB Coloring ---\n");
    int n_rounds = 0;
    t0 = std::chrono::high_resolution_clock::now();
    auto eb_color = eb_coloring_cpu(g, n_rounds);
    t1 = std::chrono::high_resolution_clock::now();
    double eb_ms = std::chrono::duration<double,std::milli>(t1-t0).count();
    bool eb_ok = verify(g, eb_color);
    printf("Rounds: %d  Time: %.2f ms  Valid: %s\n", n_rounds, eb_ms, eb_ok ? "YES" : "NO");
    print_distribution(eb_color, g.n, "EB");

    // Comparison summary
    int seq_nc = *std::max_element(seq_color.begin(), seq_color.end()) + 1;
    int eb_nc  = *std::max_element(eb_color.begin(), eb_color.end()) + 1;
    printf("\n=== Comparison Summary ===\n");
    printf("  SEQ first-fit : %d colors  %.2f ms\n", seq_nc, seq_ms);
    printf("  EB simulation : %d colors  %.2f ms  %d rounds\n", eb_nc, eb_ms, n_rounds);
    printf("  EB/SEQ ratio  : %.2fx colors\n", (double)eb_nc / seq_nc);

    return 0;
}
