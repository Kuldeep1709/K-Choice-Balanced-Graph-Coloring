#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <random>
#include <set>
#include <utility>
#include <cmath>

using EdgeList = std::vector<std::pair<int,int>>;

void deduplicate(EdgeList &edges, int n) {
    std::set<std::pair<int,int>> s(edges.begin(), edges.end());
    edges.assign(s.begin(), s.end());
}

void write_el(const EdgeList &edges, int n) {
    printf("%d %d\n", n, (int)edges.size());
    for (auto &e : edges) printf("%d %d\n", e.first, e.second);
}

// Ring
void gen_ring(int n) {
    EdgeList edges;
    for (int i = 0; i < n; i++) edges.push_back({i, (i+1)%n});
    write_el(edges, n);
}

// 2D Grid
void gen_grid(int rows, int cols) {
    int n = rows * cols;
    EdgeList edges;
    auto idx = [&](int r, int c){ return r * cols + c; };
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            if (c+1 < cols) edges.push_back({idx(r,c), idx(r,c+1)});
            if (r+1 < rows) edges.push_back({idx(r,c), idx(r+1,c)});
        }
    }
    write_el(edges, n);
}

// Random Erdos Renyi
void gen_random(int n, int m, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, n-1);
    std::set<std::pair<int,int>> s;
    int tries = 0;
    while ((int)s.size() < m && tries < m*10) {
        int u = dist(rng), v = dist(rng);
        if (u != v) {
            if (u > v) std::swap(u, v);
            s.insert({u, v});
        }
        tries++;
    }
    EdgeList edges(s.begin(), s.end());
    write_el(edges, n);
}

// R MAT (Graph500 RMAT: a=0.57, b=c=0.19, d=0.05)
void rmat_edge(int &u, int &v, int n, std::mt19937 &rng) {
    double a = 0.57, b = 0.19, c = 0.19, d = 0.05;
    int cur_n = n;
    int x = 0, y = 0;
    while (cur_n > 1) {
        cur_n /= 2;
        double r = std::uniform_real_distribution<double>(0,1)(rng);
        if (r < a) continue;
        else if (r < a+b) y += cur_n;
        else if (r < a+b+c) x += cur_n;
        else x += cur_n; y += cur_n;
    }
    u = x; v = y;
}

void gen_rmat(int n, int m, unsigned seed = 42) {
    // Round n up to next power of 2
    int p2 = 1;
    while (p2 < n) p2 <<= 1;

    std::mt19937 rng(seed);
    std::set<std::pair<int,int>> s;
    int tries = 0;
    while ((int)s.size() < m && tries < m*5) {
        int u, v;
        rmat_edge(u, v, p2, rng);
        if (u >= n || v >= n || u == v) { tries++; continue; }
        if (u > v) std::swap(u, v);
        s.insert({u, v});
        tries++;
    }
    EdgeList edges(s.begin(), s.end());
    fprintf(stderr, "RMAT: requested %d, got %d edges\n", m, (int)edges.size());
    write_el(edges, n);
}

// Clique K_n
void gen_clique(int n) {
    EdgeList edges;
    for (int u = 0; u < n; u++)
        for (int v = u+1; v < n; v++)
            edges.push_back({u,v});
    write_el(edges, n);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
            "Usage:\n"
            "  %s ring <N>\n"
            "  %s grid <rows> <cols>\n"
            "  %s random <N> <M> [seed]\n"
            "  %s rmat <N> <M> [seed]\n"
            "  %s clique <N>\n", argv[0],argv[0],argv[0],argv[0],argv[0]);
        return 1;
    }

    if (strcmp(argv[1], "ring") == 0 && argc >= 3)
        gen_ring(atoi(argv[2]));
    else if (strcmp(argv[1], "grid") == 0 && argc >= 4)
        gen_grid(atoi(argv[2]), atoi(argv[3]));
    else if (strcmp(argv[1], "random") == 0 && argc >= 4)
        gen_random(atoi(argv[2]), atoi(argv[3]), argc >= 5 ? atoi(argv[4]) : 42);
    else if (strcmp(argv[1], "rmat") == 0 && argc >= 4)
        gen_rmat(atoi(argv[2]), atoi(argv[3]), argc >= 5 ? atoi(argv[4]) : 42);
    else if (strcmp(argv[1], "clique") == 0 && argc >= 3)
        gen_clique(atoi(argv[2]));
    else {
        fprintf(stderr, "Unknown graph type or wrong arguments.\n");
        return 1;
    }
    return 0;
}
