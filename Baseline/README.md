# Edge-Based Graph Coloring — Baseline Implementation

**Paper:** Deveci, Boman, Devine, Rajamanickam — *Parallel Graph Coloring for Manycore Architectures*, IPDPS 2016

This is the **Step 1 baseline** for the DS 295 project: *Power of Two Choices in Edge-Based Graph Coloring*.

---

## Directory Layout

```
eb_coloring/
├── include/
│   └── eb_coloring.cuh         # Shared types, constants, macros
├── src/
│   ├── eb_kernels.cu            # All CUDA kernels (FORBIDCOLORS, TENTATIVECOLOR, etc.)
│   ├── eb_coloring.cu           # Host orchestration, I/O, main loop
│   ├── cpu_reference.cpp        # CPU-only: sequential first-fit + EB simulation
│   └── gen_graphs.cpp           # Synthetic graph generator
├── scripts/
│   └── plot_distribution.py     # Python: plot color class imbalance
├── graphs/                      # Put .mtx / .el files here
└── Makefile
```

---

## Build

### CPU only (no CUDA — runs on any machine)
```bash
make cpu gen
```

### GPU (requires CUDA ≥ 11, adjust arch in Makefile)
```bash
make gpu
# Default arch=sm_80 (A100). Change to sm_70 for V100, sm_86 for RTX3090, etc.
```

---

## Run

### 1. Generate synthetic graphs
```bash
./bin/gen_graphs ring   1000          > graphs/ring_1k.el
./bin/gen_graphs grid   100 100       > graphs/grid_10k.el
./bin/gen_graphs random 10000 50000   > graphs/random_10k.el
./bin/gen_graphs rmat   100000 800000 > graphs/rmat_100k.el
```

### 2. CPU reference (correctness + imbalance measurement)
```bash
./bin/cpu_ref graphs/rmat_100k.el
```
Outputs:
- Rounds, colors, timing for both sequential first-fit and EB simulation
- Color class size statistics: min/max/mean/std/imbalance ratio
- Two CSV files: `SEQ_dist.csv`, `EB_dist.csv`

### 3. GPU EB coloring (on IISc cluster)
```bash
# ATOMIC worklist:
./bin/eb_coloring graphs/rmat_100k.el

# PPS worklist:
./bin/eb_coloring graphs/rmat_100k.el --pps
```

### 4. SuiteSparse datasets
Download from https://sparse.tamu.edu/. Recommended graphs from the paper:
```
audikw_1        (0.9M vertices, 76.7M edges)   — PDE, regular
kron_g500-logn21 (2.0M, 182.1M)                — irregular, high degree variance
circuit5M       (5.5M, 53.9M)                  — highly irregular
```
```bash
# After downloading .mtx file:
./bin/eb_coloring graphs/audikw_1.mtx
./bin/eb_coloring graphs/kron_g500-logn21.mtx --pps
```

### 5. Plot color class distribution
```bash
python3 scripts/plot_distribution.py SEQ_dist.csv EB_dist.csv \
    --labels "Sequential First-Fit" "Edge-Based EB" \
    --out imbalance_plot.png
```

---

## What This Baseline Demonstrates

The sequential first-fit coloring produces **severely imbalanced** color classes:
- Color 0 absorbs the most vertices (greedy lowest-first)
- Later colors have very few vertices
- Imbalance ratio (max_size / mean_size) is typically **10x–100x** on irregular graphs

The EB algorithm produces slightly fewer colors than first-fit but exhibits the **same first-fit imbalance** since `ASSIGNCOLORS` still picks the first available bit.

This is exactly the baseline imbalance your k-choice variant will fix.

---

## Key Files to Understand

| File | What to read |
|------|-------------|
| `include/eb_coloring.cuh` | Color encoding: `cs * 32 + bit_pos` |
| `src/eb_kernels.cu` | `k_forbid_colors_full`, `k_tentative_color`, `k_assign_colors`, `k_detect_conflicts_atomic` |
| `src/cpu_reference.cpp` | `eb_coloring_cpu()` — readable single-threaded EB logic |

The `ASSIGNCOLORS` kernel (`k_assign_colors`) is the key function your proposal modifies:
```cuda
// CURRENT: first available bit (first-fit)
uint32_t tc = first_avail(allforbid);

// YOUR CHANGE (k-choice): find k available bits, pick the one with
// minimum GlobalCount[color], then atomically increment GlobalCount.
```

---

## Metrics to Collect for Report

For each graph and each algorithm variant, collect:

| Metric | How |
|--------|-----|
| Number of colors | `n_colors` in output |
| Total time (ms) | `total_time_ms` |
| Rounds | `n_rounds` |
| Color class imbalance | `max_size / mean_size` |
| Std dev of class sizes | from distribution CSV |
| Downstream throughput | Run PageRank / SSSP using color classes as wave fronts |

Compare: `SEQ` vs `EB ATOMIC` vs `EB PPS` vs (next) `EB k-choice`.
