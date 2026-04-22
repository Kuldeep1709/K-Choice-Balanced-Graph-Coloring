CXX      = g++
NVCC     = nvcc
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra
NVCCFLAGS = -O3 -std=c++17 -arch=sm_80   # adjust arch: sm_70 (V100), sm_86 (RTX30xx), sm_90 (H100)
INCLUDES  = -I include

# Thrust is bundled with CUDA; no extra -I needed

.PHONY: all cpu gpu gen clean

all: cpu gen

# ---- CPU reference (no GPU needed) ----
cpu: bin/cpu_ref

bin/cpu_ref: src/cpu_reference.cpp include/eb_coloring.cuh | bin
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@

# ---- Graph generator ----
gen: bin/gen_graphs

bin/gen_graphs: src/gen_graphs.cpp | bin
	$(CXX) $(CXXFLAGS) $< -o $@

# ---- GPU EB coloring (requires CUDA) ----
gpu: bin/eb_coloring

bin/eb_coloring: src/eb_coloring.cu src/eb_kernels.cu include/eb_coloring.cuh | bin
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) \
	    src/eb_kernels.cu src/eb_coloring.cu \
	    -o $@

bin:
	mkdir -p bin

clean:
	rm -rf bin

# ---- Quick test targets ----
test_ring: cpu gen
	./bin/gen_graphs ring 1000 > graphs/ring_1k.el
	./bin/cpu_ref graphs/ring_1k.el

test_grid: cpu gen
	./bin/gen_graphs grid 100 100 > graphs/grid_100x100.el
	./bin/cpu_ref graphs/grid_100x100.el

test_random: cpu gen
	./bin/gen_graphs random 10000 50000 > graphs/random_10k.el
	./bin/cpu_ref graphs/random_10k.el

test_rmat: cpu gen
	./bin/gen_graphs rmat 100000 800000 > graphs/rmat_100k.el
	./bin/cpu_ref graphs/rmat_100k.el

# ---- SuiteSparse download helper ----
# Download a small test matrix from SuiteSparse:
download_test:
	mkdir -p graphs
	wget -q "https://suitesparse-collection-website.herokuapp.com/MM/Pothen/mesh3e1.tar.gz" -O graphs/mesh3e1.tar.gz
	tar -xzf graphs/mesh3e1.tar.gz -C graphs/
	mv graphs/mesh3e1/mesh3e1.mtx graphs/
	rm -rf graphs/mesh3e1 graphs/mesh3e1.tar.gz
	./bin/cpu_ref graphs/mesh3e1.mtx
