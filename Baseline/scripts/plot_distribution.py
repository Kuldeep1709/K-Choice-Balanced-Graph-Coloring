#!/usr/bin/env python3
"""
plot_distribution.py

Plots color class size distributions from CSV files produced by
cpu_reference or eb_coloring executables.

Usage:
    python3 plot_distribution.py SEQ_dist.csv EB_dist.csv [--out fig.png]
    python3 plot_distribution.py *.csv --compare

Produces:
  - Bar chart of color class sizes (log scale)
  - Cumulative distribution
  - Summary statistics table
"""

import sys
import os
import csv
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def load_csv(path):
    ids, sizes = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(int(row['color_id']))
            sizes.append(int(row['size']))
    return np.array(ids), np.array(sizes)

def stats(sizes):
    s = sizes[sizes > 0]
    return {
        'n_colors': len(s),
        'total_vertices': int(s.sum()),
        'min': int(s.min()),
        'max': int(s.max()),
        'mean': float(s.mean()),
        'std': float(s.std()),
        'imbalance': float(s.max() / s.mean()),
        'p90': float(np.percentile(s, 90)),
        'p10': float(np.percentile(s, 10)),
    }

def print_stats(label, s):
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  Colors          : {s['n_colors']}")
    print(f"  Total vertices  : {s['total_vertices']}")
    print(f"  Min class size  : {s['min']}")
    print(f"  Max class size  : {s['max']}")
    print(f"  Mean class size : {s['mean']:.1f}")
    print(f"  Std deviation   : {s['std']:.1f}")
    print(f"  Imbalance ratio : {s['imbalance']:.2f}x  (max/mean)")
    print(f"  P10 / P90       : {s['p10']:.0f} / {s['p90']:.0f}")

def plot_comparison(files, labels, outfile):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Color Class Distribution: First-Fit Imbalance Analysis",
                 fontsize=14, fontweight='bold')

    colors_plot = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

    all_data = []
    for path in files:
        ids, sizes = load_csv(path)
        all_data.append((ids, sizes))

    # ---- Panel 1: Bar chart (sorted by color id) ----
    ax = axes[0]
    ax.set_title("Class Sizes by Color ID", fontsize=11)
    for i, ((ids, sizes), label) in enumerate(zip(all_data, labels)):
        ax.plot(ids, sizes, color=colors_plot[i % len(colors_plot)],
                alpha=0.7, linewidth=0.8, label=label)
    ax.set_xlabel("Color ID")
    ax.set_ylabel("Class Size")
    ax.set_yscale('log')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    # Annotate the massive imbalance (color 0 vs last)
    ax.annotate("Color 0\n(most vertices)", xy=(0, all_data[0][1][0]),
                xytext=(len(all_data[0][0])//4, all_data[0][1][0]),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=8, color='red')

    # ---- Panel 2: Histogram of class sizes ----
    ax = axes[1]
    ax.set_title("Distribution of Class Sizes", fontsize=11)
    for i, ((ids, sizes), label) in enumerate(zip(all_data, labels)):
        s = sizes[sizes > 0]
        ax.hist(s, bins=50, color=colors_plot[i % len(colors_plot)],
                alpha=0.6, label=label, density=True)
    ax.set_xlabel("Class Size")
    ax.set_ylabel("Frequency (normalized)")
    ax.set_xscale('log')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- Panel 3: Cumulative % of vertices covered ----
    ax = axes[2]
    ax.set_title("Cumulative Vertex Coverage vs # Colors Used", fontsize=11)
    for i, ((ids, sizes), label) in enumerate(zip(all_data, labels)):
        s = sizes[sizes > 0]
        sorted_s = np.sort(s)[::-1]
        cumsum = np.cumsum(sorted_s) / sorted_s.sum() * 100
        ax.plot(np.arange(1, len(cumsum)+1), cumsum,
                color=colors_plot[i % len(colors_plot)],
                linewidth=2, label=label)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50%')
    ax.axhline(y=90, color='gray', linestyle=':',  alpha=0.5, label='90%')
    ax.set_xlabel("Number of Color Classes (sorted by size)")
    ax.set_ylabel("% Vertices Covered")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {outfile}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot color class distributions")
    parser.add_argument('files', nargs='+', help='CSV files to plot')
    parser.add_argument('--out', default='distribution.png', help='Output PNG')
    parser.add_argument('--labels', nargs='+', help='Labels for each file')
    args = parser.parse_args()

    labels = args.labels or [os.path.basename(f).replace('_dist.csv','').replace('.csv','')
                              for f in args.files]

    # Print stats for each
    for path, label in zip(args.files, labels):
        _, sizes = load_csv(path)
        s = stats(sizes)
        print_stats(label, s)

    # Plot
    plot_comparison(args.files, labels, args.out)

if __name__ == '__main__':
    main()
    