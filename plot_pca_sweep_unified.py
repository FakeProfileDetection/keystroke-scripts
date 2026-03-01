#!/usr/bin/env python3
"""
Generate a single unified figure (two panels) for the PCA component sweep.

Panel (a): Accuracy vs K for all 10 scenarios, grouped by training data volume.
Panel (b): Scenario 3.2 subsampled — accuracy vs training data volume at fixed K.
           Directly shows the data volume effect without repeating panel (a)'s layout.
"""

import matplotlib
matplotlib.use('Agg')

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RESULTS_DIR = Path("pca_component_sweep_2026-02-28_132947")
K_MAX = 65

SPU_COLORS = {1: "#3274A1", 2: "#E1812C", 4: "#3A923A"}
SPU_LABELS = {1: "1 sample/user", 2: "2 samples/user", 4: "4 samples/user"}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
sweep_summary = pd.read_csv(
    sorted(RESULTS_DIR.glob("sweep_summary_*.csv"))[-1]
)
subsample_detailed = pd.read_csv(
    sorted(RESULTS_DIR.glob("subsample_detailed_*.csv"))[-1]
)
for df in [sweep_summary, subsample_detailed]:
    df['scenario_id'] = df['scenario_id'].astype(str)

# ---------------------------------------------------------------------------
# Panel (a) data: natural-group sweep curves
# ---------------------------------------------------------------------------
sweep = sweep_summary[
    (sweep_summary['K'] > 0) & (sweep_summary['K'] <= K_MAX)
].copy()

group_ribbons = {}
group_baselines = {}

for spu in [1, 2, 4]:
    spu_data = sweep[sweep['train_samples_per_user'] == spu]
    pivot = spu_data.pivot_table(index='K', columns='scenario_id',
                                 values='top1_mean')
    group_ribbons[spu] = (pivot.index.values,
                          pivot.mean(axis=1).values,
                          pivot.std(axis=1).values)
    bl = sweep_summary[sweep_summary['train_samples_per_user'] == spu]
    group_baselines[spu] = bl['baseline_top1_mean'].mean()

# ---------------------------------------------------------------------------
# Panel (b) data: 3.2 subsampled — accuracy at fixed K vs data volume
# ---------------------------------------------------------------------------
sub = subsample_detailed[
    (subsample_detailed['scenario_id'] == '3.2') &
    (~subsample_detailed['is_baseline'])
].copy()

# Pick representative K values that span the curve
K_SLICES = [3, 7, 15, 30]
K_COLORS = {3: '#e41a1c', 7: '#984ea3', 15: '#ff7f00', 30: '#4daf4a'}
K_MARKERS = {3: 's', 7: 'D', 15: '^', 30: 'o'}

# For each K, get mean accuracy at each data level
slice_data = {}
for k_val in K_SLICES:
    points = []
    for eff_spu in [1, 2, 4]:
        rows = sub[(sub['K'] == k_val) & (sub['effective_spu'] == eff_spu)]
        if len(rows) > 0:
            points.append((eff_spu, rows['top1'].mean(), rows['top1'].std()))
    slice_data[k_val] = points

# Baseline at each data level
sub_baselines = {}
bl_rows = subsample_detailed[
    (subsample_detailed['scenario_id'] == '3.2') &
    (subsample_detailed['is_baseline'])
]
for eff_spu in [1, 2, 4]:
    r = bl_rows[bl_rows['effective_spu'] == eff_spu]
    sub_baselines[eff_spu] = (r['top1'].mean(), r['top1'].std())

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4.5),
                                  gridspec_kw={'width_ratios': [1.2, 1]})

# === Panel (a): Accuracy vs K ===
for spu in [4, 2, 1]:
    k, mean, std = group_ribbons[spu]
    c = SPU_COLORS[spu]
    ax_a.plot(k, mean, color=c, linewidth=2, label=SPU_LABELS[spu])
    ax_a.axhline(group_baselines[spu], color=c, linestyle='--', linewidth=1.5,
                 alpha=0.7, dashes=(5, 3))

# Baseline labels
for spu, va in [(4, 'bottom'), (2, 'bottom'), (1, 'top')]:
    ax_a.text(K_MAX - 1, group_baselines[spu],
              f"  baseline ({group_baselines[spu]:.0%})",
              fontsize=6.5, color=SPU_COLORS[spu], va=va, ha='right',
              fontstyle='italic')

# Mark the K slices used in panel (b) with vertical lines
for k_val in K_SLICES:
    ax_a.axvline(k_val, color=K_COLORS[k_val], linewidth=0.8, alpha=0.4,
                 linestyle='--')
    ax_a.text(k_val, 0.17, f"K={k_val}", fontsize=6.5,
              color=K_COLORS[k_val], ha='center', fontweight='bold')

ax_a.set_xlabel("Number of PCA Components (K)", fontsize=10)
ax_a.set_ylabel("Mean Top-1 Accuracy", fontsize=10)
ax_a.set_title("(a)  PCA sweep: all scenarios", fontsize=10,
               loc='left', fontweight='bold')
ax_a.set_xlim(0, K_MAX)
ax_a.set_ylim(0.15, 1.05)
ax_a.grid(True, alpha=0.2)
ax_a.tick_params(labelsize=9)

legend_a = [
    Line2D([0], [0], color=SPU_COLORS[4], lw=2, label="4 samples/user"),
    Line2D([0], [0], color=SPU_COLORS[2], lw=2, label="2 samples/user"),
    Line2D([0], [0], color=SPU_COLORS[1], lw=2, label="1 sample/user"),
]
ax_a.legend(handles=legend_a, fontsize=7.5, loc='center right',
            framealpha=0.9)

# === Panel (b): Accuracy vs data volume at fixed K ===
x_positions = [1, 2, 4]

for k_val in K_SLICES:
    pts = slice_data[k_val]
    if not pts:
        continue
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    yerr = [p[2] for p in pts]
    ax_b.errorbar(xs, ys, yerr=yerr, color=K_COLORS[k_val],
                  marker=K_MARKERS[k_val], markersize=7, linewidth=1.8,
                  capsize=4, capthick=1.2, label=f"K = {k_val}")

# Baseline
bl_xs = [1, 2, 4]
bl_ys = [sub_baselines[s][0] for s in bl_xs]
bl_err = [sub_baselines[s][1] for s in bl_xs]
ax_b.errorbar(bl_xs, bl_ys, yerr=bl_err, color='#333333', marker='*',
              markersize=9, linewidth=1.5, linestyle=':', capsize=4,
              capthick=1.2, label="Baseline (all 620)")

ax_b.set_xlabel("Training Samples per User", fontsize=10)
ax_b.set_title("(b)  Scenario 3.2: data volume effect", fontsize=10,
               loc='left', fontweight='bold')
ax_b.set_xticks([1, 2, 4])
ax_b.set_xticklabels(["1\n(subsampled)", "2\n(subsampled)", "4\n(native)"],
                      fontsize=8)
ax_b.set_ylim(0.15, 1.05)
ax_b.grid(True, alpha=0.2)
ax_b.tick_params(labelsize=9)
ax_b.legend(fontsize=7.5, loc='lower right', framealpha=0.9)

plt.tight_layout()

for ext in ['png', 'pdf']:
    path = RESULTS_DIR / f"unified_pca_sweep.{ext}"
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved: {path}")

plt.close()
