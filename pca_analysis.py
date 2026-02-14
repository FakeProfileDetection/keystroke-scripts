#!/usr/bin/env python3
"""
pca_analysis.py - Per-scenario PCA analysis for keystroke biometrics features.

For each scenario, runs PCA independently on each sub-experiment's training data,
then aggregates results (median + IQR) to show the cumulative explained variance
curve per scenario.

Outputs:
- A 2x5 panel figure (one panel per scenario) showing:
    - Solid line:  median cumulative explained variance across sub-experiments
    - Shaded band: interquartile range (P25-P75)
    - Dashed red line at 95% variance threshold
    - Red dot + annotation for the elbow point
    - Green vertical line + annotation for the 95%-threshold component count
- A summary CSV with optimal component counts per scenario

Usage:
    python pca_analysis.py                          # defaults to config_rf_only.json
    python pca_analysis.py -c config_scenarios.json  # use a different config
    python pca_analysis.py --scenarios 1.1 2.1 3.1   # analyse specific scenarios only
"""

import matplotlib
matplotlib.use('Agg')

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from scenarios import generate_all_scenarios, get_scenario_by_id
from ml_utils import get_feature_columns, get_sub_experiment_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_elbow(cumulative_variance: np.ndarray) -> int:
    """Return the 0-based index of the elbow (maximum-distance-from-chord method)."""
    n = len(cumulative_variance)
    if n < 3:
        return 0

    # Normalise both axes to [0, 1] so the distance metric is scale-invariant
    x = np.linspace(0, 1, n)
    span = cumulative_variance[-1] - cumulative_variance[0]
    if span < 1e-12:
        return 0
    y = (cumulative_variance - cumulative_variance[0]) / span

    # Chord from the first to the last point
    p0 = np.array([x[0], y[0]])
    p1 = np.array([x[-1], y[-1]])
    chord = p1 - p0
    chord_len = np.linalg.norm(chord)
    if chord_len < 1e-12:
        return 0
    chord_unit = chord / chord_len

    # Perpendicular distance of each point to the chord
    dists = np.empty(n)
    for i in range(n):
        v = np.array([x[i], y[i]]) - p0
        proj = np.dot(v, chord_unit)
        dists[i] = np.linalg.norm(v - proj * chord_unit)

    return int(np.argmax(dists))


def find_threshold_component(cumulative_variance: np.ndarray,
                             threshold: float = 0.95) -> int:
    """Return the 0-based index of the first component reaching *threshold*."""
    hits = np.where(cumulative_variance >= threshold)[0]
    if len(hits) > 0:
        return int(hits[0])
    return len(cumulative_variance) - 1


def run_pca_on_data(X: np.ndarray) -> Tuple[np.ndarray, int]:
    """Fit PCA and return (cumulative_explained_variance, n_components_used)."""
    max_components = min(X.shape[0], X.shape[1]) - 1
    if max_components < 1:
        return np.array([1.0]), 1

    X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    pca = PCA(n_components=max_components)
    pca.fit(X_clean)

    return np.cumsum(pca.explained_variance_ratio_), max_components


# ---------------------------------------------------------------------------
# Per-scenario PCA
# ---------------------------------------------------------------------------

def run_scenario_pca(df: pd.DataFrame, scenario_id: str) -> Dict:
    """Run PCA for every sub-experiment in *scenario_id* and collect curves."""
    scenario = get_scenario_by_id(scenario_id)

    result: Dict = {
        "scenario_id": scenario_id,
        "scenario_name": scenario.name,
        "scenario_description": scenario.description,
        "sub_experiment_variances": [],
        "sub_experiment_names": [],
        "max_components": [],
        "n_train_samples": [],
    }

    for sub_exp in scenario.sub_experiments:
        X_train, _, y_train, _ = get_sub_experiment_data(df, sub_exp)

        if len(X_train) < 3:
            print(f"    Skipping {sub_exp.name}: only {len(X_train)} training samples")
            continue

        cum_var, max_comp = run_pca_on_data(X_train)

        result["sub_experiment_variances"].append(cum_var)
        result["sub_experiment_names"].append(sub_exp.name)
        result["max_components"].append(max_comp)
        result["n_train_samples"].append(len(X_train))

    return result


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_variance_curves(
    variances: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Align curves to the shortest length and return
    (median, p25, p75, n_components).
    """
    if not variances:
        return np.array([]), np.array([]), np.array([]), 0

    min_len = min(len(v) for v in variances)
    aligned = np.array([v[:min_len] for v in variances])

    return (
        np.median(aligned, axis=0),
        np.percentile(aligned, 25, axis=0),
        np.percentile(aligned, 75, axis=0),
        min_len,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

X_AXIS_CAP = 35  # Only show the first N components (where the action is)


def plot_scenario_panels(all_results: Dict[str, Dict], output_path: str) -> List[Dict]:
    """Create a 2x5 panel figure and return a list of summary dicts."""
    fig, axes = plt.subplots(2, 5, figsize=(26, 10), dpi=300)
    axes = axes.flatten()

    scenario_ids = sorted(all_results.keys(),
                          key=lambda s: [int(x) for x in s.split(".")])
    summary_rows: List[Dict] = []

    for idx, scenario_id in enumerate(scenario_ids):
        ax = axes[idx]
        res = all_results[scenario_id]
        variances = res["sub_experiment_variances"]

        if not variances:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12)
            ax.set_title(f"Scenario {scenario_id}", fontsize=10, fontweight="bold")
            continue

        median_c, p25_c, p75_c, n_comp = aggregate_variance_curves(variances)
        if n_comp == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12)
            ax.set_title(f"Scenario {scenario_id}", fontsize=10, fontweight="bold")
            continue

        # Cap the display range
        display_n = min(n_comp, X_AXIS_CAP)
        x = np.arange(1, display_n + 1)

        # IQR band
        ax.fill_between(x, p25_c[:display_n], p75_c[:display_n],
                        alpha=0.25, color="steelblue",
                        label="IQR (P25\u2013P75)")
        # Median curve
        ax.plot(x, median_c[:display_n], color="steelblue",
                linewidth=1.5, label="Median")

        # 95 % threshold line
        ax.axhline(y=0.95, color="red", linestyle="--", linewidth=1,
                    alpha=0.7, label="95% variance")

        # Elbow
        elbow_idx = find_elbow(median_c)
        elbow_comp = elbow_idx + 1
        elbow_var = median_c[elbow_idx]
        # Elbow (only annotate if within display range)
        if elbow_comp <= display_n:
            ax.plot(elbow_comp, elbow_var, "ro", markersize=5, zorder=5)
            ax.annotate(
                f"Elbow: {elbow_comp}",
                xy=(elbow_comp, elbow_var),
                xytext=(min(elbow_comp + 3, display_n - 1),
                        max(elbow_var - 0.08, 0.05)),
                fontsize=7, color="red",
                arrowprops=dict(arrowstyle="->", color="red", lw=0.8),
            )

        # 95 % threshold component
        thresh_idx = find_threshold_component(median_c, 0.95)
        thresh_comp = thresh_idx + 1
        if thresh_comp <= display_n:
            ax.axvline(x=thresh_comp, color="green", linestyle=":",
                        linewidth=1, alpha=0.7)
            ax.annotate(
                f"95%: {thresh_comp}",
                xy=(thresh_comp, 0.95),
                xytext=(min(thresh_comp + 2, display_n - 1), 0.88),
                fontsize=7, color="green",
                arrowprops=dict(arrowstyle="->", color="green", lw=0.8),
            )

        # Axes formatting
        ax.set_title(
            f"Scenario {scenario_id}\n{res['scenario_description']}",
            fontsize=9, fontweight="bold",
        )
        ax.set_xlabel("Components", fontsize=8)
        ax.set_ylabel("Cumul. Expl. Variance", fontsize=8)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlim(0, display_n + 1)
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend(fontsize=6, loc="lower right")

        # Summary row
        summary_rows.append({
            "scenario_id": scenario_id,
            "description": res["scenario_description"],
            "n_sub_experiments": len(variances),
            "avg_train_samples": np.mean(res["n_train_samples"]),
            "max_components_available": n_comp,
            "elbow_components": elbow_comp,
            "elbow_variance": round(float(elbow_var), 4),
            "threshold_95_components": thresh_comp,
            "threshold_95_variance": round(float(median_c[thresh_idx]), 4),
            "median_variance_at_max": round(float(median_c[-1]), 4),
        })

    # Disable unused axes (if fewer than 10 scenarios)
    for j in range(len(scenario_ids), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "PCA Explained Variance per Scenario\n"
        "(Median \u00b1 IQR across sub-experiments, fitted on training data only)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return summary_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PCA analysis per scenario for keystroke biometrics",
    )
    parser.add_argument(
        "-c", "--config", default="config_rf_only.json",
        help="Path to configuration file (default: config_rf_only.json)",
    )
    parser.add_argument(
        "-o", "--output-dir", default=None,
        help="Output directory (default: pca_results_<timestamp>)",
    )
    parser.add_argument(
        "--scenarios", nargs="+", default=None,
        help="Specific scenarios to analyse (default: all from config)",
    )
    args = parser.parse_args()

    # ---- Load dataset ----
    print("Loading dataset ...")
    with open(args.config, "r") as f:
        config = json.load(f)

    dataset_path = config["dataset_path"]
    if not Path(dataset_path).exists():
        print(f"ERROR: dataset not found: {dataset_path}")
        sys.exit(1)

    df = pl.read_csv(dataset_path).to_pandas()
    feature_cols = get_feature_columns(df.columns.tolist())
    print(f"  {df.shape[0]} rows, {len(feature_cols)} features, "
          f"{df['user_id'].nunique()} users")

    # ---- Resolve scenarios ----
    scenarios_to_run = args.scenarios or config.get(
        "scenarios_to_run",
        ["1.1", "1.2", "2.1", "2.2", "3.1", "3.2",
         "4.1", "4.2", "5.1", "5.2"],
    )
    print(f"  Scenarios: {', '.join(scenarios_to_run)}")

    # ---- Output directory ----
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = Path(args.output_dir or f"pca_results_{timestamp}")
    output_dir.mkdir(exist_ok=True)

    # ---- Run PCA per scenario ----
    all_results: Dict[str, Dict] = {}
    for sid in scenarios_to_run:
        print(f"\n  Scenario {sid} ...")
        res = run_scenario_pca(df, sid)
        all_results[sid] = res

        n = len(res["sub_experiment_variances"])
        if n > 0:
            print(f"    {n} sub-experiments, "
                  f"avg {np.mean(res['n_train_samples']):.0f} train samples, "
                  f"avg {np.mean(res['max_components']):.0f} max PCA components")

    # ---- Plot ----
    print("\nGenerating plots ...")
    plot_path = output_dir / "pca_variance_per_scenario.png"
    summary_rows = plot_scenario_panels(all_results, str(plot_path))
    print(f"  Plot saved: {plot_path}")

    # ---- Summary CSV ----
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = output_dir / "pca_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"  Summary CSV saved: {summary_path}")

        print("\n" + "=" * 100)
        print("PCA ANALYSIS SUMMARY")
        print("=" * 100)
        header = (f"{'Scenario':<10} {'Description':<38} {'SubExps':>7} "
                  f"{'AvgSamp':>8} {'MaxComp':>8} {'Elbow':>6} "
                  f"{'ElbowVar':>9} {'95%Comp':>8} {'VarAtMax':>9}")
        print(header)
        print("-" * 100)
        for r in summary_rows:
            print(f"{r['scenario_id']:<10} {r['description']:<38} "
                  f"{r['n_sub_experiments']:>7} "
                  f"{r['avg_train_samples']:>8.0f} "
                  f"{r['max_components_available']:>8} "
                  f"{r['elbow_components']:>6} "
                  f"{r['elbow_variance']:>9.4f} "
                  f"{r['threshold_95_components']:>8} "
                  f"{r['median_variance_at_max']:>9.4f}")
        print("=" * 100)

    print(f"\nAll outputs saved to: {output_dir}/")


if __name__ == "__main__":
    main()
