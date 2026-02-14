#!/usr/bin/env python3
"""
pca_vs_rf_comparison.py - Compare RF classification accuracy using PCA-derived
vs RF-derived top feature subsets.

Phase 1: Extract features consistently identified as important by PCA loadings
          across all 10 scenarios (95% variance threshold).
Phase 2: Extract RF top-10 features from existing trained model pickles.
Phase 3: Run RF classification on each feature subset and compare.
Phase 4: Aggregate results, generate bar chart, CSVs, and overlap analysis.
"""

import matplotlib
matplotlib.use('Agg')

import argparse
import json
import pickle
import sys
import warnings
from datetime import datetime
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.preprocessing import LabelEncoder

from ml_utils import (
    get_feature_columns, get_sub_experiment_data, load_config,
    apply_sub_experiment_filters
)
from scenarios import generate_all_scenarios, get_scenario_by_id

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# ---------------------------------------------------------------------------
# Phase 1 – PCA-important features
# ---------------------------------------------------------------------------

def extract_pca_important_features_for_subexp(X_train, variance_threshold=0.95):
    """Fit PCA on training data and return indices of important original features.

    1. Fit PCA, keep components up to *variance_threshold* cumulative variance.
    2. For each retained component, flag features whose |loading| exceeds the
       mean |loading| of that component.
    3. Return the union of flagged feature indices.
    """
    # Handle NaN/inf in training data
    X_clean = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

    pca = PCA()
    pca.fit(X_clean)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumvar, variance_threshold) + 1)
    n_components = min(n_components, len(cumvar))

    important_indices = set()
    for comp_idx in range(n_components):
        loadings = np.abs(pca.components_[comp_idx])
        threshold = loadings.mean()
        important_indices.update(np.where(loadings > threshold)[0])

    return important_indices


def get_pca_features_for_scenario(df, scenario, variance_threshold=0.95):
    """Return feature indices that PCA deems important in a majority of
    sub-experiments within *scenario*."""
    from collections import Counter

    feature_cols = get_feature_columns(df.columns.tolist())
    n_features = len(feature_cols)
    index_counts = Counter()
    n_subexps = 0

    for sub_exp in scenario.sub_experiments:
        train_df, _ = apply_sub_experiment_filters(df, sub_exp)
        X_train = train_df[feature_cols].values

        if len(X_train) < 2:
            continue

        important = extract_pca_important_features_for_subexp(
            X_train, variance_threshold
        )
        index_counts.update(important)
        n_subexps += 1

    if n_subexps == 0:
        return set()

    # Keep features appearing in a majority (>50%) of sub-experiments
    majority = n_subexps / 2.0
    return {idx for idx, cnt in index_counts.items() if cnt > majority}


def get_common_pca_features(df, scenarios_to_run, variance_threshold=0.95):
    """Intersect PCA-important features across all scenarios."""
    common = None
    for scenario_id in scenarios_to_run:
        scenario = get_scenario_by_id(scenario_id)
        print(f"  PCA analysis for scenario {scenario_id} "
              f"({len(scenario.sub_experiments)} sub-experiments) ...", end=" ")
        important = get_pca_features_for_scenario(
            df, scenario, variance_threshold
        )
        print(f"{len(important)} features")
        if common is None:
            common = important
        else:
            common = common.intersection(important)

    return sorted(common) if common else []


# ---------------------------------------------------------------------------
# Phase 2 – RF-important features from existing pickles
# ---------------------------------------------------------------------------

def get_rf_top_features(pickle_dir, top_n=10):
    """Average feature importances across all RF pickles and return top-N indices."""
    pkl_files = sorted(glob(str(Path(pickle_dir) / "*.pkl")))
    if not pkl_files:
        raise FileNotFoundError(f"No pickle files in {pickle_dir}")

    importances_sum = None
    count = 0

    for pf in pkl_files:
        with open(pf, 'rb') as f:
            data = pickle.load(f)
        fi = data['model'].feature_importances_
        if importances_sum is None:
            importances_sum = np.zeros_like(fi)
        importances_sum += fi
        count += 1

    avg_importances = importances_sum / count
    top_indices = np.argsort(avg_importances)[::-1][:top_n]
    return sorted(top_indices.tolist()), avg_importances


# ---------------------------------------------------------------------------
# Phase 3 – RF comparison experiments
# ---------------------------------------------------------------------------

def run_rf_on_subset(X_train, X_test, y_train, y_test, feature_indices, seed):
    """Train RF on a feature subset and return top-1 accuracy."""
    X_tr = X_train[:, feature_indices]
    X_te = X_test[:, feature_indices]

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=20, min_samples_split=2,
        min_samples_leaf=1, max_features='sqrt', bootstrap=True,
        random_state=seed, n_jobs=-1
    )
    rf.fit(X_tr, y_train)

    y_pred = rf.predict(X_te)
    acc = accuracy_score(y_test, y_pred)

    # Top-5 accuracy
    y_proba = rf.predict_proba(X_te)
    max_k = min(5, y_proba.shape[1])
    top5 = top_k_accuracy_score(y_test, y_proba, k=max_k) if max_k >= 2 else acc

    return acc, top5


def run_comparison_experiments(df, scenarios_to_run, seeds,
                               pca_indices, rf_indices, baseline_csv_path):
    """Run RF on PCA and RF feature subsets for every scenario/sub-exp/seed.

    Returns (detailed_rows, summary_rows).
    """
    feature_cols = get_feature_columns(df.columns.tolist())
    label_encoder = LabelEncoder()

    # Load baseline results for "all features" accuracy
    baseline_df = pd.read_csv(baseline_csv_path)

    detailed_rows = []

    for scenario_id in scenarios_to_run:
        scenario = get_scenario_by_id(scenario_id)
        print(f"\n  Scenario {scenario_id}: {scenario.name} "
              f"({len(scenario.sub_experiments)} sub-exps)")

        for sub_idx, sub_exp in enumerate(scenario.sub_experiments, 1):
            X_train, X_test, y_train, y_test = get_sub_experiment_data(df, sub_exp)

            if len(X_train) == 0 or len(X_test) == 0:
                continue

            # Common-user filtering (same as ml_scenario_runner.py:229-244)
            train_users = set(y_train)
            test_users = set(y_test)
            common_users = train_users.intersection(test_users)
            if len(common_users) < 2:
                continue

            train_mask = np.isin(y_train, list(common_users))
            test_mask = np.isin(y_test, list(common_users))
            X_train = X_train[train_mask]
            y_train = y_train[train_mask]
            X_test = X_test[test_mask]
            y_test = y_test[test_mask]

            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue

            y_train_enc = label_encoder.fit_transform(y_train)
            y_test_enc = label_encoder.transform(y_test)

            for seed in seeds:
                # PCA feature subset
                pca_acc, pca_top5 = run_rf_on_subset(
                    X_train, X_test, y_train_enc, y_test_enc, pca_indices, seed
                )

                # RF feature subset
                rf_acc, rf_top5 = run_rf_on_subset(
                    X_train, X_test, y_train_enc, y_test_enc, rf_indices, seed
                )

                # Baseline accuracy from existing results
                bl = baseline_df[
                    (baseline_df['sub_experiment_name'] == sub_exp.name) &
                    (baseline_df['random_seed'] == seed)
                ]
                baseline_acc = bl['test_top_1_accuracy'].values[0] if len(bl) > 0 else np.nan
                baseline_top5 = bl['test_top_5_accuracy'].values[0] if len(bl) > 0 else np.nan

                detailed_rows.append({
                    'scenario_id': scenario_id,
                    'sub_experiment': sub_exp.name,
                    'seed': seed,
                    'pca_top1': pca_acc,
                    'pca_top5': pca_top5,
                    'rf_top1': rf_acc,
                    'rf_top5': rf_top5,
                    'baseline_top1': baseline_acc,
                    'baseline_top5': baseline_top5,
                    'n_pca_features': len(pca_indices),
                    'n_rf_features': len(rf_indices),
                })

            if sub_idx % 5 == 0 or sub_idx == len(scenario.sub_experiments):
                print(f"    [{sub_idx}/{len(scenario.sub_experiments)}] done")

    # Build summary per scenario
    detailed_df = pd.DataFrame(detailed_rows)
    summary_rows = []
    for scenario_id in scenarios_to_run:
        sc = detailed_df[detailed_df['scenario_id'] == scenario_id]
        if sc.empty:
            continue
        summary_rows.append({
            'scenario_id': scenario_id,
            'pca_top1_mean': sc['pca_top1'].mean(),
            'pca_top1_std': sc['pca_top1'].std(),
            'pca_top5_mean': sc['pca_top5'].mean(),
            'rf_top1_mean': sc['rf_top1'].mean(),
            'rf_top1_std': sc['rf_top1'].std(),
            'rf_top5_mean': sc['rf_top5'].mean(),
            'baseline_top1_mean': sc['baseline_top1'].mean(),
            'baseline_top1_std': sc['baseline_top1'].std(),
            'baseline_top5_mean': sc['baseline_top5'].mean(),
            'n_sub_experiments': sc['sub_experiment'].nunique(),
            'n_pca_features': len(pca_indices),
            'n_rf_features': len(rf_indices),
        })

    summary_df = pd.DataFrame(summary_rows)
    return detailed_df, summary_df


# ---------------------------------------------------------------------------
# Phase 4 – Outputs
# ---------------------------------------------------------------------------

def plot_comparison(summary_df, output_path):
    """Grouped bar chart: scenarios on x-axis, bars per feature set."""
    scenarios = summary_df['scenario_id'].values
    x = np.arange(len(scenarios))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))

    bars1 = ax.bar(x - width, summary_df['baseline_top1_mean'], width,
                   label=f'All features (620)', color='steelblue',
                   yerr=summary_df['baseline_top1_std'], capsize=3)
    bars2 = ax.bar(x, summary_df['pca_top1_mean'], width,
                   label=f'PCA features ({summary_df["n_pca_features"].iloc[0]})',
                   color='coral',
                   yerr=summary_df['pca_top1_std'], capsize=3)
    bars3 = ax.bar(x + width, summary_df['rf_top1_mean'], width,
                   label=f'RF top-10 features', color='seagreen',
                   yerr=summary_df['rf_top1_std'], capsize=3)

    ax.set_xlabel('Scenario')
    ax.set_ylabel('Top-1 Accuracy')
    ax.set_title('PCA vs RF Feature Selection: Classification Accuracy Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1)

    # Value labels – place inside the bar to avoid colliding with error bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, h / 2),
                        ha='center', va='center', fontsize=6,
                        rotation=90, color='white', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {output_path}")


def print_summary_table(summary_df, pca_names, rf_names, overlap):
    """Print a console summary."""
    print("\n" + "=" * 80)
    print("FEATURE SELECTION COMPARISON SUMMARY")
    print("=" * 80)

    print(f"\nPCA features ({len(pca_names)}): {', '.join(pca_names[:20])}")
    if len(pca_names) > 20:
        print(f"  ... and {len(pca_names) - 20} more")
    print(f"\nRF top-10 features: {', '.join(rf_names)}")
    print(f"\nOverlap ({len(overlap)} features): {', '.join(overlap) if overlap else 'none'}")

    print(f"\n{'Scenario':<12} {'Baseline':>10} {'PCA':>10} {'RF top-10':>10} "
          f"{'PCA vs BL':>10} {'RF vs BL':>10}")
    print("-" * 62)
    for _, row in summary_df.iterrows():
        bl = row['baseline_top1_mean']
        pca = row['pca_top1_mean']
        rf = row['rf_top1_mean']
        print(f"{row['scenario_id']:<12} {bl:>10.4f} {pca:>10.4f} {rf:>10.4f} "
              f"{pca - bl:>+10.4f} {rf - bl:>+10.4f}")

    # Overall averages
    bl_avg = summary_df['baseline_top1_mean'].mean()
    pca_avg = summary_df['pca_top1_mean'].mean()
    rf_avg = summary_df['rf_top1_mean'].mean()
    print("-" * 62)
    print(f"{'Average':<12} {bl_avg:>10.4f} {pca_avg:>10.4f} {rf_avg:>10.4f} "
          f"{pca_avg - bl_avg:>+10.4f} {rf_avg - bl_avg:>+10.4f}")
    print("=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Compare PCA vs RF feature selection for keystroke biometrics'
    )
    parser.add_argument('-c', '--config', default='config_rf_only.json',
                        help='Config JSON (default: config_rf_only.json)')
    parser.add_argument('--pickle-dir',
                        default='scenario_results_rf_only_2026-01-25_223058_early_stop',
                        help='Directory with existing RF pickle files')
    parser.add_argument('--baseline-csv', default=None,
                        help='Path to baseline sub_experiment_results CSV '
                             '(auto-detected from pickle-dir if omitted)')
    parser.add_argument('--variance-threshold', type=float, default=0.95,
                        help='PCA cumulative variance threshold (default: 0.95)')
    parser.add_argument('--rf-top-n', type=int, default=10,
                        help='Number of top RF features (default: 10)')
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    # Load config
    config = load_config(args.config)
    dataset_path = config['dataset_path']
    seeds = config.get('seeds', [42, 123, 456])
    scenarios_to_run = config.get('scenarios_to_run',
                                  ["1.1", "1.2", "2.1", "2.2", "3.1", "3.2",
                                   "4.1", "4.2", "5.1", "5.2"])

    # Auto-detect baseline CSV
    baseline_csv = args.baseline_csv
    if baseline_csv is None:
        candidates = sorted(glob(str(
            Path(args.pickle_dir) / "sub_experiment_results_*.csv"
        )))
        if not candidates:
            print("Error: no sub_experiment_results CSV found in pickle dir")
            sys.exit(1)
        baseline_csv = candidates[-1]
    print(f"Baseline CSV: {baseline_csv}")

    # Output directory
    output_dir = Path(f"pca_rf_comparison_{timestamp}")
    output_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    print(f"\nLoading dataset: {dataset_path}")
    df_pl = pl.read_csv(dataset_path)
    df = df_pl.to_pandas()
    feature_cols = get_feature_columns(df.columns.tolist())
    print(f"Dataset: {df.shape[0]} rows, {len(feature_cols)} features, "
          f"{df['user_id'].nunique()} users")

    # ------------------------------------------------------------------
    # Phase 1 – PCA features
    # ------------------------------------------------------------------
    print(f"\n--- Phase 1: PCA feature extraction "
          f"(variance threshold = {args.variance_threshold}) ---")
    pca_indices = get_common_pca_features(
        df, scenarios_to_run, args.variance_threshold
    )
    pca_names = [feature_cols[i] for i in pca_indices]
    print(f"\nCommon PCA features across all scenarios: {len(pca_indices)}")

    # ------------------------------------------------------------------
    # Phase 2 – RF features
    # ------------------------------------------------------------------
    print(f"\n--- Phase 2: RF top-{args.rf_top_n} feature extraction ---")
    rf_indices, avg_importances = get_rf_top_features(
        args.pickle_dir, args.rf_top_n
    )
    rf_names = [feature_cols[i] for i in rf_indices]
    print(f"RF top-{args.rf_top_n} features: {rf_names}")

    # Overlap analysis
    overlap_indices = sorted(set(pca_indices).intersection(set(rf_indices)))
    overlap_names = [feature_cols[i] for i in overlap_indices]
    print(f"\nOverlap: {len(overlap_names)} features — {overlap_names}")

    # ------------------------------------------------------------------
    # Phase 3 – Comparison experiments
    # ------------------------------------------------------------------
    print("\n--- Phase 3: Running comparison experiments ---")
    detailed_df, summary_df = run_comparison_experiments(
        df, scenarios_to_run, seeds, pca_indices, rf_indices, baseline_csv
    )

    # ------------------------------------------------------------------
    # Phase 4 – Outputs
    # ------------------------------------------------------------------
    print("\n--- Phase 4: Generating outputs ---")

    # CSVs
    summary_path = output_dir / f"pca_vs_rf_comparison_summary_{timestamp}.csv"
    detailed_path = output_dir / f"pca_vs_rf_comparison_detailed_{timestamp}.csv"
    summary_df.to_csv(summary_path, index=False)
    detailed_df.to_csv(detailed_path, index=False)
    print(f"  Summary CSV: {summary_path}")
    print(f"  Detailed CSV: {detailed_path}")

    # Plot
    plot_path = output_dir / f"pca_vs_rf_comparison_{timestamp}.png"
    plot_comparison(summary_df, plot_path)

    # Config + feature lists JSON
    config_out = {
        'timestamp': timestamp,
        'variance_threshold': args.variance_threshold,
        'rf_top_n': args.rf_top_n,
        'seeds': seeds,
        'scenarios_to_run': scenarios_to_run,
        'dataset_path': dataset_path,
        'pickle_dir': args.pickle_dir,
        'baseline_csv': baseline_csv,
        'pca_feature_indices': [int(i) for i in pca_indices],
        'pca_feature_names': pca_names,
        'rf_feature_indices': [int(i) for i in rf_indices],
        'rf_feature_names': rf_names,
        'overlap_feature_names': overlap_names,
        'n_pca_features': len(pca_indices),
        'n_rf_features': len(rf_indices),
        'n_overlap': len(overlap_names),
    }
    config_path = output_dir / f"comparison_config_{timestamp}.json"
    with open(config_path, 'w') as f:
        json.dump(config_out, f, indent=2)
    print(f"  Config JSON: {config_path}")

    # Console summary
    print_summary_table(summary_df, pca_names, rf_names, overlap_names)

    print(f"\nAll outputs saved to: {output_dir}/")


if __name__ == "__main__":
    main()
