#!/usr/bin/env python3
"""
pca_component_sweep.py - PCA Component Sweep: Training Data Size vs. Feature Count

Experiment 1: Per-scenario PCA component sweep (K=1..max), measuring RF accuracy
              at each K using above-mean loading feature selection.
Experiment 2: Controlled subsampling of high-data scenarios (3.2, 4.2) to lower
              data sizes, then re-running the full sweep, to disentangle whether
              PCA benefits come from more components or more training data.

Outputs: detailed/summary CSVs, convergence table, and 4 plots (A-D).
"""

import matplotlib
matplotlib.use('Agg')

import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.preprocessing import LabelEncoder

from ml_utils import get_feature_columns, get_sub_experiment_data, load_config
from scenarios import generate_all_scenarios, get_scenario_by_id

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# ---------------------------------------------------------------------------
# Data-size grouping for scenarios
# ---------------------------------------------------------------------------
SCENARIO_SPU = {
    "1.1": 1, "1.2": 1, "2.1": 1, "2.2": 1,
    "3.1": 2, "3.2": 4, "4.1": 2, "4.2": 4,
    "5.1": 1, "5.2": 1,
}

SPU_GROUP_LABELS = {1: "1 sample/user", 2: "2 samples/user", 4: "4 samples/user"}


# ---------------------------------------------------------------------------
# Core helper functions
# ---------------------------------------------------------------------------

def fit_pca_full(X_train):
    """Fit PCA with all components on training data.

    Returns:
        (pca, max_k) where max_k = min(n_samples, n_features).
    """
    X_clean = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    max_k = min(X_clean.shape[0], X_clean.shape[1])
    pca = PCA(n_components=max_k)
    pca.fit(X_clean)
    return pca, max_k


def select_features_at_k(pca, k):
    """Select feature indices using above-mean loading on the first K components.

    For each of the first K components, flag features whose |loading| exceeds
    the mean |loading| of that component. Return the union of flagged indices.

    Returns:
        sorted list of selected feature indices
    """
    important_indices = set()
    for comp_idx in range(k):
        loadings = np.abs(pca.components_[comp_idx])
        threshold = loadings.mean()
        important_indices.update(np.where(loadings > threshold)[0])
    return sorted(important_indices)


def generate_k_values(max_k):
    """Produce the sweep K list: every integer 1-30, then every 5 from 35 to max_k."""
    k_values = list(range(1, min(31, max_k + 1)))
    k = 35
    while k <= max_k:
        k_values.append(k)
        k += 5
    # Always include max_k if not already present
    if max_k not in k_values and max_k > 0:
        k_values.append(max_k)
    return sorted(set(k_values))


def run_rf_classify(X_train, X_test, y_train, y_test, feature_indices, seed):
    """Train RF on a feature subset, return (top1, top3, top5) accuracy.

    Uses the standard RF hyperparameters from config_rf_only.json.
    """
    if feature_indices is not None:
        X_tr = X_train[:, feature_indices]
        X_te = X_test[:, feature_indices]
    else:
        # Baseline: use all features
        X_tr = X_train
        X_te = X_test

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=20, min_samples_split=2,
        min_samples_leaf=1, max_features='sqrt', bootstrap=True,
        random_state=seed, n_jobs=-1
    )
    rf.fit(X_tr, y_train)

    y_pred = rf.predict(X_te)
    top1 = accuracy_score(y_test, y_pred)

    y_proba = rf.predict_proba(X_te)
    n_classes = y_proba.shape[1]

    top3 = top_k_accuracy_score(y_test, y_proba, k=min(3, n_classes)) if n_classes >= 2 else top1
    top5 = top_k_accuracy_score(y_test, y_proba, k=min(5, n_classes)) if n_classes >= 2 else top1

    return top1, top3, top5


def prepare_sub_experiment(df, sub_exp):
    """Load data for a sub-experiment, apply common-user filtering, label encode.

    Returns:
        (X_train, X_test, y_train_enc, y_test_enc, train_df, test_df)
        or None if insufficient data.
    """
    X_train, X_test, y_train, y_test = get_sub_experiment_data(df, sub_exp)

    if len(X_train) == 0 or len(X_test) == 0:
        return None

    # Common-user filtering
    train_users = set(y_train)
    test_users = set(y_test)
    common_users = train_users.intersection(test_users)
    if len(common_users) < 2:
        return None

    train_mask = np.isin(y_train, list(common_users))
    test_mask = np.isin(y_test, list(common_users))
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]
    X_test = X_test[test_mask]
    y_test = y_test[test_mask]

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return None

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    return X_train, X_test, y_train_enc, y_test_enc, y_train


def subsample_training_data(X_train, y_train, target_spu, rng):
    """Subsample training data to target_spu samples per user.

    For each unique user, randomly select target_spu of their training samples.
    If a user has fewer than target_spu samples, keep all of them.

    Args:
        X_train: Training features array
        y_train: Training labels (encoded)
        target_spu: Target samples per user
        rng: numpy random Generator

    Returns:
        (X_sub, y_sub) subsampled arrays
    """
    unique_users = np.unique(y_train)
    keep_indices = []
    for user in unique_users:
        user_idx = np.where(y_train == user)[0]
        if len(user_idx) <= target_spu:
            keep_indices.extend(user_idx)
        else:
            chosen = rng.choice(user_idx, size=target_spu, replace=False)
            keep_indices.extend(chosen)
    keep_indices = sorted(keep_indices)
    return X_train[keep_indices], y_train[keep_indices]


# ---------------------------------------------------------------------------
# Experiment 1: Per-scenario PCA component sweep
# ---------------------------------------------------------------------------

def run_scenario_sweep(df, scenario_id, seeds, progress_state):
    """Full K sweep for one scenario.

    Returns:
        list of dicts, one per (sub_exp, K, seed) combination, plus baseline rows.
    """
    scenario = get_scenario_by_id(scenario_id)
    spu = SCENARIO_SPU[scenario_id]
    rows = []

    for sub_idx, sub_exp in enumerate(scenario.sub_experiments):
        result = prepare_sub_experiment(df, sub_exp)
        if result is None:
            continue
        X_train, X_test, y_train_enc, y_test_enc, y_train_raw = result

        # Clean training data for PCA
        X_clean = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

        # Fit PCA once with all components
        pca, max_k = fit_pca_full(X_clean)
        k_values = generate_k_values(max_k)

        # Also compute 95% variance threshold K for reference
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        var95_k = int(np.searchsorted(cumvar, 0.95) + 1)
        var95_k = min(var95_k, max_k)

        for k in k_values:
            feature_indices = select_features_at_k(pca, k)
            n_features = len(feature_indices)

            for seed in seeds:
                top1, top3, top5 = run_rf_classify(
                    X_clean, np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0),
                    y_train_enc, y_test_enc, feature_indices, seed
                )
                rows.append({
                    'scenario_id': scenario_id,
                    'sub_experiment': sub_exp.name,
                    'K': k,
                    'n_features_selected': n_features,
                    'top1': top1,
                    'top3': top3,
                    'top5': top5,
                    'seed': seed,
                    'train_samples_per_user': spu,
                    'max_k': max_k,
                    'var95_k': var95_k,
                    'is_baseline': False,
                    'train_samples': len(X_train),
                })

                progress_state['count'] += 1
                _report_progress(progress_state)

        # Baseline: all features
        n_all_features = X_train.shape[1]
        for seed in seeds:
            top1, top3, top5 = run_rf_classify(
                X_clean, np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0),
                y_train_enc, y_test_enc, None, seed
            )
            rows.append({
                'scenario_id': scenario_id,
                'sub_experiment': sub_exp.name,
                'K': -1,  # sentinel for baseline
                'n_features_selected': n_all_features,
                'top1': top1,
                'top3': top3,
                'top5': top5,
                'seed': seed,
                'train_samples_per_user': spu,
                'max_k': max_k,
                'var95_k': var95_k,
                'is_baseline': True,
                'train_samples': len(X_train),
            })

            progress_state['count'] += 1
            _report_progress(progress_state)

        print(f"    [{sub_idx+1}/{len(scenario.sub_experiments)}] "
              f"{sub_exp.name}: max_k={max_k}, var95_k={var95_k}, "
              f"{len(k_values)} K values")

    return rows


# ---------------------------------------------------------------------------
# Experiment 2: Controlled subsampling
# ---------------------------------------------------------------------------

def run_subsampling_experiment(df, scenario_id, seeds, subsample_seeds,
                              progress_state):
    """Sweep at 3 data sizes (4/user, 2/user, 1/user) for 3.2 or 4.2.

    Test data is kept fixed. Only training data is subsampled.

    Returns:
        list of dicts with an additional 'subsample_seed' and
        'effective_spu' column.
    """
    scenario = get_scenario_by_id(scenario_id)
    native_spu = SCENARIO_SPU[scenario_id]  # Should be 4
    target_spus = [native_spu, 2, 1]  # 4, 2, 1
    rows = []

    for sub_idx, sub_exp in enumerate(scenario.sub_experiments):
        result = prepare_sub_experiment(df, sub_exp)
        if result is None:
            continue
        X_train_full, X_test, y_train_enc_full, y_test_enc, y_train_raw = result
        X_clean_full = np.nan_to_num(X_train_full, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_clean = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        for target_spu in target_spus:
            for ss_seed in subsample_seeds:
                # Subsample training data
                if target_spu == native_spu:
                    X_train = X_clean_full
                    y_train_enc = y_train_enc_full
                else:
                    rng = np.random.default_rng(ss_seed)
                    X_train, y_train_enc = subsample_training_data(
                        X_clean_full, y_train_enc_full, target_spu, rng
                    )

                # Skip if too few samples or classes
                if len(np.unique(y_train_enc)) < 2:
                    continue

                # Fit PCA on subsampled training data
                pca, max_k = fit_pca_full(X_train)
                k_values = generate_k_values(max_k)

                cumvar = np.cumsum(pca.explained_variance_ratio_)
                var95_k = int(np.searchsorted(cumvar, 0.95) + 1)
                var95_k = min(var95_k, max_k)

                for k in k_values:
                    feature_indices = select_features_at_k(pca, k)
                    n_features = len(feature_indices)

                    for seed in seeds:
                        top1, top3, top5 = run_rf_classify(
                            X_train, X_test_clean,
                            y_train_enc, y_test_enc,
                            feature_indices, seed
                        )
                        rows.append({
                            'scenario_id': scenario_id,
                            'sub_experiment': sub_exp.name,
                            'K': k,
                            'n_features_selected': n_features,
                            'top1': top1,
                            'top3': top3,
                            'top5': top5,
                            'seed': seed,
                            'subsample_seed': ss_seed,
                            'effective_spu': target_spu,
                            'max_k': max_k,
                            'var95_k': var95_k,
                            'is_baseline': False,
                            'train_samples': len(X_train),
                        })

                        progress_state['count'] += 1
                        _report_progress(progress_state)

                # Baseline at this data level
                n_all = X_train.shape[1]
                for seed in seeds:
                    top1, top3, top5 = run_rf_classify(
                        X_train, X_test_clean,
                        y_train_enc, y_test_enc, None, seed
                    )
                    rows.append({
                        'scenario_id': scenario_id,
                        'sub_experiment': sub_exp.name,
                        'K': -1,
                        'n_features_selected': n_all,
                        'top1': top1,
                        'top3': top3,
                        'top5': top5,
                        'seed': seed,
                        'subsample_seed': ss_seed,
                        'effective_spu': target_spu,
                        'max_k': max_k,
                        'var95_k': var95_k,
                        'is_baseline': True,
                        'train_samples': len(X_train),
                    })

                    progress_state['count'] += 1
                    _report_progress(progress_state)

        print(f"    [{sub_idx+1}/{len(scenario.sub_experiments)}] "
              f"{sub_exp.name}: subsampling done")

    return rows


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------

def _report_progress(state):
    """Print progress every 100 RF trainings with ETA."""
    count = state['count']
    if count % 100 == 0 and count > 0:
        elapsed = time.time() - state['start_time']
        rate = count / elapsed
        remaining = state['total'] - count
        eta_sec = remaining / rate if rate > 0 else 0
        eta_min = eta_sec / 60
        print(f"  Progress: {count}/{state['total']} RF trainings "
              f"({count/state['total']*100:.1f}%) | "
              f"Rate: {rate:.1f}/s | ETA: {eta_min:.1f} min")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_sweep_results(detailed_df):
    """Group by (scenario_id, K, train_samples_per_user), compute mean/std.

    Returns a summary DataFrame with mean and std for top1/top3/top5 and
    mean n_features_selected.
    """
    # Separate baseline and sweep rows
    sweep = detailed_df[~detailed_df['is_baseline']].copy()
    baseline = detailed_df[detailed_df['is_baseline']].copy()

    # Aggregate sweep
    group_cols = ['scenario_id', 'K', 'train_samples_per_user']
    agg = sweep.groupby(group_cols).agg(
        top1_mean=('top1', 'mean'),
        top1_std=('top1', 'std'),
        top3_mean=('top3', 'mean'),
        top3_std=('top3', 'std'),
        top5_mean=('top5', 'mean'),
        top5_std=('top5', 'std'),
        n_features_mean=('n_features_selected', 'mean'),
        n_features_std=('n_features_selected', 'std'),
        n_observations=('top1', 'count'),
    ).reset_index()

    # Aggregate baseline
    bl_group = ['scenario_id', 'train_samples_per_user']
    bl_agg = baseline.groupby(bl_group).agg(
        baseline_top1_mean=('top1', 'mean'),
        baseline_top1_std=('top1', 'std'),
        baseline_top3_mean=('top3', 'mean'),
        baseline_top5_mean=('top5', 'mean'),
    ).reset_index()

    # Merge baseline into summary
    summary = agg.merge(bl_agg, on=bl_group, how='left')

    return summary


def aggregate_subsample_results(detailed_df):
    """Group by (scenario_id, K, effective_spu), compute mean/std."""
    sweep = detailed_df[~detailed_df['is_baseline']].copy()
    baseline = detailed_df[detailed_df['is_baseline']].copy()

    group_cols = ['scenario_id', 'K', 'effective_spu']
    agg = sweep.groupby(group_cols).agg(
        top1_mean=('top1', 'mean'),
        top1_std=('top1', 'std'),
        top3_mean=('top3', 'mean'),
        top3_std=('top3', 'std'),
        top5_mean=('top5', 'mean'),
        top5_std=('top5', 'std'),
        n_features_mean=('n_features_selected', 'mean'),
        n_features_std=('n_features_selected', 'std'),
        n_observations=('top1', 'count'),
    ).reset_index()

    bl_group = ['scenario_id', 'effective_spu']
    bl_agg = baseline.groupby(bl_group).agg(
        baseline_top1_mean=('top1', 'mean'),
        baseline_top1_std=('top1', 'std'),
        baseline_top3_mean=('top3', 'mean'),
        baseline_top5_mean=('top5', 'mean'),
    ).reset_index()

    summary = agg.merge(bl_agg, on=bl_group, how='left')
    return summary


def build_convergence_table(summary_df):
    """Find K and feature count needed to reach 90%/95% of baseline per scenario.

    Returns a DataFrame with one row per scenario.
    """
    convergence_rows = []
    for scenario_id in sorted(summary_df['scenario_id'].unique()):
        sc = summary_df[summary_df['scenario_id'] == scenario_id]
        spu = sc['train_samples_per_user'].iloc[0]
        bl_top1 = sc['baseline_top1_mean'].iloc[0]

        for pct_label, pct in [('90%', 0.90), ('95%', 0.95)]:
            target = bl_top1 * pct
            above = sc[sc['top1_mean'] >= target].sort_values('K')
            if len(above) > 0:
                first = above.iloc[0]
                convergence_rows.append({
                    'scenario_id': scenario_id,
                    'train_samples_per_user': spu,
                    'threshold': pct_label,
                    'baseline_top1': bl_top1,
                    'target_accuracy': target,
                    'K_needed': int(first['K']),
                    'n_features_at_K': int(first['n_features_mean']),
                    'accuracy_at_K': first['top1_mean'],
                })
            else:
                convergence_rows.append({
                    'scenario_id': scenario_id,
                    'train_samples_per_user': spu,
                    'threshold': pct_label,
                    'baseline_top1': bl_top1,
                    'target_accuracy': target,
                    'K_needed': np.nan,
                    'n_features_at_K': np.nan,
                    'accuracy_at_K': np.nan,
                })

    return pd.DataFrame(convergence_rows)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

# Scenario-specific colors for consistent plotting
SCENARIO_COLORS = {
    "1.1": "#1f77b4", "1.2": "#aec7e8",
    "2.1": "#ff7f0e", "2.2": "#ffbb78",
    "3.1": "#2ca02c", "3.2": "#98df8a",
    "4.1": "#d62728", "4.2": "#ff9896",
    "5.1": "#9467bd", "5.2": "#c5b0d5",
}


def plot_accuracy_vs_k(summary_df, output_path):
    """Plot A: 1x3 panels by data-size group, lines per scenario, x=K, y=top-1."""
    spu_groups = sorted(summary_df['train_samples_per_user'].unique())
    n_panels = len(spu_groups)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5), squeeze=False)
    axes = axes[0]

    for ax_idx, spu in enumerate(spu_groups):
        ax = axes[ax_idx]
        panel = summary_df[summary_df['train_samples_per_user'] == spu]

        for scenario_id in sorted(panel['scenario_id'].unique()):
            sc = panel[panel['scenario_id'] == scenario_id].sort_values('K')
            color = SCENARIO_COLORS.get(scenario_id, 'gray')

            ax.plot(sc['K'], sc['top1_mean'], label=scenario_id,
                    color=color, linewidth=1.5)
            ax.fill_between(sc['K'],
                            sc['top1_mean'] - sc['top1_std'],
                            sc['top1_mean'] + sc['top1_std'],
                            alpha=0.15, color=color)

            # Baseline dashed line
            bl = sc['baseline_top1_mean'].iloc[0]
            ax.axhline(bl, color=color, linestyle='--', alpha=0.5, linewidth=0.8)

        # Vertical markers at 95% variance threshold K values
        var95_ks = panel.groupby('scenario_id').apply(
            lambda g: g.iloc[0] if len(g) > 0 else None
        )
        # Gather unique var95_k values from the detailed data if available
        ax.set_xlabel('Number of PCA Components (K)')
        ax.set_ylabel('Mean Top-1 Accuracy')
        ax.set_title(f'{SPU_GROUP_LABELS.get(spu, f"{spu} spu")}')
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    fig.suptitle('Accuracy vs. Number of PCA Components', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Plot A saved: {output_path}")


def plot_accuracy_vs_nfeatures(summary_df, output_path):
    """Plot B: Same as A but x-axis = number of features selected."""
    spu_groups = sorted(summary_df['train_samples_per_user'].unique())
    n_panels = len(spu_groups)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5), squeeze=False)
    axes = axes[0]

    for ax_idx, spu in enumerate(spu_groups):
        ax = axes[ax_idx]
        panel = summary_df[summary_df['train_samples_per_user'] == spu]

        for scenario_id in sorted(panel['scenario_id'].unique()):
            sc = panel[panel['scenario_id'] == scenario_id].sort_values('n_features_mean')
            color = SCENARIO_COLORS.get(scenario_id, 'gray')

            ax.plot(sc['n_features_mean'], sc['top1_mean'], label=scenario_id,
                    color=color, linewidth=1.5)
            ax.fill_between(sc['n_features_mean'],
                            sc['top1_mean'] - sc['top1_std'],
                            sc['top1_mean'] + sc['top1_std'],
                            alpha=0.15, color=color)

            bl = sc['baseline_top1_mean'].iloc[0]
            ax.axhline(bl, color=color, linestyle='--', alpha=0.5, linewidth=0.8)

        ax.set_xlabel('Number of Selected Original Features')
        ax.set_ylabel('Mean Top-1 Accuracy')
        ax.set_title(f'{SPU_GROUP_LABELS.get(spu, f"{spu} spu")}')
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    fig.suptitle('Accuracy vs. Number of Selected Features', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Plot B saved: {output_path}")


def plot_convergence_analysis(convergence_df, output_path):
    """Plot C: Bar chart of K needed to reach 90%/95% of baseline per scenario."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, threshold in enumerate(['90%', '95%']):
        ax = axes[ax_idx]
        ct = convergence_df[convergence_df['threshold'] == threshold].copy()
        ct = ct.sort_values('scenario_id')

        scenarios = ct['scenario_id'].values
        x = np.arange(len(scenarios))
        colors = [SCENARIO_COLORS.get(s, 'gray') for s in scenarios]

        k_vals = ct['K_needed'].values
        bars = ax.bar(x, k_vals, color=colors, edgecolor='black', linewidth=0.5)

        # Value labels
        for bar, val in zip(bars, k_vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f'{int(val)}', ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Scenario')
        ax.set_ylabel('K (PCA Components) Needed')
        ax.set_title(f'Components to Reach {threshold} of Baseline Accuracy')
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Plot C saved: {output_path}")


def plot_subsampling_comparison(subsample_summary, output_path):
    """Plot D: 1x2 panels (3.2, 4.2) with 3 lines each (4/2/1 spu)."""
    scenarios = sorted(subsample_summary['scenario_id'].unique())
    n_panels = len(scenarios)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5), squeeze=False)
    axes = axes[0]

    spu_colors = {4: '#2ca02c', 2: '#ff7f0e', 1: '#1f77b4'}
    spu_styles = {4: '-', 2: '--', 1: ':'}

    for ax_idx, scenario_id in enumerate(scenarios):
        ax = axes[ax_idx]
        panel = subsample_summary[subsample_summary['scenario_id'] == scenario_id]

        for eff_spu in sorted(panel['effective_spu'].unique(), reverse=True):
            sc = panel[panel['effective_spu'] == eff_spu].sort_values('K')
            color = spu_colors.get(eff_spu, 'gray')
            style = spu_styles.get(eff_spu, '-')

            ax.plot(sc['K'], sc['top1_mean'],
                    label=f'{eff_spu}/user (~{sc["n_observations"].iloc[0]//3} sub-exp obs)',
                    color=color, linestyle=style, linewidth=1.5)
            ax.fill_between(sc['K'],
                            sc['top1_mean'] - sc['top1_std'],
                            sc['top1_mean'] + sc['top1_std'],
                            alpha=0.12, color=color)

            # Baseline
            bl = sc['baseline_top1_mean'].iloc[0]
            ax.axhline(bl, color=color, linestyle='--', alpha=0.4, linewidth=0.8)

        ax.set_xlabel('Number of PCA Components (K)')
        ax.set_ylabel('Mean Top-1 Accuracy')
        ax.set_title(f'Scenario {scenario_id}')
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    fig.suptitle('Subsampling Comparison: Training Data Size Effect on PCA Sweep',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Plot D saved: {output_path}")


# ---------------------------------------------------------------------------
# Dry run estimation
# ---------------------------------------------------------------------------

def estimate_rf_trainings(scenarios_to_run, skip_subsampling):
    """Estimate the total number of RF trainings for progress tracking."""
    n_seeds = 3
    n_subsample_seeds = 3
    total_exp1 = 0
    total_exp2 = 0

    for scenario_id in scenarios_to_run:
        scenario = get_scenario_by_id(scenario_id)
        n_sub = len(scenario.sub_experiments)
        # Estimate max_k ~ 60 for most scenarios (min of samples, features)
        # K sweep: 30 (1-30) + ~6 (35,40,...,60) + 1 baseline = ~37 per sub-exp
        est_k_values = 37
        total_exp1 += n_sub * est_k_values * n_seeds

    if not skip_subsampling:
        for scenario_id in ['3.2', '4.2']:
            if scenario_id in scenarios_to_run:
                scenario = get_scenario_by_id(scenario_id)
                n_sub = len(scenario.sub_experiments)
                # 3 data levels x 3 subsample seeds x ~37 K values x 3 RF seeds
                # But native spu only uses 1 subsample seed effectively
                n_levels = 3  # 4/user, 2/user, 1/user
                total_exp2 += n_sub * n_levels * n_subsample_seeds * est_k_values * n_seeds

    return total_exp1, total_exp2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='PCA Component Sweep: Training Data Size vs. Feature Count'
    )
    parser.add_argument('-c', '--config', default='config_rf_only.json',
                        help='Config JSON (default: config_rf_only.json)')
    parser.add_argument('--skip-subsampling', action='store_true',
                        help='Skip Experiment 2 (subsampling)')
    parser.add_argument('--scenarios', nargs='+', default=None,
                        help='Run specific scenarios (e.g., --scenarios 1.1 3.2)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Estimate RF trainings and exit')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    dataset_path = config['dataset_path']
    seeds = config.get('seeds', [42, 123, 456])
    subsample_seeds = [0, 7, 13]

    all_scenario_ids = ["1.1", "1.2", "2.1", "2.2", "3.1", "3.2",
                        "4.1", "4.2", "5.1", "5.2"]
    scenarios_to_run = args.scenarios if args.scenarios else all_scenario_ids

    # Validate scenario IDs
    for sid in scenarios_to_run:
        if sid not in all_scenario_ids:
            print(f"Error: Unknown scenario '{sid}'. Valid: {all_scenario_ids}")
            sys.exit(1)

    # Dry run
    est_exp1, est_exp2 = estimate_rf_trainings(scenarios_to_run, args.skip_subsampling)
    total_est = est_exp1 + est_exp2
    print(f"PCA Component Sweep")
    print(f"  Scenarios: {scenarios_to_run}")
    print(f"  RF seeds: {seeds}")
    print(f"  Subsample seeds: {subsample_seeds}")
    print(f"  Estimated RF trainings:")
    print(f"    Experiment 1 (sweep): ~{est_exp1:,}")
    if not args.skip_subsampling:
        print(f"    Experiment 2 (subsampling): ~{est_exp2:,}")
    print(f"    Total: ~{total_est:,}")

    if args.dry_run:
        print("\n[Dry run] Exiting.")
        return

    # Timestamp and output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = Path(f"pca_component_sweep_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Load dataset
    print(f"\nLoading dataset: {dataset_path}")
    df_pl = pl.read_csv(dataset_path)
    df = df_pl.to_pandas()
    feature_cols = get_feature_columns(df.columns.tolist())
    print(f"Dataset: {df.shape[0]} rows, {len(feature_cols)} features, "
          f"{df['user_id'].nunique()} users")

    # =====================================================================
    # Experiment 1: Per-scenario PCA component sweep
    # =====================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Per-Scenario PCA Component Sweep")
    print("=" * 70)

    progress = {
        'count': 0,
        'total': total_est,
        'start_time': time.time(),
    }

    all_sweep_rows = []
    for scenario_id in scenarios_to_run:
        scenario = get_scenario_by_id(scenario_id)
        print(f"\n  Scenario {scenario_id}: {scenario.name} "
              f"({len(scenario.sub_experiments)} sub-exps, "
              f"{SCENARIO_SPU[scenario_id]} spu)")
        rows = run_scenario_sweep(df, scenario_id, seeds, progress)
        all_sweep_rows.extend(rows)
        print(f"  -> {len(rows)} result rows")

    sweep_detailed = pd.DataFrame(all_sweep_rows)

    # Aggregate
    sweep_summary = aggregate_sweep_results(sweep_detailed)
    convergence = build_convergence_table(sweep_summary)

    # =====================================================================
    # Experiment 2: Subsampling (3.2 and 4.2 only)
    # =====================================================================
    subsample_detailed = None
    subsample_summary = None

    subsample_candidates = [s for s in ['3.2', '4.2'] if s in scenarios_to_run]

    if not args.skip_subsampling and subsample_candidates:
        print("\n" + "=" * 70)
        print("EXPERIMENT 2: Controlled Subsampling")
        print("=" * 70)

        all_subsample_rows = []
        for scenario_id in subsample_candidates:
            scenario = get_scenario_by_id(scenario_id)
            print(f"\n  Scenario {scenario_id}: {scenario.name} "
                  f"({len(scenario.sub_experiments)} sub-exps)")
            rows = run_subsampling_experiment(
                df, scenario_id, seeds, subsample_seeds, progress
            )
            all_subsample_rows.extend(rows)
            print(f"  -> {len(rows)} result rows")

        subsample_detailed = pd.DataFrame(all_subsample_rows)
        subsample_summary = aggregate_subsample_results(subsample_detailed)
    elif args.skip_subsampling:
        print("\n[Skipping Experiment 2 as requested]")
    else:
        print("\n[No 3.2/4.2 in selected scenarios, skipping Experiment 2]")

    # =====================================================================
    # Save outputs
    # =====================================================================
    print("\n" + "=" * 70)
    print("SAVING OUTPUTS")
    print("=" * 70)

    ts = timestamp

    # Detailed sweep CSV
    sweep_det_path = output_dir / f"sweep_detailed_{ts}.csv"
    sweep_detailed.to_csv(sweep_det_path, index=False)
    print(f"  {sweep_det_path} ({len(sweep_detailed)} rows)")

    # Summary sweep CSV
    sweep_sum_path = output_dir / f"sweep_summary_{ts}.csv"
    sweep_summary.to_csv(sweep_sum_path, index=False)
    print(f"  {sweep_sum_path} ({len(sweep_summary)} rows)")

    # Convergence table
    conv_path = output_dir / f"convergence_table_{ts}.csv"
    convergence.to_csv(conv_path, index=False)
    print(f"  {conv_path}")

    # Subsampling CSVs
    if subsample_detailed is not None:
        ss_det_path = output_dir / f"subsample_detailed_{ts}.csv"
        subsample_detailed.to_csv(ss_det_path, index=False)
        print(f"  {ss_det_path} ({len(subsample_detailed)} rows)")

    # Config dump
    config_out = {
        'timestamp': timestamp,
        'dataset_path': dataset_path,
        'seeds': seeds,
        'subsample_seeds': subsample_seeds,
        'scenarios_to_run': scenarios_to_run,
        'skip_subsampling': args.skip_subsampling,
        'rf_hyperparameters': {
            'n_estimators': 300,
            'max_depth': 20,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'bootstrap': True,
        },
        'k_sweep': 'dense 1-30, then every 5 to max_k',
        'feature_selection': 'above-mean |loading| per component, union across K',
        'n_features_total': len(feature_cols),
        'scenario_spu': SCENARIO_SPU,
        'estimated_trainings': total_est,
        'actual_trainings': progress['count'],
        'runtime_seconds': time.time() - progress['start_time'],
    }
    config_path = output_dir / f"sweep_config_{ts}.json"
    with open(config_path, 'w') as f:
        json.dump(config_out, f, indent=2)
    print(f"  {config_path}")

    # =====================================================================
    # Plots
    # =====================================================================
    print("\n--- Generating plots ---")

    # Plot A: Accuracy vs K
    plot_a_path = output_dir / f"accuracy_vs_k_{ts}.png"
    plot_accuracy_vs_k(sweep_summary, plot_a_path)

    # Plot B: Accuracy vs number of features
    plot_b_path = output_dir / f"accuracy_vs_nfeatures_{ts}.png"
    plot_accuracy_vs_nfeatures(sweep_summary, plot_b_path)

    # Plot C: Convergence analysis
    if len(convergence.dropna(subset=['K_needed'])) > 0:
        plot_c_path = output_dir / f"convergence_analysis_{ts}.png"
        plot_convergence_analysis(convergence, plot_c_path)
    else:
        print("  [Skipping Plot C: no convergence data]")

    # Plot D: Subsampling comparison
    if subsample_summary is not None and len(subsample_summary) > 0:
        plot_d_path = output_dir / f"subsampling_comparison_{ts}.png"
        plot_subsampling_comparison(subsample_summary, plot_d_path)
    else:
        print("  [Skipping Plot D: no subsampling data]")

    # =====================================================================
    # Console summary
    # =====================================================================
    elapsed = time.time() - progress['start_time']
    print(f"\n{'=' * 70}")
    print(f"DONE  |  {progress['count']} RF trainings in {elapsed/60:.1f} min")
    print(f"All outputs saved to: {output_dir}/")
    print(f"{'=' * 70}")

    # Print convergence highlights
    print("\nConvergence highlights (K needed for 95% of baseline):")
    c95 = convergence[convergence['threshold'] == '95%']
    for _, row in c95.iterrows():
        k_str = f"K={int(row['K_needed'])}" if not np.isnan(row['K_needed']) else "N/A"
        nf_str = f"{int(row['n_features_at_K'])} feats" if not np.isnan(row['n_features_at_K']) else ""
        print(f"  {row['scenario_id']}: {k_str} ({nf_str}) "
              f"[baseline={row['baseline_top1']:.4f}]")


if __name__ == "__main__":
    main()
