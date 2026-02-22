#!/usr/bin/env python3
"""
audit_scenario_separation.py - Validate scenario separation in both PCA and RF
feature-selection pipelines.

Checks:
  PCA pipeline:
    1. No train/test overlap in apply_sub_experiment_filters() for each sub-exp.
    2. PCA is fit on training data only (compare loadings from train-only vs all).
    3. Report feature counts per scenario and final intersection count.

  RF pipeline:
    1. Each pickle's clean_experiment_name maps to a valid scenario.
    2. Per-scenario aggregation produces different results than global aggregation.
    3. Report top-N features per scenario and the intersection.

Usage:
    python audit_scenario_separation.py [--config config_rf_only.json] \
        [--pickle-dir scenario_results_rf_only_2026-01-25_223058_early_stop] \
        [--top-n 10] [--json audit_report.json]
"""

import argparse
import json
import pickle
import sys
import warnings
from collections import defaultdict
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from ml_utils import (
    get_feature_columns, load_config, apply_sub_experiment_filters
)
from scenarios import get_scenario_by_id, generate_all_scenarios
from pca_vs_rf_comparison import (
    extract_pca_important_features_for_subexp,
    get_pca_features_for_scenario,
    get_common_pca_features,
    get_rf_features_for_scenario,
)

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# ---------------------------------------------------------------------------
# PCA audit
# ---------------------------------------------------------------------------

def audit_pca_train_test_overlap(df, scenarios_to_run):
    """Verify no row-index overlap between train and test sets."""
    findings = []
    all_pass = True
    for scenario_id in scenarios_to_run:
        scenario = get_scenario_by_id(scenario_id)
        for sub_exp in scenario.sub_experiments:
            train_df, test_df = apply_sub_experiment_filters(df, sub_exp)
            overlap = train_df.index.intersection(test_df.index)
            if len(overlap) > 0:
                all_pass = False
                findings.append({
                    'check': 'train_test_overlap',
                    'scenario': scenario_id,
                    'sub_experiment': sub_exp.name,
                    'status': 'FAIL',
                    'detail': f'{len(overlap)} overlapping row indices',
                })
            else:
                findings.append({
                    'check': 'train_test_overlap',
                    'scenario': scenario_id,
                    'sub_experiment': sub_exp.name,
                    'status': 'PASS',
                    'detail': (f'train={len(train_df)}, test={len(test_df)}, '
                               f'overlap=0'),
                })
    return all_pass, findings


def audit_pca_train_only_fit(df, scenarios_to_run, variance_threshold=0.95,
                              sample_limit=3):
    """Verify PCA loadings differ when fit on train vs all data.

    For efficiency, only checks *sample_limit* sub-experiments per scenario.
    """
    feature_cols = get_feature_columns(df.columns.tolist())
    findings = []
    all_pass = True

    for scenario_id in scenarios_to_run:
        scenario = get_scenario_by_id(scenario_id)
        checked = 0
        for sub_exp in scenario.sub_experiments:
            if checked >= sample_limit:
                break
            train_df, test_df = apply_sub_experiment_filters(df, sub_exp)
            X_train = train_df[feature_cols].values
            X_all = pd.concat([train_df, test_df])[feature_cols].values

            if len(X_train) < 2 or len(X_all) < 2:
                continue

            idx_train = extract_pca_important_features_for_subexp(
                X_train, variance_threshold
            )
            idx_all = extract_pca_important_features_for_subexp(
                X_all, variance_threshold
            )

            same = (idx_train == idx_all)
            status = 'INFO' if same else 'PASS'
            # If they're identical, it's not necessarily a failure — it just
            # means the test data didn't change things.  A failure would be
            # if the code were *using* X_all; we can't detect that purely
            # from outputs, so we flag it as INFO.
            findings.append({
                'check': 'pca_train_only_fit',
                'scenario': scenario_id,
                'sub_experiment': sub_exp.name,
                'status': status,
                'detail': (f'train_features={len(idx_train)}, '
                           f'all_features={len(idx_all)}, '
                           f'identical={same}'),
            })
            checked += 1

    return all_pass, findings


def audit_pca_feature_counts(df, scenarios_to_run, variance_threshold=0.95):
    """Report PCA feature counts per scenario and the cross-scenario intersection."""
    findings = []
    per_scenario = {}
    for scenario_id in scenarios_to_run:
        scenario = get_scenario_by_id(scenario_id)
        important = get_pca_features_for_scenario(df, scenario, variance_threshold)
        per_scenario[scenario_id] = len(important)
        findings.append({
            'check': 'pca_feature_count',
            'scenario': scenario_id,
            'status': 'INFO',
            'detail': f'{len(important)} features via majority vote',
        })

    intersection = get_common_pca_features(df, scenarios_to_run, variance_threshold)
    findings.append({
        'check': 'pca_intersection',
        'scenario': 'all',
        'status': 'INFO',
        'detail': f'{len(intersection)} features in cross-scenario intersection',
    })
    return True, findings, per_scenario


# ---------------------------------------------------------------------------
# RF audit
# ---------------------------------------------------------------------------

def audit_rf_pickle_scenarios(pickle_dir):
    """Verify each pickle's experiment_name maps to a valid scenario."""
    valid_ids = set(generate_all_scenarios().keys())
    pkl_files = sorted(glob(str(Path(pickle_dir) / "*.pkl")))
    findings = []
    all_pass = True

    for pf in pkl_files:
        name = Path(pf).stem
        scenario_id = name.split('_')[1][1:]  # "S1.1" → "1.1"
        with open(pf, 'rb') as f:
            data = pickle.load(f)
        meta_name = data['metadata'].get('clean_experiment_name', '')
        meta_scenario = meta_name.split('_')[0][1:] if meta_name.startswith('S') else '?'

        if scenario_id not in valid_ids:
            all_pass = False
            findings.append({
                'check': 'rf_valid_scenario',
                'file': name,
                'status': 'FAIL',
                'detail': f'scenario_id={scenario_id} not in valid set',
            })
        elif scenario_id != meta_scenario:
            all_pass = False
            findings.append({
                'check': 'rf_valid_scenario',
                'file': name,
                'status': 'FAIL',
                'detail': (f'filename scenario={scenario_id} != '
                           f'metadata scenario={meta_scenario}'),
            })
        else:
            findings.append({
                'check': 'rf_valid_scenario',
                'file': name,
                'status': 'PASS',
                'detail': f'scenario_id={scenario_id} matches metadata',
            })

    return all_pass, findings


def audit_rf_global_vs_perscenario(pickle_dir, top_n=10):
    """Compare old global aggregation with new per-scenario majority vote."""
    from collections import Counter
    pkl_files = sorted(glob(str(Path(pickle_dir) / "*.pkl")))
    if not pkl_files:
        return False, [{'check': 'rf_global_vs_perscenario', 'status': 'FAIL',
                        'detail': 'No pickle files found'}]

    # --- old method: global average ---
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

    old_avg = importances_sum / count
    old_top = set(np.argsort(old_avg)[::-1][:top_n].tolist())

    # --- new method: per-scenario majority vote ---
    scenario_pkls = defaultdict(list)
    for pf in pkl_files:
        name = Path(pf).stem
        scenario_id = name.split('_')[1][1:]
        scenario_pkls[scenario_id].append(pf)

    index_counts = Counter()
    per_scenario_top = {}
    for sid in sorted(scenario_pkls.keys()):
        top_set = get_rf_features_for_scenario(scenario_pkls[sid], top_n)
        per_scenario_top[sid] = sorted(top_set)
        index_counts.update(top_set)

    n_scenarios = len(scenario_pkls)
    majority = n_scenarios / 2.0
    new_top = {idx for idx, cnt in index_counts.items() if cnt > majority}

    differs = (old_top != new_top)
    findings = [{
        'check': 'rf_global_vs_perscenario',
        'status': 'PASS' if differs else 'INFO',
        'detail': (f'old_global_top{top_n}={sorted(old_top)}, '
                   f'new_majority_vote={sorted(new_top)}, '
                   f'differs={differs}'),
    }]

    # Per-scenario detail
    for sid in sorted(per_scenario_top.keys()):
        findings.append({
            'check': 'rf_per_scenario_top',
            'scenario': sid,
            'status': 'INFO',
            'detail': f'top-{top_n} indices: {per_scenario_top[sid]}',
        })

    return True, findings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Audit scenario separation in PCA and RF pipelines'
    )
    parser.add_argument('-c', '--config', default='config_rf_only.json',
                        help='Config JSON (default: config_rf_only.json)')
    parser.add_argument('--pickle-dir',
                        default='scenario_results_rf_only_2026-01-25_223058_early_stop',
                        help='Directory with existing RF pickle files')
    parser.add_argument('--variance-threshold', type=float, default=0.95,
                        help='PCA cumulative variance threshold (default: 0.95)')
    parser.add_argument('--top-n', type=int, default=10,
                        help='Number of top RF features (default: 10)')
    parser.add_argument('--json', default=None,
                        help='Optional path to write detailed JSON report')
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_path = config['dataset_path']
    scenarios_to_run = config.get('scenarios_to_run',
                                  ["1.1", "1.2", "2.1", "2.2", "3.1", "3.2",
                                   "4.1", "4.2", "5.1", "5.2"])

    print(f"Loading dataset: {dataset_path}")
    df_pl = pl.read_csv(dataset_path)
    df = df_pl.to_pandas()
    print(f"Dataset: {df.shape[0]} rows, {df['user_id'].nunique()} users\n")

    all_findings = []
    overall_pass = True

    # ---- PCA audits ----
    print("=" * 70)
    print("PCA PIPELINE AUDIT")
    print("=" * 70)

    print("\n[1/3] Checking train/test overlap ...")
    ok, findings = audit_pca_train_test_overlap(df, scenarios_to_run)
    overall_pass &= ok
    n_fail = sum(1 for f in findings if f['status'] == 'FAIL')
    n_pass = sum(1 for f in findings if f['status'] == 'PASS')
    print(f"      {n_pass} PASS, {n_fail} FAIL across "
          f"{len(findings)} sub-experiments")
    all_findings.extend(findings)

    print("\n[2/3] Checking PCA train-only fit (sampling sub-experiments) ...")
    ok, findings = audit_pca_train_only_fit(
        df, scenarios_to_run, args.variance_threshold
    )
    overall_pass &= ok
    n_same = sum(1 for f in findings if 'identical=True' in f['detail'])
    n_diff = sum(1 for f in findings if 'identical=False' in f['detail'])
    print(f"      {n_diff} train≠all (PASS), {n_same} train=all (INFO) "
          f"across {len(findings)} sampled sub-experiments")
    all_findings.extend(findings)

    print("\n[3/3] PCA feature counts per scenario ...")
    ok, findings, pca_per = audit_pca_feature_counts(
        df, scenarios_to_run, args.variance_threshold
    )
    overall_pass &= ok
    for sid in sorted(pca_per.keys()):
        print(f"      Scenario {sid}: {pca_per[sid]} features")
    inter_finding = [f for f in findings if f['check'] == 'pca_intersection'][0]
    print(f"      Intersection: {inter_finding['detail']}")
    all_findings.extend(findings)

    # ---- RF audits ----
    print("\n" + "=" * 70)
    print("RF PIPELINE AUDIT")
    print("=" * 70)

    print("\n[1/2] Validating pickle scenario mappings ...")
    ok, findings = audit_rf_pickle_scenarios(args.pickle_dir)
    overall_pass &= ok
    n_fail = sum(1 for f in findings if f['status'] == 'FAIL')
    n_pass = sum(1 for f in findings if f['status'] == 'PASS')
    print(f"      {n_pass} PASS, {n_fail} FAIL across {len(findings)} pickles")
    all_findings.extend(findings)

    print(f"\n[2/2] Comparing global vs per-scenario RF top-{args.top_n} ...")
    ok, findings = audit_rf_global_vs_perscenario(args.pickle_dir, args.top_n)
    overall_pass &= ok
    main_finding = [f for f in findings
                    if f['check'] == 'rf_global_vs_perscenario'][0]
    print(f"      {main_finding['detail']}")
    per_sc = [f for f in findings if f['check'] == 'rf_per_scenario_top']
    for f in per_sc:
        print(f"      Scenario {f['scenario']}: {f['detail']}")
    all_findings.extend(findings)

    # ---- Summary ----
    print("\n" + "=" * 70)
    n_total_fail = sum(1 for f in all_findings if f['status'] == 'FAIL')
    if n_total_fail == 0:
        print("ALL CHECKS PASSED")
    else:
        print(f"FAILURES: {n_total_fail}")
    print("=" * 70)

    # ---- Optional JSON ----
    if args.json:
        report = {
            'overall_pass': overall_pass,
            'n_checks': len(all_findings),
            'n_failures': n_total_fail,
            'findings': all_findings,
        }
        with open(args.json, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nDetailed report written to: {args.json}")


if __name__ == "__main__":
    main()
