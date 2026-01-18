#!/usr/bin/env python3
"""
Score-level fusion of 4 model results using Random Forest weighted averaging.
Combines k=1 through k=5 accuracy values from per-model TSV files.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path

# Paths
RESULTS_DIR = Path("scenario_results_scenarios_2025-12-29_113757_early_stop")
MODELS = ["RandomForest", "CatBoost", "NaiveBayes", "ExtraTrees"]
K_COLS = ['k=1', 'k=2', 'k=3', 'k=4', 'k=5']

def load_model_tsv(model_name):
    """Load a per-model TSV file."""
    path = RESULTS_DIR / f"ml_results_{model_name}_2025-12-29_113757.tsv"
    return pd.read_csv(path, sep='\t')

def extract_data_rows(df):
    """Extract only data rows (not mean/std/empty rows)."""
    # Data rows have non-empty Train column that isn't 'mean' or 'std'
    mask = df['Train'].notna() & ~df['Train'].isin(['mean', 'std', ''])
    return df[mask].copy()

def train_rf_for_weights(model_dfs):
    """
    Train RF regressor to determine model weights.
    Features: k values from each model
    Target: max value across models (best achievable)
    """
    # Collect training data
    X_list = []
    y_list = []

    # Get data rows from first model as reference
    ref_df = extract_data_rows(model_dfs[MODELS[0]])
    n_rows = len(ref_df)

    for row_idx in range(n_rows):
        for k_col in K_COLS:
            # Features: k value from each model for this row
            features = []
            for model in MODELS:
                data_df = extract_data_rows(model_dfs[model])
                val = data_df.iloc[row_idx][k_col]
                if isinstance(val, str):
                    val = float(val)
                features.append(val)

            # Target: max value (best achievable)
            target = max(features)

            X_list.append(features)
            y_list.append(target)

    X = np.array(X_list)
    y = np.array(y_list)

    print(f"Training RF on {len(X)} samples...")

    # Train RF
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # Get feature importances as weights
    weights = rf.feature_importances_

    print("\nModel Weights (RF Feature Importances):")
    for model, weight in zip(MODELS, weights):
        print(f"  {model}: {weight:.4f}")

    return weights

def apply_weighted_fusion(model_dfs, weights):
    """Apply weighted average fusion using RF-derived weights."""
    # Normalize weights
    weights = np.array(weights) / np.sum(weights)

    # Use first model's TSV as template
    template = model_dfs[MODELS[0]].copy()

    # Get data row indices
    data_mask = template['Train'].notna() & ~template['Train'].isin(['mean', 'std', ''])

    # Process each data row
    for idx in template[data_mask].index:
        for k_col in K_COLS:
            # Get values from all models
            values = []
            for model in MODELS:
                val = model_dfs[model].loc[idx, k_col]
                if isinstance(val, str):
                    val = float(val)
                values.append(val)

            # Compute weighted average
            fused_val = np.sum(np.array(values) * weights)
            template.loc[idx, k_col] = f"{fused_val:.4f}"

    return template

def recalculate_stats(df):
    """Recalculate mean and std rows based on fused values."""
    result_rows = []
    current_scenario_data = []

    for idx, row in df.iterrows():
        train_val = row['Train'] if pd.notna(row['Train']) else ''

        if train_val == 'mean':
            # Calculate mean from accumulated data
            if current_scenario_data:
                k_arrays = np.array(current_scenario_data)
                means = np.mean(k_arrays, axis=0)
                new_row = row.copy()
                for i, k_col in enumerate(K_COLS):
                    new_row[k_col] = f"{means[i]:.4f}"
                result_rows.append(new_row)
            else:
                result_rows.append(row)

        elif train_val == 'std':
            # Calculate std from accumulated data
            if current_scenario_data:
                k_arrays = np.array(current_scenario_data)
                stds = np.std(k_arrays, axis=0)
                new_row = row.copy()
                for i, k_col in enumerate(K_COLS):
                    new_row[k_col] = f"{stds[i]:.4f}"
                result_rows.append(new_row)
            current_scenario_data = []  # Reset for next scenario

        elif train_val == '':
            # Empty separator row
            result_rows.append(row)

        else:
            # Data row - accumulate for stats
            k_values = []
            for k_col in K_COLS:
                val = row[k_col]
                if isinstance(val, str):
                    val = float(val)
                k_values.append(val)
            current_scenario_data.append(k_values)
            result_rows.append(row)

    return pd.DataFrame(result_rows)

def main():
    print("Loading per-model TSV files...")
    model_dfs = {}
    for model in MODELS:
        model_dfs[model] = load_model_tsv(model)
        print(f"  Loaded {model}: {len(model_dfs[model])} rows")

    print("\nTraining Random Forest to determine model weights...")
    weights = train_rf_for_weights(model_dfs)

    print("\nApplying weighted fusion...")
    fused_df = apply_weighted_fusion(model_dfs, weights)

    print("Recalculating mean/std rows...")
    fused_df = recalculate_stats(fused_df)

    # Save output
    output_path = RESULTS_DIR / "ml_results_RF_Fusion_2025-12-29_113757.tsv"
    fused_df.to_csv(output_path, sep='\t', index=False)
    print(f"\nSaved fused results to: {output_path}")

    # Print sample comparison
    print("\nSample comparison (Scenario 1.1, first row):")
    data_mask = fused_df['Train'].notna() & ~fused_df['Train'].isin(['mean', 'std', ''])
    first_data_row = fused_df[data_mask].iloc[0]
    print(f"  Fused: k=1={first_data_row['k=1']}, k=5={first_data_row['k=5']}")

    for model in MODELS:
        model_data = extract_data_rows(model_dfs[model]).iloc[0]
        print(f"  {model}: k=1={model_data['k=1']}, k=5={model_data['k=5']}")

if __name__ == '__main__':
    main()
