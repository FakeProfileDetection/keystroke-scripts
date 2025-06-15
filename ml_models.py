import os
import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
    confusion_matrix,
)
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt
import seaborn as sns
import bob.measure
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Global configuration and results storage
with open(os.path.join(os.getcwd(), "classifier_config.json"), "r") as f:
    config = json.load(f)

# Global variables that will be set by the main script
EARLY_STOP = False
NUM_SEEDS = 1

# Global results storage
EXPERIMENT_RESULTS = []
DETAILED_RESULTS = []
TIMESTAMP = datetime.now().isoformat().replace(":", "-").replace(".", "-")[:19]
EARLY_STOP_SUFFIX = "_early_stop" if EARLY_STOP else ""
OUTPUT_DIR = Path(f"experiment_results_{TIMESTAMP}{EARLY_STOP_SUFFIX}")
OUTPUT_DIR.mkdir(exist_ok=True)

# Progress tracking
TOTAL_EXPERIMENTS = 0
CURRENT_EXPERIMENT = 0


def set_global_params(early_stop=False, num_seeds=1):
    """Set global parameters from the main script"""
    global EARLY_STOP, NUM_SEEDS, EARLY_STOP_SUFFIX, OUTPUT_DIR
    EARLY_STOP = early_stop
    NUM_SEEDS = num_seeds
    # Update the suffix and output directory
    EARLY_STOP_SUFFIX = "_early_stop" if EARLY_STOP else ""
    OUTPUT_DIR = Path(f"experiment_results_{TIMESTAMP}{EARLY_STOP_SUFFIX}")
    OUTPUT_DIR.mkdir(exist_ok=True)


def validate_data_quality(X_train, X_test, y_train, y_test):
    """Validate data before training"""
    issues = []

    # Check for NaN values
    if X_train.isnull().any().any():
        issues.append("X_train contains NaN values")
    if X_test.isnull().any().any():
        issues.append("X_test contains NaN values")

    # Check for infinite values
    if not np.isfinite(X_train.select_dtypes(include=[np.number])).all().all():
        issues.append("X_train contains infinite values")
    if not np.isfinite(X_test.select_dtypes(include=[np.number])).all().all():
        issues.append("X_test contains infinite values")

    # Check class distribution
    min_samples = y_train.value_counts().min()
    unique_classes = len(y_train.unique())

    print(f"📊 Classes: {unique_classes}, Min samples per class: {min_samples}")

    if min_samples < 2:
        issues.append("Very low sample count - results may be unreliable")

    return len(issues) == 0, issues


def calculate_comprehensive_metrics(
    y_true, y_pred, y_pred_proba, label_encoder, dataset_type="test"
):
    """Calculate comprehensive metrics for user identification task"""
    metrics = {}

    # Basic metrics
    metrics[f"{dataset_type}_accuracy"] = accuracy_score(y_true, y_pred)
    metrics[f"{dataset_type}_f1_weighted"] = f1_score(
        y_true, y_pred, average="weighted", zero_division=0
    )
    metrics[f"{dataset_type}_f1_macro"] = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics[f"{dataset_type}_precision_weighted"] = precision_score(
        y_true, y_pred, average="weighted", zero_division=0
    )
    metrics[f"{dataset_type}_precision_macro"] = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics[f"{dataset_type}_recall_weighted"] = recall_score(
        y_true, y_pred, average="weighted", zero_division=0
    )
    metrics[f"{dataset_type}_recall_macro"] = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    # Encode labels if needed for top-k calculations
    if isinstance(y_true.iloc[0] if hasattr(y_true, "iloc") else y_true[0], str):
        y_true_encoded = label_encoder.transform(y_true)
    else:
        y_true_encoded = y_true

    # Top-k accuracy (critical for user identification)
    max_k = min(5, len(label_encoder.classes_))
    for k in range(1, max_k + 1):
        try:
            top_k_acc = top_k_accuracy_score(y_true_encoded, y_pred_proba, k=k)
            metrics[f"{dataset_type}_top_{k}_accuracy"] = top_k_acc
        except Exception as e:
            print(f"⚠️ Could not calculate top-{k} accuracy: {e}")
            metrics[f"{dataset_type}_top_{k}_accuracy"] = 0.0

    # Recognition rate using bob.measure (handle errors gracefully)
    try:
        rr_scores = []
        for i, true_label in enumerate(y_true_encoded):
            if true_label < y_pred_proba.shape[1]:  # Check bounds
                pos_score = y_pred_proba[i, true_label]
                neg_scores = np.delete(y_pred_proba[i], true_label)
                rr_scores.append((neg_scores, [pos_score]))

        if rr_scores:
            recognition_rate = bob.measure.recognition_rate(rr_scores, rank=1)
            metrics[f"{dataset_type}_recognition_rate"] = recognition_rate
        else:
            metrics[f"{dataset_type}_recognition_rate"] = 0.0
    except Exception as e:
        print(f"⚠️ Could not calculate recognition rate: {e}")
        metrics[f"{dataset_type}_recognition_rate"] = 0.0

    return metrics


def create_top_k_confusion_matrices(
    y_true, y_pred_proba, label_encoder, title, filename, k_values=[1, 5]
):
    """Create confusion matrices for top-k predictions with improved formatting"""

    for k in k_values:
        if k > y_pred_proba.shape[1]:
            continue

        # Get top-k predictions
        top_k_indices = np.argsort(y_pred_proba, axis=1)[:, -k:]

        # Create binary prediction array (1 if true class in top-k, 0 otherwise)
        y_true_encoded = (
            label_encoder.transform(y_true) if hasattr(y_true, "iloc") else y_true
        )
        top_k_correct = np.array(
            [
                true_label in top_k_pred
                for true_label, top_k_pred in zip(y_true_encoded, top_k_indices)
            ]
        )

        # For top-1, use actual predicted class
        if k == 1:
            y_pred_top1 = np.argmax(y_pred_proba, axis=1)
            y_pred_decoded = label_encoder.inverse_transform(y_pred_top1)

            # Create confusion matrix with actual user IDs
            cm = confusion_matrix(y_true, y_pred_decoded)

            # Get unique labels for proper labeling
            unique_labels = sorted(list(set(y_true) | set(y_pred_decoded)))

            # Calculate figure size based on number of users (minimum 12x10, scale up for more users)
            n_users = len(unique_labels)
            fig_width = max(12, min(50, n_users * 0.8))
            fig_height = max(10, min(40, n_users * 0.6))

            plt.figure(figsize=(fig_width, fig_height))

            # Show all users up to 50, then show every other label for readability
            if len(unique_labels) <= 50:
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=unique_labels,
                    yticklabels=unique_labels,
                    cbar_kws={"shrink": 0.8},
                )
                plt.xticks(rotation=45, ha="right")
                plt.yticks(rotation=0)
            else:
                # Show every other label for readability
                x_labels = [
                    label if i % 2 == 0 else "" for i, label in enumerate(unique_labels)
                ]
                y_labels = [
                    label if i % 2 == 0 else "" for i, label in enumerate(unique_labels)
                ]

                sns.heatmap(
                    cm,
                    annot=False,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=x_labels,
                    yticklabels=y_labels,
                    cbar_kws={"shrink": 0.6},
                )
                plt.xticks(rotation=90, ha="center", fontsize=8)
                plt.yticks(rotation=0, fontsize=8)

            plt.title(f"Top-1 Confusion Matrix - {title}")
            plt.ylabel("True User ID")
            plt.xlabel("Predicted User ID")
            plt.tight_layout()
            plt.savefig(
                OUTPUT_DIR / f"{filename}_top_1_confusion_matrix.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

        else:
            # For top-k (k>1), create binary confusion matrix
            binary_true = np.ones_like(
                top_k_correct
            )  # All should be correctly identified
            binary_pred = top_k_correct.astype(int)

            cm_binary = confusion_matrix(binary_true, binary_pred, labels=[0, 1])

            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm_binary,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=[f"Not in Top-{k}", f"In Top-{k}"],
                yticklabels=["User Not Found", "User Found"],
            )
            plt.title(f"Top-{k} Identification Success - {title}")
            plt.ylabel("Expected Outcome")
            plt.xlabel("Actual Outcome")
            plt.tight_layout()
            plt.savefig(
                OUTPUT_DIR / f"{filename}_top_{k}_identification_matrix.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()


def save_model_with_metadata(
    model, model_name, experiment_name, hyperparams, metrics, random_seed=42
):
    """Save model with comprehensive metadata"""
    early_stop_suffix = "_early_stop" if EARLY_STOP else ""

    # Clean experiment name for filename
    clean_experiment_name = experiment_name.replace(f"_{model_name}", "").replace(
        "_no_scaling", ""
    )

    # Create filename with all relevant info
    filename = f"{model_name.lower()}_{clean_experiment_name}_{TIMESTAMP}_seed{random_seed}{early_stop_suffix}.pkl"

    # Prepare metadata
    metadata = {
        "model_name": model_name,
        "experiment_name": experiment_name,
        "clean_experiment_name": clean_experiment_name,
        "timestamp": TIMESTAMP,
        "random_seed": random_seed,
        "early_stopping_used": EARLY_STOP,
        "hyperparameters": hyperparams,
        "performance_metrics": metrics,
        "config_used": config,
        "task_type": "user_identification",
        "data_preprocessing": "pre_normalized",
    }

    # Save model and metadata
    model_data = {"model": model, "metadata": metadata}

    filepath = OUTPUT_DIR / filename
    with open(filepath, "wb") as f:
        pickle.dump(model_data, f)

    print(f"💾 Model saved: {filename}")
    return str(filepath)


def run_xgboost_model(
    X_train, X_test, y_train, y_test, experiment_name="unknown", random_seed=42
):
    """Fast XGBoost model with streamlined parameters"""
    global CURRENT_EXPERIMENT
    CURRENT_EXPERIMENT += 1

    print(
        f"\n🚀 Running XGBoost - {experiment_name} ({CURRENT_EXPERIMENT}/{TOTAL_EXPERIMENTS})"
    )

    # Validate data quality
    is_valid, issues = validate_data_quality(X_train, X_test, y_train, y_test)
    if not is_valid:
        print(f"⚠️ Data quality issues: {issues}")

    # Use pre-normalized data directly
    X_train_scaled = X_train.values
    X_test_scaled = X_test.values

    # Handle NaN/inf values
    if not np.isfinite(X_train_scaled).all():
        print("⚠️ Warning: X_train contains NaN/inf values, filling with median")
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=np.nanmedian(X_train_scaled))
    if not np.isfinite(X_test_scaled).all():
        print("⚠️ Warning: X_test contains NaN/inf values, filling with median")
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=np.nanmedian(X_test_scaled))

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    min_samples_per_class = y_train.value_counts().min()
    print(f"Minimum samples per class: {min_samples_per_class}")

    # Configure early stopping and parallelism
    early_stopping_rounds = 50 if EARLY_STOP else None
    eval_set = [(X_test_scaled, y_test_encoded)] if EARLY_STOP else None
    n_jobs = min(4, os.cpu_count() // 2) if os.cpu_count() > 4 else 1

    if min_samples_per_class < 2:
        print("⚠️ Not enough samples for CV → using default parameters")
        best_params = {
            "n_estimators": 1000 if EARLY_STOP else 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1,
            "random_state": random_seed,
            "n_jobs": n_jobs,
        }
        best_xgb = XGBClassifier(**best_params)

        with tqdm(desc="Training XGBoost", leave=False) as pbar:
            if EARLY_STOP:
                best_xgb.fit(
                    X_train_scaled,
                    y_train_encoded,
                    early_stopping_rounds=early_stopping_rounds,
                    eval_set=eval_set,
                    verbose=False,
                )
            else:
                best_xgb.fit(X_train_scaled, y_train_encoded)
            pbar.update(1)
    else:
        stratified_kfold = StratifiedKFold(
            n_splits=2, shuffle=True, random_state=random_seed
        )

        # STREAMLINED parameter grid for speed
        param_grid = {
            "n_estimators": [1000] if EARLY_STOP else [100, 200],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "reg_lambda": [1, 3],
        }

        base_estimator = XGBClassifier(random_state=random_seed, n_jobs=n_jobs)

        with tqdm(desc="Grid Search XGBoost", leave=False) as pbar:
            grid_search_xgb = GridSearchCV(
                base_estimator,
                param_grid,
                scoring="accuracy",
                cv=stratified_kfold,
                n_jobs=n_jobs,
                verbose=0,
            )
            grid_search_xgb.fit(X_train_scaled, y_train_encoded)
            pbar.update(1)

        best_xgb = grid_search_xgb.best_estimator_
        best_params = grid_search_xgb.best_params_
        print(f"Best XGBoost params: {best_params}")

        # Refit with early stopping if requested
        if EARLY_STOP:
            best_xgb.set_params(n_estimators=1000)
            with tqdm(desc="Final XGBoost Training", leave=False) as pbar:
                best_xgb.fit(
                    X_train_scaled,
                    y_train_encoded,
                    early_stopping_rounds=early_stopping_rounds,
                    eval_set=eval_set,
                    verbose=False,
                )
                pbar.update(1)

    # Predictions and metrics
    y_pred_train = best_xgb.predict(X_train_scaled)
    y_pred_test = best_xgb.predict(X_test_scaled)

    y_pred_train_decoded = label_encoder.inverse_transform(y_pred_train)
    y_pred_test_decoded = label_encoder.inverse_transform(y_pred_test)

    y_pred_proba_train = best_xgb.predict_proba(X_train_scaled)
    y_pred_proba_test = best_xgb.predict_proba(X_test_scaled)

    # Calculate metrics
    train_metrics = calculate_comprehensive_metrics(
        y_train, y_pred_train_decoded, y_pred_proba_train, label_encoder, "train"
    )
    test_metrics = calculate_comprehensive_metrics(
        y_test, y_pred_test_decoded, y_pred_proba_test, label_encoder, "test"
    )

    all_metrics = {**train_metrics, **test_metrics}

    # Print results
    print(f"XGBoost Train Top-1: {train_metrics.get('train_top_1_accuracy', 0):.4f}")
    print(f"XGBoost Test Top-1: {test_metrics.get('test_top_1_accuracy', 0):.4f}")
    print(f"XGBoost Test Top-5: {test_metrics.get('test_top_5_accuracy', 0):.4f}")

    # Create visualizations
    clean_exp_name = experiment_name.replace("_no_scaling", "")
    create_top_k_confusion_matrices(
        y_test,
        y_pred_proba_test,
        label_encoder,
        f"XGBoost {clean_exp_name}",
        f"xgboost_{clean_exp_name}",
    )

    # Feature importance
    if config.get("draw_feature_importance_graph", False):
        plt.figure(figsize=(12, 8))
        plot_importance(best_xgb, importance_type="weight", max_num_features=15)
        plt.title(f"XGBoost Feature Importance - {clean_exp_name}")
        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / f"xgboost_feature_importance_{clean_exp_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # Save model
    hyperparams = best_params if "best_params" in locals() else best_xgb.get_params()
    model_path = save_model_with_metadata(
        best_xgb, "XGBoost", experiment_name, hyperparams, all_metrics, random_seed
    )

    # Store results
    result_record = {
        "model": "XGBoost",
        "experiment": experiment_name,
        "random_seed": random_seed,
        "early_stopping": EARLY_STOP,
        "model_path": model_path,
        "hyperparameters": str(hyperparams),
        **all_metrics,
    }
    EXPERIMENT_RESULTS.append(result_record)

    # Store detailed results
    for k in range(1, 6):
        detailed_record = {
            "model": "XGBoost",
            "experiment": experiment_name,
            "random_seed": random_seed,
            "early_stopping": EARLY_STOP,
            "k_value": k,
            "train_top_k_accuracy": train_metrics.get(f"train_top_{k}_accuracy", 0),
            "test_top_k_accuracy": test_metrics.get(f"test_top_{k}_accuracy", 0),
            "model_path": model_path,
            "hyperparameters": str(hyperparams),
            "timestamp": TIMESTAMP,
        }
        DETAILED_RESULTS.append(detailed_record)

    return all_metrics


def run_random_forest_model(
    X_train, X_test, y_train, y_test, experiment_name="unknown", random_seed=42
):
    """Fast Random Forest model with streamlined parameters"""
    global CURRENT_EXPERIMENT
    CURRENT_EXPERIMENT += 1

    print(
        f"\n🌲 Running Random Forest - {experiment_name} ({CURRENT_EXPERIMENT}/{TOTAL_EXPERIMENTS})"
    )

    # Validate data and use pre-normalized values
    is_valid, issues = validate_data_quality(X_train, X_test, y_train, y_test)
    if not is_valid:
        print(f"⚠️ Data quality issues: {issues}")

    X_train_scaled = X_train.values
    X_test_scaled = X_test.values

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    min_samples_per_class = y_train.value_counts().min()
    print(f"Minimum samples per class: {min_samples_per_class}")

    n_jobs = min(4, os.cpu_count() // 2) if os.cpu_count() > 4 else 1

    if min_samples_per_class < 2:
        print("⚠️ Not enough samples for CV → using default parameters")
        best_rf = RandomForestClassifier(
            n_estimators=500, random_state=random_seed, n_jobs=n_jobs
        )
        with tqdm(desc="Training Random Forest", leave=False) as pbar:
            best_rf.fit(X_train_scaled, y_train_encoded)
            pbar.update(1)
        best_params = best_rf.get_params()
    else:
        stratified_kfold = StratifiedKFold(
            n_splits=2, shuffle=True, random_state=random_seed
        )

        # STREAMLINED parameter grid for speed
        param_grid = {
            "n_estimators": [100, 300, 500],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 10],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt", "log2"],
            "bootstrap": [True, False],
        }

        with tqdm(desc="Grid Search Random Forest", leave=False) as pbar:
            grid_search_rf = GridSearchCV(
                RandomForestClassifier(random_state=random_seed, n_jobs=n_jobs),
                param_grid,
                scoring="accuracy",
                cv=stratified_kfold,
                n_jobs=n_jobs,
                verbose=0,
            )
            grid_search_rf.fit(X_train_scaled, y_train_encoded)
            pbar.update(1)

        best_rf = grid_search_rf.best_estimator_
        best_params = grid_search_rf.best_params_
        print(f"Best RF params: {best_params}")

    # Predictions and metrics
    y_pred_train = best_rf.predict(X_train_scaled)
    y_pred_test = best_rf.predict(X_test_scaled)

    y_pred_train_decoded = label_encoder.inverse_transform(y_pred_train)
    y_pred_test_decoded = label_encoder.inverse_transform(y_pred_test)

    y_pred_proba_train = best_rf.predict_proba(X_train_scaled)
    y_pred_proba_test = best_rf.predict_proba(X_test_scaled)

    train_metrics = calculate_comprehensive_metrics(
        y_train, y_pred_train_decoded, y_pred_proba_train, label_encoder, "train"
    )
    test_metrics = calculate_comprehensive_metrics(
        y_test, y_pred_test_decoded, y_pred_proba_test, label_encoder, "test"
    )

    all_metrics = {**train_metrics, **test_metrics}

    print(
        f"Random Forest Train Top-1: {train_metrics.get('train_top_1_accuracy', 0):.4f}"
    )
    print(f"Random Forest Test Top-1: {test_metrics.get('test_top_1_accuracy', 0):.4f}")
    print(f"Random Forest Test Top-5: {test_metrics.get('test_top_5_accuracy', 0):.4f}")

    clean_exp_name = experiment_name.replace("_no_scaling", "")
    create_top_k_confusion_matrices(
        y_test,
        y_pred_proba_test,
        label_encoder,
        f"Random Forest {clean_exp_name}",
        f"rf_{clean_exp_name}",
    )

    # Feature importance
    if config.get("draw_feature_importance_graph", False):
        plt.figure(figsize=(12, 8))
        feature_importances = best_rf.feature_importances_
        feature_names = X_train.columns
        indices = np.argsort(feature_importances)[-15:]

        plt.barh(range(len(indices)), feature_importances[indices], align="center")
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel("Feature Importance")
        plt.title(f"Random Forest Feature Importance - {clean_exp_name}")
        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / f"rf_feature_importance_{clean_exp_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # Save model and results
    model_path = save_model_with_metadata(
        best_rf, "RandomForest", experiment_name, best_params, all_metrics, random_seed
    )

    result_record = {
        "model": "RandomForest",
        "experiment": experiment_name,
        "random_seed": random_seed,
        "early_stopping": False,
        "model_path": model_path,
        "hyperparameters": str(best_params),
        **all_metrics,
    }
    EXPERIMENT_RESULTS.append(result_record)

    for k in range(1, 6):
        detailed_record = {
            "model": "RandomForest",
            "experiment": experiment_name,
            "random_seed": random_seed,
            "early_stopping": False,
            "k_value": k,
            "train_top_k_accuracy": train_metrics.get(f"train_top_{k}_accuracy", 0),
            "test_top_k_accuracy": test_metrics.get(f"test_top_{k}_accuracy", 0),
            "model_path": model_path,
            "hyperparameters": str(best_params),
            "timestamp": TIMESTAMP,
        }
        DETAILED_RESULTS.append(detailed_record)

    return all_metrics


def run_catboost_model(
    X_train, X_test, y_train, y_test, experiment_name="unknown", random_seed=42
):
    """Fast CatBoost model with streamlined parameters"""
    global CURRENT_EXPERIMENT
    CURRENT_EXPERIMENT += 1

    print(
        f"\n🐱 Running CatBoost - {experiment_name} ({CURRENT_EXPERIMENT}/{TOTAL_EXPERIMENTS})"
    )

    # Validate data and use pre-normalized values
    is_valid, issues = validate_data_quality(X_train, X_test, y_train, y_test)
    if not is_valid:
        print(f"⚠️ Data quality issues: {issues}")

    X_train_scaled = X_train.values
    X_test_scaled = X_test.values

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    min_samples_per_class = y_train.value_counts().min()
    print(f"Minimum samples per class: {min_samples_per_class}")

    # Configure early stopping
    early_stopping_rounds = 50 if EARLY_STOP else None
    eval_set = Pool(X_test_scaled, y_test_encoded) if EARLY_STOP else None
    thread_count = min(4, os.cpu_count() // 2) if os.cpu_count() > 4 else 1

    if min_samples_per_class < 2:
        print("⚠️ Not enough samples for CV → using default parameters")
        best_params = {
            "iterations": 1000 if EARLY_STOP else 100,
            "depth": 6,
            "learning_rate": 0.1,
            "l2_leaf_reg": 3,
            "border_count": 64,
            "random_seed": random_seed,
            "verbose": 0,
            "thread_count": thread_count,
        }

        best_catboost_model = CatBoostClassifier(**best_params)

        with tqdm(desc="Training CatBoost", leave=False) as pbar:
            if EARLY_STOP and eval_set:
                best_catboost_model.fit(
                    X_train_scaled,
                    y_train_encoded,
                    early_stopping_rounds=early_stopping_rounds,
                    eval_set=eval_set,
                    use_best_model=True,
                )
            else:
                best_catboost_model.fit(X_train_scaled, y_train_encoded)
            pbar.update(1)
    else:
        stratified_kfold = StratifiedKFold(
            n_splits=2, shuffle=True, random_state=random_seed
        )

        # STREAMLINED parameter grid for speed
        param_grid = {
            "iterations": [1000] if EARLY_STOP else [100, 200],
            "depth": [4, 6, 8],
            "learning_rate": [0.05, 0.1],
            "l2_leaf_reg": [1, 3, 5],
            "border_count": [32, 64],
        }

        with tqdm(desc="Grid Search CatBoost", leave=False) as pbar:
            grid_search = GridSearchCV(
                estimator=CatBoostClassifier(
                    verbose=0, random_seed=random_seed, thread_count=thread_count
                ),
                param_grid=param_grid,
                scoring="accuracy",
                cv=stratified_kfold,
                verbose=0,
                n_jobs=1,  # CatBoost handles parallelism internally
            )
            grid_search.fit(X_train_scaled, y_train_encoded)
            pbar.update(1)

        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        print(f"Best CatBoost params: {best_params}")

        # Train final model with best params
        best_catboost_model = CatBoostClassifier(
            **best_params, random_seed=random_seed, verbose=0, thread_count=thread_count
        )

        with tqdm(desc="Final CatBoost Training", leave=False) as pbar:
            if EARLY_STOP and eval_set:
                best_catboost_model.fit(
                    X_train_scaled,
                    y_train_encoded,
                    early_stopping_rounds=early_stopping_rounds,
                    eval_set=eval_set,
                    use_best_model=True,
                )
            else:
                best_catboost_model.fit(X_train_scaled, y_train_encoded)
            pbar.update(1)

    # Predictions and metrics
    y_pred_train = best_catboost_model.predict(X_train_scaled)
    y_pred_test = best_catboost_model.predict(X_test_scaled)

    y_pred_train_decoded = label_encoder.inverse_transform(y_pred_train)
    y_pred_test_decoded = label_encoder.inverse_transform(y_pred_test)

    y_pred_proba_train = best_catboost_model.predict_proba(X_train_scaled)
    y_pred_proba_test = best_catboost_model.predict_proba(X_test_scaled)

    train_metrics = calculate_comprehensive_metrics(
        y_train, y_pred_train_decoded, y_pred_proba_train, label_encoder, "train"
    )
    test_metrics = calculate_comprehensive_metrics(
        y_test, y_pred_test_decoded, y_pred_proba_test, label_encoder, "test"
    )

    all_metrics = {**train_metrics, **test_metrics}

    print(f"CatBoost Train Top-1: {train_metrics.get('train_top_1_accuracy', 0):.4f}")
    print(f"CatBoost Test Top-1: {test_metrics.get('test_top_1_accuracy', 0):.4f}")
    print(f"CatBoost Test Top-5: {test_metrics.get('test_top_5_accuracy', 0):.4f}")

    clean_exp_name = experiment_name.replace("_no_scaling", "")
    create_top_k_confusion_matrices(
        y_test,
        y_pred_proba_test,
        label_encoder,
        f"CatBoost {clean_exp_name}",
        f"catboost_{clean_exp_name}",
    )

    # Feature importance
    if config.get("draw_feature_importance_graph", False):
        try:
            feature_importances = best_catboost_model.get_feature_importance(
                Pool(X_train_scaled, label=y_train_encoded)
            )
            feature_names = X_train.columns
            plt.figure(figsize=(12, 8))
            sorted_indices = feature_importances.argsort()[-15:]
            plt.barh(
                range(len(sorted_indices)),
                feature_importances[sorted_indices],
                align="center",
            )
            plt.yticks(
                range(len(sorted_indices)), [feature_names[i] for i in sorted_indices]
            )
            plt.xlabel("Feature Importance")
            plt.title(f"CatBoost Feature Importance - {clean_exp_name}")
            plt.tight_layout()
            plt.savefig(
                OUTPUT_DIR / f"catboost_feature_importance_{clean_exp_name}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
        except Exception as e:
            print(f"⚠️ Could not generate feature importance plot: {e}")

    # Save model and results
    hyperparams = (
        best_params if "best_params" in locals() else best_catboost_model.get_params()
    )
    model_path = save_model_with_metadata(
        best_catboost_model,
        "CatBoost",
        experiment_name,
        hyperparams,
        all_metrics,
        random_seed,
    )

    result_record = {
        "model": "CatBoost",
        "experiment": experiment_name,
        "random_seed": random_seed,
        "early_stopping": EARLY_STOP,
        "model_path": model_path,
        "hyperparameters": str(hyperparams),
        **all_metrics,
    }
    EXPERIMENT_RESULTS.append(result_record)

    for k in range(1, 6):
        detailed_record = {
            "model": "CatBoost",
            "experiment": experiment_name,
            "random_seed": random_seed,
            "early_stopping": EARLY_STOP,
            "k_value": k,
            "train_top_k_accuracy": train_metrics.get(f"train_top_{k}_accuracy", 0),
            "test_top_k_accuracy": test_metrics.get(f"test_top_{k}_accuracy", 0),
            "model_path": model_path,
            "hyperparameters": str(hyperparams),
            "timestamp": TIMESTAMP,
        }
        DETAILED_RESULTS.append(detailed_record)

    return all_metrics


def run_svm_model(
    X_train, X_test, y_train, y_test, experiment_name="unknown", random_seed=42
):
    """Fast SVM model with streamlined parameters"""
    global CURRENT_EXPERIMENT
    CURRENT_EXPERIMENT += 1

    print(
        f"\n⚙️ Running SVM - {experiment_name} ({CURRENT_EXPERIMENT}/{TOTAL_EXPERIMENTS})"
    )

    # Validate data and use pre-normalized values
    is_valid, issues = validate_data_quality(X_train, X_test, y_train, y_test)
    if not is_valid:
        print(f"⚠️ Data quality issues: {issues}")

    X_train_scaled = X_train.values
    X_test_scaled = X_test.values

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    min_samples_per_class = y_train.value_counts().min()
    print(f"Minimum samples per class: {min_samples_per_class}")

    if min_samples_per_class < 2:
        print("⚠️ Not enough samples for CV → using default parameters")
        best_svm = SVC(
            C=1,
            kernel="rbf",
            gamma="scale",
            probability=True,
            random_state=random_seed,
        )
        with tqdm(desc="Training SVM", leave=False) as pbar:
            best_svm.fit(X_train_scaled, y_train_encoded)
            pbar.update(1)
        best_params = best_svm.get_params()
    else:
        stratified_kfold = StratifiedKFold(
            n_splits=2, shuffle=True, random_state=random_seed
        )

        # STREAMLINED parameter grid for speed
        param_grid = {
            "C": [0.1, 1, 10],
            "kernel": ["rbf"],  # Only RBF for speed
            "gamma": ["scale", "auto"],
        }

        with tqdm(desc="Grid Search SVM", leave=False) as pbar:
            grid_search_svm = GridSearchCV(
                SVC(probability=True, random_state=random_seed),
                param_grid,
                scoring="accuracy",
                cv=stratified_kfold,
                n_jobs=1,  # SVM doesn't parallelize well
                verbose=0,
            )
            grid_search_svm.fit(X_train_scaled, y_train_encoded)
            pbar.update(1)

        best_svm = grid_search_svm.best_estimator_
        best_params = grid_search_svm.best_params_
        print(f"Best SVM params: {best_params}")

    # Predictions and metrics
    y_pred_train = best_svm.predict(X_train_scaled)
    y_pred_test = best_svm.predict(X_test_scaled)

    y_pred_train_decoded = label_encoder.inverse_transform(y_pred_train)
    y_pred_test_decoded = label_encoder.inverse_transform(y_pred_test)

    y_pred_proba_train = best_svm.predict_proba(X_train_scaled)
    y_pred_proba_test = best_svm.predict_proba(X_test_scaled)

    train_metrics = calculate_comprehensive_metrics(
        y_train, y_pred_train_decoded, y_pred_proba_train, label_encoder, "train"
    )
    test_metrics = calculate_comprehensive_metrics(
        y_test, y_pred_test_decoded, y_pred_proba_test, label_encoder, "test"
    )

    all_metrics = {**train_metrics, **test_metrics}

    print(f"SVM Train Top-1: {train_metrics.get('train_top_1_accuracy', 0):.4f}")
    print(f"SVM Test Top-1: {test_metrics.get('test_top_1_accuracy', 0):.4f}")
    print(f"SVM Test Top-5: {test_metrics.get('test_top_5_accuracy', 0):.4f}")

    clean_exp_name = experiment_name.replace("_no_scaling", "")
    create_top_k_confusion_matrices(
        y_test,
        y_pred_proba_test,
        label_encoder,
        f"SVM {clean_exp_name}",
        f"svm_{clean_exp_name}",
    )

    # Save model and results
    model_path = save_model_with_metadata(
        best_svm, "SVM", experiment_name, best_params, all_metrics, random_seed
    )

    result_record = {
        "model": "SVM",
        "experiment": experiment_name,
        "random_seed": random_seed,
        "early_stopping": False,
        "model_path": model_path,
        "hyperparameters": str(best_params),
        **all_metrics,
    }
    EXPERIMENT_RESULTS.append(result_record)

    for k in range(1, 6):
        detailed_record = {
            "model": "SVM",
            "experiment": experiment_name,
            "random_seed": random_seed,
            "early_stopping": False,
            "k_value": k,
            "train_top_k_accuracy": train_metrics.get(f"train_top_{k}_accuracy", 0),
            "test_top_k_accuracy": test_metrics.get(f"test_top_{k}_accuracy", 0),
            "model_path": model_path,
            "hyperparameters": str(best_params),
            "timestamp": TIMESTAMP,
        }
        DETAILED_RESULTS.append(detailed_record)

    return all_metrics


def create_performance_plots(results_df):
    """Create comprehensive performance plots with proper data"""
    if results_df.empty:
        print("⚠️ No results to plot")
        return

    # Create comprehensive performance plots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Top-1 Accuracy by Model",
            "Top-K Accuracy Trends",
            "F1 Score vs Top-1 Accuracy",
            "Model Performance Overview",
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
        ],
    )

    models = results_df["model"].unique()
    colors = px.colors.qualitative.Set1[: len(models)]

    # 1. Top-1 accuracy by model (by experiment)
    for i, model in enumerate(models):
        model_data = results_df[results_df["model"] == model]

        if "test_top_1_accuracy" in model_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=model_data["experiment"],
                    y=model_data["test_top_1_accuracy"],
                    name=f"{model}",
                    line=dict(color=colors[i]),
                    mode="lines+markers",
                ),
                row=1,
                col=1,
            )

    # 2. Top-K accuracy trends (average across all experiments for each model)
    k_values = [1, 2, 3, 4, 5]
    for i, model in enumerate(models):
        model_data = results_df[results_df["model"] == model]
        k_accuracies = []
        for k in k_values:
            col_name = f"test_top_{k}_accuracy"
            if col_name in model_data.columns:
                avg_acc = model_data[col_name].fillna(0).mean()
                k_accuracies.append(avg_acc)
            else:
                k_accuracies.append(0)

        fig.add_trace(
            go.Scatter(
                x=k_values,
                y=k_accuracies,
                name=f"{model}",
                line=dict(color=colors[i]),
                mode="lines+markers",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    # 3. F1 vs Top-1 Accuracy scatter plot
    for i, model in enumerate(models):
        model_data = results_df[results_df["model"] == model]
        if (
            "test_top_1_accuracy" in model_data.columns
            and "test_f1_weighted" in model_data.columns
        ):
            fig.add_trace(
                go.Scatter(
                    x=model_data["test_f1_weighted"].fillna(0),
                    y=model_data["test_top_1_accuracy"].fillna(0),
                    name=f"{model}",
                    mode="markers",
                    marker=dict(color=colors[i], size=8),
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

    # 4. Model Performance Overview (average Top-1 and Top-5 for each model)
    model_names = []
    avg_top1_scores = []
    avg_top5_scores = []

    for model in models:
        model_data = results_df[results_df["model"] == model]
        avg_top1 = (
            model_data.get("test_top_1_accuracy", pd.Series([0])).fillna(0).mean()
        )
        avg_top5 = (
            model_data.get("test_top_5_accuracy", pd.Series([0])).fillna(0).mean()
        )

        model_names.append(model)
        avg_top1_scores.append(avg_top1)
        avg_top5_scores.append(avg_top5)

    # Add Top-1 bars (in blue/teal)
    fig.add_trace(
        go.Bar(
            x=model_names,
            y=avg_top1_scores,
            name="Avg Top-1",
            marker_color="teal",
            showlegend=True,
        ),
        row=2,
        col=2,
    )

    # Add Top-5 bars (in purple)
    fig.add_trace(
        go.Bar(
            x=model_names,
            y=avg_top5_scores,
            name="Avg Top-5",
            marker_color="purple",
            showlegend=True,
        ),
        row=2,
        col=2,
    )

    # Update layout
    fig.update_layout(
        height=800,
        title_text="User Identification Performance Analysis",
        barmode="group",  # Group bars side by side
    )

    # Add axis labels
    fig.update_xaxes(title_text="Experiment", row=1, col=1)
    fig.update_yaxes(title_text="Top-1 Accuracy", row=1, col=1)

    fig.update_xaxes(title_text="K Value", row=1, col=2)
    fig.update_yaxes(title_text="Top-K Accuracy", row=1, col=2)

    fig.update_xaxes(title_text="F1 Score (Weighted)", row=2, col=1)
    fig.update_yaxes(title_text="Top-1 Accuracy", row=2, col=1)

    fig.update_xaxes(title_text="Model", row=2, col=2)
    fig.update_yaxes(title_text="Average Accuracy", row=2, col=2)

    fig.write_html(OUTPUT_DIR / "performance_plots.html")
    print(f"📈 Performance plots saved to: {OUTPUT_DIR / 'performance_plots.html'}")


def generate_html_report():
    """Generate comprehensive HTML report"""
    if not EXPERIMENT_RESULTS:
        print("⚠️ No experiment results to report")
        return

    # Convert to DataFrames
    results_df = pd.DataFrame(EXPERIMENT_RESULTS)
    detailed_df = pd.DataFrame(DETAILED_RESULTS)

    # Save CSV files (without scaler column as requested)
    csv_path = OUTPUT_DIR / f"experiment_results_{TIMESTAMP}.csv"
    detailed_csv_path = OUTPUT_DIR / f"detailed_topk_results_{TIMESTAMP}.csv"

    results_df.to_csv(csv_path, index=False)
    detailed_df.to_csv(detailed_csv_path, index=False)

    print(f"📊 Results saved to: {csv_path}")
    print(f"📊 Detailed Top-K results saved to: {detailed_csv_path}")

    # Create performance plots
    create_performance_plots(results_df)

    # Generate HTML report
    best_top1_idx = results_df["test_top_1_accuracy"].fillna(0).idxmax()
    best_top5_idx = results_df["test_top_5_accuracy"].fillna(0).idxmax()

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>User Identification Results - {TIMESTAMP}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
            .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 10px; margin-bottom: 30px; }}
            .section {{ background-color: white; padding: 20px; margin-bottom: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #34495e; color: white; }}
            .metric-highlight {{ font-weight: bold; color: #27ae60; }}
            .info-box {{ background-color: #ecf0f1; padding: 15px; border-left: 4px solid #3498db; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>User Identification Analysis</h1>
            <p><strong>Timestamp:</strong> {TIMESTAMP}</p>
            <p><strong>Task:</strong> Cross-Platform User Identification</p>
            <p><strong>Early Stopping:</strong> {'Enabled' if EARLY_STOP else 'Disabled'}</p>
            <p><strong>Total Model Runs:</strong> {len(results_df)}</p>
            <p><strong>Data:</strong> Pre-normalized (no additional scaling)</p>
        </div>
        
        <div class="section">
            <h2>Best Performance Summary</h2>
            <div class="info-box">
                <p><strong>Best Top-1 Accuracy:</strong> <span class="metric-highlight">{results_df.loc[best_top1_idx, 'test_top_1_accuracy']:.4f}</span></p>
                <p><strong>Best Top-1 Model:</strong> <span class="metric-highlight">{results_df.loc[best_top1_idx, 'model']} ({results_df.loc[best_top1_idx, 'experiment']})</span></p>
                <p><strong>Best Top-5 Accuracy:</strong> <span class="metric-highlight">{results_df.loc[best_top5_idx, 'test_top_5_accuracy']:.4f}</span></p>
                <p><strong>Best Top-5 Model:</strong> <span class="metric-highlight">{results_df.loc[best_top5_idx, 'model']} ({results_df.loc[best_top5_idx, 'experiment']})</span></p>
            </div>
        </div>
        
        <div class="section">
            <h2>Top-K Performance Summary</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Experiment</th>
                    <th>Top-1</th>
                    <th>Top-2</th>
                    <th>Top-3</th>
                    <th>Top-4</th>
                    <th>Top-5</th>
                </tr>
    """

    # Add top-k performance table
    for _, row in results_df.iterrows():
        html_content += f"""
                <tr>
                    <td>{row['model']}</td>
                    <td>{row['experiment']}</td>
                    <td>{row.get('test_top_1_accuracy', 0):.4f}</td>
                    <td>{row.get('test_top_2_accuracy', 0):.4f}</td>
                    <td>{row.get('test_top_3_accuracy', 0):.4f}</td>
                    <td>{row.get('test_top_4_accuracy', 0):.4f}</td>
                    <td>{row.get('test_top_5_accuracy', 0):.4f}</td>
                </tr>
        """

    html_content += f"""
            </table>
        </div>
        
        <div class="section">
            <h2>Generated Files</h2>
            <ul>
                <li><strong>Summary Results:</strong> experiment_results_{TIMESTAMP}.csv</li>
                <li><strong>Detailed Top-K Results:</strong> detailed_topk_results_{TIMESTAMP}.csv</li>
                <li><strong>Performance Plots:</strong> performance_plots.html</li>
                <li><strong>Trained Models:</strong> {len([r for r in EXPERIMENT_RESULTS if 'model_path' in r])} models with metadata</li>
                <li><strong>Confusion Matrices:</strong> Enhanced Top-1 and Top-5 visualizations</li>
                <li><strong>Feature Importance:</strong> Available for tree-based models</li>
            </ul>
        </div>
    </body>
    </html>
    """

    # Save HTML report
    html_path = OUTPUT_DIR / f"user_identification_report_{TIMESTAMP}.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"📋 HTML report: {html_path}")
    return html_path


def set_total_experiments(total):
    """Set total experiments for progress tracking"""
    global TOTAL_EXPERIMENTS
    TOTAL_EXPERIMENTS = total


def finalize_experiments():
    """Generate all reports and analysis"""
    if EXPERIMENT_RESULTS:
        generate_html_report()
        print(f"\n🎉 User identification analysis complete!")
        print(f"📁 All results in: {OUTPUT_DIR}")
    else:
        print("⚠️ No experiments were run")


if __name__ == "__main__":
    print(f"🚀 Fast User Identification ML Models loaded")
    print(f"⚡ Early stopping: {EARLY_STOP}")
    print(f"📁 Results: {OUTPUT_DIR}")
    print(f"🔧 Data preprocessing: Pre-normalized")
    print(f"⚡ Speed optimized: Streamlined parameter grids")
