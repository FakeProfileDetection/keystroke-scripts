import os
import json
import enum
import pickle
import argparse
import warnings
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
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
import plotly.offline as pyo

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Global configuration and results storage
with open(os.path.join(os.getcwd(), "classifier_config.json"), "r") as f:
    config = json.load(f)

# Add command line argument parsing
parser = argparse.ArgumentParser(description='Run ML experiments with optional early stopping')
parser.add_argument('-e', '--early-stop', action='store_true', 
                   help='Use early stopping for XGBoost and CatBoost models')
parser.add_argument('-s', '--seeds', type=int, default=1,
                   help='Number of different random seeds to test (default: 1)')
args = parser.parse_args()

# Global results storage
EXPERIMENT_RESULTS = []
DETAILED_RESULTS = []
HYPERPARAMETER_ANALYSIS = []  # New: Store all CV results for analysis
TIMESTAMP = datetime.now().isoformat().replace(':', '-').replace('.', '-')[:19]
EARLY_STOP_SUFFIX = "_early_stop" if args.early_stop else ""
OUTPUT_DIR = Path(f"experiment_results_{TIMESTAMP}{EARLY_STOP_SUFFIX}")
OUTPUT_DIR.mkdir(exist_ok=True)

# Progress tracking
TOTAL_EXPERIMENTS = 0
CURRENT_EXPERIMENT = 0


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


def calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba, label_encoder, dataset_type="test"):
    """Calculate comprehensive metrics for user identification task"""
    metrics = {}
    
    # Basic metrics
    metrics[f'{dataset_type}_accuracy'] = accuracy_score(y_true, y_pred)
    metrics[f'{dataset_type}_f1_weighted'] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics[f'{dataset_type}_f1_macro'] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics[f'{dataset_type}_precision_weighted'] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics[f'{dataset_type}_precision_macro'] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics[f'{dataset_type}_recall_weighted'] = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics[f'{dataset_type}_recall_macro'] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    
    # Encode labels if needed for top-k calculations
    if isinstance(y_true.iloc[0] if hasattr(y_true, 'iloc') else y_true[0], str):
        y_true_encoded = label_encoder.transform(y_true)
    else:
        y_true_encoded = y_true
    
    # Top-k accuracy (critical for user identification)
    max_k = min(5, len(label_encoder.classes_))
    for k in range(1, max_k + 1):
        try:
            top_k_acc = top_k_accuracy_score(y_true_encoded, y_pred_proba, k=k)
            metrics[f'{dataset_type}_top_{k}_accuracy'] = top_k_acc
        except Exception as e:
            print(f"⚠️ Could not calculate top-{k} accuracy: {e}")
            metrics[f'{dataset_type}_top_{k}_accuracy'] = 0.0
    
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
            metrics[f'{dataset_type}_recognition_rate'] = recognition_rate
        else:
            metrics[f'{dataset_type}_recognition_rate'] = 0.0
    except Exception as e:
        print(f"⚠️ Could not calculate recognition rate: {e}")
        metrics[f'{dataset_type}_recognition_rate'] = 0.0
    
    return metrics


def create_top_k_confusion_matrices(y_true, y_pred_proba, label_encoder, title, filename, k_values=[1, 5]):
    """Create confusion matrices for top-k predictions with improved formatting"""
    
    for k in k_values:
        if k > y_pred_proba.shape[1]:
            continue
            
        # Get top-k predictions
        top_k_indices = np.argsort(y_pred_proba, axis=1)[:, -k:]
        
        # Create binary prediction array (1 if true class in top-k, 0 otherwise)
        y_true_encoded = label_encoder.transform(y_true) if hasattr(y_true, 'iloc') else y_true
        top_k_correct = np.array([true_label in top_k_pred for true_label, top_k_pred in zip(y_true_encoded, top_k_indices)])
        
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
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=unique_labels, yticklabels=unique_labels,
                           cbar_kws={'shrink': 0.8})
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
            else:
                # Show every other label for readability
                x_labels = [label if i % 2 == 0 else '' for i, label in enumerate(unique_labels)]
                y_labels = [label if i % 2 == 0 else '' for i, label in enumerate(unique_labels)]
                
                sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                           xticklabels=x_labels, yticklabels=y_labels,
                           cbar_kws={'shrink': 0.6})
                plt.xticks(rotation=90, ha='center', fontsize=8)
                plt.yticks(rotation=0, fontsize=8)
            
            plt.title(f'Top-1 Confusion Matrix - {title}')
            plt.ylabel('True User ID')
            plt.xlabel('Predicted User ID')
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f'{filename}_top_1_confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        else:
            # For top-k (k>1), create binary confusion matrix
            binary_true = np.ones_like(top_k_correct)  # All should be correctly identified
            binary_pred = top_k_correct.astype(int)
            
            cm_binary = confusion_matrix(binary_true, binary_pred, labels=[0, 1])
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_binary, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=[f'Not in Top-{k}', f'In Top-{k}'], 
                       yticklabels=['User Not Found', 'User Found'])
            plt.title(f'Top-{k} Identification Success - {title}')
            plt.ylabel('Expected Outcome')
            plt.xlabel('Actual Outcome')
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f'{filename}_top_{k}_identification_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()


def analyze_hyperparameter_performance():
    """Analyze hyperparameter search results to suggest improvements"""
    if not HYPERPARAMETER_ANALYSIS:
        return
    
    analysis_df = pd.DataFrame(HYPERPARAMETER_ANALYSIS)
    
    # Save detailed hyperparameter results
    hp_csv_path = OUTPUT_DIR / f"hyperparameter_analysis_{TIMESTAMP}.csv"
    analysis_df.to_csv(hp_csv_path, index=False)
    
    print(f"\n📊 Hyperparameter Analysis Summary:")
    print(f"Total grid search combinations tested: {len(analysis_df)}")
    
    # Group by model and analyze
    for model_name in analysis_df['model'].unique():
        model_data = analysis_df[analysis_df['model'] == model_name]
        
        print(f"\n🔍 {model_name} Analysis:")
        print(f"  • Combinations tested: {len(model_data)}")
        print(f"  • Best CV score: {model_data['cv_score'].max():.4f}")
        print(f"  • Worst CV score: {model_data['cv_score'].min():.4f}")
        print(f"  • Score variance: {model_data['cv_score'].var():.6f}")
        
        # Check if we're hitting parameter boundaries
        best_params = model_data.loc[model_data['cv_score'].idxmax(), 'params']
        print(f"  • Best parameters: {best_params}")
        
        # Analyze parameter distributions
        param_cols = [col for col in model_data.columns if col.startswith('param_')]
        if param_cols:
            print(f"  • Parameter analysis:")
            for param_col in param_cols:
                unique_vals = model_data[param_col].dropna().unique()
                if len(unique_vals) > 1:
                    # Check if best value is at boundary
                    best_val = model_data.loc[model_data['cv_score'].idxmax(), param_col]
                    if pd.notna(best_val):
                        sorted_vals = sorted(unique_vals)
                        if (best_val == sorted_vals[0] or best_val == sorted_vals[-1]) and len(sorted_vals) > 2:
                            print(f"    ⚠️ {param_col}: Best value ({best_val}) at boundary - consider expanding range")
                        else:
                            print(f"    ✅ {param_col}: Best value ({best_val}) not at boundary")
    
    # Generate recommendations
    print(f"\n💡 Recommendations:")
    
    # Check variance in scores
    overall_variance = analysis_df['cv_score'].var()
    if overall_variance > 0.01:
        print("  • High score variance detected - hyperparameter tuning is important!")
    else:
        print("  • Low score variance - current hyperparameters may be sufficient")
    
    # Model comparison
    model_scores = analysis_df.groupby('model')['cv_score'].max()
    best_model = model_scores.idxmax()
    print(f"  • Best performing model: {best_model} (CV score: {model_scores[best_model]:.4f})")
    
    # Suggest additional models
    tested_models = set(analysis_df['model'].unique())
    additional_models = {
        'ExtraTrees': 'Extra Trees Classifier',
        'GradientBoosting': 'Gradient Boosting Classifier', 
        'KNN': 'K-Nearest Neighbors',
        'LogisticRegression': 'Logistic Regression'
    } 
    
    untested = set(additional_models.keys()) - tested_models
    if untested:
        print(f"  • Consider testing additional models: {', '.join([additional_models[m] for m in untested])}")


def save_model_with_metadata(model, model_name, experiment_name, hyperparams, metrics, random_seed=42):
    """Save model with comprehensive metadata"""
    early_stop_suffix = "_early_stop" if args.early_stop else ""
    
    # Clean experiment name for filename (remove redundant model name)
    clean_experiment_name = experiment_name.replace(f"_{model_name}", "").replace("_no_scaling", "")
    
    # Create filename with all relevant info
    filename = f"{model_name.lower()}_{clean_experiment_name}_{TIMESTAMP}_seed{random_seed}{early_stop_suffix}.pkl"
    
    # Prepare metadata
    metadata = {
        'model_name': model_name,
        'experiment_name': experiment_name,
        'clean_experiment_name': clean_experiment_name,
        'timestamp': TIMESTAMP,
        'random_seed': random_seed,
        'early_stopping_used': args.early_stop,
        'hyperparameters': hyperparams,
        'performance_metrics': metrics,
        'config_used': config,
        'task_type': 'user_identification',
        'data_preprocessing': 'pre_normalized'
    }
    
    # Save model and metadata
    model_data = {
        'model': model,
        'metadata': metadata
    }
    
    filepath = OUTPUT_DIR / filename
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"💾 Model saved: {filename}")
    return str(filepath)


def run_with_multiple_seeds(model_func, X_train, X_test, y_train, y_test, experiment_name, model_name):
    """Run model with multiple random seeds if specified"""
    all_results = []
    
    seeds = [42, 123, 456, 789, 999] if args.seeds > 1 else [42]
    seeds = seeds[:args.seeds]  # Use only requested number of seeds
    
    for seed_idx, seed in enumerate(seeds):
        print(f"  🎲 Running with random seed {seed} ({seed_idx + 1}/{len(seeds)})")
        
        # Set random seeds for reproducibility
        np.random.seed(seed)
        
        try:
            result = model_func(X_train, X_test, y_train, y_test, experiment_name, seed)
            result['random_seed'] = seed
            all_results.append(result)
        except Exception as e:
            print(f"❌ Failed with seed {seed}: {e}")
    
    return all_results


def run_xgboost_model(X_train, X_test, y_train, y_test, experiment_name="unknown", random_seed=42):
    """Enhanced XGBoost model with comprehensive tracking"""
    global CURRENT_EXPERIMENT
    CURRENT_EXPERIMENT += 1
    
    print(f"\n🚀 Running XGBoost - {experiment_name} ({CURRENT_EXPERIMENT}/{TOTAL_EXPERIMENTS})")
    
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
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    min_samples_per_class = y_train.value_counts().min()
    print(f"Minimum samples per class: {min_samples_per_class}")

    # Configure early stopping and parallelism
    early_stopping_rounds = 50 if args.early_stop else None
    eval_set = [(X_test_scaled, y_test_encoded)] if args.early_stop else None
    n_jobs = min(4, os.cpu_count() // 2) if os.cpu_count() > 4 else 1

    if min_samples_per_class < 2:
        print("⚠️ Not enough samples for CV → using default parameters")
        best_params = {
            'n_estimators': 1000 if args.early_stop else 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_lambda': 1,
            'random_state': random_seed,
            'n_jobs': n_jobs,
        }
        best_xgb = XGBClassifier(**best_params)
        
        with tqdm(desc="Training XGBoost", unit="epoch") as pbar:
            if args.early_stop:
                best_xgb.fit(X_train_scaled, y_train_encoded, 
                             early_stopping_rounds=early_stopping_rounds,
                             eval_set=eval_set, verbose=False)
            else:
                best_xgb.fit(X_train_scaled, y_train_encoded)
            pbar.update(1)
    else:
        stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_seed)

        # Expanded parameter grid
        param_grid = {
            "n_estimators": [1000] if args.early_stop else [100, 200, 500],
            "max_depth": [3, 4, 6, 8],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "reg_lambda": [0.1, 1, 3, 10],
            "reg_alpha": [0, 0.1, 1],
        }

        base_estimator = XGBClassifier(random_state=random_seed, n_jobs=n_jobs)
        
        with tqdm(desc="Grid Search XGBoost") as pbar:
            grid_search_xgb = GridSearchCV(
                base_estimator,
                param_grid,
                scoring="accuracy",
                cv=stratified_kfold,
                n_jobs=n_jobs,
                verbose=0,
                return_train_score=True
            )
            grid_search_xgb.fit(X_train_scaled, y_train_encoded)
            pbar.update(1)
            
        # Store hyperparameter analysis data
        cv_results = pd.DataFrame(grid_search_xgb.cv_results_)
        for idx, row in cv_results.iterrows():
            hp_record = {
                'model': 'XGBoost',
                'experiment': experiment_name,
                'random_seed': random_seed,
                'cv_score': row['mean_test_score'],
                'cv_std': row['std_test_score'],
                'params': str(row['params']),
                'rank': row['rank_test_score']
            }
            # Add individual parameters for analysis
            for param, value in row['params'].items():
                hp_record[f'param_{param}'] = value
            HYPERPARAMETER_ANALYSIS.append(hp_record)
            
        best_xgb = grid_search_xgb.best_estimator_
        best_params = grid_search_xgb.best_params_
        print(f"Best XGBoost params: {best_params}")
        
        # Refit with early stopping if requested
        if args.early_stop:
            best_xgb.set_params(n_estimators=1000)
            with tqdm(desc="Final XGBoost Training") as pbar:
                best_xgb.fit(X_train_scaled, y_train_encoded,
                            early_stopping_rounds=early_stopping_rounds,
                            eval_set=eval_set, verbose=False)
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
    create_top_k_confusion_matrices(y_test, y_pred_proba_test, label_encoder,
                                   f"XGBoost {clean_exp_name}", f"xgboost_{clean_exp_name}")
    
    # Feature importance
    if config.get("draw_feature_importance_graph", False):
        plt.figure(figsize=(12, 8))
        plot_importance(best_xgb, importance_type="weight", max_num_features=15)
        plt.title(f"XGBoost Feature Importance - {clean_exp_name}\nBest Params: {str(best_params)[:100]}...")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"xgboost_feature_importance_{clean_exp_name}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()

    # Save model
    hyperparams = best_params if 'best_params' in locals() else best_xgb.get_params()
    model_path = save_model_with_metadata(
        best_xgb, "XGBoost", experiment_name, hyperparams, all_metrics, random_seed
    )
    
    # Store results
    result_record = {
        'model': 'XGBoost',
        'experiment': experiment_name,
        'random_seed': random_seed,
        'early_stopping': args.early_stop,
        'model_path': model_path,
        'hyperparameters': str(hyperparams),
        **all_metrics
    }
    EXPERIMENT_RESULTS.append(result_record)
    
    # Store detailed results
    for k in range(1, 6):
        detailed_record = {
            'model': 'XGBoost',
            'experiment': experiment_name,
            'random_seed': random_seed,
            'early_stopping': args.early_stop,
            'k_value': k,
            'train_top_k_accuracy': train_metrics.get(f'train_top_{k}_accuracy', 0),
            'test_top_k_accuracy': test_metrics.get(f'test_top_{k}_accuracy', 0),
            'model_path': model_path,
            'hyperparameters': str(hyperparams),
            'timestamp': TIMESTAMP
        }
        DETAILED_RESULTS.append(detailed_record)
    
    return all_metrics


# I'll continue with the other models in the next response due to length limits...
def run_random_forest_model(X_train, X_test, y_train, y_test, experiment_name="unknown", random_seed=42):
    """Enhanced Random Forest model"""
    global CURRENT_EXPERIMENT
    CURRENT_EXPERIMENT += 1
    
    print(f"\n🌲 Running Random Forest - {experiment_name} ({CURRENT_EXPERIMENT}/{TOTAL_EXPERIMENTS})")
    
    # Validate data and use pre-normalized values
    is_valid, issues = validate_data_quality(X_train, X_test, y_train, y_test)
    if not is_valid:
        print(f"⚠️ Data quality issues: {issues}")
    
    X_train_scaled = X_train.values
    X_test_scaled = X_test.values

    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    min_samples_per_class = y_train.value_counts().min()
    print(f"Minimum samples per class: {min_samples_per_class}")
    
    n_jobs = min(4, os.cpu_count() // 2) if os.cpu_count() > 4 else 1
    
    if min_samples_per_class < 2:
        print("⚠️ Not enough samples for CV → using default parameters")
        best_rf = RandomForestClassifier(
            n_estimators=500,
            random_state=random_seed,
            n_jobs=n_jobs
        )
        with tqdm(desc="Training Random Forest") as pbar:
            best_rf.fit(X_train_scaled, y_train_encoded)
            pbar.update(1)
        best_params = best_rf.get_params()
    else:
        stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_seed)

        # Expanded parameter grid
        param_grid = {
            "n_estimators": [100, 300, 500, 800],
            "max_depth": [5, 10, 15, 20, None],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", "log2", 0.5, 0.7],
            "bootstrap": [True, False],
        }

        with tqdm(desc="Grid Search Random Forest") as pbar:
            grid_search_rf = GridSearchCV(
                RandomForestClassifier(random_state=random_seed, n_jobs=n_jobs),
                param_grid,
                scoring="accuracy",
                cv=stratified_kfold,
                n_jobs=n_jobs,
                verbose=0,
                return_train_score=True
            )
            grid_search_rf.fit(X_train_scaled, y_train_encoded)
            pbar.update(1)

        # Store hyperparameter analysis
        cv_results = pd.DataFrame(grid_search_rf.cv_results_)
        for idx, row in cv_results.iterrows():
            hp_record = {
                'model': 'RandomForest',
                'experiment': experiment_name,
                'random_seed': random_seed,
                'cv_score': row['mean_test_score'],
                'cv_std': row['std_test_score'],
                'params': str(row['params']),
                'rank': row['rank_test_score']
            }
            for param, value in row['params'].items():
                hp_record[f'param_{param}'] = value
            HYPERPARAMETER_ANALYSIS.append(hp_record)

        best_rf = grid_search_rf.best_estimator_
        best_params = grid_search_rf.best_params_
        print(f"Best RF params: {best_params}")

    # Predictions and metrics (similar structure as XGBoost)
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
    
    print(f"Random Forest Train Top-1: {train_metrics.get('train_top_1_accuracy', 0):.4f}")
    print(f"Random Forest Test Top-1: {test_metrics.get('test_top_1_accuracy', 0):.4f}")
    print(f"Random Forest Test Top-5: {test_metrics.get('test_top_5_accuracy', 0):.4f}")
    
    clean_exp_name = experiment_name.replace("_no_scaling", "")
    create_top_k_confusion_matrices(y_test, y_pred_proba_test, label_encoder,
                                   f"Random Forest {clean_exp_name}", f"rf_{clean_exp_name}")

    # Feature importance with hyperparameter info
    if config.get("draw_feature_importance_graph", False):
        plt.figure(figsize=(12, 8))
        feature_importances = best_rf.feature_importances_
        feature_names = X_train.columns
        indices = np.argsort(feature_importances)[-15:]
        
        plt.barh(range(len(indices)), feature_importances[indices], align="center")
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel("Feature Importance")
        plt.title(f"Random Forest Feature Importance - {clean_exp_name}\nBest Params: {str(best_params)[:100]}...")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"rf_feature_importance_{clean_exp_name}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()

    # Save model and results (similar to XGBoost pattern)
    model_path = save_model_with_metadata(
        best_rf, "RandomForest", experiment_name, best_params, all_metrics, random_seed
    )
    
    result_record = {
        'model': 'RandomForest',
        'experiment': experiment_name,
        'random_seed': random_seed,
        'early_stopping': False,
        'model_path': model_path,
        'hyperparameters': str(best_params),
        **all_metrics
    }
    EXPERIMENT_RESULTS.append(result_record)
    
    for k in range(1, 6):
        detailed_record = {
            'model': 'RandomForest',
            'experiment': experiment_name,
            'random_seed': random_seed,
            'early_stopping': False,
            'k_value': k,
            'train_top_k_accuracy': train_metrics.get(f'train_top_{k}_accuracy', 0),
            'test_top_k_accuracy': test_metrics.get(f'test_top_{k}_accuracy', 0),
            'model_path': model_path,
            'hyperparameters': str(best_params),
            'timestamp': TIMESTAMP
        }
        DETAILED_RESULTS.append(detailed_record)
    
    return all_metrics


# Similar implementations for CatBoost, SVM, and new models...
# [The rest of the models follow the same pattern but I need to continue in another artifact due to length]

def create_performance_plots(results_df):
    """Create comprehensive performance plots"""
    if results_df.empty:
        print("⚠️ No results to plot")
        return
        
    # Enhanced plotting with multiple seeds analysis
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Top-1 Accuracy by Model', 'Top-K Accuracy Progression', 
                       'Random Seed Stability', 'Hyperparameter Impact'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    models = results_df['model'].unique()
    colors = px.colors.qualitative.Set1[:len(models)]
    
    # Top-1 accuracy by model (with error bars if multiple seeds)
    for i, model in enumerate(models):
        model_data = results_df[results_df['model'] == model]
        
        if 'random_seed' in model_data.columns and len(model_data['random_seed'].unique()) > 1:
            # Multiple seeds - show mean and std
            seed_stats = model_data.groupby('experiment')['test_top_1_accuracy'].agg(['mean', 'std']).reset_index()
            fig.add_trace(
                go.Scatter(x=seed_stats['experiment'], y=seed_stats['mean'],
                          error_y=dict(type='data', array=seed_stats['std']),
                          name=f'{model} (avg)', line=dict(color=colors[i]),
                          mode='lines+markers'),
                row=1, col=1
            )
        else:
            # Single seed
            fig.add_trace(
                go.Scatter(x=model_data['experiment'], y=model_data['test_top_1_accuracy'],
                          name=f'{model}', line=dict(color=colors[i]),
                          mode='lines+markers'),
                row=1, col=1
            )
    
    fig.update_layout(height=800, title_text="Enhanced User Identification Performance Analysis")
    fig.write_html(OUTPUT_DIR / "performance_plots.html")


def generate_html_report():
    """Generate comprehensive HTML report with hyperparameter analysis"""
    if not EXPERIMENT_RESULTS:
        print("⚠️ No experiment results to report")
        return
    
    # Convert to DataFrames
    results_df = pd.DataFrame(EXPERIMENT_RESULTS)
    detailed_df = pd.DataFrame(DETAILED_RESULTS)
    
    # Save CSV files
    csv_path = OUTPUT_DIR / f"experiment_results_{TIMESTAMP}.csv"
    detailed_csv_path = OUTPUT_DIR / f"detailed_topk_results_{TIMESTAMP}.csv"
    
    results_df.to_csv(csv_path, index=False)
    detailed_df.to_csv(detailed_csv_path, index=False)
    
    print(f"📊 Results saved to: {csv_path}")
    print(f"📊 Detailed Top-K results saved to: {detailed_csv_path}")
    
    # Analyze hyperparameters
    analyze_hyperparameter_performance()
    
    # Create performance plots
    create_performance_plots(results_df)
    
    # Generate enhanced HTML report
    best_top1_idx = results_df['test_top_1_accuracy'].fillna(0).idxmax()
    best_top5_idx = results_df['test_top_5_accuracy'].fillna(0).idxmax()
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Enhanced User Identification Results - {TIMESTAMP}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
            .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 10px; margin-bottom: 30px; }}
            .section {{ background-color: white; padding: 20px; margin-bottom: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #34495e; color: white; }}
            .metric-highlight {{ font-weight: bold; color: #27ae60; }}
            .warning {{ color: #e74c3c; font-weight: bold; }}
            .info-box {{ background-color: #ecf0f1; padding: 15px; border-left: 4px solid #3498db; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔐 Enhanced User Identification Analysis</h1>
            <p><strong>Timestamp:</strong> {TIMESTAMP}</p>
            <p><strong>Task:</strong> Cross-Platform User Identification</p>
            <p><strong>Random Seeds Tested:</strong> {args.seeds}</p>
            <p><strong>Early Stopping:</strong> {'✅ Enabled' if args.early_stop else '❌ Disabled'}</p>
            <p><strong>Total Model Runs:</strong> {len(results_df)}</p>
            <p><strong>Hyperparameter Combinations:</strong> {len(HYPERPARAMETER_ANALYSIS)}</p>
        </div>
        
        <div class="section">
            <h2>🎯 Peak Performance Summary</h2>
            <div class="info-box">
                <p><strong>Best Top-1 Accuracy:</strong> <span class="metric-highlight">{results_df.loc[best_top1_idx, 'test_top_1_accuracy']:.4f}</span></p>
                <p><strong>Best Top-1 Model:</strong> <span class="metric-highlight">{results_df.loc[best_top1_idx, 'model']} ({results_df.loc[best_top1_idx, 'experiment']})</span></p>
                <p><strong>Best Top-5 Accuracy:</strong> <span class="metric-highlight">{results_df.loc[best_top5_idx, 'test_top_5_accuracy']:.4f}</span></p>
                <p><strong>Best Top-5 Model:</strong> <span class="metric-highlight">{results_df.loc[best_top5_idx, 'model']} ({results_df.loc[best_top5_idx, 'experiment']})</span></p>
            </div>
        </div>
        
        <div class="section">
            <h2>🔬 Hyperparameter Analysis Insights</h2>
            <p>Grid search tested <strong>{len(HYPERPARAMETER_ANALYSIS)}</strong> hyperparameter combinations across all models.</p>
            <p>📊 <strong>Detailed Analysis:</strong> See hyperparameter_analysis_{TIMESTAMP}.csv for complete results</p>
            <p>💡 <strong>Recommendations:</strong> Check console output for specific parameter expansion suggestions</p>
        </div>
        
        <div class="section">
            <h2>📁 Complete File Inventory</h2>
            <ul>
                <li>📊 <strong>Summary Results:</strong> experiment_results_{TIMESTAMP}.csv</li>
                <li>📋 <strong>Detailed Top-K Results:</strong> detailed_topk_results_{TIMESTAMP}.csv</li>
                <li>🔬 <strong>Hyperparameter Analysis:</strong> hyperparameter_analysis_{TIMESTAMP}.csv</li>
                <li>📈 <strong>Interactive Plots:</strong> performance_plots.html</li>
                <li>🤖 <strong>Trained Models:</strong> {len([r for r in EXPERIMENT_RESULTS if 'model_path' in r])} models with full metadata</li>
                <li>🖼️ <strong>Confusion Matrices:</strong> Enhanced Top-1 and Top-5 visualizations</li>
                <li>📊 <strong>Feature Importance:</strong> With hyperparameter annotations</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    html_path = OUTPUT_DIR / f"enhanced_user_identification_report_{TIMESTAMP}.html"
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"📋 Enhanced HTML report: {html_path}")
    return html_path


def set_total_experiments(total):
    """Set total experiments for progress tracking"""
    global TOTAL_EXPERIMENTS
    TOTAL_EXPERIMENTS = total


def finalize_experiments():
    """Generate all reports and analysis"""
    if EXPERIMENT_RESULTS:
        generate_html_report()
        print(f"\n🎉 Enhanced user identification analysis complete!")
        print(f"📁 All results in: {OUTPUT_DIR}")
        print(f"🔍 Key insights:")
        print(f"   - Models tested: {len(set([r['model'] for r in EXPERIMENT_RESULTS]))}")
        print(f"   - Hyperparameter combinations: {len(HYPERPARAMETER_ANALYSIS)}")
        print(f"   - Random seeds: {args.seeds}")
    else:
        print("⚠️ No experiments were run")


if __name__ == "__main__":
    print(f"🚀 Enhanced User Identification ML Models loaded")
    print(f"⚡ Early stopping: {args.early_stop}")
    print(f"🎲 Random seeds: {args.seeds}")
    print(f"📁 Results: {OUTPUT_DIR}")