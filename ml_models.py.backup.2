import os
import json
import enum
import pickle
import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.svm import SVC
from extended_minmax import ExtendedMinMaxScalar
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
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


class ScalarType(enum.Enum):
    """Enum class representing the different types of scalers available."""
    STANDARD = 1
    MIN_MAX = 2
    EXTENDED_MIN_MAX = 3


# Global configuration and results storage
with open(os.path.join(os.getcwd(), "classifier_config.json"), "r") as f:
    config = json.load(f)

# Add command line argument parsing
parser = argparse.ArgumentParser(description='Run ML experiments with optional early stopping')
parser.add_argument('-e', '--early-stop', action='store_true', 
                   help='Use early stopping for XGBoost and CatBoost models')
args = parser.parse_args()

# Global results storage
EXPERIMENT_RESULTS = []
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
EARLY_STOP_SUFFIX = "_early_stop" if args.early_stop else ""
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


def calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba, label_encoder, dataset_type="test"):
    """Calculate comprehensive metrics for both train and test sets"""
    metrics = {}
    
    # Basic metrics
    metrics[f'{dataset_type}_accuracy'] = accuracy_score(y_true, y_pred)
    metrics[f'{dataset_type}_f1_weighted'] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics[f'{dataset_type}_f1_macro'] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics[f'{dataset_type}_precision_weighted'] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics[f'{dataset_type}_precision_macro'] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics[f'{dataset_type}_recall_weighted'] = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics[f'{dataset_type}_recall_macro'] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    
    # Encode labels if needed
    if isinstance(y_true.iloc[0] if hasattr(y_true, 'iloc') else y_true[0], str):
        y_true_encoded = label_encoder.transform(y_true)
        y_pred_encoded = label_encoder.transform(y_pred)
    else:
        y_true_encoded = y_true
        y_pred_encoded = y_pred
    
    # Top-k accuracy
    for k in range(1, min(6, len(label_encoder.classes_) + 1)):
        top_k_acc = top_k_accuracy_score(y_true_encoded, y_pred_proba, k=k)
        metrics[f'{dataset_type}_top_{k}_accuracy'] = top_k_acc
    
    # Recognition rate using bob.measure
    try:
        rr_scores = []
        for i, true_label in enumerate(y_true_encoded):
            pos_score = y_pred_proba[i, true_label]
            neg_scores = np.delete(y_pred_proba[i], true_label)
            rr_scores.append((neg_scores, [pos_score]))
        
        recognition_rate = bob.measure.recognition_rate(rr_scores, rank=1)
        metrics[f'{dataset_type}_recognition_rate'] = recognition_rate
    except Exception as e:
        print(f"⚠️ Could not calculate recognition rate: {e}")
        metrics[f'{dataset_type}_recognition_rate'] = 0.0
    
    return metrics


def create_confusion_matrix_plot(y_true, y_pred, title, filename):
    """Create confusion matrix visualization"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {title}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{filename}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_performance_plots(results_df):
    """Create comprehensive performance plots"""
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Accuracy Comparison', 'F1 Score Comparison', 
                       'Precision vs Recall', 'Top-k Accuracy'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    models = results_df['model'].unique()
    colors = px.colors.qualitative.Set1[:len(models)]
    
    # Accuracy comparison
    for i, model in enumerate(models):
        model_data = results_df[results_df['model'] == model]
        fig.add_trace(
            go.Scatter(x=model_data['experiment'], y=model_data['test_accuracy'],
                      name=f'{model} (Test)', line=dict(color=colors[i]),
                      mode='lines+markers'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=model_data['experiment'], y=model_data['train_accuracy'],
                      name=f'{model} (Train)', line=dict(color=colors[i], dash='dash'),
                      mode='lines+markers'),
            row=1, col=1
        )
    
    # F1 Score comparison
    for i, model in enumerate(models):
        model_data = results_df[results_df['model'] == model]
        fig.add_trace(
            go.Scatter(x=model_data['experiment'], y=model_data['test_f1_weighted'],
                      name=f'{model} F1', line=dict(color=colors[i]),
                      mode='lines+markers', showlegend=False),
            row=1, col=2
        )
    
    # Precision vs Recall
    for i, model in enumerate(models):
        model_data = results_df[results_df['model'] == model]
        fig.add_trace(
            go.Scatter(x=model_data['test_precision_weighted'], y=model_data['test_recall_weighted'],
                      name=f'{model}', mode='markers', marker=dict(color=colors[i], size=10),
                      showlegend=False),
            row=2, col=1
        )
    
    # Top-k accuracy (using top-1 and top-3 if available)
    for i, model in enumerate(models):
        model_data = results_df[results_df['model'] == model]
        if 'test_top_1_accuracy' in model_data.columns:
            fig.add_trace(
                go.Scatter(x=model_data['experiment'], y=model_data['test_top_1_accuracy'],
                          name=f'{model} Top-1', line=dict(color=colors[i]),
                          mode='lines+markers', showlegend=False),
                row=2, col=2
            )
    
    fig.update_layout(height=800, title_text="Model Performance Overview")
    fig.write_html(OUTPUT_DIR / "performance_plots.html")


def save_model_with_metadata(model, model_name, experiment_name, scaler_type, hyperparams, metrics):
    """Save model with comprehensive metadata"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    early_stop_suffix = "_early_stop" if args.early_stop else ""
    
    # Create filename with all relevant info
    filename = f"{model_name}_{experiment_name}_{scaler_type.name}_{timestamp}{early_stop_suffix}.pkl"
    
    # Prepare metadata
    metadata = {
        'model_name': model_name,
        'experiment_name': experiment_name,
        'scaler_type': scaler_type.name,
        'timestamp': timestamp,
        'early_stopping_used': args.early_stop,
        'hyperparameters': hyperparams,
        'performance_metrics': metrics,
        'config_used': config
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


def run_xgboost_model(X_train, X_test, y_train, y_test, scalar_obj: ScalarType, 
                     experiment_name="unknown", max_k=5):
    """Enhanced XGBoost model with comprehensive tracking"""
    print(f"\n🚀 Running XGBoost - {experiment_name} - {scalar_obj.name}")
    
    # Validate data quality
    is_valid, issues = validate_data_quality(X_train, X_test, y_train, y_test)
    if not is_valid:
        print(f"⚠️ Data quality issues: {issues}")
    
    # Scale the features
    if scalar_obj == ScalarType.MIN_MAX:
        scaler = MinMaxScaler()
    elif scalar_obj == ScalarType.STANDARD:
        scaler = StandardScaler()
    elif scalar_obj == ScalarType.EXTENDED_MIN_MAX:
        scaler = ExtendedMinMaxScalar()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Sanity check for NaNs or infinities
    if not np.isfinite(X_train_scaled).all():
        raise ValueError("❌ X_train_scaled contains NaNs or infinite values")
    if not np.isfinite(X_test_scaled).all():
        raise ValueError("❌ X_test_scaled contains NaNs or infinite values")

    # Encode the labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    min_samples_per_class = y_train.value_counts().min()
    print(f"Minimum samples per class in training set: {min_samples_per_class}")

    # Configure early stopping if requested
    early_stopping_rounds = 50 if args.early_stop else None
    eval_set = [(X_test_scaled, y_test_encoded)] if args.early_stop else None

    if min_samples_per_class < 2:
        print("⚠️ Not enough samples for StratifiedKFold → fitting XGBoost without CV.")
        best_params = {
            'n_estimators': 1000 if args.early_stop else 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 1.0,
            'colsample_bytree': 1.0,
            'reg_lambda': 1,
            'n_jobs': -1,
        }
        best_xgb = XGBClassifier(**best_params)
        
        if args.early_stop:
            best_xgb.fit(X_train_scaled, y_train_encoded, 
                         early_stopping_rounds=early_stopping_rounds,
                         eval_set=eval_set, verbose=False)
        else:
            best_xgb.fit(X_train_scaled, y_train_encoded)
    else:
        stratified_kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

        param_grid = {
            "n_estimators": [1000] if args.early_stop else [100, 200, 300],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.01, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "reg_lambda": [1, 3, 5],
        }

        base_estimator = XGBClassifier(n_jobs=-1)
        
        grid_search_xgb = GridSearchCV(
            base_estimator,
            param_grid,
            scoring="accuracy",
            cv=stratified_kfold,
            n_jobs=-1,
            verbose=1,
        )
        grid_search_xgb.fit(X_train_scaled, y_train_encoded)
        best_xgb = grid_search_xgb.best_estimator_
        best_params = grid_search_xgb.best_params_
        
        # Refit with early stopping if requested
        if args.early_stop:
            best_xgb.set_params(n_estimators=1000)
            best_xgb.fit(X_train_scaled, y_train_encoded,
                        early_stopping_rounds=early_stopping_rounds,
                        eval_set=eval_set, verbose=False)

    # Predictions
    y_pred_train = best_xgb.predict(X_train_scaled)
    y_pred_test = best_xgb.predict(X_test_scaled)
    
    y_pred_train_decoded = label_encoder.inverse_transform(y_pred_train)
    y_pred_test_decoded = label_encoder.inverse_transform(y_pred_test)
    
    y_pred_proba_train = best_xgb.predict_proba(X_train_scaled)
    y_pred_proba_test = best_xgb.predict_proba(X_test_scaled)

    # Calculate comprehensive metrics
    train_metrics = calculate_comprehensive_metrics(
        y_train, y_pred_train_decoded, y_pred_proba_train, label_encoder, "train"
    )
    test_metrics = calculate_comprehensive_metrics(
        y_test, y_pred_test_decoded, y_pred_proba_test, label_encoder, "test"
    )
    
    # Combine metrics
    all_metrics = {**train_metrics, **test_metrics}
    
    # Print key results
    print(f"XGBoost Train Accuracy: {train_metrics['train_accuracy']:.4f}")
    print(f"XGBoost Test Accuracy: {test_metrics['test_accuracy']:.4f}")
    print(f"Overfitting Check - Accuracy Gap: {train_metrics['train_accuracy'] - test_metrics['test_accuracy']:.4f}")
    
    # Create confusion matrix
    create_confusion_matrix_plot(y_test, y_pred_test_decoded, 
                               f"XGBoost {experiment_name}", f"xgboost_{experiment_name}")
    
    # Feature importance plot
    if config.get("draw_feature_importance_graph", False):
        plt.figure(figsize=(12, 8))
        plot_importance(best_xgb, importance_type="weight", max_num_features=15)
        plt.title(f"XGBoost Feature Importance - {experiment_name}")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"xgboost_feature_importance_{experiment_name}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()

    # Save model
    hyperparams = best_params if 'best_params' in locals() else best_xgb.get_params()
    model_path = save_model_with_metadata(
        best_xgb, "XGBoost", experiment_name, scalar_obj, hyperparams, all_metrics
    )
    
    # Store results for reporting
    result_record = {
        'model': 'XGBoost',
        'experiment': experiment_name,
        'scaler': scalar_obj.name,
        'early_stopping': args.early_stop,
        'model_path': model_path,
        **all_metrics
    }
    EXPERIMENT_RESULTS.append(result_record)
    
    return all_metrics


def run_random_forest_model(X_train, X_test, y_train, y_test, scalar_obj: ScalarType, 
                           experiment_name="unknown", max_k=5):
    """Enhanced Random Forest model with comprehensive tracking"""
    print(f"\n🌲 Running Random Forest - {experiment_name} - {scalar_obj.name}")
    
    # Validate data quality
    is_valid, issues = validate_data_quality(X_train, X_test, y_train, y_test)
    if not is_valid:
        print(f"⚠️ Data quality issues: {issues}")
    
    # Scale the features
    if scalar_obj == ScalarType.MIN_MAX:
        scaler = MinMaxScaler()
    elif scalar_obj == ScalarType.STANDARD:
        scaler = StandardScaler()
    elif scalar_obj == ScalarType.EXTENDED_MIN_MAX:
        scaler = ExtendedMinMaxScalar()
        
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Encode the labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    train_min_samples_per_class = y_train.value_counts().min()
    test_min_samples_per_class = y_test.value_counts().min()
    print(f"Minimum samples per class in training set: {train_min_samples_per_class}")
    print(f"Minimum samples per class in test set: {test_min_samples_per_class}")
    
    if train_min_samples_per_class < 2:
        print("⚠️ Not enough samples for StratifiedKFold → fitting without CV.")
        best_rf = RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            n_jobs=-1
        ).fit(X_train_scaled, y_train_encoded)
        best_params = best_rf.get_params()
    else:
        stratified_kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

        # Define the hyperparameter grid
        param_grid = {
            "n_estimators": [100, 200, 500],
            "max_depth": [10, 20, 30, None],
            "min_samples_split": [2, 10, 20],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
            "bootstrap": [True, False],
        }

        # Perform GridSearchCV for hyperparameter tuning
        grid_search_rf = GridSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=-1),
            param_grid,
            scoring="accuracy",
            cv=stratified_kfold,
            n_jobs=-1,
            verbose=1,
        )
        grid_search_rf.fit(X_train_scaled, y_train_encoded)

        # Get the best estimator
        best_rf = grid_search_rf.best_estimator_
        best_params = grid_search_rf.best_params_

    # Predictions
    y_pred_train = best_rf.predict(X_train_scaled)
    y_pred_test = best_rf.predict(X_test_scaled)
    
    y_pred_train_decoded = label_encoder.inverse_transform(y_pred_train)
    y_pred_test_decoded = label_encoder.inverse_transform(y_pred_test)
    
    y_pred_proba_train = best_rf.predict_proba(X_train_scaled)
    y_pred_proba_test = best_rf.predict_proba(X_test_scaled)

    # Calculate comprehensive metrics
    train_metrics = calculate_comprehensive_metrics(
        y_train, y_pred_train_decoded, y_pred_proba_train, label_encoder, "train"
    )
    test_metrics = calculate_comprehensive_metrics(
        y_test, y_pred_test_decoded, y_pred_proba_test, label_encoder, "test"
    )
    
    # Combine metrics
    all_metrics = {**train_metrics, **test_metrics}
    
    # Print key results
    print(f"Random Forest Train Accuracy: {train_metrics['train_accuracy']:.4f}")
    print(f"Random Forest Test Accuracy: {test_metrics['test_accuracy']:.4f}")
    print(f"Overfitting Check - Accuracy Gap: {train_metrics['train_accuracy'] - test_metrics['test_accuracy']:.4f}")
    
    # Create confusion matrix
    create_confusion_matrix_plot(y_test, y_pred_test_decoded, 
                               f"Random Forest {experiment_name}", f"rf_{experiment_name}")

    # Plot feature importance
    if config.get("draw_feature_importance_graph", False):
        plt.figure(figsize=(12, 8))
        feature_importances = best_rf.feature_importances_
        feature_names = X_train.columns
        indices = np.argsort(feature_importances)[-15:]
        
        plt.barh(range(len(indices)), feature_importances[indices], align="center")
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel("Feature Importance")
        plt.title(f"Random Forest Feature Importance - {experiment_name}")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"rf_feature_importance_{experiment_name}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()

    # Save model
    model_path = save_model_with_metadata(
        best_rf, "RandomForest", experiment_name, scalar_obj, best_params, all_metrics
    )
    
    # Store results for reporting
    result_record = {
        'model': 'RandomForest',
        'experiment': experiment_name,
        'scaler': scalar_obj.name,
        'early_stopping': False,  # RF doesn't support early stopping
        'model_path': model_path,
        **all_metrics
    }
    EXPERIMENT_RESULTS.append(result_record)
    
    return all_metrics


def run_catboost_model(X_train, X_test, y_train, y_test, scalar_obj: ScalarType, 
                      experiment_name="unknown", max_k=5):
    """Enhanced CatBoost model with comprehensive tracking"""
    print(f"\n🐱 Running CatBoost - {experiment_name} - {scalar_obj.name}")
    
    # Validate data quality
    is_valid, issues = validate_data_quality(X_train, X_test, y_train, y_test)
    if not is_valid:
        print(f"⚠️ Data quality issues: {issues}")
    
    # Scale the features
    if scalar_obj == ScalarType.MIN_MAX:
        scaler = MinMaxScaler()
    elif scalar_obj == ScalarType.STANDARD:
        scaler = StandardScaler()
    elif scalar_obj == ScalarType.EXTENDED_MIN_MAX:
        scaler = ExtendedMinMaxScalar()
        
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    min_samples_per_class = y_train.value_counts().min()
    print(f"Minimum samples per class in training set: {min_samples_per_class}")

    # Configure early stopping if requested
    early_stopping_rounds = 50 if args.early_stop else None
    eval_set = Pool(X_test_scaled, y_test_encoded) if args.early_stop else None

    if min_samples_per_class < 2:
        print("⚠️ Not enough samples for StratifiedKFold → fitting CatBoost without CV.")
        
        best_params = {
            'iterations': 1000 if args.early_stop else 100,
            'depth': 6,
            'learning_rate': 0.1,
            'l2_leaf_reg': 3,
            'border_count': 64,
            'random_seed': 42,
            'verbose': 0,
        }
        
        best_catboost_model = CatBoostClassifier(**best_params)
        
        if args.early_stop and eval_set:
            best_catboost_model.fit(
                X_train_scaled, y_train_encoded,
                early_stopping_rounds=early_stopping_rounds,
                eval_set=eval_set,
                use_best_model=True
            )
        else:
            best_catboost_model.fit(X_train_scaled, y_train_encoded)
    else:
        stratified_kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

        param_grid = {
            "iterations": [1000] if args.early_stop else [100, 200, 300],
            "depth": [4, 6, 8],
            "learning_rate": [0.01, 0.05, 0.1],
            "l2_leaf_reg": [1, 3, 5],
            "border_count": [32, 64, 128],
        }

        grid_search = GridSearchCV(
            estimator=CatBoostClassifier(verbose=0, random_seed=42),
            param_grid=param_grid,
            scoring="accuracy",
            cv=stratified_kfold,
            verbose=1,
            n_jobs=-1,
        )

        grid_search.fit(X_train_scaled, y_train_encoded)
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        print(f"Best Parameters: {best_params}")
        print(f"Best Cross-Validation Score: {best_score}")

        # Train final model with best params
        best_catboost_model = CatBoostClassifier(**best_params, random_seed=42, verbose=0)
        
        if args.early_stop and eval_set:
            best_catboost_model.fit(
                X_train_scaled, y_train_encoded,
                early_stopping_rounds=early_stopping_rounds,
                eval_set=eval_set,
                use_best_model=True
            )
        else:
            best_catboost_model.fit(X_train_scaled, y_train_encoded)

    # Predictions
    y_pred_train = best_catboost_model.predict(X_train_scaled)
    y_pred_test = best_catboost_model.predict(X_test_scaled)
    
    y_pred_train_decoded = label_encoder.inverse_transform(y_pred_train)
    y_pred_test_decoded = label_encoder.inverse_transform(y_pred_test)
    
    y_pred_proba_train = best_catboost_model.predict_proba(X_train_scaled)
    y_pred_proba_test = best_catboost_model.predict_proba(X_test_scaled)

    # Calculate comprehensive metrics
    train_metrics = calculate_comprehensive_metrics(
        y_train, y_pred_train_decoded, y_pred_proba_train, label_encoder, "train"
    )
    test_metrics = calculate_comprehensive_metrics(
        y_test, y_pred_test_decoded, y_pred_proba_test, label_encoder, "test"
    )
    
    # Combine metrics
    all_metrics = {**train_metrics, **test_metrics}
    
    # Print key results
    print(f"CatBoost Train Accuracy: {train_metrics['train_accuracy']:.4f}")
    print(f"CatBoost Test Accuracy: {test_metrics['test_accuracy']:.4f}")
    print(f"Overfitting Check - Accuracy Gap: {train_metrics['train_accuracy'] - test_metrics['test_accuracy']:.4f}")
    
    # Create confusion matrix
    create_confusion_matrix_plot(y_test, y_pred_test_decoded, 
                               f"CatBoost {experiment_name}", f"catboost_{experiment_name}")

    # Feature importance plot
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
            plt.title(f"CatBoost Feature Importance - {experiment_name}")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"catboost_feature_importance_{experiment_name}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"⚠️ Could not generate feature importance plot: {e}")

    # Save model
    hyperparams = best_params if 'best_params' in locals() else best_catboost_model.get_params()
    model_path = save_model_with_metadata(
        best_catboost_model, "CatBoost", experiment_name, scalar_obj, hyperparams, all_metrics
    )
    
    # Store results for reporting
    result_record = {
        'model': 'CatBoost',
        'experiment': experiment_name,
        'scaler': scalar_obj.name,
        'early_stopping': args.early_stop,
        'model_path': model_path,
        **all_metrics
    }
    EXPERIMENT_RESULTS.append(result_record)
    
    return all_metrics


def run_svm_model(X_train, X_test, y_train, y_test, experiment_name="unknown", max_k=5):
    """Enhanced SVM model with comprehensive tracking"""
    print(f"\n⚙️ Running SVM - {experiment_name}")
    
    # Validate data quality
    is_valid, issues = validate_data_quality(X_train, X_test, y_train, y_test)
    if not is_valid:
        print(f"⚠️ Data quality issues: {issues}")
    
    # Scale features
    scaler = StandardScaler()  # SVM typically works best with StandardScaler
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    min_samples_per_class = y_train.value_counts().min()
    print(f"Minimum samples per class in training set: {min_samples_per_class}")

    if min_samples_per_class < 2:
        print("⚠️ Not enough samples for StratifiedKFold → fitting SVM without CV.")
        best_svm = SVC(
            C=1,
            kernel="rbf",
            gamma="scale",
            probability=True,
            random_state=42,
        ).fit(X_train_scaled, y_train_encoded)
        best_params = best_svm.get_params()
    else:
        stratified_kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

        param_grid = {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"],
        }

        grid_search_svm = GridSearchCV(
            SVC(probability=True, random_state=42),
            param_grid,
            scoring="accuracy",
            cv=stratified_kfold,
            n_jobs=-1,
            verbose=1,
        )
        grid_search_svm.fit(X_train_scaled, y_train_encoded)
        best_svm = grid_search_svm.best_estimator_
        best_params = grid_search_svm.best_params_

    # Predictions
    y_pred_train = best_svm.predict(X_train_scaled)
    y_pred_test = best_svm.predict(X_test_scaled)
    
    y_pred_train_decoded = label_encoder.inverse_transform(y_pred_train)
    y_pred_test_decoded = label_encoder.inverse_transform(y_pred_test)
    
    y_pred_proba_train = best_svm.predict_proba(X_train_scaled)
    y_pred_proba_test = best_svm.predict_proba(X_test_scaled)

    # Calculate comprehensive metrics
    train_metrics = calculate_comprehensive_metrics(
        y_train, y_pred_train_decoded, y_pred_proba_train, label_encoder, "train"
    )
    test_metrics = calculate_comprehensive_metrics(
        y_test, y_pred_test_decoded, y_pred_proba_test, label_encoder, "test"
    )
    
    # Combine metrics
    all_metrics = {**train_metrics, **test_metrics}
    
    # Print key results
    print(f"SVM Train Accuracy: {train_metrics['train_accuracy']:.4f}")
    print(f"SVM Test Accuracy: {test_metrics['test_accuracy']:.4f}")
    print(f"Overfitting Check - Accuracy Gap: {train_metrics['train_accuracy'] - test_metrics['test_accuracy']:.4f}")
    
    # Create confusion matrix
    create_confusion_matrix_plot(y_test, y_pred_test_decoded, 
                               f"SVM {experiment_name}", f"svm_{experiment_name}")

    # Feature importance for linear SVM
    if config.get("draw_feature_importance_graph", False) and best_svm.kernel == "linear":
        try:
            coef = best_svm.coef_[0]
            sorted_idx = np.argsort(np.abs(coef))[::-1][:15]
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(sorted_idx)), coef[sorted_idx])
            plt.xticks(range(len(sorted_idx)), X_train.columns[sorted_idx], rotation=45)
            plt.title(f"SVM Linear Coefficients (Top Features) - {experiment_name}")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"svm_feature_importance_{experiment_name}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"⚠️ Could not generate SVM feature importance plot: {e}")
    elif best_svm.kernel != "linear":
        print("⚠️ Note: SVM with non-linear kernel does not provide feature importance.")

    # Save model
    model_path = save_model_with_metadata(
        best_svm, "SVM", experiment_name, ScalarType.STANDARD, best_params, all_metrics
    )
    
    # Store results for reporting
    result_record = {
        'model': 'SVM',
        'experiment': experiment_name,
        'scaler': 'STANDARD',
        'early_stopping': False,  # SVM doesn't support early stopping
        'model_path': model_path,
        **all_metrics
    }
    EXPERIMENT_RESULTS.append(result_record)
    
    return all_metrics


def generate_html_report():
    """Generate comprehensive HTML report"""
    if not EXPERIMENT_RESULTS:
        print("⚠️ No experiment results to report")
        return
    
    # Convert to DataFrame
    results_df = pd.DataFrame(EXPERIMENT_RESULTS)
    
    # Save to CSV
    csv_path = OUTPUT_DIR / f"experiment_results_{TIMESTAMP}{EARLY_STOP_SUFFIX}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"📊 Results saved to: {csv_path}")
    
    # Create performance plots
    create_performance_plots(results_df)
    
    # Generate HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ML Experiment Results - {TIMESTAMP}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                background-color: #f5f5f5;
            }}
            .header {{
                background-color: #2c3e50;
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 30px;
            }}
            .section {{
                background-color: white;
                padding: 20px;
                margin-bottom: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #34495e;
                color: white;
            }}
            .metric-highlight {{
                font-weight: bold;
                color: #27ae60;
            }}
            .warning {{
                color: #e74c3c;
                font-weight: bold;
            }}
            .info-box {{
                background-color: #ecf0f1;
                padding: 15px;
                border-left: 4px solid #3498db;
                margin-bottom: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🤖 ML Experiment Results</h1>
            <p><strong>Timestamp:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p><strong>Early Stopping:</strong> {'✅ Enabled' if args.early_stop else '❌ Disabled'}</p>
            <p><strong>Total Experiments:</strong> {len(results_df)}</p>
        </div>
        
        <div class="section">
            <h2>📊 Summary Statistics</h2>
            <div class="info-box">
                <p><strong>Best Test Accuracy:</strong> <span class="metric-highlight">{results_df['test_accuracy'].max():.4f}</span></p>
                <p><strong>Best Model:</strong> <span class="metric-highlight">{results_df.loc[results_df['test_accuracy'].idxmax(), 'model']}</span></p>
                <p><strong>Best Experiment:</strong> <span class="metric-highlight">{results_df.loc[results_df['test_accuracy'].idxmax(), 'experiment']}</span></p>
            </div>
        </div>
        
        <div class="section">
            <h2>🔍 Overfitting Analysis</h2>
            <p>Models with high train-test accuracy gap (>0.1) may be overfitting:</p>
    """
    
    # Add overfitting analysis
    results_df['accuracy_gap'] = results_df['train_accuracy'] - results_df['test_accuracy']
    overfitting_models = results_df[results_df['accuracy_gap'] > 0.1]
    
    if len(overfitting_models) > 0:
        html_content += """
            <table>
                <tr>
                    <th>Model</th>
                    <th>Experiment</th>
                    <th>Train Accuracy</th>
                    <th>Test Accuracy</th>
                    <th>Gap</th>
                </tr>
        """
        for _, row in overfitting_models.iterrows():
            html_content += f"""
                <tr>
                    <td>{row['model']}</td>
                    <td>{row['experiment']}</td>
                    <td>{row['train_accuracy']:.4f}</td>
                    <td>{row['test_accuracy']:.4f}</td>
                    <td class="warning">{row['accuracy_gap']:.4f}</td>
                </tr>
            """
        html_content += "</table>"
    else:
        html_content += '<p class="metric-highlight">✅ No significant overfitting detected!</p>'
    
    html_content += """
        </div>
        
        <div class="section">
            <h2>📈 Complete Results Table</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Experiment</th>
                    <th>Scaler</th>
                    <th>Train Acc</th>
                    <th>Test Acc</th>
                    <th>Test F1</th>
                    <th>Test Precision</th>
                    <th>Test Recall</th>
                </tr>
    """
    
    # Add all results
    for _, row in results_df.iterrows():
        html_content += f"""
            <tr>
                <td>{row['model']}</td>
                <td>{row['experiment']}</td>
                <td>{row['scaler']}</td>
                <td>{row['train_accuracy']:.4f}</td>
                <td>{row['test_accuracy']:.4f}</td>
                <td>{row['test_f1_weighted']:.4f}</td>
                <td>{row['test_precision_weighted']:.4f}</td>
                <td>{row['test_recall_weighted']:.4f}</td>
            </tr>
        """
    
    html_content += f"""
            </table>
        </div>
        
        <div class="section">
            <h2>📁 Generated Files</h2>
            <ul>
                <li>📊 <strong>CSV Results:</strong> experiment_results_{TIMESTAMP}{EARLY_STOP_SUFFIX}.csv</li>
                <li>📈 <strong>Performance Plots:</strong> performance_plots.html</li>
                <li>🤖 <strong>Saved Models:</strong> {len([r for r in EXPERIMENT_RESULTS if 'model_path' in r])} models saved</li>
                <li>🖼️ <strong>Visualizations:</strong> Confusion matrices and feature importance plots</li>
            </ul>
        </div>
        
        <div class="section">
            <h2>⚙️ Configuration Used</h2>
            <pre>{json.dumps(config, indent=2)}</pre>
        </div>
        
    </body>
    </html>
    """
    
    # Save HTML report
    html_path = OUTPUT_DIR / f"experiment_report_{TIMESTAMP}{EARLY_STOP_SUFFIX}.html"
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"📋 HTML report generated: {html_path}")
    print(f"🎯 Open this file in your browser to view the complete report!")
    
    return html_path


# Export the generate_report function so it can be called from the main script
def finalize_experiments():
    """Call this at the end of your experiments to generate reports"""
    if EXPERIMENT_RESULTS:
        generate_html_report()
        print(f"\n🎉 Experiment complete! Check the '{OUTPUT_DIR}' folder for all results.")
    else:
        print("⚠️ No experiments were run - no reports generated.")


if __name__ == "__main__":
    print(f"🚀 ML Models module loaded with early stopping: {args.early_stop}")
    print(f"📁 Results will be saved to: {OUTPUT_DIR}")