"""
ml_core.py - Core ML functionality for keystroke biometrics experiments.
Contains model training, evaluation, and visualization functions.
"""

import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score, precision_score,
    recall_score, top_k_accuracy_score, confusion_matrix
)
from xgboost import XGBClassifier, plot_importance
import bob.measure
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


@dataclass
class ExperimentConfig:
    """Configuration for experiments."""
    dataset_path: str
    early_stopping: bool = False
    num_seeds: int = 1
    output_affix: str = ""
    random_seeds: List[int] = None
    show_class_distributions: bool = False
    draw_feature_importance: bool = True
    
    def __post_init__(self):
        if self.random_seeds is None:
            base_seeds = [42, 123, 456, 789, 999]
            self.random_seeds = base_seeds[:self.num_seeds]


@dataclass
class ExperimentResult:
    """Container for experiment results."""
    model_name: str
    experiment_name: str
    random_seed: int
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    model_path: str
    hyperparameters: Dict[str, Any]


class ModelTrainer:
    """Handles training and evaluation of ML models."""
    
    def __init__(self, config: ExperimentConfig, output_dir: Path, timestamp: str):
        self.config = config
        self.output_dir = output_dir
        self.timestamp = timestamp
        self.label_encoder = LabelEncoder()
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_pred_proba: np.ndarray, prefix: str) -> Dict[str, float]:
        """Calculate comprehensive metrics."""
        metrics = {
            f'{prefix}_accuracy': accuracy_score(y_true, y_pred),
            f'{prefix}_f1_weighted': f1_score(y_true, y_pred, average="weighted", zero_division=0),
            f'{prefix}_f1_macro': f1_score(y_true, y_pred, average="macro", zero_division=0),
            f'{prefix}_precision_weighted': precision_score(y_true, y_pred, average="weighted", zero_division=0),
            f'{prefix}_precision_macro': precision_score(y_true, y_pred, average="macro", zero_division=0),
            f'{prefix}_recall_weighted': recall_score(y_true, y_pred, average="weighted", zero_division=0),
            f'{prefix}_recall_macro': recall_score(y_true, y_pred, average="macro", zero_division=0),
        }
        
        # Top-k accuracy
        max_k = min(5, y_pred_proba.shape[1])
        for k in range(1, max_k + 1):
            try:
                top_k_acc = top_k_accuracy_score(y_true, y_pred_proba, k=k)
                metrics[f'{prefix}_top_{k}_accuracy'] = top_k_acc
            except Exception:
                metrics[f'{prefix}_top_{k}_accuracy'] = 0.0
        
        # Recognition rate using bob.measure
        try:
            rr_scores = []
            for i, true_label in enumerate(y_true):
                if true_label < y_pred_proba.shape[1]:
                    pos_score = y_pred_proba[i, true_label]
                    neg_scores = np.delete(y_pred_proba[i], true_label)
                    rr_scores.append((neg_scores, [pos_score]))
            
            if rr_scores:
                recognition_rate = bob.measure.recognition_rate(rr_scores, rank=1)
                metrics[f'{prefix}_recognition_rate'] = recognition_rate
            else:
                metrics[f'{prefix}_recognition_rate'] = 0.0
        except Exception as e:
            print(f"⚠️ Could not calculate recognition rate: {e}")
            metrics[f'{prefix}_recognition_rate'] = 0.0
        
        return metrics
    
    def save_model(self, model: Any, model_name: str, experiment_name: str, 
                   hyperparams: Dict, metrics: Dict, seed: int) -> str:
        """Save model with comprehensive metadata."""
        clean_experiment_name = experiment_name.replace(f"_{model_name}", "").replace("_no_scaling", "")
        filename = f"{model_name.lower()}_{clean_experiment_name}_{self.timestamp}_seed{seed}.pkl"
        
        if self.config.early_stopping:
            filename = filename.replace('.pkl', '_early_stop.pkl')
        
        filepath = self.output_dir / filename
        
        metadata = {
            'model_name': model_name,
            'experiment_name': experiment_name,
            'clean_experiment_name': clean_experiment_name,
            'timestamp': self.timestamp,
            'random_seed': seed,
            'early_stopping_used': self.config.early_stopping,
            'hyperparameters': hyperparams,
            'performance_metrics': metrics,
            'task_type': 'user_identification',
            'data_preprocessing': 'pre_normalized'
        }
        
        model_data = {
            'model': model,
            'metadata': metadata
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Model saved: {filename}")
        return str(filepath)
    
    def train_random_forest(self, X_train: np.ndarray, X_test: np.ndarray, 
                           y_train: np.ndarray, y_test: np.ndarray, 
                           experiment_name: str, seed: int) -> ExperimentResult:
        """Train Random Forest model."""
        print(f"🌲 Training Random Forest - {experiment_name}")
        
        min_samples_per_class = np.bincount(y_train).min()
        
        if min_samples_per_class < 2:
            model = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1)
            model.fit(X_train, y_train)
            best_params = model.get_params()
        else:
            param_grid = {
                "n_estimators": [100, 300, 500],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 10],
                "min_samples_leaf": [1, 2],
                "max_features": ["sqrt", "log2"],
                "bootstrap": [True, False],
            }
            
            cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
            grid_search = GridSearchCV(
                RandomForestClassifier(random_state=seed, n_jobs=-1),
                param_grid, cv=cv, scoring="accuracy", n_jobs=-1, verbose=0
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        
        # Predictions and metrics
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        train_proba = model.predict_proba(X_train)
        test_proba = model.predict_proba(X_test)
        
        train_metrics = self.calculate_metrics(y_train, train_pred, train_proba, "train")
        test_metrics = self.calculate_metrics(y_test, test_pred, test_proba, "test")
        all_metrics = {**train_metrics, **test_metrics}
        
        model_path = self.save_model(model, "RandomForest", experiment_name, best_params, all_metrics, seed)
        
        return ExperimentResult("RandomForest", experiment_name, seed, train_metrics, 
                              test_metrics, model_path, best_params)
    
    def train_xgboost(self, X_train: np.ndarray, X_test: np.ndarray, 
                      y_train: np.ndarray, y_test: np.ndarray, 
                      experiment_name: str, seed: int) -> ExperimentResult:
        """Train XGBoost model."""
        print(f"🚀 Training XGBoost - {experiment_name}")
        
        eval_set = [(X_test, y_test)] if self.config.early_stopping else None
        early_stopping_rounds = 50 if self.config.early_stopping else None
        min_samples_per_class = np.bincount(y_train).min()
        
        if min_samples_per_class < 2:
            model = XGBClassifier(
                n_estimators=1000 if self.config.early_stopping else 200,
                max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                reg_lambda=1, random_state=seed, n_jobs=-1
            )
            if self.config.early_stopping:
                model.fit(X_train, y_train, early_stopping_rounds=early_stopping_rounds,
                         eval_set=eval_set, verbose=False)
            else:
                model.fit(X_train, y_train)
            best_params = model.get_params()
        else:
            param_grid = {
                "n_estimators": [1000] if self.config.early_stopping else [100, 200],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.05, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
                "reg_lambda": [1, 3],
            }
            
            cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
            grid_search = GridSearchCV(
                XGBClassifier(random_state=seed, n_jobs=-1),
                param_grid, cv=cv, scoring="accuracy", n_jobs=-1, verbose=0
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
            if self.config.early_stopping:
                model.set_params(n_estimators=1000)
                model.fit(X_train, y_train, early_stopping_rounds=early_stopping_rounds,
                         eval_set=eval_set, verbose=False)
        
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        train_proba = model.predict_proba(X_train)
        test_proba = model.predict_proba(X_test)
        
        train_metrics = self.calculate_metrics(y_train, train_pred, train_proba, "train")
        test_metrics = self.calculate_metrics(y_test, test_pred, test_proba, "test")
        all_metrics = {**train_metrics, **test_metrics}
        
        model_path = self.save_model(model, "XGBoost", experiment_name, best_params, all_metrics, seed)
        
        return ExperimentResult("XGBoost", experiment_name, seed, train_metrics, 
                              test_metrics, model_path, best_params)
    
    def train_catboost(self, X_train: np.ndarray, X_test: np.ndarray, 
                       y_train: np.ndarray, y_test: np.ndarray, 
                       experiment_name: str, seed: int) -> ExperimentResult:
        """Train CatBoost model."""
        print(f"🐱 Training CatBoost - {experiment_name}")
        
        eval_set = Pool(X_test, y_test) if self.config.early_stopping else None
        early_stopping_rounds = 50 if self.config.early_stopping else None
        min_samples_per_class = np.bincount(y_train).min()
        
        if min_samples_per_class < 2:
            model = CatBoostClassifier(
                iterations=1000 if self.config.early_stopping else 100,
                depth=6, learning_rate=0.1, l2_leaf_reg=3, border_count=64,
                random_seed=seed, verbose=False, thread_count=-1
            )
            if self.config.early_stopping:
                model.fit(X_train, y_train, early_stopping_rounds=early_stopping_rounds,
                         eval_set=eval_set, use_best_model=True)
            else:
                model.fit(X_train, y_train)
            best_params = model.get_params()
        else:
            param_grid = {
                "iterations": [1000] if self.config.early_stopping else [100, 200],
                "depth": [4, 6, 8],
                "learning_rate": [0.05, 0.1],
                "l2_leaf_reg": [1, 3, 5],
                "border_count": [32, 64],
            }
            
            cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
            grid_search = GridSearchCV(
                CatBoostClassifier(verbose=False, random_seed=seed, thread_count=-1),
                param_grid, cv=cv, scoring="accuracy", n_jobs=1, verbose=0
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
            if self.config.early_stopping:
                model.set_params(iterations=1000)
                model.fit(X_train, y_train, early_stopping_rounds=early_stopping_rounds,
                         eval_set=eval_set, use_best_model=True)
        
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        train_proba = model.predict_proba(X_train)
        test_proba = model.predict_proba(X_test)
        
        train_metrics = self.calculate_metrics(y_train, train_pred, train_proba, "train")
        test_metrics = self.calculate_metrics(y_test, test_pred, test_proba, "test")
        all_metrics = {**train_metrics, **test_metrics}
        
        model_path = self.save_model(model, "CatBoost", experiment_name, best_params, all_metrics, seed)
        
        return ExperimentResult("CatBoost", experiment_name, seed, train_metrics, 
                              test_metrics, model_path, best_params)
    
    def train_svm(self, X_train: np.ndarray, X_test: np.ndarray, 
                  y_train: np.ndarray, y_test: np.ndarray, 
                  experiment_name: str, seed: int) -> ExperimentResult:
        """Train SVM model."""
        print(f"⚙️ Training SVM - {experiment_name}")
        
        min_samples_per_class = np.bincount(y_train).min()
        
        if min_samples_per_class < 2:
            model = SVC(C=1, kernel="rbf", gamma="scale", probability=True, random_state=seed)
            model.fit(X_train, y_train)
            best_params = model.get_params()
        else:
            param_grid = {
                "C": [0.1, 1, 10],
                "kernel": ["rbf"],
                "gamma": ["scale", "auto"],
            }
            
            cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
            grid_search = GridSearchCV(
                SVC(probability=True, random_state=seed),
                param_grid, cv=cv, scoring="accuracy", n_jobs=1, verbose=0
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        train_proba = model.predict_proba(X_train)
        test_proba = model.predict_proba(X_test)
        
        train_metrics = self.calculate_metrics(y_train, train_pred, train_proba, "train")
        test_metrics = self.calculate_metrics(y_test, test_pred, test_proba, "test")
        all_metrics = {**train_metrics, **test_metrics}
        
        model_path = self.save_model(model, "SVM", experiment_name, best_params, all_metrics, seed)
        
        return ExperimentResult("SVM", experiment_name, seed, train_metrics, 
                              test_metrics, model_path, best_params)


class Visualizer:
    """Handles all visualization and reporting functionality."""
    
    def __init__(self, config: ExperimentConfig, output_dir: Path, timestamp: str):
        self.config = config
        self.output_dir = output_dir
        self.timestamp = timestamp
    
    def plot_platform_leakage(self, features: List[Tuple[str, float]]):
        """Plot platform leakage analysis."""
        plt.figure(figsize=(12, 8))
        names, scores = zip(*features)
        colors = ['red' if score > 0.1 else 'orange' if score > 0.05 else 'green' for score in scores]
        
        feature_names = [name[:30] + '...' if len(name) > 30 else name for name in names]
        
        plt.barh(range(len(feature_names)), scores, color=colors)
        plt.yticks(range(len(feature_names)), feature_names)
        plt.xlabel("Feature Importance")
        plt.title("Top Platform-Leaking Features\nRed: High Risk | Orange: Moderate Risk | Green: Low Risk")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(self.output_dir / "platform_leakage_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_confusion_matrices(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                                  title: str, filename: str, k_values=[1, 5]):
        """Create confusion matrices for top-k predictions."""
        for k in k_values:
            if k > y_pred_proba.shape[1]:
                continue
            
            if k == 1:
                # Top-1 confusion matrix
                y_pred_top1 = np.argmax(y_pred_proba, axis=1)
                cm = confusion_matrix(y_true, y_pred_top1)
                
                # Get unique labels
                unique_labels = sorted(list(set(y_true) | set(y_pred_top1)))
                n_users = len(unique_labels)
                fig_width = max(12, min(50, n_users * 0.8))
                fig_height = max(10, min(40, n_users * 0.6))
                
                plt.figure(figsize=(fig_width, fig_height))
                
                if len(unique_labels) <= 50:
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                               xticklabels=unique_labels, yticklabels=unique_labels,
                               cbar_kws={'shrink': 0.8})
                    plt.xticks(rotation=45, ha='right')
                    plt.yticks(rotation=0)
                else:
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
                plt.savefig(self.output_dir / f'{filename}_top_1_confusion_matrix.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            else:
                # Top-k identification success matrix
                top_k_indices = np.argsort(y_pred_proba, axis=1)[:, -k:]
                top_k_correct = np.array([true_label in top_k_pred for true_label, top_k_pred in zip(y_true, top_k_indices)])
                
                binary_true = np.ones_like(top_k_correct)
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
                plt.savefig(self.output_dir / f'{filename}_top_{k}_identification_matrix.png', dpi=300, bbox_inches='tight')
                plt.close()
    
    def plot_feature_importance(self, model: Any, model_name: str, experiment_name: str, 
                               feature_names: List[str]):
        """Plot feature importance for tree-based models."""
        if not self.config.draw_feature_importance:
            return
        
        try:
            plt.figure(figsize=(12, 8))
            
            if model_name == "XGBoost":
                plot_importance(model, importance_type="weight", max_num_features=15)
                plt.title(f"XGBoost Feature Importance - {experiment_name}")
            
            elif model_name == "RandomForest":
                importances = model.feature_importances_
                indices = np.argsort(importances)[-15:]
                plt.barh(range(len(indices)), importances[indices], align="center")
                plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
                plt.xlabel("Feature Importance")
                plt.title(f"Random Forest Feature Importance - {experiment_name}")
            
            elif model_name == "CatBoost":
                # Create a dummy pool for feature importance
                dummy_X = np.random.randn(10, len(feature_names))
                dummy_y = np.random.randint(0, 2, 10)
                pool = Pool(dummy_X, label=dummy_y)
                
                importances = model.get_feature_importance(pool)
                sorted_indices = importances.argsort()[-15:]
                plt.barh(range(len(sorted_indices)), importances[sorted_indices], align="center")
                plt.yticks(range(len(sorted_indices)), [feature_names[i] for i in sorted_indices])
                plt.xlabel("Feature Importance")
                plt.title(f"CatBoost Feature Importance - {experiment_name}")
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f"{model_name.lower()}_feature_importance_{experiment_name}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"⚠️ Could not generate feature importance plot for {model_name}: {e}")
    
    def create_performance_plots(self, results_df: pd.DataFrame):
        """Create comprehensive performance plots with Plotly."""
        if results_df.empty:
            print("⚠️ No results to plot")
            return
        
        # Create comprehensive performance plots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Top-1 Accuracy by Model', 'Top-K Accuracy Trends', 
                           'F1 Score vs Top-1 Accuracy', 'Model Performance Overview'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        models = results_df['model'].unique()
        colors = px.colors.qualitative.Set1[:len(models)]
        
        # 1. Top-1 accuracy by model (by experiment)
        for i, model in enumerate(models):
            model_data = results_df[results_df['model'] == model]
            
            if 'test_top_1_accuracy' in model_data.columns:
                fig.add_trace(
                    go.Scatter(x=model_data['experiment'], y=model_data['test_top_1_accuracy'],
                              name=f'{model}', line=dict(color=colors[i]),
                              mode='lines+markers'),
                    row=1, col=1
                )
        
        # 2. Top-K accuracy trends
        k_values = [1, 2, 3, 4, 5]
        for i, model in enumerate(models):
            model_data = results_df[results_df['model'] == model]
            k_accuracies = []
            for k in k_values:
                col_name = f'test_top_{k}_accuracy'
                if col_name in model_data.columns:
                    avg_acc = model_data[col_name].fillna(0).mean()
                    k_accuracies.append(avg_acc)
                else:
                    k_accuracies.append(0)
            
            fig.add_trace(
                go.Scatter(x=k_values, y=k_accuracies,
                          name=f'{model}', line=dict(color=colors[i]),
                          mode='lines+markers', showlegend=False),
                row=1, col=2
            )
        
        # 3. F1 vs Top-1 Accuracy scatter plot
        for i, model in enumerate(models):
            model_data = results_df[results_df['model'] == model]
            if 'test_top_1_accuracy' in model_data.columns and 'test_f1_weighted' in model_data.columns:
                fig.add_trace(
                    go.Scatter(x=model_data['test_f1_weighted'].fillna(0), 
                              y=model_data['test_top_1_accuracy'].fillna(0),
                              name=f'{model}', mode='markers', 
                              marker=dict(color=colors[i], size=8),
                              showlegend=False),
                    row=2, col=1
                )
        
        # 4. Model Performance Overview
        model_names = []
        avg_top1_scores = []
        avg_top5_scores = []
        
        for model in models:
            model_data = results_df[results_df['model'] == model]
            avg_top1 = model_data.get('test_top_1_accuracy', pd.Series([0])).fillna(0).mean()
            avg_top5 = model_data.get('test_top_5_accuracy', pd.Series([0])).fillna(0).mean()
            
            model_names.append(model)
            avg_top1_scores.append(avg_top1)
            avg_top5_scores.append(avg_top5)
        
        # Add bars
        fig.add_trace(
            go.Bar(x=model_names, y=avg_top1_scores, 
                   name='Avg Top-1', marker_color='teal'),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Bar(x=model_names, y=avg_top5_scores, 
                   name='Avg Top-5', marker_color='purple'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800, 
            title_text="User Identification Performance Analysis",
            barmode='group'
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
        
        fig.write_html(self.output_dir / "performance_plots.html")
        print(f"📈 Performance plots saved to: {self.output_dir / 'performance_plots.html'}")


def analyze_platform_leakage(df: pd.DataFrame, output_dir: Path) -> Tuple[float, List[Tuple[str, float]]]:
    """Analyze potential platform leakage in features."""
    print("\n🔍 Running comprehensive platform leakage diagnostic...")
    
    feature_cols = [col for col in df.columns if col not in {"user_id", "platform_id", "session_id"}]
    X = df[feature_cols].fillna(df[feature_cols].median())
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["platform_id"])
    
    # Sample if dataset is large
    platform_counts = df["platform_id"].value_counts()
    min_platform_size = platform_counts.min()
    if min_platform_size > 300:
        df_sampled = df.groupby("platform_id").sample(n=300, random_state=42)
        X = df_sampled[feature_cols].fillna(df_sampled[feature_cols].median())
        y = label_encoder.fit_transform(df_sampled["platform_id"])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    
    # Train classifier to predict platform
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    clf.fit(X_train, y_train)
    
    accuracy = clf.score(X_test, y_test)
    print(f"Platform Prediction Accuracy: {accuracy:.4f}")
    
    # Risk assessment
    if accuracy > 0.8:
        print("HIGH LEAKAGE RISK - Platform can be predicted with high accuracy!")
    elif accuracy > 0.6:
        print("MODERATE LEAKAGE RISK - Some platform-specific patterns detected")
    else:
        print("LOW LEAKAGE RISK - Platform prediction is difficult")
    
    # Get feature importance
    importances = clf.feature_importances_
    indices = importances.argsort()[-15:][::-1]
    top_features = [(feature_cols[i], importances[i]) for i in indices]
    
    print("\nTop 15 platform-leaking features:")
    for name, score in top_features:
        risk_level = "HIGH" if score > 0.1 else "MODERATE" if score > 0.05 else "LOW"
        print(f"  {risk_level:8} {name:35} -> {score:.4f}")
    
    return accuracy, top_features