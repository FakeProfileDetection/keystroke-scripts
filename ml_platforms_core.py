"""
ml_platforms_core.py - Core ML functionality for keystroke biometrics experiments.
Contains model training, evaluation, and configuration classes.
FIXED VERSION - Addresses XGBoost early stopping and CatBoost parameter issues.
"""

import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
    confusion_matrix,
)
from xgboost import XGBClassifier
import bob.measure

from ml_platforms_visualizer import Visualizer

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


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
    debug_mode: bool = False  # New debug flag

    def __post_init__(self):
        if self.random_seeds is None:
            base_seeds = [42, 123, 456, 789, 999]
            self.random_seeds = base_seeds[: self.num_seeds]


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

    def __init__(
        self,
        config: ExperimentConfig,
        output_dir: Path,
        timestamp: str,
        use_gpu: bool = True,
    ):
        self.config = config
        self.output_dir = output_dir
        self.timestamp = timestamp
        self.label_encoder = LabelEncoder()

        # Check if GPU is available and use_gpu, else self.use_gpu = False
        self.use_gpu = False
        if self.use_gpu:
            import torch

            if not torch.cuda.is_available():
                print("⚠️ GPU not available, switching to CPU mode.")
                self.use_gpu = False

    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        prefix: str,
    ) -> Dict[str, float]:
        """Calculate comprehensive metrics."""
        metrics = {
            f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
            f"{prefix}_f1_weighted": f1_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            f"{prefix}_f1_macro": f1_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            f"{prefix}_precision_weighted": precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            f"{prefix}_precision_macro": precision_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            f"{prefix}_recall_weighted": recall_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            f"{prefix}_recall_macro": recall_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
        }

        # Top-k accuracy
        max_k = min(5, y_pred_proba.shape[1])
        for k in range(1, max_k + 1):
            try:
                top_k_acc = top_k_accuracy_score(y_true, y_pred_proba, k=k)
                metrics[f"{prefix}_top_{k}_accuracy"] = top_k_acc
            except Exception:
                metrics[f"{prefix}_top_{k}_accuracy"] = 0.0

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
                metrics[f"{prefix}_recognition_rate"] = recognition_rate
            else:
                metrics[f"{prefix}_recognition_rate"] = 0.0
        except Exception as e:
            print(f"⚠️ Could not calculate recognition rate: {e}")
            metrics[f"{prefix}_recognition_rate"] = 0.0

        return metrics

    def save_model(
        self,
        model: Any,
        model_name: str,
        experiment_name: str,
        hyperparams: Dict,
        metrics: Dict,
        seed: int,
    ) -> str:
        """Save model with comprehensive metadata."""
        clean_experiment_name = experiment_name.replace(f"_{model_name}", "").replace(
            "_no_scaling", ""
        )
        filename = f"{model_name.lower()}_{clean_experiment_name}_{self.timestamp}_seed{seed}.pkl"

        if self.config.early_stopping:
            filename = filename.replace(".pkl", "_early_stop.pkl")

        filepath = self.output_dir / filename

        metadata = {
            "model_name": model_name,
            "experiment_name": experiment_name,
            "clean_experiment_name": clean_experiment_name,
            "timestamp": self.timestamp,
            "random_seed": seed,
            "early_stopping_used": self.config.early_stopping,
            "debug_mode": self.config.debug_mode,
            "hyperparameters": hyperparams,
            "performance_metrics": metrics,
            "task_type": "user_identification",
            "data_preprocessing": "pre_normalized",
        }

        model_data = {"model": model, "metadata": metadata}

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        print(f"💾 Model saved: {filename}")
        return str(filepath)

    def get_param_grid(self, model_type: str, debug_mode: bool = False):
        """Get parameter grids - minimal for debug mode, full for production."""
        if debug_mode:
            # Minimal grids for fast debugging
            if model_type == "random_forest":
                return {"n_estimators": [50], "max_depth": [10]}
            elif model_type == "xgboost":
                return {"n_estimators": [50], "max_depth": [6], "learning_rate": [0.1]}
            elif model_type == "catboost":
                return {"iterations": [50], "depth": [6], "learning_rate": [0.1]}
            elif model_type == "svm":
                return {"C": [1], "kernel": ["rbf"], "gamma": ["scale"]}
            elif model_type == "mlp":
                return {
                    "hidden_layer_sizes": [(50,)],
                    "activation": ["relu"],
                    "solver": ["adam"],
                    "alpha": [0.0001],
                    "learning_rate": ["constant"],
                    "batch_size": [16],
                    "learning_rate_init": [0.001],
                    "max_iter": [1000],
                    "early_stopping": [False],
                }
            elif model_type == "naive_bayes":
                # Naive Bayes does not require hyperparameter tuning
                return {}
        else:
            # Full grids matching the original ml_models.py
            if model_type == "random_forest":
                return {
                    "n_estimators": [100, 300, 500],
                    "max_depth": [10, 20, None],
                    "min_samples_split": [2, 10],
                    "min_samples_leaf": [1, 2],
                    "max_features": ["sqrt", "log2"],
                    "bootstrap": [True, False],
                }
            elif model_type == "xgboost":
                return {
                    "n_estimators": (
                        [1000] if self.config.early_stopping else [100, 200]
                    ),
                    "max_depth": [4, 6, 8],
                    "learning_rate": [0.05, 0.1],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0],
                    "reg_lambda": [1, 3],
                    "device": ["cuda"] if self.use_gpu else ["cpu"],
                }
            elif model_type == "catboost":
                return {
                    "iterations": [1000] if self.config.early_stopping else [100, 200],
                    "depth": [4, 6, 8],
                    "learning_rate": [0.05, 0.1],
                    "l2_leaf_reg": [1, 3, 5],
                    "border_count": [32, 64],
                }
            elif model_type == "svm":
                return {
                    "decision_function_shape": ["ovo", "ovr"],
                    "C": [0.1, 1, 10, 100],
                    "kernel": ["rbf", "linear", "poly", "sigmoid"],
                    "gamma": [
                        "scale",
                        "auto",
                    ],
                }
            elif model_type == "mlp":
                return {
                    "hidden_layer_sizes": [(50,), (100,), (50, 50)],
                    "activation": ["relu", "tanh"],
                    "solver": ["adam", "lbfgs"],
                    "alpha": [0.0001, 0.001],
                    "learning_rate": ["constant", "adaptive"],
                    "batch_size": [36, 16, 8, 2, 1],
                    "learning_rate_init": [0.001, 0.01],
                    "max_iter": [5000, 2000],
                    "early_stopping": [True] if self.config.early_stopping else [False],
                }
            elif model_type == "naive_bayes":
                # Naive Bayes does not require hyperparameter tuning
                return {}
        return {}

    def train_naive_bayes(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        experiment_name: str,
        seed: int,
    ) -> ExperimentResult:
        """Train Naive Bayes model."""
        print(f"🔍 Training Naive Bayes - {experiment_name}")

        min_samples_per_class = np.bincount(y_train).min()

        if min_samples_per_class < 2:
            model = GaussianNB()
            model.fit(X_train, y_train)
            best_params = model.get_params()
        else:
            param_grid = self.get_param_grid("naive_bayes", self.config.debug_mode)

            cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
            grid_search = GridSearchCV(
                GaussianNB(),
                param_grid,
                cv=cv,
                scoring="accuracy",
                n_jobs=-1,
                verbose=0,
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            best_params = grid_search.best_params_

        # Predictions and metrics
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        train_proba = model.predict_proba(X_train)
        test_proba = model.predict_proba(X_test)

        train_metrics = self.calculate_metrics(
            y_train, train_pred, train_proba, "train"
        )
        test_metrics = self.calculate_metrics(y_test, test_pred, test_proba, "test")
        all_metrics = {**train_metrics, **test_metrics}

        # Create confusion matrices
        clean_exp_name = experiment_name.replace("_no_scaling", "")
        visualizer = Visualizer(self.config, self.output_dir, self.timestamp)
        visualizer.create_confusion_matrices(
            y_test,
            test_proba,
            f"NaiveBayes {clean_exp_name}",
            f"naive_bayes_{clean_exp_name}",
        )

        # Feature importance plot
        if self.config.draw_feature_importance:
            visualizer.plot_feature_importance(
                model,
                "NaiveBayes",
                experiment_name,
                [f"feature_{i}" for i in range(X_train.shape[1])],
            )
        model_path = self.save_model(
            model, "NaiveBayes", experiment_name, best_params, all_metrics, seed
        )

        return ExperimentResult(
            "NaiveBayes",
            experiment_name,
            seed,
            train_metrics,
            test_metrics,
            model_path,
            best_params,
        )

    def train_random_forest(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        experiment_name: str,
        seed: int,
    ) -> ExperimentResult:
        """Train Random Forest model."""
        print(f"🌲 Training Random Forest - {experiment_name}")

        min_samples_per_class = np.bincount(y_train).min()

        if min_samples_per_class < 2:
            model = RandomForestClassifier(
                n_estimators=500, random_state=seed, n_jobs=-1
            )
            model.fit(X_train, y_train)
            best_params = model.get_params()
        else:
            param_grid = self.get_param_grid("random_forest", self.config.debug_mode)

            cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
            grid_search = GridSearchCV(
                RandomForestClassifier(random_state=seed, n_jobs=-1),
                param_grid,
                cv=cv,
                scoring="accuracy",
                n_jobs=-1,
                verbose=0,
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            best_params = grid_search.best_params_

        # Predictions and metrics
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        train_proba = model.predict_proba(X_train)
        test_proba = model.predict_proba(X_test)

        train_metrics = self.calculate_metrics(
            y_train, train_pred, train_proba, "train"
        )
        test_metrics = self.calculate_metrics(y_test, test_pred, test_proba, "test")
        all_metrics = {**train_metrics, **test_metrics}

        # Create confusion matrices
        clean_exp_name = experiment_name.replace("_no_scaling", "")
        visualizer = Visualizer(self.config, self.output_dir, self.timestamp)
        visualizer.create_confusion_matrices(
            y_test,
            test_proba,
            f"RandomForest {clean_exp_name}",
            f"RandomForest_{clean_exp_name}",
        )

        # Feature importance plot
        if self.config.draw_feature_importance:
            visualizer.plot_feature_importance(
                model,
                "RandomForest",
                experiment_name,
                [f"feature_{i}" for i in range(X_train.shape[1])],
            )

        model_path = self.save_model(
            model, "RandomForest", experiment_name, best_params, all_metrics, seed
        )

        return ExperimentResult(
            "RandomForest",
            experiment_name,
            seed,
            train_metrics,
            test_metrics,
            model_path,
            best_params,
        )

    def train_xgboost(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        experiment_name: str,
        seed: int,
    ) -> ExperimentResult:
        """Train XGBoost model - FIXED VERSION."""
        print(f"🚀 Training XGBoost - {experiment_name}")

        min_samples_per_class = np.bincount(y_train).min()

        if min_samples_per_class < 2:
            # Direct training without grid search
            if self.config.early_stopping:
                model = XGBClassifier(
                    n_estimators=1000,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1,
                    random_state=seed,
                    n_jobs=-1,
                    eval_metric="mlogloss",
                    early_stopping_rounds=50,  # Set here for XGBoost >= 2.0
                )
                # For XGBoost >= 2.0, early_stopping_rounds is set in the model init
                model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            else:
                model = XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1,
                    random_state=seed,
                    n_jobs=-1,
                )
                model.fit(X_train, y_train)
            best_params = model.get_params()
        else:
            # Grid search
            param_grid = self.get_param_grid("xgboost", self.config.debug_mode)

            cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
            grid_search = GridSearchCV(
                XGBClassifier(random_state=seed, n_jobs=-1),
                param_grid,
                cv=cv,
                scoring="accuracy",
                n_jobs=-1,
                verbose=0,
            )
            grid_search.fit(X_train, y_train)

            best_params = grid_search.best_params_

            if self.config.early_stopping:
                # Create new model with best params and early stopping
                # Remove n_estimators from best_params to avoid conflict
                early_stop_params = {
                    k: v for k, v in best_params.items() if k != "n_estimators"
                }

                model = XGBClassifier(
                    **early_stop_params,
                    n_estimators=1000,
                    random_state=seed,
                    n_jobs=-1,
                    eval_metric="mlogloss",
                    early_stopping_rounds=50,
                )
                model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            else:
                model = grid_search.best_estimator_

        # Predictions and metrics
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        train_proba = model.predict_proba(X_train)
        test_proba = model.predict_proba(X_test)

        train_metrics = self.calculate_metrics(
            y_train, train_pred, train_proba, "train"
        )
        test_metrics = self.calculate_metrics(y_test, test_pred, test_proba, "test")
        all_metrics = {**train_metrics, **test_metrics}

        # Create confusion matrices
        clean_exp_name = experiment_name.replace("_no_scaling", "")
        visualizer = Visualizer(self.config, self.output_dir, self.timestamp)
        visualizer.create_confusion_matrices(
            y_test, test_proba, f"XGBoost {clean_exp_name}", f"xgboost_{clean_exp_name}"
        )

        # Feature importance plot
        if self.config.draw_feature_importance:
            visualizer.plot_feature_importance(
                model,
                "XGBoost",
                experiment_name,
                [f"feature_{i}" for i in range(X_train.shape[1])],
            )

        model_path = self.save_model(
            model, "XGBoost", experiment_name, best_params, all_metrics, seed
        )

        return ExperimentResult(
            "XGBoost",
            experiment_name,
            seed,
            train_metrics,
            test_metrics,
            model_path,
            best_params,
        )

    def train_catboost(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        experiment_name: str,
        seed: int,
    ) -> ExperimentResult:
        """Train CatBoost model - FIXED VERSION."""
        print(f"🐱 Training CatBoost - {experiment_name}")

        eval_set = Pool(X_test, y_test) if self.config.early_stopping else None
        min_samples_per_class = np.bincount(y_train).min()

        if min_samples_per_class < 2:
            # Direct training without grid search
            if self.config.early_stopping:
                model = CatBoostClassifier(
                    iterations=1000,
                    depth=6,
                    learning_rate=0.1,
                    l2_leaf_reg=3,
                    border_count=64,
                    random_seed=seed,
                    verbose=False,
                    thread_count=-1,
                    early_stopping_rounds=50,
                )
                model.fit(X_train, y_train, eval_set=eval_set, use_best_model=True)
            else:
                model = CatBoostClassifier(
                    iterations=100,
                    depth=6,
                    learning_rate=0.1,
                    l2_leaf_reg=3,
                    border_count=64,
                    random_seed=seed,
                    verbose=False,
                    thread_count=-1,
                )
                model.fit(X_train, y_train)
            best_params = model.get_params()
        else:
            # Grid search - first without early stopping
            param_grid = self.get_param_grid("catboost", self.config.debug_mode)

            cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
            grid_search = GridSearchCV(
                CatBoostClassifier(verbose=False, random_seed=seed, thread_count=-1),
                param_grid,
                cv=cv,
                scoring="accuracy",
                n_jobs=1,
                verbose=0,
            )
            grid_search.fit(X_train, y_train)

            best_params = grid_search.best_params_

            if self.config.early_stopping:
                # Create new model with best params and early stopping
                # Remove 'iterations' from best_params to avoid conflict
                early_stop_params = {
                    k: v for k, v in best_params.items() if k != "iterations"
                }

                model = CatBoostClassifier(
                    **early_stop_params,
                    iterations=1000,
                    random_seed=seed,
                    verbose=False,
                    thread_count=-1,
                    early_stopping_rounds=50,
                )
                model.fit(X_train, y_train, eval_set=eval_set, use_best_model=True)
            else:
                model = grid_search.best_estimator_

        # Predictions and metrics
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        train_proba = model.predict_proba(X_train)
        test_proba = model.predict_proba(X_test)

        train_metrics = self.calculate_metrics(
            y_train, train_pred, train_proba, "train"
        )
        test_metrics = self.calculate_metrics(y_test, test_pred, test_proba, "test")
        all_metrics = {**train_metrics, **test_metrics}

        # Create confusion matrices
        clean_exp_name = experiment_name.replace("_no_scaling", "")
        visualizer = Visualizer(self.config, self.output_dir, self.timestamp)
        visualizer.create_confusion_matrices(
            y_test,
            test_proba,
            f"CatBoost {clean_exp_name}",
            f"catboost_{clean_exp_name}",
        )

        # Feature importance plot
        if self.config.draw_feature_importance:
            visualizer.plot_feature_importance(
                model,
                "CatBoost",
                experiment_name,
                [f"feature_{i}" for i in range(X_train.shape[1])],
            )

        model_path = self.save_model(
            model, "CatBoost", experiment_name, best_params, all_metrics, seed
        )

        return ExperimentResult(
            "CatBoost",
            experiment_name,
            seed,
            train_metrics,
            test_metrics,
            model_path,
            best_params,
        )

    def train_mlp(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        experiment_name: str,
        seed: int,
    ) -> ExperimentResult:
        """Train Random Forest model."""
        print(f"🌲 Training Random Forest - {experiment_name}")

        min_samples_per_class = np.bincount(y_train).min()

        if min_samples_per_class < 2:
            model = MLPClassifier(n_estimators=500, random_state=seed)
            model.fit(X_train, y_train)
            best_params = model.get_params()
        else:
            param_grid = self.get_param_grid("mlp", self.config.debug_mode)

            cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
            grid_search = GridSearchCV(
                MLPClassifier(random_state=seed),
                param_grid,
                cv=cv,
                scoring="accuracy",
                n_jobs=-1,
                verbose=0,
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            best_params = grid_search.best_params_

        # Predictions and metrics
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        train_proba = model.predict_proba(X_train)
        test_proba = model.predict_proba(X_test)

        train_metrics = self.calculate_metrics(
            y_train, train_pred, train_proba, "train"
        )
        test_metrics = self.calculate_metrics(y_test, test_pred, test_proba, "test")
        all_metrics = {**train_metrics, **test_metrics}

        # Create confusion matrices
        clean_exp_name = experiment_name.replace("_no_scaling", "")
        visualizer = Visualizer(self.config, self.output_dir, self.timestamp)
        visualizer.create_confusion_matrices(
            y_test, test_proba, f"train_mlp {clean_exp_name}", f"mlp_{clean_exp_name}"
        )

        # Feature importance plot
        if self.config.draw_feature_importance:
            visualizer.plot_feature_importance(
                model,
                "mlp",
                experiment_name,
                [f"feature_{i}" for i in range(X_train.shape[1])],
            )

        model_path = self.save_model(
            model, "mlp", experiment_name, best_params, all_metrics, seed
        )

        return ExperimentResult(
            "mlp",
            experiment_name,
            seed,
            train_metrics,
            test_metrics,
            model_path,
            best_params,
        )

    def train_svm(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        experiment_name: str,
        seed: int,
    ) -> ExperimentResult:
        """Train SVM model."""
        print(f"⚙️ Training SVM - {experiment_name}")

        min_samples_per_class = np.bincount(y_train).min()

        if min_samples_per_class < 2:
            model = SVC(
                C=1, kernel="rbf", gamma="scale", probability=True, random_state=seed
            )
            model.fit(X_train, y_train)
            best_params = model.get_params()
        else:
            param_grid = self.get_param_grid("svm", self.config.debug_mode)

            cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
            grid_search = GridSearchCV(
                SVC(probability=True, random_state=seed),
                param_grid,
                cv=cv,
                scoring="accuracy",
                n_jobs=1,
                verbose=0,
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            best_params = grid_search.best_params_

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        train_proba = model.predict_proba(X_train)
        test_proba = model.predict_proba(X_test)

        train_metrics = self.calculate_metrics(
            y_train, train_pred, train_proba, "train"
        )
        test_metrics = self.calculate_metrics(y_test, test_pred, test_proba, "test")
        all_metrics = {**train_metrics, **test_metrics}

        # Create confusion matrices
        clean_exp_name = experiment_name.replace("_no_scaling", "")
        visualizer = Visualizer(self.config, self.output_dir, self.timestamp)
        visualizer.create_confusion_matrices(
            y_test, test_proba, f"SVM {clean_exp_name}", f"svm_{clean_exp_name}"
        )

        model_path = self.save_model(
            model, "SVM", experiment_name, best_params, all_metrics, seed
        )

        return ExperimentResult(
            "SVM",
            experiment_name,
            seed,
            train_metrics,
            test_metrics,
            model_path,
            best_params,
        )


def analyze_platform_leakage(
    df: pd.DataFrame, output_dir: Path
) -> Tuple[float, List[Tuple[str, float]]]:
    """Analyze potential platform leakage in features."""
    print("\n🔍 Running comprehensive platform leakage diagnostic...")

    feature_cols = [
        col for col in df.columns if col not in {"user_id", "platform_id", "session_id"}
    ]
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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

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
