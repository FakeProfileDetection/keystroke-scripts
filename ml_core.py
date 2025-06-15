"""
ml_core.py - Core ML functionality for keystroke biometrics experiments
"""

import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass

import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier, Pool
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
from lightgbm import LGBMClassifier

import bob.measure

from ml_visualizer import Visualizer
from ml_utils import prepare_param_grid, perform_grid_search, check_gpu_availability

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class ExperimentConfig:
    """Configuration for experiments."""

    config_dict: Dict[str, Any]

    def __getattr__(self, name):
        return self.config_dict.get(name)

    @property
    def dataset_path(self):
        return self.config_dict.get("dataset_path", "")

    @property
    def early_stopping(self):
        return self.config_dict.get("early_stopping", False)

    @property
    def random_seeds(self):
        return self.config_dict.get("seeds", [42])

    @property
    def output_affix(self):
        return self.config_dict.get("output_affix", "")

    @property
    def show_class_distributions(self):
        return self.config_dict.get("show_class_distributions", False)

    @property
    def draw_feature_importance(self):
        return self.config_dict.get("draw_feature_importance", True)

    @property
    def debug_mode(self):
        return self.config_dict.get("debug", False)

    @property
    def models_to_train(self):
        return self.config_dict.get("models_to_train", [])

    @property
    def experiments(self):
        return self.config_dict.get("experiments", [])

    @property
    def param_grids(self):
        return self.config_dict.get("param_grids", {})

    @property
    def use_gpu(self):
        return self.config_dict.get("use_gpu", True)


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

        # Check GPU availability
        # self.use_gpu = config.use_gpu and check_gpu_availability()
        # if config.use_gpu and not self.use_gpu:
        #     print("⚠️ GPU not available, switching to CPU mode.")

        self.use_gpu = False  # Force CPU mode for compatibility with bob.measure--seems to be running slow with GPU

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

    def train_model_generic(
        self,
        model_class: Any,
        model_name: str,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        experiment_name: str,
        seed: int,
        custom_fit_func: Optional[Callable] = None,
    ) -> ExperimentResult:
        """Generic model training function to reduce code duplication."""
        print(f"🔍 Training {model_name} - {experiment_name}")

        # Get parameter grid
        param_grid = self.config.param_grids.get(model_name.lower(), {})
        param_grid = prepare_param_grid(
            param_grid, self.config.early_stopping, self.use_gpu, model_name.lower()
        )

        # Initialize model with seed if applicable
        model_init_params = {}
        if model_name.lower() in [
            "randomforest",
            "xgboost",
            "svm",
            "mlp",
            "lightgbm",
            "extratrees",
            "gradientboosting",
            "logisticregression",
        ]:
            model_init_params["random_state"] = seed
        if model_name.lower() == "catboost":
            # CatBoost uses random_seed, not random_state
            model_init_params["random_seed"] = seed
            model_init_params["verbose"] = False
            model_init_params["thread_count"] = -1
        if model_name.lower() in ["randomforest", "extratrees"]:
            model_init_params["n_jobs"] = -1
        if model_name.lower() == "xgboost":
            model_init_params["n_jobs"] = -1
        if model_name.lower() == "lightgbm":
            model_init_params["n_jobs"] = -1
            model_init_params["verbose"] = -1
        if model_name.lower() == "gradientboosting":
            model_init_params["verbose"] = 0
        if model_name.lower() == "svm":
            model_init_params["probability"] = True
        if model_name.lower() == "knn":
            model_init_params["n_jobs"] = (
                -1
            )  # KNN supports parallel distance computation
        if model_name.lower() == "logisticregression":
            model_init_params["n_jobs"] = (
                -1
            )  # LogisticRegression supports parallel computation

        # Create model instance
        model_instance = model_class(**model_init_params)

        # Perform grid search or direct training
        if param_grid and len(param_grid) > 0:
            model, best_params = perform_grid_search(
                model_instance,
                param_grid,
                X_train,
                y_train,
                cv_folds=2,
                random_seed=seed,
                n_jobs=1 if model_name.lower() in ["catboost", "svm"] else -1,
            )
        else:
            # Direct training without grid search
            model = model_instance
            if custom_fit_func:
                custom_fit_func(model, X_train, y_train, X_test, y_test)
            else:
                model.fit(X_train, y_train)
            best_params = model.get_params()

        # Handle early stopping for specific models
        if self.config.early_stopping and model_name.lower() in ["xgboost", "catboost"]:
            model = self._train_with_early_stopping(
                model_class,
                model_name,
                best_params,
                X_train,
                y_train,
                X_test,
                y_test,
                seed,
            )

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

        # Create visualizations
        clean_exp_name = experiment_name.replace("_no_scaling", "")
        visualizer = Visualizer(self.config, self.output_dir, self.timestamp)
        visualizer.create_confusion_matrices(
            y_test,
            test_proba,
            f"{model_name} {clean_exp_name}",
            f"{model_name.lower()}_{clean_exp_name}",
        )

        # Feature importance plot for tree-based models
        if self.config.draw_feature_importance and model_name.lower() in [
            "randomforest",
            "xgboost",
            "catboost",
            "extratrees",
            "gradientboosting",
            "lightgbm",
        ]:
            visualizer.plot_feature_importance(
                model,
                model_name,
                experiment_name,
                [f"feature_{i}" for i in range(X_train.shape[1])],
            )

        # Save model
        model_path = self.save_model(
            model, model_name, experiment_name, best_params, all_metrics, seed
        )

        return ExperimentResult(
            model_name,
            experiment_name,
            seed,
            train_metrics,
            test_metrics,
            model_path,
            best_params,
        )

    def _train_with_early_stopping(
        self,
        model_class: Any,
        model_name: str,
        best_params: Dict,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        seed: int,
    ) -> Any:
        """Handle early stopping for XGBoost and CatBoost."""
        if model_name.lower() == "xgboost":
            # Remove n_estimators from params to avoid conflict
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

        elif model_name.lower() == "catboost":
            # Remove iterations from params to avoid conflict
            early_stop_params = {
                k: v for k, v in best_params.items() if k != "iterations"
            }

            eval_set = Pool(X_test, y_test)
            model = CatBoostClassifier(
                **early_stop_params,
                iterations=1000,
                random_seed=seed,
                verbose=False,
                thread_count=-1,
                early_stopping_rounds=50,
            )
            model.fit(X_train, y_train, eval_set=eval_set, use_best_model=True)

        return model

    # Specific model training functions (simplified wrappers)
    def train_naive_bayes(self, *args, **kwargs) -> ExperimentResult:
        return self.train_model_generic(GaussianNB, "NaiveBayes", *args, **kwargs)

    def train_random_forest(self, *args, **kwargs) -> ExperimentResult:
        return self.train_model_generic(
            RandomForestClassifier, "RandomForest", *args, **kwargs
        )

    def train_xgboost(self, *args, **kwargs) -> ExperimentResult:
        return self.train_model_generic(XGBClassifier, "XGBoost", *args, **kwargs)

    def train_catboost(self, *args, **kwargs) -> ExperimentResult:
        return self.train_model_generic(CatBoostClassifier, "CatBoost", *args, **kwargs)

    def train_mlp(self, *args, **kwargs) -> ExperimentResult:
        return self.train_model_generic(MLPClassifier, "MLP", *args, **kwargs)

    def train_svm(self, *args, **kwargs) -> ExperimentResult:
        return self.train_model_generic(SVC, "SVM", *args, **kwargs)

    def train_lightgbm(self, *args, **kwargs) -> ExperimentResult:
        return self.train_model_generic(LGBMClassifier, "LightGBM", *args, **kwargs)

    def train_extratrees(self, *args, **kwargs) -> ExperimentResult:
        return self.train_model_generic(
            ExtraTreesClassifier, "ExtraTrees", *args, **kwargs
        )

    def train_gradientboosting(self, *args, **kwargs) -> ExperimentResult:
        return self.train_model_generic(
            GradientBoostingClassifier, "GradientBoosting", *args, **kwargs
        )

    def train_knn(self, *args, **kwargs) -> ExperimentResult:
        return self.train_model_generic(KNeighborsClassifier, "KNN", *args, **kwargs)

    def train_logisticregression(self, *args, **kwargs) -> ExperimentResult:
        return self.train_model_generic(
            LogisticRegression, "LogisticRegression", *args, **kwargs
        )
