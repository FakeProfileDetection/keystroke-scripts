"""
ml_utils.py - Common utilities for ML experiments
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.base import BaseEstimator
import torch


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def merge_configs(base_config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Merge configuration with command line overrides."""
    config = base_config.copy()
    
    # Handle nested updates for param_grids
    if 'param_grids' in overrides and 'param_grids' in config:
        config['param_grids'] = base_config.get('param_grids', {}).copy()
        config['param_grids'].update(overrides['param_grids'])
        overrides = {k: v for k, v in overrides.items() if k != 'param_grids'}
    
    # Update top-level configs
    config.update(overrides)
    
    return config


def check_gpu_availability() -> bool:
    """Check if CUDA GPU is available."""
    try:
        return torch.cuda.is_available()
    except:
        return False


def prepare_param_grid(base_params: Dict[str, Any], 
                      early_stopping: bool = False,
                      use_gpu: bool = False,
                      model_type: str = "") -> Dict[str, Any]:
    """Prepare parameter grid with runtime modifications."""
    params = base_params.copy()
    
    # Model-specific modifications
    if model_type == "xgboost":
        if early_stopping:
            params["n_estimators"] = [1000]
        if use_gpu:
            params["device"] = ["cuda"]
        else:
            params["device"] = ["cpu"]
            
    elif model_type == "catboost":
        if early_stopping:
            params["iterations"] = [1000]
            
    elif model_type == "mlp":
        if early_stopping:
            params["early_stopping"] = [True]
        else:
            params["early_stopping"] = [False]
    
    elif model_type == "logisticregression":
        # Ensure solver compatibility with penalty
        if "penalty" in params and "solver" in params:
            # If L1 is in penalties, ensure compatible solvers
            if "l1" in params["penalty"]:
                compatible_solvers = ["liblinear", "saga"]
                params["solver"] = [s for s in params["solver"] if s in compatible_solvers]
    
    return params


# def perform_grid_search(model_class: BaseEstimator,
#                        param_grid: Dict[str, Any],
#                        X_train: np.ndarray,
#                        y_train: np.ndarray,
#                        cv_folds: int = 2,
#                        random_seed: int = 42,
#                        n_jobs: int = -1) -> Tuple[BaseEstimator, Dict[str, Any]]:
#     """Perform grid search and return best model and parameters."""
#     min_samples_per_class = np.bincount(y_train).min()
    
#     if min_samples_per_class < cv_folds:
#         # Not enough samples for cross-validation
#         # Use default parameters or first value from grid
#         default_params = {}
#         for key, values in param_grid.items():
#             if isinstance(values, list) and len(values) > 0:
#                 default_params[key] = values[0]
#             else:
#                 default_params[key] = values
        
#         model = model_class(**default_params)
#         model.fit(X_train, y_train)
#         return model, default_params

def perform_grid_search(model_instance: BaseEstimator,  # <-- Changed parameter name for clarity
                       param_grid: Dict[str, Any],
                       X_train: np.ndarray,
                       y_train: np.ndarray,
                       cv_folds: int = 2,
                       random_seed: int = 42,
                       n_jobs: int = -1) -> Tuple[BaseEstimator, Dict[str, Any], bool]:
    """Perform grid search and return best model and parameters."""
    min_samples_per_class = np.bincount(y_train).min()
    
    if min_samples_per_class < cv_folds:
        # Not enough samples for cross-validation
        # Use the provided instance directly
        model_instance.fit(X_train, y_train)
        return model_instance, model_instance.get_params(), False
    
    # Perform grid search
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    grid_search = GridSearchCV(
        model_instance,
        param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=n_jobs,
        verbose=0
    )
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_, True


def get_experiment_filters(experiment_config: Dict[str, Any], 
                          df_columns: List[str]) -> Tuple[List[Any], Any, str]:
    """
    Parse experiment configuration to get train/test filters.
    
    Returns:
        tuple: (train_values, test_value, column_name)
    """
    # Check experiment type
    if experiment_config.get("session", False):
        return (experiment_config["train"], 
                experiment_config["test"], 
                "session_id")
    elif "platform" in experiment_config:
        return (experiment_config["train"], 
                experiment_config["test"], 
                "platform_id")
    else:
        raise ValueError(f"Unknown experiment type in config: {experiment_config}")


def validate_dataset(df, required_columns: List[str]) -> None:
    """Validate that dataset contains required columns."""
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")


def get_feature_columns(df_columns: List[str]) -> List[str]:
    """Get feature columns by excluding metadata columns."""
    metadata_cols = {"user_id", "platform_id", "session_id", "video_id"}
    return [col for col in df_columns if col not in metadata_cols]

