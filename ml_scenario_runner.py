#!/usr/bin/env python3
"""
ml_scenario_runner.py - Scenario-based ML experiment runner for keystroke biometrics.

This runner uses the scenario-based experiment definitions from scenarios.py.
Results are aggregated at the scenario level (averaging across sub-experiments).
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ml_core import ExperimentConfig, ExperimentResult, ModelTrainer
from ml_visualizer import Visualizer
from ml_utils import (
    load_config, merge_configs, validate_dataset,
    get_feature_columns, get_sub_experiment_data
)
from scenarios import (
    Scenario, SubExperiment, generate_all_scenarios,
    get_scenario_by_id, PLATFORM_NAMES
)


@dataclass
class SubExperimentResult:
    """Result from a single sub-experiment run."""
    sub_experiment_name: str
    scenario_id: str
    model_name: str
    random_seed: int
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    model_path: str
    train_samples: int
    test_samples: int
    cross_validation_used: bool = False


@dataclass
class ScenarioResult:
    """Aggregated results for a scenario across all sub-experiments."""
    scenario_id: str
    scenario_name: str
    scenario_description: str
    model_name: str
    random_seed: int
    num_sub_experiments: int

    # Aggregated metrics (mean across sub-experiments)
    mean_test_accuracy: float = 0.0
    std_test_accuracy: float = 0.0
    mean_test_top_1_accuracy: float = 0.0
    std_test_top_1_accuracy: float = 0.0
    mean_test_top_5_accuracy: float = 0.0
    std_test_top_5_accuracy: float = 0.0
    mean_test_f1_weighted: float = 0.0
    std_test_f1_weighted: float = 0.0

    # Train metrics
    mean_train_accuracy: float = 0.0
    mean_train_top_1_accuracy: float = 0.0
    mean_train_top_5_accuracy: float = 0.0

    # Sub-experiment details
    sub_experiment_results: List[SubExperimentResult] = field(default_factory=list)


class MLScenarioRunner:
    """Main orchestrator for scenario-based ML experiments."""

    def __init__(self, config: ExperimentConfig, max_workers: int = None):
        self.config = config
        self.max_workers = max_workers or __import__('os').cpu_count()
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.output_dir = self._create_output_dir()

        # Results storage
        self.sub_experiment_results: List[SubExperimentResult] = []
        self.scenario_results: List[ScenarioResult] = []

        # Initialize components
        self.trainer = ModelTrainer(config, self.output_dir, self.timestamp)
        self.visualizer = Visualizer(config, self.output_dir, self.timestamp)

        # Get scenarios to run from config
        self.scenarios_to_run = config.config_dict.get("scenarios_to_run", ["1.1", "1.2", "2.1", "2.2", "3.1", "3.2", "4.1", "4.2", "5.1", "5.2"])
        self.models_to_train = config.models_to_train

        # Model training functions mapping
        self.model_train_funcs = {
            "RandomForest": self.trainer.train_random_forest,
            "XGBoost": self.trainer.train_xgboost,
            "CatBoost": self.trainer.train_catboost,
            "SVM": self.trainer.train_svm,
            "MLP": self.trainer.train_mlp,
            "NaiveBayes": self.trainer.train_naive_bayes,
            "LightGBM": self.trainer.train_lightgbm,
            "ExtraTrees": self.trainer.train_extratrees,
            "GradientBoosting": self.trainer.train_gradientboosting,
            "KNN": self.trainer.train_knn,
            "LogisticRegression": self.trainer.train_logisticregression
        }

        print(f"📋 Configuration loaded:")
        print(f"  - Debug mode: {config.debug_mode}")
        print(f"  - Early stopping: {config.early_stopping}")
        print(f"  - Models to train: {', '.join(self.models_to_train)}")
        print(f"  - Scenarios to run: {', '.join(self.scenarios_to_run)}")
        print(f"  - Random seeds: {config.random_seeds}")
        print(f"  - Dataset path: {config.dataset_path}")

        # File paths for results
        self.sub_exp_results_path = self.output_dir / f"sub_experiment_results_{self.timestamp}.csv"
        self.scenario_results_path = self.output_dir / f"scenario_results_{self.timestamp}.csv"
        self.detailed_results_path = self.output_dir / f"detailed_topk_results_{self.timestamp}.csv"

    def _create_output_dir(self) -> Path:
        """Create output directory with timestamp."""
        suffix = "_early_stop" if self.config.early_stopping else ""
        debug_suffix = "_debug" if self.config.debug_mode else ""
        affix = f"_{self.config.output_affix}" if self.config.output_affix else ""
        dir_name = f"scenario_results{affix}_{self.timestamp}{suffix}{debug_suffix}"
        output_dir = Path(dir_name)
        output_dir.mkdir(exist_ok=True)
        return output_dir

    def load_data(self) -> pd.DataFrame:
        """Load and validate dataset."""
        print(f"📂 Loading dataset: {self.config.dataset_path}")

        if not Path(self.config.dataset_path).exists():
            raise FileNotFoundError(f"Dataset not found: {self.config.dataset_path}")

        df = pl.read_csv(self.config.dataset_path)
        print(f"📊 Dataset shape: {df.shape}")

        # Validate required columns
        required_cols = ["user_id", "platform_id", "video_id", "session_id"]
        validate_dataset(df, required_cols)

        # Convert to pandas for sklearn compatibility
        df_pd = df.to_pandas()

        print(f"📊 Dataset info:")
        print(f"  - Platforms: {sorted(df_pd['platform_id'].unique())}")
        print(f"  - Videos: {sorted(df_pd['video_id'].unique())}")
        print(f"  - Sessions: {sorted(df_pd['session_id'].unique())}")
        print(f"  - Users: {df_pd['user_id'].nunique()}")

        return df_pd

    def run_scenarios(self, df: pd.DataFrame):
        """Run all scenarios and their sub-experiments."""
        total_scenarios = len(self.scenarios_to_run)

        print(f"\n🚀 Running {total_scenarios} scenarios")

        for scenario_idx, scenario_id in enumerate(self.scenarios_to_run, 1):
            scenario = get_scenario_by_id(scenario_id)
            print(f"\n{'='*70}")
            print(f"📊 Scenario {scenario_idx}/{total_scenarios}: {scenario.name}")
            print(f"   {scenario.description}")
            print(f"   Sub-experiments: {len(scenario.sub_experiments)}")
            print(f"{'='*70}")

            self._run_scenario(df, scenario)

        # Aggregate results
        self._aggregate_scenario_results()

        print(f"\n🎉 All scenarios completed! Results saved to: {self.output_dir}")

    def _run_scenario(self, df: pd.DataFrame, scenario: Scenario):
        """Run all sub-experiments for a single scenario."""

        for seed in self.config.random_seeds:
            print(f"\n🎲 Random seed: {seed}")

            for model_name in self.models_to_train:
                if model_name not in self.model_train_funcs:
                    print(f"❌ Unknown model: {model_name}")
                    continue

                print(f"\n🤖 Model: {model_name}")

                for sub_idx, sub_exp in enumerate(scenario.sub_experiments, 1):
                    print(f"  [{sub_idx}/{len(scenario.sub_experiments)}] {sub_exp.name}", end=" ")

                    try:
                        result = self._run_sub_experiment(df, sub_exp, model_name, seed)
                        if result:
                            self.sub_experiment_results.append(result)
                            print(f"✅ Top-1: {result.test_metrics.get('test_top_1_accuracy', 0):.4f}")
                        else:
                            print("⚠️ Skipped (insufficient data)")
                    except Exception as e:
                        print(f"❌ Failed: {e}")
                        import traceback
                        traceback.print_exc()

                # Save intermediate results after each model
                self._save_intermediate_results()

    def _run_sub_experiment(self, df: pd.DataFrame, sub_exp: SubExperiment,
                           model_name: str, seed: int) -> Optional[SubExperimentResult]:
        """Run a single sub-experiment and return the result."""

        # Get train/test data using the sub-experiment filters
        X_train, X_test, y_train, y_test = get_sub_experiment_data(df, sub_exp)

        # Check if we have enough data
        if len(X_train) == 0 or len(X_test) == 0:
            return None

        # Filter to only users present in both train and test sets (for identification)
        train_users = set(y_train)
        test_users = set(y_test)
        common_users = train_users.intersection(test_users)
        
        if len(common_users) < 2:
            return None
            
        # Filter train and test data to only include common users
        train_mask = np.isin(y_train, list(common_users))
        test_mask = np.isin(y_test, list(common_users))
        
        X_train = X_train[train_mask]
        y_train = y_train[train_mask]
        X_test = X_test[test_mask]
        y_test = y_test[test_mask]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            return None

        # Encode labels
        y_train_encoded = self.trainer.label_encoder.fit_transform(y_train)
        y_test_encoded = self.trainer.label_encoder.transform(y_test)

        # Train model
        train_func = self.model_train_funcs[model_name]
        experiment_result = train_func(
            X_train, X_test,
            y_train_encoded, y_test_encoded,
            sub_exp.name, seed
        )

        # Convert to SubExperimentResult
        sub_result = SubExperimentResult(
            sub_experiment_name=sub_exp.name,
            scenario_id=sub_exp.scenario_id,
            model_name=model_name,
            random_seed=seed,
            train_metrics=experiment_result.train_metrics,
            test_metrics=experiment_result.test_metrics,
            hyperparameters=experiment_result.hyperparameters,
            model_path=experiment_result.model_path,
            train_samples=len(X_train),
            test_samples=len(X_test),
            cross_validation_used=experiment_result.cross_validation_used
        )

        return sub_result

    def _aggregate_scenario_results(self):
        """Aggregate sub-experiment results into scenario-level results."""

        print("\n📊 Aggregating results by scenario...")

        # Group by (scenario_id, model_name, seed)
        results_df = pd.DataFrame([
            {
                "scenario_id": r.scenario_id,
                "sub_experiment_name": r.sub_experiment_name,
                "model_name": r.model_name,
                "random_seed": r.random_seed,
                "test_accuracy": r.test_metrics.get("test_accuracy", 0),
                "test_top_1_accuracy": r.test_metrics.get("test_top_1_accuracy", 0),
                "test_top_2_accuracy": r.test_metrics.get("test_top_2_accuracy", 0),
                "test_top_3_accuracy": r.test_metrics.get("test_top_3_accuracy", 0),
                "test_top_4_accuracy": r.test_metrics.get("test_top_4_accuracy", 0),
                "test_top_5_accuracy": r.test_metrics.get("test_top_5_accuracy", 0),
                "test_f1_weighted": r.test_metrics.get("test_f1_weighted", 0),
                "train_accuracy": r.train_metrics.get("train_accuracy", 0),
                "train_top_1_accuracy": r.train_metrics.get("train_top_1_accuracy", 0),
                "train_top_5_accuracy": r.train_metrics.get("train_top_5_accuracy", 0),
                "train_samples": r.train_samples,
                "test_samples": r.test_samples,
            }
            for r in self.sub_experiment_results
        ])

        if results_df.empty:
            print("⚠️ No results to aggregate")
            return

        # Get scenario metadata
        all_scenarios = generate_all_scenarios()

        # Aggregate by scenario, model, and seed
        grouped = results_df.groupby(["scenario_id", "model_name", "random_seed"])

        for (scenario_id, model_name, seed), group in grouped:
            scenario = all_scenarios.get(scenario_id)
            if not scenario:
                continue

            scenario_result = ScenarioResult(
                scenario_id=scenario_id,
                scenario_name=scenario.name,
                scenario_description=scenario.description,
                model_name=model_name,
                random_seed=seed,
                num_sub_experiments=len(group),

                # Test metrics
                mean_test_accuracy=group["test_accuracy"].mean(),
                std_test_accuracy=group["test_accuracy"].std(),
                mean_test_top_1_accuracy=group["test_top_1_accuracy"].mean(),
                std_test_top_1_accuracy=group["test_top_1_accuracy"].std(),
                mean_test_top_5_accuracy=group["test_top_5_accuracy"].mean(),
                std_test_top_5_accuracy=group["test_top_5_accuracy"].std(),
                mean_test_f1_weighted=group["test_f1_weighted"].mean(),
                std_test_f1_weighted=group["test_f1_weighted"].std(),

                # Train metrics
                mean_train_accuracy=group["train_accuracy"].mean(),
                mean_train_top_1_accuracy=group["train_top_1_accuracy"].mean(),
                mean_train_top_5_accuracy=group["train_top_5_accuracy"].mean(),
            )

            self.scenario_results.append(scenario_result)

        print(f"✅ Aggregated into {len(self.scenario_results)} scenario-level results")

    def _save_intermediate_results(self):
        """Save intermediate results to CSV."""
        if self.sub_experiment_results:
            results_data = []
            for r in self.sub_experiment_results:
                row = {
                    "scenario_id": r.scenario_id,
                    "sub_experiment_name": r.sub_experiment_name,
                    "model_name": r.model_name,
                    "random_seed": r.random_seed,
                    "train_samples": r.train_samples,
                    "test_samples": r.test_samples,
                    "hyperparameters": str(r.hyperparameters),
                    "model_path": r.model_path,
                    **r.train_metrics,
                    **r.test_metrics
                }
                results_data.append(row)

            pd.DataFrame(results_data).to_csv(self.sub_exp_results_path, index=False)

    def generate_comprehensive_report(self):
        """Generate comprehensive reports and visualizations."""
        if not self.scenario_results:
            print("⚠️ No results to report")
            return

        print("\n📋 Generating comprehensive reports...")

        # Save scenario-level results
        scenario_data = []
        for r in self.scenario_results:
            row = {
                "scenario_id": r.scenario_id,
                "scenario_name": r.scenario_name,
                "scenario_description": r.scenario_description,
                "model_name": r.model_name,
                "random_seed": r.random_seed,
                "num_sub_experiments": r.num_sub_experiments,
                "mean_test_accuracy": r.mean_test_accuracy,
                "std_test_accuracy": r.std_test_accuracy,
                "mean_test_top_1_accuracy": r.mean_test_top_1_accuracy,
                "std_test_top_1_accuracy": r.std_test_top_1_accuracy,
                "mean_test_top_5_accuracy": r.mean_test_top_5_accuracy,
                "std_test_top_5_accuracy": r.std_test_top_5_accuracy,
                "mean_test_f1_weighted": r.mean_test_f1_weighted,
                "std_test_f1_weighted": r.std_test_f1_weighted,
                "mean_train_accuracy": r.mean_train_accuracy,
                "mean_train_top_1_accuracy": r.mean_train_top_1_accuracy,
                "mean_train_top_5_accuracy": r.mean_train_top_5_accuracy,
            }
            scenario_data.append(row)

        scenario_df = pd.DataFrame(scenario_data)
        scenario_df.to_csv(self.scenario_results_path, index=False)
        print(f"📊 Scenario results saved to: {self.scenario_results_path}")

        # Generate scenario comparison plot
        self._plot_scenario_comparison(scenario_df)

        # Generate model comparison plot per scenario
        self._plot_model_comparison_by_scenario(scenario_df)

        # Print summary
        print(f"\n📊 Final Summary:")

        # Best results per scenario
        for scenario_id in sorted(scenario_df["scenario_id"].unique()):
            scenario_subset = scenario_df[scenario_df["scenario_id"] == scenario_id]
            best_idx = scenario_subset["mean_test_top_1_accuracy"].idxmax()
            best_row = scenario_subset.loc[best_idx]
            print(f"  {best_row['scenario_name']}: Best = {best_row['model_name']} "
                  f"(Top-1: {best_row['mean_test_top_1_accuracy']:.4f} ± {best_row['std_test_top_1_accuracy']:.4f})")

        print(f"\n📁 All outputs saved to: {self.output_dir}")

        # Generate TSV template output
        self.generate_template_tsv()

    def generate_template_tsv(self):
        """Generate TSV output matching the results template format.

        Output format matches results_template.xlsx with columns:
        - Scenario Group, Scenario, Train, Train samples, Test, Test samples
        - k=1, k=2, k=3, k=4, k=5 (top-k accuracy for best model)
        - Mean and std rows per scenario
        """
        if not self.sub_experiment_results:
            print("⚠️ No results to export to TSV")
            return

        tsv_path = self.output_dir / f"ml_baseline_results_{self.timestamp}.tsv"

        # Convert results to DataFrame
        results_data = []
        for r in self.sub_experiment_results:
            results_data.append({
                "scenario_id": r.scenario_id,
                "sub_experiment_name": r.sub_experiment_name,
                "model_name": r.model_name,
                "random_seed": r.random_seed,
                "train_samples": r.train_samples,
                "test_samples": r.test_samples,
                "test_top_1_accuracy": r.test_metrics.get("test_top_1_accuracy", 0),
                "test_top_2_accuracy": r.test_metrics.get("test_top_2_accuracy", 0),
                "test_top_3_accuracy": r.test_metrics.get("test_top_3_accuracy", 0),
                "test_top_4_accuracy": r.test_metrics.get("test_top_4_accuracy", 0),
                "test_top_5_accuracy": r.test_metrics.get("test_top_5_accuracy", 0),
            })

        results_df = pd.DataFrame(results_data)

        # Find best model per scenario (by mean top-1 accuracy across all sub-experiments)
        best_models = {}
        for scenario_id in results_df["scenario_id"].unique():
            scenario_data = results_df[results_df["scenario_id"] == scenario_id]
            model_perf = scenario_data.groupby("model_name")["test_top_1_accuracy"].mean()
            best_models[scenario_id] = model_perf.idxmax()

        print(f"\n📊 Best models per scenario:")
        for sid, model in best_models.items():
            print(f"  {sid}: {model}")

        # Get scenario metadata
        all_scenarios = generate_all_scenarios()

        # Generate TSV rows
        tsv_rows = []

        # Header
        tsv_rows.append([
            "Scenario Group", "Scenario", "Train", "Train samples/user",
            "Test", "Test samples/user", "Notes",
            "k=1", "k=2", "k=3", "k=4", "k=5", "Best Model"
        ])

        # Process each scenario
        scenario_groups = {
            "1.1": "Same platform, same topic",
            "1.2": "Same platform, same topic",
            "2.1": "Cross platform, same topic",
            "2.2": "Cross platform, same topic",
            "3.1": "Cross platform, same topic (1→2)",
            "3.2": "Cross platform, same topic (2→1)",
            "4.1": "Cross platform, cross topic (1→2)",
            "4.2": "Cross platform, cross topic (2→1)",
            "5.1": "Same platform, cross topic (S1→S2)",
            "5.2": "Same platform, cross topic (S2→S1)",
        }

        for scenario_id in sorted(results_df["scenario_id"].unique()):
            scenario = all_scenarios.get(scenario_id)
            if not scenario:
                continue

            best_model = best_models[scenario_id]
            scenario_group = scenario_groups.get(scenario_id, "")

            # Filter to best model and average across seeds
            scenario_best = results_df[
                (results_df["scenario_id"] == scenario_id) &
                (results_df["model_name"] == best_model)
            ]

            # Group by sub-experiment (average across seeds)
            for sub_exp in scenario.sub_experiments:
                sub_data = scenario_best[scenario_best["sub_experiment_name"] == sub_exp.name]

                if sub_data.empty:
                    continue

                # Average across seeds
                avg_k1 = sub_data["test_top_1_accuracy"].mean()
                avg_k2 = sub_data["test_top_2_accuracy"].mean()
                avg_k3 = sub_data["test_top_3_accuracy"].mean()
                avg_k4 = sub_data["test_top_4_accuracy"].mean()
                avg_k5 = sub_data["test_top_5_accuracy"].mean()

                train_samples = sub_data["train_samples"].iloc[0]
                test_samples = sub_data["test_samples"].iloc[0]

                row = [
                    scenario_group if sub_exp == scenario.sub_experiments[0] else "",
                    f"Scenario {scenario_id}" if sub_exp == scenario.sub_experiments[0] else "",
                    sub_exp.train_notation,
                    scenario.train_samples_per_user,
                    sub_exp.test_notation,
                    scenario.test_samples_per_user,
                    f"train={train_samples}, test={test_samples}",
                    f"{avg_k1:.4f}",
                    f"{avg_k2:.4f}",
                    f"{avg_k3:.4f}",
                    f"{avg_k4:.4f}",
                    f"{avg_k5:.4f}",
                    best_model if sub_exp == scenario.sub_experiments[0] else ""
                ]
                tsv_rows.append(row)

            # Add mean row
            scenario_all = scenario_best.groupby("sub_experiment_name").agg({
                "test_top_1_accuracy": "mean",
                "test_top_2_accuracy": "mean",
                "test_top_3_accuracy": "mean",
                "test_top_4_accuracy": "mean",
                "test_top_5_accuracy": "mean",
            })

            mean_k1 = scenario_all["test_top_1_accuracy"].mean()
            mean_k2 = scenario_all["test_top_2_accuracy"].mean()
            mean_k3 = scenario_all["test_top_3_accuracy"].mean()
            mean_k4 = scenario_all["test_top_4_accuracy"].mean()
            mean_k5 = scenario_all["test_top_5_accuracy"].mean()

            tsv_rows.append([
                "", "", "mean", "", "", "", "",
                f"{mean_k1:.4f}", f"{mean_k2:.4f}", f"{mean_k3:.4f}",
                f"{mean_k4:.4f}", f"{mean_k5:.4f}", ""
            ])

            # Add std row
            std_k1 = scenario_all["test_top_1_accuracy"].std()
            std_k2 = scenario_all["test_top_2_accuracy"].std()
            std_k3 = scenario_all["test_top_3_accuracy"].std()
            std_k4 = scenario_all["test_top_4_accuracy"].std()
            std_k5 = scenario_all["test_top_5_accuracy"].std()

            tsv_rows.append([
                "", "", "std", "", "", "", "",
                f"{std_k1:.4f}", f"{std_k2:.4f}", f"{std_k3:.4f}",
                f"{std_k4:.4f}", f"{std_k5:.4f}", ""
            ])

            # Add blank separator row
            tsv_rows.append([""] * 13)

        # Write TSV
        with open(tsv_path, "w") as f:
            for row in tsv_rows:
                f.write("\t".join(str(x) for x in row) + "\n")

        print(f"📋 Template TSV saved to: {tsv_path}")

    def _plot_scenario_comparison(self, scenario_df: pd.DataFrame):
        """Plot comparison of scenarios (average across models)."""

        # Average across models and seeds for each scenario
        scenario_avg = scenario_df.groupby(["scenario_id", "scenario_name"]).agg({
            "mean_test_top_1_accuracy": "mean",
            "mean_test_top_5_accuracy": "mean",
            "std_test_top_1_accuracy": "mean",
        }).reset_index()

        scenario_avg = scenario_avg.sort_values("scenario_id")

        fig, ax = plt.subplots(figsize=(12, 6))

        x = range(len(scenario_avg))
        width = 0.35

        bars1 = ax.bar([i - width/2 for i in x], scenario_avg["mean_test_top_1_accuracy"],
                      width, label="Top-1 Accuracy", color="steelblue")
        bars2 = ax.bar([i + width/2 for i in x], scenario_avg["mean_test_top_5_accuracy"],
                      width, label="Top-5 Accuracy", color="coral")

        ax.set_xlabel("Scenario")
        ax.set_ylabel("Accuracy")
        ax.set_title("Scenario Performance Comparison (averaged across models)")
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_avg["scenario_name"], rotation=45, ha="right")
        ax.legend()
        ax.set_ylim(0, 1)

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(self.output_dir / f"scenario_comparison_{self.timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"📈 Scenario comparison plot saved")

    def _plot_model_comparison_by_scenario(self, scenario_df: pd.DataFrame):
        """Plot model comparison for each scenario."""

        for scenario_id in sorted(scenario_df["scenario_id"].unique()):
            scenario_subset = scenario_df[scenario_df["scenario_id"] == scenario_id]
            scenario_name = scenario_subset["scenario_name"].iloc[0]

            # Average across seeds for each model
            model_avg = scenario_subset.groupby("model_name").agg({
                "mean_test_top_1_accuracy": "mean",
                "mean_test_top_5_accuracy": "mean",
                "std_test_top_1_accuracy": "mean",
            }).reset_index()

            model_avg = model_avg.sort_values("mean_test_top_1_accuracy", ascending=False)

            fig, ax = plt.subplots(figsize=(12, 6))

            x = range(len(model_avg))
            width = 0.35

            ax.bar([i - width/2 for i in x], model_avg["mean_test_top_1_accuracy"],
                  width, label="Top-1 Accuracy", color="steelblue",
                  yerr=model_avg["std_test_top_1_accuracy"], capsize=3)
            ax.bar([i + width/2 for i in x], model_avg["mean_test_top_5_accuracy"],
                  width, label="Top-5 Accuracy", color="coral")

            ax.set_xlabel("Model")
            ax.set_ylabel("Accuracy")
            ax.set_title(f"{scenario_name}: Model Comparison")
            ax.set_xticks(x)
            ax.set_xticklabels(model_avg["model_name"], rotation=45, ha="right")
            ax.legend()
            ax.set_ylim(0, 1)

            plt.tight_layout()
            safe_scenario_name = scenario_id.replace(".", "_")
            plt.savefig(self.output_dir / f"model_comparison_scenario_{safe_scenario_name}_{self.timestamp}.png",
                       dpi=300, bbox_inches='tight')
            plt.close()

        print(f"📈 Model comparison plots saved for all scenarios")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Scenario-based ML experiments for keystroke biometrics')
    parser.add_argument('-c', '--config', default='config_scenarios.json',
                        help='Path to configuration file (default: config_scenarios.json)')
    parser.add_argument('-d', '--dataset', help='Path to dataset CSV (overrides config)')
    parser.add_argument('-e', '--early-stop', action='store_true',
                        help='Use early stopping (overrides config)')
    parser.add_argument('-s', '--seeds', type=int, nargs='+',
                        help='Random seeds (overrides config)')
    parser.add_argument('-o', '--output-affix', help='Output directory suffix (overrides config)')
    parser.add_argument('--scenarios', nargs='+',
                        help='Specific scenarios to run (e.g., 1 2 3.1)')
    parser.add_argument('--models', nargs='+',
                        help='Specific models to train (overrides config)')
    parser.add_argument('--max-workers', type=int, help='Max CPU workers')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU acceleration (overrides config)')
    parser.add_argument('--debug', action='store_true',
                        help='Use debug configuration')

    args = parser.parse_args()

    # Load configuration
    config_path = args.config

    try:
        config_dict = load_config(config_path)
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {config_path}")
        print("Creating default config_scenarios.json...")

        # Create default config
        default_config = {
            "dataset_path": "dataset/lori_typenet_features.csv",
            "early_stopping": True,
            "seeds": [42],
            "output_affix": "scenarios",
            "show_class_distributions": False,
            "draw_feature_importance": True,
            "debug": False,
            "use_gpu": True,
            "scenarios_to_run": ["1.1", "1.2", "2.1", "2.2", "3.1", "3.2", "4.1", "4.2", "5.1", "5.2"],
            "models_to_train": ["RandomForest", "XGBoost", "CatBoost"],
            "param_grids": {
                "randomforest": {"n_estimators": [100, 300], "max_depth": [10, 20]},
                "xgboost": {"n_estimators": [100], "max_depth": [4, 6]},
                "catboost": {"iterations": [100], "depth": [4, 6]}
            }
        }

        import json
        with open("config_scenarios.json", "w") as f:
            json.dump(default_config, f, indent=2)

        config_dict = default_config
        print("✅ Default configuration created")

    # Build overrides from command line arguments
    overrides = {}
    if args.dataset:
        overrides['dataset_path'] = args.dataset
    if args.early_stop:
        overrides['early_stopping'] = True
    if args.seeds:
        overrides['seeds'] = args.seeds
    if args.output_affix:
        overrides['output_affix'] = args.output_affix
    if args.scenarios:
        overrides['scenarios_to_run'] = args.scenarios
    if args.models:
        overrides['models_to_train'] = args.models
    if args.no_gpu:
        overrides['use_gpu'] = False
    if args.debug:
        overrides['debug'] = True
        overrides['seeds'] = [42]
        overrides['scenarios_to_run'] = ["1.1"]
        overrides['models_to_train'] = ["RandomForest"]

    # Merge configurations
    final_config = merge_configs(config_dict, overrides)

    # Create configuration object
    config = ExperimentConfig(final_config)

    # Validate configuration
    if not config.dataset_path:
        print("❌ No dataset path specified. Use -d flag or set in config file.")
        sys.exit(1)

    print(f"📋 Configuration loaded from: {config_path}")

    # Run experiments
    runner = MLScenarioRunner(config, max_workers=args.max_workers)

    try:
        df = runner.load_data()

        # Save config to output directory
        config_file = runner.output_dir / f"config_{runner.timestamp}.json"
        import json
        with open(config_file, 'w') as f:
            json.dump(final_config, f, indent=4)

        # Run scenarios
        runner.run_scenarios(df)

        # Generate comprehensive report
        runner.generate_comprehensive_report()

        print("\n🎊 Pipeline completed successfully!")
        print(f"📁 Results directory: {runner.output_dir}")

    except Exception as e:
        print(f"❌ Error running experiments: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
