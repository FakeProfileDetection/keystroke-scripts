#!/usr/bin/env python3
"""
ml_platforms_runner.py - Quick fix version that works with current ml_core.py
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import List

import polars as pl
import pandas as pd
import matplotlib.pyplot as plt

from ml_platforms_core import (
    ExperimentConfig, ExperimentResult, ModelTrainer, Visualizer
)


class MLExperimentRunner:
    """Main orchestrator for ML experiments."""
    
    def __init__(self, config: ExperimentConfig, max_workers: int = None, use_gpu: bool = True):
        self.config = config
        self.max_workers = max_workers or 16  # Default to 16 workers
        self.use_gpu = use_gpu
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.output_dir = self._create_output_dir()
        self.results: List[ExperimentResult] = []
        self.detailed_results: List[dict] = []
        
        # Initialize components
        self.trainer = ModelTrainer(config, self.output_dir, self.timestamp)
        self.visualizer = Visualizer(config, self.output_dir, self.timestamp)
        
        # Experiment configurations: (train_platforms, test_platform, name)
        self.experiments = [
            ([1, 2], 3, "FI_vs_T"), ([1, 3], 2, "FT_vs_I"), ([2, 1], 3, "IF_vs_T"),
            ([2, 3], 1, "IT_vs_F"), ([3, 1], 2, "TF_vs_I"), ([3, 2], 1, "TI_vs_F"),
            ([1], 2, "F_vs_I"), ([1], 3, "F_vs_T"), ([2], 1, "I_vs_F"),
            ([2], 3, "I_vs_T"), ([3], 1, "T_vs_F"), ([3], 2, "T_vs_I"),
        ]
    
    def _create_output_dir(self) -> Path:
        """Create output directory with timestamp."""
        suffix = "_early_stop" if self.config.early_stopping else ""
        affix = f"_{self.config.output_affix}" if self.config.output_affix else ""
        dir_name = f"experiment_results{affix}_{self.timestamp}{suffix}"
        output_dir = Path(dir_name)
        output_dir.mkdir(exist_ok=True)
        return output_dir
    
    def load_data(self) -> pl.DataFrame:
        """Load and validate dataset."""
        print(f"📂 Loading dataset: {self.config.dataset_path}")
        
        if not Path(self.config.dataset_path).exists():
            raise FileNotFoundError(f"Dataset not found: {self.config.dataset_path}")
        
        df = pl.read_csv(self.config.dataset_path)
        print(f"📊 Dataset shape: {df.shape}")
        
        # Validate required columns
        required_cols = {"user_id", "platform_id"}
        if not required_cols.issubset(set(df.columns)):
            raise ValueError(f"Missing required columns: {required_cols - set(df.columns)}")
        
        return df
    
    def run_experiments(self, df: pl.DataFrame):
        """Run all experiments with comprehensive tracking."""
        print(f"\n🚀 Running {len(self.experiments)} experiments with {self.config.num_seeds} seeds each")
        print(f"Total model runs: {len(self.experiments) * 4 * self.config.num_seeds}")
        print(f"🖥️  Using {self.max_workers} CPU workers")
        print(f"🎮 GPU acceleration: {'Enabled' if self.use_gpu else 'Disabled'}")
        
        df_pd = df.to_pandas()  # Convert once for sklearn compatibility
        
        for exp_idx, (train_platforms, test_platform, exp_name) in enumerate(self.experiments, 1):
            print(f"\n{'='*60}")
            print(f"🎯 Experiment {exp_idx}/{len(self.experiments)}: {exp_name}")
            print(f"Training on platforms: {train_platforms}, Testing on: {test_platform}")
            print(f"{'='*60}")
            
            # Create train/test splits
            train_mask = df_pd["platform_id"].isin(train_platforms)
            test_mask = df_pd["platform_id"] == test_platform
            
            feature_cols = [col for col in df_pd.columns if col not in {"user_id", "platform_id", "session_id"}]
            X_train = df_pd.loc[train_mask, feature_cols]
            X_test = df_pd.loc[test_mask, feature_cols]
            y_train = df_pd.loc[train_mask, "user_id"]
            y_test = df_pd.loc[test_mask, "user_id"]
            
            print(f"📊 Train samples: {len(X_train)}, Test samples: {len(X_test)}")
            print(f"📊 Train features: {len(X_train.columns)}")
            
            # Check class distribution
            train_class_counts = y_train.value_counts()
            test_class_counts = y_test.value_counts()
            
            print(f"📊 Train classes: {len(train_class_counts)}, Test classes: {len(test_class_counts)}")
            print(f"📊 Min samples per class (train): {train_class_counts.min()}")
            print(f"📊 Min samples per class (test): {test_class_counts.min()}")
            
            if train_class_counts.min() < 2:
                print("⚠️ Very low sample count - some models may not work properly")
            
            # Show class distribution if requested
            if self.config.show_class_distributions:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                
                train_class_counts.head(20).plot(kind="bar", ax=ax1, title=f"Train Classes - {exp_name}")
                ax1.set_xlabel("User ID")
                ax1.set_ylabel("Sample Count")
                
                test_class_counts.head(20).plot(kind="bar", ax=ax2, title=f"Test Classes - {exp_name}")
                ax2.set_xlabel("User ID")
                ax2.set_ylabel("Sample Count")
                
                plt.tight_layout()
                plt.savefig(self.output_dir / f"class_distribution_{exp_name}.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            # Encode labels for sklearn
            y_train_encoded = self.trainer.label_encoder.fit_transform(y_train)
            y_test_encoded = self.trainer.label_encoder.transform(y_test)
            
            # Convert to numpy arrays
            X_train_np = X_train.values
            X_test_np = X_test.values
            
            # Run models with different seeds
            for seed in self.config.random_seeds:
                seed_suffix = f"_seed{seed}" if self.config.num_seeds > 1 else ""
                full_exp_name = f"{exp_name}{seed_suffix}"
                
                print(f"\n🎲 Running with random seed {seed}")
                
                # Train all models
                models_to_run = [
                    ("RandomForest", self.trainer.train_random_forest),
                    ("XGBoost", self.trainer.train_xgboost),
                    ("CatBoost", self.trainer.train_catboost),
                    ("SVM", self.trainer.train_svm),
                ]
                
                for model_name, train_func in models_to_run:
                    try:
                        result = train_func(X_train_np, X_test_np, y_train_encoded, y_test_encoded, 
                                          full_exp_name, seed)
                        self.results.append(result)
                        
                        # Store detailed results for top-k analysis
                        for k in range(1, 6):
                            detailed_record = {
                                'model': result.model_name,
                                'experiment': result.experiment_name,
                                'random_seed': result.random_seed,
                                'early_stopping': self.config.early_stopping,
                                'k_value': k,
                                'train_top_k_accuracy': result.train_metrics.get(f'train_top_{k}_accuracy', 0),
                                'test_top_k_accuracy': result.test_metrics.get(f'test_top_{k}_accuracy', 0),
                                'model_path': result.model_path,
                                'hyperparameters': str(result.hyperparameters),
                                'timestamp': self.timestamp
                            }
                            self.detailed_results.append(detailed_record)
                        
                        print(f"✅ {model_name}: Top-1 = {result.test_metrics.get('test_top_1_accuracy', 0):.4f}, "
                              f"Top-5 = {result.test_metrics.get('test_top_5_accuracy', 0):.4f}")
                    except Exception as e:
                        print(f"❌ {model_name} failed: {e}")
        
        print(f"\n🎉 All experiments completed! Results saved to: {self.output_dir}")
    
    def generate_comprehensive_report(self):
        """Generate comprehensive HTML report and all visualizations - exactly like original."""
        if not self.results:
            print("⚠️ No results to report")
            return
        
        print("\n📋 Generating comprehensive reports...")
        
        # Convert results to DataFrames
        results_data = []
        for r in self.results:
            row = {
                'model': r.model_name,
                'experiment': r.experiment_name,
                'random_seed': r.random_seed,
                'early_stopping': self.config.early_stopping,
                'model_path': r.model_path,
                'hyperparameters': str(r.hyperparameters),
                **r.train_metrics,
                **r.test_metrics
            }
            results_data.append(row)
        
        results_df = pd.DataFrame(results_data)
        detailed_df = pd.DataFrame(self.detailed_results)
        
        # Save CSV files - EXACTLY like original
        csv_path = self.output_dir / f"experiment_results_{self.timestamp}.csv"
        detailed_csv_path = self.output_dir / f"detailed_topk_results_{self.timestamp}.csv"
        
        results_df.to_csv(csv_path, index=False)
        detailed_df.to_csv(detailed_csv_path, index=False)
        
        print(f"📊 Results saved to: {csv_path}")
        print(f"📊 Detailed Top-K results saved to: {detailed_csv_path}")
        
        # Create performance plots - EXACTLY like original
        self.visualizer.create_performance_plots(results_df)
        
        # Generate HTML report - EXACTLY like original
        self.visualizer.generate_comprehensive_html_report(results_df, detailed_df)
        
        # Print summary
        print(f"\n📊 Final Summary:")
        print(f"  📈 Best Top-1: {results_df['test_top_1_accuracy'].max():.4f}")
        print(f"  📈 Best Top-5: {results_df['test_top_5_accuracy'].max():.4f}")
        print(f"  🗂️ Files generated: {len(list(self.output_dir.iterdir()))}")
        print(f"  📊 Models trained: {len(results_df)}")
        print(f"📁 All outputs saved to: {self.output_dir}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Clean ML experiments for keystroke biometrics')
    parser.add_argument('-d', '--dataset', required=True, help='Path to dataset CSV')
    parser.add_argument('-e', '--early-stop', action='store_true', help='Use early stopping')
    parser.add_argument('-s', '--seeds', type=int, default=1, help='Number of random seeds')
    parser.add_argument('-o', '--output-affix', default='', help='Output directory suffix')
    parser.add_argument('--show-class-dist', action='store_true', help='Show class distribution plots')
    parser.add_argument('--no-feature-importance', action='store_true', help='Skip feature importance plots')
    parser.add_argument('--max-workers', type=int, default=16, help='Max CPU workers (default: 16)')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration')
    
    args = parser.parse_args()
    
    # Create configuration with only the parameters that ExperimentConfig accepts
    config = ExperimentConfig(
        dataset_path=args.dataset,
        early_stopping=args.early_stop,
        num_seeds=args.seeds,
        output_affix=args.output_affix,
        show_class_distributions=args.show_class_dist,
        draw_feature_importance=not args.no_feature_importance
    )
    
    # Run experiments
    runner = MLExperimentRunner(config, max_workers=args.max_workers, use_gpu=not args.no_gpu)
    df = runner.load_data()
    
    print(f"📋 Dataset info:")
    print(f"  Shape: {df.shape}")
    print(f"  Platforms: {sorted(df['platform_id'].unique().to_list())}")
    print(f"  Users: {df['user_id'].n_unique()}")
    
    # Run experiments
    runner.run_experiments(df)
    
    # Generate comprehensive report - exactly like original
    runner.generate_comprehensive_report()
    
    print("\n🎊 Pipeline completed successfully!")
    print(f"🌐 Open the HTML report to view results: {runner.output_dir}/user_identification_report_{runner.timestamp}.html")


if __name__ == "__main__":
    main()