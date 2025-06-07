#!/usr/bin/env python3
"""
ml_runner.py - Main orchestrator for ML experiments with configuration support
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import polars as pl
import pandas as pd
import matplotlib.pyplot as plt

from ml_core import (
    ExperimentConfig, ExperimentResult, ModelTrainer
)
from ml_visualizer import Visualizer
from ml_utils import (
    load_config, merge_configs, get_experiment_filters,
    validate_dataset, get_feature_columns
)


class MLExperimentRunner:
    """Main orchestrator for ML experiments."""
    
    def __init__(self, config: ExperimentConfig, max_workers: int = None):
        self.config = config
        self.max_workers = max_workers or __import__('os').cpu_count()
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.output_dir = self._create_output_dir()
        self.results: List[ExperimentResult] = []
        self.detailed_results: List[dict] = []
        
        # Initialize components
        self.trainer = ModelTrainer(config, self.output_dir, self.timestamp)
        self.visualizer = Visualizer(config, self.output_dir, self.timestamp)
        
        # Get experiments from config
        self.experiments = config.experiments
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
        print(f"  - Number of experiments: {len(self.experiments)}")
        print(f"  - Random seeds: {config.random_seeds}")
        print(f"  - Dataset path: {config.dataset_path}")
        
         # Define file paths once
        self.detailed_backup_path = self.output_dir / f"detailed_topk_results_{self.timestamp}.csv"
        self.summary_backup_path = self.output_dir / f"experiment_results_{self.timestamp}.csv"
        
        # Initialize intermediate save files with headers
        self._init_intermediate_files()
        
    def _init_intermediate_files(self):
        """Initialize CSV files with headers for intermediate saves."""
        # Detailed results header
        detailed_columns = ['model', 'experiment', 'random_seed', 'early_stopping', 
                        'k_value', 'train_top_k_accuracy', 'test_top_k_accuracy', 
                        'model_path', 'hyperparameters', 'timestamp']
        pd.DataFrame(columns=detailed_columns).to_csv(self.detailed_backup_path, index=False)
        
        # Summary results header - just create empty file, will be overwritten with full data
        with open(self.summary_backup_path, 'w') as f:
            f.write("")
    
    def _create_output_dir(self) -> Path:
        """Create output directory with timestamp."""
        suffix = "_early_stop" if self.config.early_stopping else ""
        debug_suffix = "_debug" if self.config.debug_mode else ""
        affix = f"_{self.config.output_affix}" if self.config.output_affix else ""
        dir_name = f"experiment_results{affix}_{self.timestamp}{suffix}{debug_suffix}"
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
        required_cols = {"user_id"}
        validate_dataset(df, list(required_cols))
        
        return df
    
    def run_experiments(self, df: pl.DataFrame):
        """Run all experiments with comprehensive tracking."""
        print(f"\n🚀 Running {len(self.experiments)} experiments with {len(self.config.random_seeds)} seeds each")
        print(f"Total model runs: {len(self.experiments) * len(self.models_to_train) * len(self.config.random_seeds)}")
        print(f"🖥️  Using {self.max_workers} CPU workers")
        print(f"🎮 GPU acceleration: {'Enabled' if self.config.use_gpu else 'Disabled'}")
        
        df_pd = df.to_pandas()  # Convert once for sklearn compatibility
        
        for exp_idx, experiment in enumerate(self.experiments, 1):
            print(f"\n{'='*60}")
            print(f"🎯 Experiment {exp_idx}/{len(self.experiments)}: {experiment['name']}")
            
            # Get train/test filters based on experiment type
            train_values, test_value, filter_column = get_experiment_filters(
                experiment, df_pd.columns.tolist()
            )
            
            print(f"Training on {filter_column}: {train_values}, Testing on: {test_value}")
            
                        
            # Create train/test splits
            if filter_column == "session_id":
                train_platforms = experiment.get("platform", "All")
                if str(train_platforms).lower() == "all":
                    print("Using  all platforms")
                    train_mask = df_pd[filter_column].isin(train_values)
                    test_mask = df_pd[filter_column] == test_value
                else:
                    if not isinstance(train_platforms, list):
                        train_platforms = [train_platforms]
                    print(f"Using platforms: {train_platforms}")
                    train_mask = df_pd[filter_column].isin(train_values) & df_pd['platform_id'].isin(train_platforms)
                    test_mask = df_pd[filter_column] == test_value
            else:
                train_mask = df_pd[filter_column].isin(train_values)
                test_mask = df_pd[filter_column] == test_value
                
            print(f"{'='*60}")
            
            feature_cols = get_feature_columns(df_pd.columns.tolist())
            X_train = df_pd.loc[train_mask, feature_cols].values
            X_test = df_pd.loc[test_mask, feature_cols].values
            y_train = df_pd.loc[train_mask, "user_id"].values
            y_test = df_pd.loc[test_mask, "user_id"].values
            
            print(f"📊 Train samples: {len(X_train)}, Test samples: {len(X_test)}")
            print(f"📊 Train features: {X_train.shape[1]}")
            
            # Check class distribution
            train_class_counts = pd.Series(y_train).value_counts()
            test_class_counts = pd.Series(y_test).value_counts()
            
            print(f"📊 Train classes: {len(train_class_counts)}, Test classes: {len(test_class_counts)}")
            print(f"📊 Min samples per class (train): {train_class_counts.min()}")
            print(f"📊 Min samples per class (test): {test_class_counts.min()}")
            
            if train_class_counts.min() < 2:
                print("⚠️ Very low sample count - some models may not work properly")
            
            # Show class distribution if requested
            if self.config.show_class_distributions:
                self._plot_class_distribution(train_class_counts, test_class_counts, experiment['name'])
            
            # Encode labels for sklearn
            y_train_encoded = self.trainer.label_encoder.fit_transform(y_train)
            y_test_encoded = self.trainer.label_encoder.transform(y_test)
            
            # Run models with different seeds
            for seed in self.config.random_seeds:
                print(f"\n🎲 Running with random seed {seed}")
                
                # Train selected models
                for model_name in self.models_to_train:
                    if model_name not in self.model_train_funcs:
                        print(f"❌ Unknown model: {model_name}")
                        continue
                    
                    print(f"\n\n🤖 Model: {model_name}")
                    try:
                        train_func = self.model_train_funcs[model_name]
                        result = train_func(
                            X_train, X_test, 
                            y_train_encoded, y_test_encoded,
                            experiment['name'], seed
                        )
                        self.results.append(result)
                        
                        # Store detailed results for top-k analysis
                        self._store_detailed_results(result)
                        
                        print(f"✅ {model_name}: Top-1 = {result.test_metrics.get('test_top_1_accuracy', 0):.4f}, "
                              f"Top-5 = {result.test_metrics.get('test_top_5_accuracy', 0):.4f}")
                    except Exception as e:
                        print(f"❌ {model_name} failed: {e}")
                        import traceback
                        traceback.print_exc()
        
        print(f"\n🎉 All experiments completed! Results saved to: {self.output_dir}")
    
    def _plot_class_distribution(self, train_counts: pd.Series, test_counts: pd.Series, exp_name: str):
        """Plot class distribution for train and test sets."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        train_counts.head(20).plot(kind="bar", ax=ax1, title=f"Train Classes - {exp_name}")
        ax1.set_xlabel("User ID")
        ax1.set_ylabel("Sample Count")
        
        test_counts.head(20).plot(kind="bar", ax=ax2, title=f"Test Classes - {exp_name}")
        ax2.set_xlabel("User ID")
        ax2.set_ylabel("Sample Count")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"class_distribution_{exp_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _store_detailed_results(self, result: ExperimentResult):
        """Store detailed results for top-k analysis."""
        new_records = []
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
            
        # Append new records to file
        pd.DataFrame(new_records).to_csv(self.detailed_backup_path, mode='a', header=False, index=False)
        
        # Save complete summary (overwrite)
        self._save_summary_results()
        
    def _save_summary_results(self):
        """Save summary results by overwriting the file."""
        if self.results:
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
            
            pd.DataFrame(results_data).to_csv(self.summary_backup_path, index=False)
    
    def generate_comprehensive_report(self):
        """Generate comprehensive HTML report and all visualizations."""
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
        
        # Save final CSV files (these are the same paths as the backups)
        # The backup files already contain the data, so this is just ensuring
        # the final versions are complete
        results_df.to_csv(self.summary_backup_path, index=False)
        detailed_df.to_csv(self.detailed_backup_path, index=False)
        
        print(f"📊 Results saved to: {self.summary_backup_path}")
        print(f"📊 Detailed Top-K results saved to: {self.detailed_backup_path}")
        
        # Create performance plots
        self.visualizer.create_performance_plots(results_df)
        
        # Generate HTML report
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
    parser = argparse.ArgumentParser(description='ML experiments for keystroke biometrics')
    parser.add_argument('-c', '--config', default='config_full.json', 
                        help='Path to configuration file (default: config_full.json)')
    parser.add_argument('-d', '--dataset', help='Path to dataset CSV (overrides config)')
    parser.add_argument('-e', '--early-stop', action='store_true', 
                        help='Use early stopping (overrides config)')
    parser.add_argument('-s', '--seeds', type=int, nargs='+', 
                        help='Random seeds (overrides config)')
    parser.add_argument('-o', '--output-affix', help='Output directory suffix (overrides config)')
    parser.add_argument('--show-class-dist', action='store_true', 
                        help='Show class distribution plots (overrides config)')
    parser.add_argument('--no-feature-importance', action='store_true', 
                        help='Skip feature importance plots (overrides config)')
    parser.add_argument('--max-workers', type=int, help='Max CPU workers')
    parser.add_argument('--no-gpu', action='store_true', 
                        help='Disable GPU acceleration (overrides config)')
    parser.add_argument('--debug', action='store_true', 
                        help='Use debug configuration (loads config_debug.json)')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.debug:
        config_path = 'config_debug.json'
    else:
        config_path = args.config
    
    try:
        config_dict = load_config(config_path)
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {config_path}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        sys.exit(1)
    
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
    if args.show_class_dist:
        overrides['show_class_distributions'] = True
    if args.no_feature_importance:
        overrides['draw_feature_importance'] = False
    if args.no_gpu:
        overrides['use_gpu'] = False
    
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
    runner = MLExperimentRunner(config, max_workers=args.max_workers)
    
    try:
        df = runner.load_data()
        
        print(f"📋 Dataset info:")
        print(f"  Shape: {df.shape}")
        if 'platform_id' in df.columns:
            print(f"  Platforms: {sorted(df['platform_id'].unique().to_list())}")
        if 'session_id' in df.columns:
            print(f"  Sessions: {sorted(df['session_id'].unique().to_list())}")
        print(f"  Users: {df['user_id'].n_unique()}")
        
        # Save config to output directory
        config_file = runner.output_dir / f"config_{runner.timestamp}.json"
        with open(config_file, 'w') as f:
            import json
            json.dump(final_config, f, indent=4)
        
        # Run experiments
        runner.run_experiments(df)
        
        # Generate comprehensive report
        runner.generate_comprehensive_report()
        
        print("\n🎊 Pipeline completed successfully!")
        print(f"🌐 Open the HTML report to view results: "
              f"{runner.output_dir}/user_identification_report_{runner.timestamp}.html")
        
    except Exception as e:
        print(f"❌ Error running experiments: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()