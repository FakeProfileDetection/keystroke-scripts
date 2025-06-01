#!/usr/bin/env python3
"""
test_fixed_ml_platforms.py - Test script to verify the fixes work correctly
"""

import numpy as np
import pandas as pd
from pathlib import Path
from ml_platforms_core import ExperimentConfig, ModelTrainer
import warnings
warnings.filterwarnings('ignore')

def create_test_data():
    """Create a small test dataset."""
    n_samples = 300
    n_features = 10
    n_users = 10
    n_platforms = 3
    
    # Create synthetic data
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    
    # Assign users and platforms
    users = np.repeat(range(n_users), n_samples // n_users)
    platforms = np.random.choice(range(1, n_platforms + 1), size=n_samples)
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    df['user_id'] = users
    df['platform_id'] = platforms
    
    return df

def test_model_training():
    """Test that all models can be trained without errors."""
    print("Creating test data...")
    df = create_test_data()
    
    # Setup config
    config = ExperimentConfig(
        dataset_path="test_data.csv",
        early_stopping=True,
        num_seeds=1,
        debug_mode=True  # Use debug mode for faster testing
    )
    
    # Create output directory
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize trainer
    trainer = ModelTrainer(config, output_dir, "test_timestamp")
    
    # Prepare data
    train_mask = df['platform_id'] != 3
    test_mask = df['platform_id'] == 3
    
    feature_cols = [col for col in df.columns if col not in ['user_id', 'platform_id']]
    
    X_train = df.loc[train_mask, feature_cols].values
    X_test = df.loc[test_mask, feature_cols].values
    y_train = trainer.label_encoder.fit_transform(df.loc[train_mask, 'user_id'])
    y_test = trainer.label_encoder.transform(df.loc[test_mask, 'user_id'])
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    # Test each model
    models_to_test = [
        ("RandomForest", trainer.train_random_forest),
        ("XGBoost", trainer.train_xgboost),
        ("CatBoost", trainer.train_catboost),
        ("SVM", trainer.train_svm),
    ]
    
    results = []
    for model_name, train_func in models_to_test:
        print(f"\nTesting {model_name}...")
        try:
            result = train_func(X_train, X_test, y_train, y_test, "test_experiment", 42)
            print(f"✅ {model_name} training successful!")
            print(f"   Top-1 Accuracy: {result.test_metrics.get('test_top_1_accuracy', 0):.4f}")
            print(f"   Top-5 Accuracy: {result.test_metrics.get('test_top_5_accuracy', 0):.4f}")
            results.append((model_name, "SUCCESS"))
        except Exception as e:
            print(f"❌ {model_name} training failed: {e}")
            results.append((model_name, f"FAILED: {str(e)}"))
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY:")
    for model_name, status in results:
        print(f"  {model_name}: {status}")
    
    # Cleanup
    import shutil
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    return all(status == "SUCCESS" for _, status in results)

if __name__ == "__main__":
    print("Testing fixed ML platforms implementation...")
    success = test_model_training()
    
    if success:
        print("\n🎉 All tests passed! The fixes are working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")

