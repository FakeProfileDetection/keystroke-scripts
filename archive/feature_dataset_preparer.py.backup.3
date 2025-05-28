import ast
import os
import json
import re
from ml_models import (
    ScalarType,
    run_random_forest_model,
    run_xgboost_model,
    run_catboost_model,
    run_svm_model,
    finalize_experiments,
    set_total_experiments,
    OUTPUT_DIR,
)
from feature_table import (
    CKP_SOURCE,
    columns_to_remove,
    create_full_user_and_platform_table,
    table_to_cleaned_df,
)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


# Deserialization of columns
def deserialize_column(df, column_name):
    def clean_and_eval(x):
        if isinstance(x, str):
            # turn "np.float64(123.0)" → "123.0"
            # TODO: we need to check if this negatively affects the alpha_words bigrams
            no_np = re.sub(r"np\.float64\(([^)]+)\)", r"\1", x)
            return ast.literal_eval(no_np)
        else:
            return x

    df[column_name] = df[column_name].apply(clean_and_eval)
    return df


# Flattening the columns with lists into separate columns
def flatten_column(df, column_name):
    new_cols = pd.DataFrame(df[column_name].tolist(), index=df.index)
    new_cols.columns = [
        f"{column_name}_mean",
        f"{column_name}_median",
        f"{column_name}_q1",
        f"{column_name}_q3",
        f"{column_name}_std",
    ]

    df = pd.concat([df.drop(columns=[column_name]), new_cols], axis=1)
    return df


def remove_leaky_features(df, threshold=0.1):
    """Remove features that are too predictive of platform_id"""
    print(f"\n🔍 Analyzing platform leakage with threshold {threshold}...")
    
    feature_cols = df.drop(columns=["user_id", "platform_id", "session_id"], errors="ignore").columns
    X = df[feature_cols]
    y = df["platform_id"]
    
    # Handle missing values
    X_clean = X.fillna(X.median())
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Sample for speed if dataset is large
    if len(df) > 1000:
        sample_size = min(1000, len(df))
        sample_idx = np.random.choice(len(df), sample_size, replace=False)
        X_sample = X_clean.iloc[sample_idx]
        y_sample = y_encoded[sample_idx]
    else:
        X_sample = X_clean
        y_sample = y_encoded
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_sample, y_sample)
    
    # Identify leaky features
    importances = clf.feature_importances_
    leaky_features = X.columns[importances > threshold]
    
    print(f"🗑️ Found {len(leaky_features)} potentially leaky features (importance > {threshold})")
    if len(leaky_features) > 0:
        print("Top leaky features:")
        leaky_importance = [(feat, imp) for feat, imp in zip(X.columns, importances) if imp > threshold]
        leaky_importance.sort(key=lambda x: x[1], reverse=True)
        for feat, imp in leaky_importance[:10]:
            print(f"  {feat}: {imp:.4f}")
    
    # Remove leaky features
    df_clean = df.drop(columns=leaky_features, errors="ignore")
    
    print(f"📊 Dataset shape before: {df.shape}, after: {df_clean.shape}")
    return df_clean


def analyze_platform_leakage(df: pd.DataFrame):
    """Enhanced platform leakage analysis with better visualization and fixed warnings"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split

    print("\n🔍 Running comprehensive platform leakage diagnostic...")

    # Drop ID columns to isolate features
    feature_cols = df.drop(columns=["user_id", "platform_id", "session_id"], errors="ignore").columns
    X = df[feature_cols]
    y = df["platform_id"]

    # Handle missing values
    X_clean = X.fillna(X.median())

    # Encode platform ID
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Sample for speed if needed
    platform_counts = df["platform_id"].value_counts()
    min_platform_size = platform_counts.min()
    if min_platform_size > 300:
        df_sampled = df.groupby("platform_id").sample(n=300, random_state=42)
        X_clean = df_sampled[feature_cols].fillna(df_sampled[feature_cols].median())
        y_encoded = label_encoder.fit_transform(df_sampled["platform_id"])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_encoded, stratify=y_encoded, test_size=0.3, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)
    print(f"Platform Prediction Accuracy: {acc:.4f}")
    
    if acc > 0.8:
        print("HIGH LEAKAGE RISK - Platform can be predicted with high accuracy!")
    elif acc > 0.6:
        print("MODERATE LEAKAGE RISK - Some platform-specific patterns detected")
    else:
        print("LOW LEAKAGE RISK - Platform prediction is difficult")

    # Top features
    importances = clf.feature_importances_
    indices = importances.argsort()[-15:][::-1]
    top_features = [(X_clean.columns[i], importances[i]) for i in indices]
    
    print("\nTop 15 platform-leaking features:")
    for name, score in top_features:
        risk_level = "HIGH" if score > 0.1 else "MODERATE" if score > 0.05 else "LOW"
        print(f"  {risk_level:8} {name:35} -> {score:.4f}")

    # Enhanced visualization with fixed warnings
    plt.figure(figsize=(12, 8))
    
    # Create data for plotting
    feature_names = [name[:30] + '...' if len(name) > 30 else name for name, _ in top_features]
    feature_scores = [score for _, score in top_features]
    
    # Create colors based on risk level
    colors = ['red' if score > 0.1 else 'orange' if score > 0.05 else 'green' 
              for score in feature_scores]
    
    # Use matplotlib instead of seaborn to avoid palette warning
    bars = plt.barh(range(len(feature_names)), feature_scores, color=colors)
    plt.yticks(range(len(feature_names)), feature_names)
    
    # Add risk level text to title without emojis to avoid font warnings
    plt.title("Top Platform-Leaking Features\nRed: High Risk | Orange: Moderate Risk | Green: Low Risk")
    plt.xlabel("Feature Importance")
    plt.gca().invert_yaxis()  # Highest importance at top
    plt.tight_layout()
    
    # Save plot to output directory
    plt.savefig(OUTPUT_DIR / "platform_leakage_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    return acc, top_features


def setup_experiments():
    """Enhanced experiment setup with leakage detection"""
    print("🔧 Setting up experiments...")
    
    # This is the csv saving logic for the features, we shouldn't have to always run this unless we change this with the users
    # or the features
    # We will also need to rerun this if we ever change the source because currently the columns are fixed
    #################
    # source = CKP_SOURCE.FAKE_PROFILE_DATASET
    # rows = create_full_user_and_platform_table(source)
    # cleaned = table_to_cleaned_df(rows, source)
    # cleaned.to_csv("fp_features_data.csv", mode="w+")
    #################
    
    df = pd.read_csv(os.path.join(os.getcwd(), "fp_features_data.csv"))
    print(f"📊 Loaded dataset with shape: {df.shape}")
    
    df = df.dropna()
    df.drop(columns_to_remove(), inplace=True, axis=1, errors="ignore")
    
    print(f"📊 After cleaning: {df.shape}")
    
    # Columns to deserialize
    # NOTE: deserialization here is different than dropping the unnecessary columns before they are passed to the model.
    #       Here deserialization is to make sure the feature lists get reinterpreted from str to python lists
    #       But we are not removing them from the df here because we still need them to setup the experiments
    # Apply deserialization and flattening to each relevant column
    for col in list(df.columns):
        if col in ["user_id", "platform_id", "session_id"]:
            continue
            
        print(f"Processing column: {col}")
        df = deserialize_column(df, col)
        df = flatten_column(df, col)

    # Converting 'user_id' and 'platform_id' to numeric values
    df["user_id"] = df["user_id"].apply(
        lambda x: int(ast.literal_eval(x)[0])
        if isinstance(x, str)
        else int(x[0])
        if isinstance(x, list)
        else x
    )

    df["platform_id"] = df["platform_id"].apply(
        lambda x: int(ast.literal_eval(x)[0])
        if isinstance(x, str)
        else int(x[0])
        if isinstance(x, list)
        else x
    )
    
    print(f"📊 Final dataset shape: {df.shape}")
    df.to_csv("cleaned_features_data.csv", mode="w+")
    return df


def run_experiments(df: pd.DataFrame):
    """Enhanced experiment runner with comprehensive tracking"""
    print("\n🚀 Starting comprehensive User Identification ML experiments...")
    
    # Validate dataset
    print("🔍 Validating dataset...")
    print(f"Dataset shape: {df.shape}")
    print(f"Platforms: {sorted(df['platform_id'].unique())}")
    print(f"Users per platform:")
    platform_user_counts = df.groupby('platform_id')['user_id'].nunique()
    for platform, count in platform_user_counts.items():
        print(f"  Platform {platform}: {count} users")
    
    # Check for minimum viable samples
    min_users_per_platform = platform_user_counts.min()
    if min_users_per_platform < 5:
        print("⚠️ Warning: Very few users per platform - results may not be reliable")
    
    # Run platform leakage analysis
    leakage_acc, leaky_features = analyze_platform_leakage(df)
    
    # Optionally remove leaky features (uncomment if needed)
    # if leakage_acc > 0.7:
    #     print("🔧 Removing highly leaky features...")
    #     df = remove_leaky_features(df, threshold=0.1)

    experiments = [
        # Dual-platform training tests (original ones)
        ([1, 2], 3, "FI_vs_T"),
        ([1, 3], 2, "FT_vs_I"),
        ([2, 1], 3, "IF_vs_T"),
        ([2, 3], 1, "IT_vs_F"),
        ([3, 1], 2, "TF_vs_I"),
        ([3, 2], 1, "TI_vs_F"),
        # Single-platform training tests
        ([1], 2, "F_vs_I"),
        ([1], 3, "F_vs_T"),
        ([2], 1, "I_vs_F"),
        ([2], 3, "I_vs_T"),
        ([3], 1, "T_vs_F"),
        ([3], 2, "T_vs_I"),
    ]
    
    # Calculate total experiments: each experiment runs 4 models (RF, XGB, CatBoost, SVM)
    total_model_runs = len(experiments) * 4
    set_total_experiments(total_model_runs)
    print(f"📊 Total model runs planned: {total_model_runs}")
    
    with open(os.path.join(os.getcwd(), "classifier_config.json"), "r") as f:
        config = json.load(f)

    total_experiments = len(experiments)
    
    for exp_idx, (train_platforms, test_platform, experiment_name) in enumerate(experiments, 1):
        print(f"\n{'='*60}")
        print(f"🎯 Experiment {exp_idx}/{total_experiments}: {experiment_name}")
        print(f"Training on platforms: {train_platforms}, Testing on: {test_platform}")
        print(f"{'='*60}")

        # Create train/test splits
        train_mask = df["platform_id"].isin(train_platforms)
        test_mask = df["platform_id"] == test_platform
        
        X_train = df[train_mask].drop(
            columns=["platform_id", "user_id", "session_id"], errors="ignore"
        )
        y_train = df[train_mask]["user_id"]
        
        X_test = df[test_mask].drop(
            columns=["platform_id", "user_id", "session_id"], errors="ignore"
        )
        y_test = df[test_mask]["user_id"]
        
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

        # Plot class distribution if needed
        if config.get("show_class_distributions", False):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            train_class_counts.head(20).plot(kind="bar", ax=ax1, title=f"Train Classes - {experiment_name}")
            ax1.set_xlabel("User ID")
            ax1.set_ylabel("Sample Count")
            
            test_class_counts.head(20).plot(kind="bar", ax=ax2, title=f"Test Classes - {experiment_name}")
            ax2.set_xlabel("User ID")
            ax2.set_ylabel("Sample Count")
            
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"class_distribution_{experiment_name}.png", dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()

        # Run all models WITHOUT scaling (data is pre-normalized)
        # Commented out scaling options as requested - data is pre-normalized elsewhere
        # scalers_to_test = [ScalarType.STANDARD, ScalarType.MIN_MAX]
        scalers_to_test = [ScalarType.NONE]  # No additional scaling
        
        print(f"\n🤖 Running Random Forest models...")
        for scaler in scalers_to_test:
            print(f"  📊 Running without additional scaling (data pre-normalized)...")
            try:
                run_random_forest_model(
                    X_train, X_test, y_train, y_test, 
                    scalar_obj=scaler, experiment_name=f"{experiment_name}_no_scaling"
                )
            except Exception as e:
                print(f"❌ Random Forest failed: {e}")
        
        print(f"\n🚀 Running XGBoost models...")
        for scaler in scalers_to_test:
            print(f"  📊 Running without additional scaling (data pre-normalized)...")
            try:
                run_xgboost_model(
                    X_train, X_test, y_train, y_test, 
                    scalar_obj=scaler, experiment_name=f"{experiment_name}_no_scaling"
                )
            except Exception as e:
                print(f"❌ XGBoost failed: {e}")
        
        print(f"\n🐱 Running CatBoost models...")
        for scaler in scalers_to_test:
            print(f"  📊 Running without additional scaling (data pre-normalized)...")
            try:
                run_catboost_model(
                    X_train, X_test, y_train, y_test, 
                    scalar_obj=scaler, experiment_name=f"{experiment_name}_no_scaling"
                )
            except Exception as e:
                print(f"❌ CatBoost failed: {e}")
        
        # Run SVM (no scaling needed as data is pre-normalized)
        print(f"\n⚙️ Running SVM model...")
        try:
            run_svm_model(
                X_train, X_test, y_train, y_test, 
                experiment_name=f"{experiment_name}_SVM"
            )
        except Exception as e:
            print(f"❌ SVM failed: {e}")
        
        print(f"✅ Completed experiment {exp_idx}/{total_experiments}: {experiment_name}")
    
    print(f"\n🎉 All experiments completed!")
    
    # Generate final reports
    finalize_experiments()


def main():
    """Main execution function"""
    print("🚀 Starting User Identification ML Pipeline...")
    print("=" * 60)
    
    # Option 1: Setup experiments from scratch (uncomment if needed)
    # print("🔧 Setting up experiments from raw data...")
    # final_df = setup_experiments()
    # analyze_platform_leakage(final_df)
    
    # Option 2: Load pre-processed data (current approach)
    print("📂 Loading pre-processed dataset...")
    
    # Check if the direct dataset exists
    if os.path.exists("dataset_2_full_IL_filtred.csv"):
        print("✅ Using dataset_2_full_IL_filtred.csv")
        final_df = pd.read_csv("dataset_2_full_IL_filtred.csv")
    elif os.path.exists("cleaned_features_data.csv"):
        print("✅ Using cleaned_features_data.csv")
        final_df = pd.read_csv("cleaned_features_data.csv")
    else:
        print("❌ No suitable dataset found. Setting up from scratch...")
        final_df = setup_experiments()
    
    print(f"📊 Dataset loaded with shape: {final_df.shape}")
    
    # Run platform leakage analysis
    analyze_platform_leakage(final_df)
    
    # Run all experiments
    run_experiments(final_df)
    
    print("\n🎊 User Identification Pipeline completed successfully!")
    print("Check the experiment_results_* folder for all outputs.")


if __name__ == "__main__":
    main()