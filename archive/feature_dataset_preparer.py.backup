import ast
import os
import json
import re
from ml_models import (
    ScalarType,
    run_random_forest_model,
    run_xgboost_model,
    run_catboost_model,
)
from feature_table import (
    CKP_SOURCE,
    columns_to_remove,
    create_full_user_and_platform_table,
    table_to_cleaned_df,
)
import pandas as pd
import matplotlib.pyplot as plt

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


def analyze_platform_leakage(df: pd.DataFrame):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split

    print("\n🔍 Running platform leakage diagnostic...")

    # Drop ID columns to isolate features
    feature_cols = df.drop(columns=["user_id", "platform_id"], errors="ignore").columns
    X = df[feature_cols]
    y = df["platform_id"]

    # Encode platform ID
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Sample for speed if needed
    platform_counts = df["platform_id"].value_counts()
    min_platform_size = platform_counts.min()
    if min_platform_size > 300:
        df = df.groupby("platform_id").sample(n=300, random_state=42)
        X = df[feature_cols]
        y_encoded = label_encoder.fit_transform(df["platform_id"])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, stratify=y_encoded, test_size=0.3, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)
    print(f"⚠️ Platform Prediction Accuracy: {acc:.4f}")

    # Top features
    importances = clf.feature_importances_
    indices = importances.argsort()[-10:][::-1]
    top_features = [(X.columns[i], importances[i]) for i in indices]
    print("🧬 Top 10 platform-leaking features:")
    for name, score in top_features:
        print(f"  {name:30} → {score:.4f}")

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=[score for _, score in top_features],
        y=[name for name, _ in top_features],
    )
    plt.title("Top Platform-Leaking Features")
    plt.xlabel("Feature Importance")
    plt.tight_layout()
    plt.show()


def setup_experiments():
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
    df = df.dropna()
    df.drop(columns_to_remove(), inplace=True, axis=1, errors="ignore")
    # print(df.columns)
    # input("Printing dataframe columns")
    # Columns to deserialize
    # NOTE: deserialization here is different than dropping the unnecessary columns before they are passed to the model.
    #       Here deserialization is to make sure the feature lists get reinterpreted from str to python lists
    #       But we are not removing them from the df here because we still need them to setup the experiments
    # Apply deserialization and flattening to each relevant column
    for col in list(df.columns):
        df = deserialize_column(df, col)
        if col == "user_id" or col == "platform_id" or col == "session_id":
            pass
        else:
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
    # print(list(df.columns))
    # input()
    df.to_csv("cleaned_features_data.csv", mode="w+")
    return df


def run_experiments(df: pd.DataFrame):
    experiments = [
        # Dual-platform training tests (original ones)
        ([1, 2], 3, "FI vs. T"),
        ([1, 3], 2, "FT vs. I"),
        ([2, 1], 3, "IF vs. T"),
        ([2, 3], 1, "IT vs. F"),
        ([3, 1], 2, "TF vs. I"),
        ([3, 2], 1, "TI vs. F"),
        # Single-platform training tests
        ([1], 2, "F vs. I"),
        ([1], 3, "F vs. T"),
        ([2], 1, "I vs. F"),
        ([2], 3, "I vs. T"),
        ([3], 1, "T vs. F"),
        ([3], 2, "T vs. I"),
    ]
    with open(os.path.join(os.getcwd(), "classifier_config.json"), "r") as f:
        config = json.load(f)

    for train_platforms, test_platform, experiment_name in experiments:
        print(experiment_name)

        X_train = df[df["platform_id"].isin(list(train_platforms))].drop(
            columns=["platform_id", "user_id", "session_id"], errors="raise", axis=1
        )
        print(X_train.columns)
        print(len(X_train))
        y_train = df[df["platform_id"].isin(list(train_platforms))]["user_id"]

        X_test = df[df["platform_id"] == test_platform].drop(
            columns=["platform_id", "user_id", "session_id"], errors="raise", axis=1
        )
        y_test = df[df["platform_id"] == test_platform]["user_id"]

        # print("Number of samples per class in training set:")
        # print(y_train.value_counts())
        # print("Number of samples per class in testing set:")
        # print(y_test.value_counts())
        # input()
        # Plot class distribution if needed
        if config["show_class_distributions"]:
            y_train.value_counts().plot(kind="bar", title=f"Train {experiment_name}")
            plt.show()
            y_test.value_counts().plot(kind="bar", title=f"Test {experiment_name}")
            plt.show()
        # TODO: ExtendedMinMax is having some issues with xgb?
        print(f"Random Forest: {experiment_name} results:")
        print("With StandardScaler")
        run_random_forest_model(
            X_train, X_test, y_train, y_test, scalar_obj=ScalarType.STANDARD
        )
        print("With MinMaxScaler")
        run_random_forest_model(
            X_train, X_test, y_train, y_test, scalar_obj=ScalarType.MIN_MAX
        )
        # run_random_forest_model(X_train, X_test, y_train, y_test, scalar_obj=ScalarType.EXTENDED_MIN_MAX)
        print(f"XGBoost: {experiment_name} results")
        print("With StandardScaler")
        run_xgboost_model(
            X_train, X_test, y_train, y_test, scalar_obj=ScalarType.STANDARD
        )
        print("With MinMaxScaler")
        run_xgboost_model(
            X_train, X_test, y_train, y_test, scalar_obj=ScalarType.MIN_MAX
        )
        # input(f"{experiment_name} results")

# NOTE: this function does not need to be run on the dataset_2_full_IL_filtred.csv because that already gives us the statistical features which 
#       can be fed directly to the models
# final_df = setup_experiments()
# analyze_platform_leakage(final_df)
final_df = pd.read_csv("dataset_2_full_IL_filtred.csv")
run_experiments(final_df)
