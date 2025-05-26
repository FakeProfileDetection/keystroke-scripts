import os
import json
import enum
import pandas as pd
import numpy as np

from sklearn.svm import SVC
from extended_minmax import ExtendedMinMaxScalar
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt
import bob.measure


class ScalarType(enum.Enum):
    """Enum class representing the different types of verifiers available."""

    STANDARD = 1
    MIN_MAX = 2
    EXTENDED_MIN_MAX = 3


with open(os.path.join(os.getcwd(), "classifier_config.json"), "r") as f:
    config = json.load(f)


def analyze_top_k_distribution(y_test_encoded, y_pred_proba_xgb, label_encoder):
    # Convert encoded labels back to original for readability
    y_test_decoded = label_encoder.inverse_transform(y_test_encoded)

    # Create a DataFrame to store results
    analysis_df = pd.DataFrame(
        {"True Label": y_test_decoded, "True Label Encoded": y_test_encoded}
    )

    # Rank each probability for each sample and locate the rank of the true label
    true_label_ranks = []
    for i, true_label in enumerate(y_test_encoded):
        sorted_indices = np.argsort(
            -y_pred_proba_xgb[i]
        )  # Sort probabilities in descending order
        true_label_rank = (
            np.where(sorted_indices == true_label)[0][0] + 1
        )  # Get 1-based rank of true label
        true_label_ranks.append(true_label_rank)

    # Add ranks to the DataFrame
    analysis_df["True Label Rank"] = true_label_ranks

    # Display distribution of ranks to see where the true label typically falls
    # rank_distribution = analysis_df["True Label Rank"].value_counts().sort_index()
    # print("True Label Rank Distribution:\n", rank_distribution)

    # Display average and median rank for additional insight
    # avg_rank = analysis_df["True Label Rank"].mean()
    # median_rank = analysis_df["True Label Rank"].median()
    # print(f"\nAverage Rank of True Labels: {avg_rank:.2f}")
    # print(f"Median Rank of True Labels: {median_rank}")

    return analysis_df


def run_xgboost_model(
    X_train, X_test, y_train, y_test, scalar_obj: ScalarType, max_k=5
):
    # Scale the features
    if scalar_obj == ScalarType.MIN_MAX:
        scaler = MinMaxScaler()
    elif scalar_obj == ScalarType.STANDARD:
        scaler = StandardScaler()
    elif scalar_obj == ScalarType.EXTENDED_MIN_MAX:
        scaler = ExtendedMinMaxScalar()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Sanity check for NaNs or infinities
    if not np.isfinite(X_train_scaled).all():
        raise ValueError("❌ X_train_scaled contains NaNs or infinite values")
    if not np.isfinite(X_test_scaled).all():
        raise ValueError("❌ X_test_scaled contains NaNs or infinite values")

    # Encode the labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    min_samples_per_class = y_train.value_counts().min()
    print(f"Minimum samples per class in training set: {min_samples_per_class}")

    if min_samples_per_class < 2:
        print("⚠️ Not enough samples for StratifiedKFold → fitting XGBoost without CV.")
        best_xgb = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_lambda=1,
            n_jobs=-1,
        ).fit(X_train_scaled, y_train_encoded)
    else:
        stratified_kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.01, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "reg_lambda": [1, 3, 5],
        }

        grid_search_xgb = GridSearchCV(
            XGBClassifier(n_jobs=-1),
            param_grid,
            scoring="accuracy",
            cv=stratified_kfold,
            n_jobs=-1,
            verbose=1,
        )
        grid_search_xgb.fit(X_train_scaled, y_train_encoded)
        best_xgb = grid_search_xgb.best_estimator_

    y_pred_xgb = best_xgb.predict(X_test_scaled)
    y_pred_xgb_decoded = label_encoder.inverse_transform(y_pred_xgb)
    accuracy_xgb = accuracy_score(y_test, y_pred_xgb_decoded)
    print(f"XGBoost Accuracy: {accuracy_xgb}")

    y_pred_proba_xgb = best_xgb.predict_proba(X_test_scaled)

    # Suppress warnings from undefined precision
    classification_rep = classification_report(
        y_test, y_pred_xgb_decoded, zero_division=0
    )
    print(f"Classification Report:\n{classification_rep}")

    # Optional: check for classes that were never predicted
    missing_preds = set(y_test) - set(y_pred_xgb_decoded)
    if missing_preds:
        print(f"⚠️ Labels not predicted at all: {missing_preds}")

    for k in range(1, max_k + 1):
        top_k_acc_xgb = top_k_accuracy_score(y_test_encoded, y_pred_proba_xgb, k=k)
        print(f"XGBoost Top-{k} Accuracy: {top_k_acc_xgb}")

    rr_scores = []
    for i, true_label in enumerate(y_test_encoded):
        pos_score = y_pred_proba_xgb[i, true_label]
        neg_scores = np.delete(y_pred_proba_xgb[i], true_label)
        rr_scores.append((neg_scores, [pos_score]))

    recognition_rate = bob.measure.recognition_rate(rr_scores, rank=1)
    print(f"Recognition Rate at rank 1: {recognition_rate:.2f}")

    analyze_top_k_distribution(y_test_encoded, y_pred_proba_xgb, label_encoder)

    f1 = f1_score(y_test, y_pred_xgb_decoded, average="weighted", zero_division=0)
    precision = precision_score(
        y_test, y_pred_xgb_decoded, average="weighted", zero_division=0
    )
    recall = recall_score(
        y_test, y_pred_xgb_decoded, average="weighted", zero_division=0
    )

    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    if config["draw_feature_importance_graph"]:
        plt.figure(figsize=(10, 8))
        plot_importance(best_xgb, importance_type="weight", max_num_features=10)
        plt.title("XGBoost Feature Importance")
        plt.savefig("XGBoost Feature Importance.png")


def run_random_forest_model(
    X_train, X_test, y_train, y_test, scalar_obj: ScalarType, max_k=5
):
    # Scale the features
    if scalar_obj == ScalarType.MIN_MAX:
        scaler = MinMaxScaler()
    elif scalar_obj == ScalarType.STANDARD:
        scaler = StandardScaler()
    elif scalar_obj == ScalarType.EXTENDED_MIN_MAX:
        scaler = ExtendedMinMaxScalar()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Encode the labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    train_min_samples_per_class = y_train.value_counts().min()
    test_min_samples_per_class = y_test.value_counts().min()
    print(f"Minimum samples per class in training set: {train_min_samples_per_class}")
    print(f"Minimum samples per class in test set: {test_min_samples_per_class}")
    if train_min_samples_per_class < 2:
        print("⚠️ Not enough samples for StratifiedKFold → fitting without CV.")
        best_rf = RandomForestClassifier(random_state=42).fit(
            X_train_scaled, y_train_encoded
        )
    else:
        stratified_kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

        # Define the hyperparameter grid
        param_grid = {
            "n_estimators": [100, 200, 500],
            "max_depth": [10, 20, 30, None],
            "min_samples_split": [2, 10, 20],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
            "bootstrap": [True, False],
        }

        # Perform GridSearchCV for hyperparameter tuning
        grid_search_rf = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            scoring="accuracy",
            cv=stratified_kfold,
            n_jobs=-1,
            verbose=1,
        )
        grid_search_rf.fit(X_train_scaled, y_train_encoded)

        # Get the best estimator
        best_rf = grid_search_rf.best_estimator_

    # Predict using the test set
    y_pred_rf = best_rf.predict(X_test_scaled)

    # Decode the predictions back to original labels
    y_pred_rf_decoded = label_encoder.inverse_transform(y_pred_rf)

    # Compute accuracy using the decoded predictions
    accuracy_rf = accuracy_score(y_test, y_pred_rf_decoded)
    print(f"Random Forest Accuracy: {accuracy_rf}")

    # Get the probabilities for top-k accuracy
    y_pred_proba_rf = best_rf.predict_proba(X_test_scaled)

    # Compute classification report
    # classification_rep = classification_report(y_test, y_pred_rf_decoded)
    # print(f"Classification Report:\n{classification_rep}")

    # Top-k accuracy loop
    for k in range(1, max_k + 1):
        top_k_acc_rf = top_k_accuracy_score(y_test_encoded, y_pred_proba_rf, k=k)
        print(f"Random Forest Top-{k} Accuracy: {top_k_acc_rf}")

    # Prepare the structure for computing recognition rate using bob.measure
    rr_scores = []
    for i, true_label in enumerate(y_test_encoded):
        # Get positive (true class score) and negative (all other class scores)
        pos_score = y_pred_proba_rf[i, true_label]  # The score for the correct class
        neg_scores = np.delete(
            y_pred_proba_rf[i], true_label
        )  # The scores for all other classes

        # Append as (negative scores, positive score) tuple for each probe
        rr_scores.append((neg_scores, [pos_score]))

    # Calculate recognition rate at rank 1
    # recognition_rate = bob.measure.recognition_rate(rr_scores, rank=1)
    # print(f"Recognition Rate at rank 1: {recognition_rate:.2f}")

    f1 = f1_score(y_test, y_pred_rf_decoded, average="weighted", zero_division=0)
    precision = precision_score(y_test, y_pred_rf_decoded, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred_rf_decoded, average="weighted", zero_division=0)
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    # Plot feature importance
    if config["draw_feature_importance_graph"]:
        # Plot feature importance
        plt.figure(figsize=(10, 8))
        feature_importances = best_rf.feature_importances_

        # Assuming X_train is a DataFrame, use its columns as feature names
        feature_names = X_train.columns

        indices = np.argsort(feature_importances)[-10:]
        plt.barh(range(len(indices)), feature_importances[indices], align="center")
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel("Feature Importance")
        plt.title("Random Forest Feature Importance")
        plt.savefig("Random Forest Feature Importance.png")


def run_catboost_model(
    X_train, X_test, y_train, y_test, scalar_obj: ScalarType, max_k=5
):
    # Scale the features
    if scalar_obj == ScalarType.MIN_MAX:
        scaler = MinMaxScaler()
    elif scalar_obj == ScalarType.STANDARD:
        scaler = StandardScaler()
    elif scalar_obj == ScalarType.EXTENDED_MIN_MAX:
        scaler = ExtendedMinMaxScalar()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # ✅ Calculate min samples per class
    min_samples_per_class = y_train.value_counts().min()
    print(f"Minimum samples per class in training set: {min_samples_per_class}")

    if min_samples_per_class < 2:
        print("⚠️ Not enough samples for StratifiedKFold → fitting CatBoost without CV.")

        # Use default parameters or predefined params when CV can't be used
        best_catboost_model = CatBoostClassifier(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            l2_leaf_reg=3,
            border_count=64,
            random_seed=42,
            verbose=0,
        )
        best_catboost_model.fit(X_train_scaled, y_train_encoded)

    else:
        stratified_kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

        param_grid = {
            "iterations": [100, 200, 300],
            "depth": [4, 6, 8],
            "learning_rate": [0.01, 0.05, 0.1],
            "l2_leaf_reg": [1, 3, 5],
            "border_count": [32, 64, 128],
        }

        grid_search = GridSearchCV(
            estimator=CatBoostClassifier(verbose=0, random_seed=42),
            param_grid=param_grid,
            scoring="accuracy",
            cv=stratified_kfold,
            verbose=1,
            n_jobs=-1,
        )

        grid_search.fit(X_train_scaled, y_train_encoded)

        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        print(f"Best Parameters: {best_params}")
        print(f"Best Cross-Validation Score: {best_score}")

        # Train final model with best params
        best_catboost_model = CatBoostClassifier(
            **best_params, random_seed=42, verbose=0
        )
        best_catboost_model.fit(X_train_scaled, y_train_encoded)

    # Predict and evaluate
    y_pred = best_catboost_model.predict(X_test_scaled)
    y_pred_proba = best_catboost_model.predict_proba(X_test_scaled)

    classification_rep = classification_report(y_test_encoded, y_pred, zero_division=0)
    print(f"Classification Report:\n{classification_rep}")

    # Top-k accuracy
    for k in range(1, max_k + 1):
        top_k_acc = top_k_accuracy_score(y_test_encoded, y_pred_proba, k=k)
        print(f"Catboost Top-{k} Accuracy: {top_k_acc}")

    # Prepare recognition rate
    rr_scores = []
    for i, true_label in enumerate(y_test_encoded):
        pos_score = y_pred_proba[i, true_label]
        neg_scores = np.delete(y_pred_proba[i], true_label)
        rr_scores.append((neg_scores, [pos_score]))

    # Recognition rate loop
    for k in range(1, max_k + 1):
        recognition_rate = bob.measure.recognition_rate(rr_scores, rank=k)
        print(f"Catboost Top-{k} Recognition Rate: {recognition_rate}")

    # Feature importance plot
    if config["draw_feature_importance_graph"]:
        feature_importances = best_catboost_model.get_feature_importance(
            Pool(X_train_scaled, label=y_train_encoded)
        )
        feature_names = X_train.columns
        plt.figure(figsize=(10, 8))
        sorted_indices = feature_importances.argsort()[-10:]
        plt.barh(
            range(len(sorted_indices)),
            feature_importances[sorted_indices],
            align="center",
        )
        plt.yticks(
            range(len(sorted_indices)), [feature_names[i] for i in sorted_indices]
        )
        plt.xlabel("Feature Importance")
        plt.title("CatBoost Feature Importance")
        plt.savefig("Catboost Feature Importance.png")


def run_svm_model(X_train, X_test, y_train, y_test, max_k=5):
    # Scale features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # ✅ Calculate min samples per class
    min_samples_per_class = y_train.value_counts().min()
    print(f"Minimum samples per class in training set: {min_samples_per_class}")

    if min_samples_per_class < 2:
        print("⚠️ Not enough samples for StratifiedKFold → fitting SVM without CV.")
        best_svm = SVC(
            C=1,
            kernel="rbf",
            gamma="scale",
            probability=True,
            random_state=42,
        ).fit(X_train_scaled, y_train_encoded)
    else:
        stratified_kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

        param_grid = {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"],
        }

        grid_search_svm = GridSearchCV(
            SVC(probability=True),
            param_grid,
            scoring="accuracy",
            cv=stratified_kfold,
            n_jobs=-1,
            verbose=1,
        )
        grid_search_svm.fit(X_train_scaled, y_train_encoded)
        best_svm = grid_search_svm.best_estimator_

    # Predict on test set
    y_pred_svm = best_svm.predict(X_test_scaled)
    y_pred_svm_decoded = label_encoder.inverse_transform(y_pred_svm)

    accuracy_svm = accuracy_score(y_test, y_pred_svm_decoded)
    print(f"SVM Accuracy: {accuracy_svm}")

    y_pred_proba_svm = best_svm.predict_proba(X_test_scaled)

    classification_rep = classification_report(y_test, y_pred_svm_decoded,zero_division=0)
    print(f"Classification Report:\n{classification_rep}")

    for k in range(1, max_k + 1):
        top_k_acc_svm = top_k_accuracy_score(y_test_encoded, y_pred_proba_svm, k=k)
        print(f"SVM Top-{k} Accuracy: {top_k_acc_svm}")

    rr_scores = []
    for i, true_label in enumerate(y_test_encoded):
        pos_score = y_pred_proba_svm[i, true_label]
        neg_scores = np.delete(y_pred_proba_svm[i], true_label)
        rr_scores.append((neg_scores, [pos_score]))

    recognition_rate = bob.measure.recognition_rate(rr_scores, rank=1)
    print(f"Recognition Rate at rank 1: {recognition_rate:.2f}")

    analyze_top_k_distribution(y_test_encoded, y_pred_proba_svm, label_encoder)

    f1 = f1_score(y_test, y_pred_svm_decoded, average="weighted", zero_division=0)
    precision = precision_score(y_test, y_pred_svm_decoded, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred_svm_decoded, average="weighted", zero_division=0)

    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    if config and config.get("draw_feature_importance_graph"):
        print(
            "⚠️ Note: SVM does not provide native feature importance like tree models."
        )
        if best_svm.kernel == "linear":
            coef = best_svm.coef_[0]
            sorted_idx = np.argsort(np.abs(coef))[::-1][:10]
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(sorted_idx)), coef[sorted_idx])
            plt.xticks(range(len(sorted_idx)), X_train.columns[sorted_idx], rotation=45)
            plt.title("SVM Linear Coefficients (Top Features)")
            plt.tight_layout()
            plt.savefig("SVM Feature Importance.png")
