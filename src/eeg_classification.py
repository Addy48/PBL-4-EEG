"""
EEG Stress Classification Pipeline (Demo)
-------------------------------------------
Demonstrates the classification methodology for EEG-based stress detection
using the SAM-40 dataset with dual validation (SD + SI).

This is a demonstration script. Full ensemble analysis and statistical
validation code are withheld as the associated paper is under peer review.

Dataset structure expected:
    BASE_PATH/
    ├── Relax/          (40 CSV files, one per subject)
    ├── Arithmetic/     (40 CSV files)
    ├── Mirror_image/   (40 CSV files)
    └── Stroop/         (40 CSV files)

Each CSV: rows = samples, columns = extracted EEG features (14 channels x 10 features).

Usage:
    python eeg_classification.py --data_path /path/to/dataset --results_path /path/to/results


"""

import os
import glob
import time
import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import xgboost as xgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")


# =============================================================================
#  CONFIGURATION
# =============================================================================

COMPARISONS = [
    ("Relax", "Arithmetic", "Relax-Arithmetic"),
    ("Relax", "Mirror", "Relax-Mirror"),
    ("Relax", "Stroop", "Relax-Stroop"),
]

MODELS = {
    "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "RandomForest": RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
    ),
    "CatBoost": CatBoostClassifier(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        random_state=42,
        verbose=False,
    ),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    "LinearSVC": LinearSVC(max_iter=2000, random_state=42),
    "LogisticRegression": LogisticRegression(
        max_iter=2000, random_state=42, n_jobs=-1
    ),
}


# =============================================================================
#  DATA LOADING
# =============================================================================


def load_data_with_groups(folder_path, label):
    """
    Load all CSV files from a task folder.
    Each CSV file = one subject. This mapping is critical for GroupKFold
    to ensure no subject appears in both train and test sets.

    Parameters
    ----------
    folder_path : str
        Path to folder containing per-subject CSV files.
    label : int
        Class label (0 = Relax, 1 = Task).

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    groups : np.ndarray, shape (n_samples,)
        Subject identifiers for group-based splitting.
    """
    all_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {folder_path}")

    df_list = []
    groups_list = []

    for group_id, file_path in enumerate(all_files):
        df = pd.read_csv(file_path)
        df_list.append(df)
        groups_list.append(np.full(len(df), f"label{label}_sub{group_id}"))

    data = pd.concat(df_list, axis=0, ignore_index=True)
    X = data.values
    y = np.full(len(X), label)
    groups = np.hstack(groups_list)

    return X, y, groups


# =============================================================================
#  VALIDATION METHODS
# =============================================================================


def evaluate_subject_independent(X, y, groups, model_name, model, n_splits=5):
    """
    Subject-Independent validation using GroupKFold.
    Ensures complete subject separation between train and test.
    StandardScaler is fit on training fold only (no data leakage).
    """
    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": [], "auc": []}
    all_y_true, all_y_pred = [], []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale: fit on train only
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Clone and train
        model_clone = type(model)(**model.get_params())
        model_clone.fit(X_train_scaled, y_train)

        y_pred = model_clone.predict(X_test_scaled)

        # AUC computation
        if hasattr(model_clone, "predict_proba"):
            y_proba = model_clone.predict_proba(X_test_scaled)[:, 1]
        else:
            y_proba = model_clone.decision_function(X_test_scaled)

        fold_metrics["accuracy"].append(accuracy_score(y_test, y_pred))
        fold_metrics["precision"].append(
            precision_score(y_test, y_pred, zero_division=0)
        )
        fold_metrics["recall"].append(recall_score(y_test, y_pred, zero_division=0))
        fold_metrics["f1"].append(f1_score(y_test, y_pred, zero_division=0))
        fold_metrics["auc"].append(roc_auc_score(y_test, y_proba))

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        print(
            f"      Fold {fold_idx}/{n_splits}: "
            f"Acc={fold_metrics['accuracy'][-1]:.4f}, "
            f"AUC={fold_metrics['auc'][-1]:.4f}"
        )

    avg = {k: np.mean(v) for k, v in fold_metrics.items()}
    std = {k: np.std(v) for k, v in fold_metrics.items()}
    cm = confusion_matrix(all_y_true, all_y_pred)

    return avg, std, cm


def evaluate_subject_dependent(X, y, model_name, model, n_splits=5):
    """
    Subject-Dependent validation using StratifiedKFold.
    Samples from same subject may appear in both train and test.
    This is the idealized (optimistic) evaluation scenario.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": [], "auc": []}
    all_y_true, all_y_pred = [], []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model_clone = type(model)(**model.get_params())
        model_clone.fit(X_train_scaled, y_train)

        y_pred = model_clone.predict(X_test_scaled)

        if hasattr(model_clone, "predict_proba"):
            y_proba = model_clone.predict_proba(X_test_scaled)[:, 1]
        else:
            y_proba = model_clone.decision_function(X_test_scaled)

        fold_metrics["accuracy"].append(accuracy_score(y_test, y_pred))
        fold_metrics["precision"].append(
            precision_score(y_test, y_pred, zero_division=0)
        )
        fold_metrics["recall"].append(recall_score(y_test, y_pred, zero_division=0))
        fold_metrics["f1"].append(f1_score(y_test, y_pred, zero_division=0))
        fold_metrics["auc"].append(roc_auc_score(y_test, y_proba))

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        print(
            f"      Fold {fold_idx}/{n_splits}: "
            f"Acc={fold_metrics['accuracy'][-1]:.4f}, "
            f"AUC={fold_metrics['auc'][-1]:.4f}"
        )

    avg = {k: np.mean(v) for k, v in fold_metrics.items()}
    std = {k: np.std(v) for k, v in fold_metrics.items()}
    cm = confusion_matrix(all_y_true, all_y_pred)

    return avg, std, cm


# =============================================================================
#  RESULTS SAVING
# =============================================================================


def save_results(save_dir, model_name, task_name, val_type, avg, std, cm):
    """Save metrics CSV and confusion matrix plot."""
    os.makedirs(save_dir, exist_ok=True)

    # Metrics CSV
    pd.DataFrame(
        [
            {
                "Model": model_name,
                "Task": task_name,
                "Validation": val_type,
                "Accuracy": f"{avg['accuracy']:.4f}",
                "Precision": f"{avg['precision']:.4f}",
                "Recall": f"{avg['recall']:.4f}",
                "F1": f"{avg['f1']:.4f}",
                "AUC": f"{avg['auc']:.4f}",
                "Accuracy_Std": f"{std['accuracy']:.4f}",
            }
        ]
    ).to_csv(os.path.join(save_dir, "metrics.csv"), index=False)

    # Confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Relax", "Task"],
        yticklabels=["Relax", "Task"],
    )
    plt.title(f"{model_name} | {task_name} | {val_type}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
#  MAIN PIPELINE
# =============================================================================


def run_pipeline(data_path, results_path):
    """Execute the full classification pipeline."""
    datasets = {
        "Relax": os.path.join(data_path, "Relax"),
        "Arithmetic": os.path.join(data_path, "Arithmetic"),
        "Mirror": os.path.join(data_path, "Mirror_image"),
        "Stroop": os.path.join(data_path, "Stroop"),
    }

    # Verify data exists
    for name, path in datasets.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing folder: {path}")
    print("All data folders verified.\n")

    all_results = []

    for task_idx, (d1_name, d2_name, task_name) in enumerate(COMPARISONS, 1):
        print(f"\n{'='*60}")
        print(f"TASK {task_idx}/3: {task_name}")
        print(f"{'='*60}")

        # Load data
        X1, y1, groups1 = load_data_with_groups(datasets[d1_name], label=0)
        X2, y2, groups2 = load_data_with_groups(datasets[d2_name], label=1)
        X = np.vstack((X1, X2))
        y = np.hstack((y1, y2))
        groups = np.hstack((groups1, groups2))

        print(
            f"  Loaded: {len(X):,} samples, {X.shape[1]} features, "
            f"{len(np.unique(groups))} subjects"
        )

        for model_name, model in MODELS.items():
            print(f"\n  [{model_name}]")
            start = time.time()

            # --- Subject-Independent (GroupKFold) ---
            print(f"    Subject-Independent (GroupKFold, 5-fold):")
            si_avg, si_std, si_cm = evaluate_subject_independent(
                X, y, groups, model_name, model
            )
            save_dir_si = os.path.join(results_path, "SI", model_name, task_name)
            save_results(save_dir_si, model_name, task_name, "SI", si_avg, si_std, si_cm)

            # --- Subject-Dependent (StratifiedKFold) ---
            print(f"    Subject-Dependent (StratifiedKFold, 5-fold):")
            sd_avg, sd_std, sd_cm = evaluate_subject_dependent(
                X, y, model_name, model
            )
            save_dir_sd = os.path.join(results_path, "SD", model_name, task_name)
            save_results(save_dir_sd, model_name, task_name, "SD", sd_avg, sd_std, sd_cm)

            elapsed = time.time() - start
            print(
                f"    Result: SD={sd_avg['accuracy']:.4f}, "
                f"SI={si_avg['accuracy']:.4f}, "
                f"Gap={sd_avg['accuracy'] - si_avg['accuracy']:.4f} "
                f"({elapsed:.1f}s)"
            )

            all_results.append(
                {
                    "Model": model_name,
                    "Task": task_name,
                    "SD_Accuracy": round(sd_avg["accuracy"] * 100, 2),
                    "SI_Accuracy": round(si_avg["accuracy"] * 100, 2),
                    "Gap_pp": round(
                        (sd_avg["accuracy"] - si_avg["accuracy"]) * 100, 2
                    ),
                }
            )

    # Save summary
    summary_df = pd.DataFrame(all_results)
    summary_path = os.path.join(results_path, "summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"\nSummary:\n{summary_df.to_string(index=False)}")
    print(f"\nResults saved to: {results_path}")

    return summary_df


# =============================================================================
#  ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EEG Stress Classification Pipeline"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to SAM-40 dataset root (containing Relax/, Arithmetic/, etc.)",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="./results",
        help="Path to save results",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("EEG STRESS CLASSIFICATION PIPELINE")
    print("=" * 60)
    print(f"Data:    {args.data_path}")
    print(f"Results: {args.results_path}")
    print(f"Models:  {', '.join(MODELS.keys())}")
    print(f"Tasks:   {len(COMPARISONS)}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    run_pipeline(args.data_path, args.results_path)
