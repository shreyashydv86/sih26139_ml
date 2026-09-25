"""
SIH26139 -- Classical ML baselines.

Uses nested cross-validation with leakage-free preprocessing.

For every outer fold:
1. Split raw data into training and test sets.
2. Fit feature selection ONLY on training data.
3. Fit scaling ONLY on training data.
4. Tune hyperparameters using ONLY the training data.
5. Train the selected model on the training data.
6. Evaluate once on the untouched test data.
"""

import numpy as np

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    recall_score,
    accuracy_score,
)

from src.preprocessing import preprocess_fold


# ============================================================
# HYPERPARAMETER GRIDS
# ============================================================

PARAM_GRIDS = {

    "SVM": {
        "estimator": SVC(
            probability=True,
            random_state=42
        ),
        "params": {
            "C": [0.1, 1, 10],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"],
        },
    },

    "Random Forest": {
        "estimator": RandomForestClassifier(
            random_state=42
        ),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, None],
            "min_samples_split": [2, 5],
        },
    },

    "MLP": {
        "estimator": MLPClassifier(
            max_iter=2000,
            random_state=42
        ),
        "params": {
            "hidden_layer_sizes": [(32,), (64, 32)],
            "alpha": [0.0001, 0.001, 0.01],
            "learning_rate_init": [0.001, 0.01],
        },
    },
}


# ============================================================
# HYPERPARAMETER TUNING
# ============================================================

def tune_model(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    inner_cv: int = 3
):
    """
    Tune one model using ONLY the outer training data.
    """

    config = PARAM_GRIDS[name]

    inner_skf = StratifiedKFold(
        n_splits=inner_cv,
        shuffle=True,
        random_state=42
    )

    search = GridSearchCV(
        estimator=config["estimator"],
        param_grid=config["params"],
        cv=inner_skf,
        scoring="roc_auc",
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    return (
        search.best_estimator_,
        search.best_params_
    )


# ============================================================
# NESTED CROSS-VALIDATION
# ============================================================

def evaluate_with_nested_cv(
    name: str,
    X_raw: np.ndarray,
    y: np.ndarray,
    folds: list,
    k_features: int = 4,
    inner_cv: int = 3
):
    """
    Perform nested CV with leakage-free preprocessing.
    """

    metrics = {
        "auc": [],
        "f1": [],
        "sensitivity": [],
        "accuracy": [],
    }

    fold_params = []

    for fold_number, (train_idx, test_idx) in enumerate(
        folds,
        start=1
    ):

        print(f"\n    Fold {fold_number}")

        # ----------------------------------------------------
        # Raw outer train/test split
        # ----------------------------------------------------

        X_train_raw = X_raw[train_idx]
        X_test_raw = X_raw[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        # ----------------------------------------------------
        # Leakage-free preprocessing
        # ----------------------------------------------------

        (
            X_train,
            X_test,
            selector,
            scaler
        ) = preprocess_fold(
            X_train_raw,
            X_test_raw,
            y_train,
            k_features=k_features
        )

        # ----------------------------------------------------
        # Inner CV hyperparameter tuning
        # ----------------------------------------------------

        best_estimator, best_params = tune_model(
            name,
            X_train,
            y_train,
            inner_cv=inner_cv
        )

        # ----------------------------------------------------
        # Train on outer training fold
        # ----------------------------------------------------

        best_estimator.fit(
            X_train,
            y_train
        )

        # ----------------------------------------------------
        # Evaluate on untouched outer test fold
        # ----------------------------------------------------

        probabilities = best_estimator.predict_proba(
            X_test
        )[:, 1]

        predictions = best_estimator.predict(
            X_test
        )

        auc = roc_auc_score(
            y_test,
            probabilities
        )

        f1 = f1_score(
            y_test,
            predictions
        )

        sensitivity = recall_score(
            y_test,
            predictions
        )

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        metrics["auc"].append(float(auc))
        metrics["f1"].append(float(f1))
        metrics["sensitivity"].append(float(sensitivity))
        metrics["accuracy"].append(float(accuracy))

        fold_params.append(best_params)

        print(
            f"      AUC: {auc:.3f}"
        )

        print(
            f"      F1: {f1:.3f}"
        )

        print(
            f"      Sensitivity: {sensitivity:.3f}"
        )

        print(
            f"      Accuracy: {accuracy:.3f}"
        )

        print(
            f"      Parameters: {best_params}"
        )

    # --------------------------------------------------------
    # Calculate mean and standard deviation
    # --------------------------------------------------------

    summary = {}

    for metric_name, values in metrics.items():

        summary[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "folds": values,
        }

    return summary, fold_params


# ============================================================
# RUN ALL CLASSICAL BASELINES
# ============================================================

def run_all_baselines(
    X_raw: np.ndarray,
    y: np.ndarray,
    folds: list,
    k_features: int = 4
):
    """
    Run SVM, Random Forest and MLP using nested CV.
    """

    results = {}

    for name in PARAM_GRIDS:

        print("\n" + "=" * 60)
        print(f"Running {name}")
        print("=" * 60)

        scores, fold_params = evaluate_with_nested_cv(
            name=name,
            X_raw=X_raw,
            y=y,
            folds=folds,
            k_features=k_features,
            inner_cv=3
        )

        results[name] = {
            "metrics": scores,
            "fold_params": fold_params,
        }

        print(
            f"\n{name} final result:"
        )

        print(
            f"AUC = "
            f"{scores['auc']['mean']:.3f} "
            f"+/- "
            f"{scores['auc']['std']:.3f}"
        )

    return results


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    from preprocessing import prepare_dataset

    data = prepare_dataset(
        dataset="breast_cancer",
        k_features=4,
        n_splits=5
    )

    results = run_all_baselines(
        X_raw=data["X_raw"],
        y=data["y"],
        folds=data["folds"],
        k_features=data["k_features"]
    )