"""
SIH26139 -- Classical ML baselines.

Trains and cross-validates SVM, Random Forest, and MLP with proper
hyperparameter tuning (GridSearchCV) rather than default settings.
An under-tuned baseline makes the quantum model look artificially
better -- this module exists specifically to prevent that.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, recall_score, accuracy_score

PARAM_GRIDS = {
    "SVM": {
        "estimator": SVC(probability=True, random_state=42),
        "params": {
            "C": [0.1, 1, 10],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"],
        },
    },
    "Random Forest": {
        "estimator": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, None],
            "min_samples_split": [2, 5],
        },
    },
    "MLP": {
        "estimator": MLPClassifier(max_iter=2000, random_state=42),
        "params": {
            "hidden_layer_sizes": [(32,), (64, 32)],
            "alpha": [0.0001, 0.001, 0.01],
            "learning_rate_init": [0.001, 0.01],
        },
    },
}


def tune_model(name: str, X: np.ndarray, y: np.ndarray, inner_cv: int = 3):
    """
    Grid-search the best hyperparameters for one model using an inner
    cross-validation loop on the full dataset. Returns the best
    estimator, ready to be evaluated with the outer CV folds.
    """
    cfg = PARAM_GRIDS[name]
    inner_skf = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=42)
    search = GridSearchCV(cfg["estimator"], cfg["params"], cv=inner_skf,
                           scoring="roc_auc", n_jobs=-1)
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


def evaluate_with_cv(estimator, X: np.ndarray, y: np.ndarray, folds: list):
    """
    Evaluate a (tuned) estimator across the SAME outer folds used for
    the quantum model, so the comparison is apples-to-apples. Returns
    per-fold metrics plus mean and std, which is what should be
    reported -- never a single train/test split number.
    """
    metrics = {"auc": [], "f1": [], "sensitivity": [], "accuracy": []}

    for train_idx, test_idx in folds:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        estimator.fit(X_train, y_train)
        proba = estimator.predict_proba(X_test)[:, 1]
        preds = estimator.predict(X_test)

        metrics["auc"].append(roc_auc_score(y_test, proba))
        metrics["f1"].append(f1_score(y_test, preds))
        metrics["sensitivity"].append(recall_score(y_test, preds))
        metrics["accuracy"].append(accuracy_score(y_test, preds))

    summary = {
        k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "folds": v}
        for k, v in metrics.items()
    }
    return summary


def run_all_baselines(X: np.ndarray, y: np.ndarray, folds: list):
    """
    Tune and evaluate all three classical baselines. Returns a dict
    keyed by model name, each containing the fold-level metrics and
    the fitted best hyperparameters (for the technical report).
    """
    results = {}
    for name in PARAM_GRIDS:
        best_estimator, best_params = tune_model(name, X, y)
        scores = evaluate_with_cv(best_estimator, X, y, folds)
        results[name] = {"metrics": scores, "best_params": best_params}
        print(f"  {name}: AUC = {scores['auc']['mean']:.3f} +/- {scores['auc']['std']:.3f}  "
              f"(best params: {best_params})")
    return results


if __name__ == "__main__":
    from preprocessing import prepare_dataset
    data = prepare_dataset()
    results = run_all_baselines(data["X"], data["y"], data["folds"])
