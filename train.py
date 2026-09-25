"""
SIH26139 -- Main training pipeline.

Pipeline:
1. Load raw dataset
2. Create outer cross-validation folds
3. Run leakage-free classical baselines
4. Run leakage-free hybrid QML
5. Compare QML with classical models
6. Save results to JSON
"""

import argparse
import json

from src.preprocessing import prepare_dataset
from src.classical_baselines import run_all_baselines
from src.quantum_model import evaluate_hybrid_with_cv
from src.statistical_test import compare_models


def run_experiment(
    dataset,
    source=None,
    k_features=4,
    n_splits=5,
    epochs=50
):

    print("\n" + "=" * 70)
    print("SIH26139 HYBRID QUANTUM-CLASSICAL ML")
    print("=" * 70)

    print(f"\nDataset: {dataset}")
    print(f"Features selected per fold: {k_features}")
    print(f"Outer CV folds: {n_splits}")
    print(f"QML epochs: {epochs}")

    # ==========================================================
    # STEP 1: DATASET PREPARATION
    # ==========================================================

    print("\n" + "=" * 70)
    print("STEP 1: DATASET PREPARATION")
    print("=" * 70)

    data = prepare_dataset(
        dataset=dataset,
        source=source,
        k_features=k_features,
        n_splits=n_splits
    )

    print(f"\nDataset: {data['display_name']}")
    print(f"Category: {data['disease_category']}")
    print(f"Patients: {data['n_patients']}")
    print(
        f"Original features: "
        f"{len(data['feature_names'])}"
    )
    print(f"CV folds: {len(data['folds'])}")

    # ==========================================================
    # STEP 2: CLASSICAL BASELINES
    # ==========================================================

    print("\n" + "=" * 70)
    print("STEP 2: CLASSICAL BASELINES")
    print("=" * 70)

    classical_results = run_all_baselines(
        X_raw=data["X_raw"],
        y=data["y"],
        folds=data["folds"],
        k_features=data["k_features"]
    )

    # ==========================================================
    # STEP 3: HYBRID QML
    # ==========================================================

    print("\n" + "=" * 70)
    print("STEP 3: HYBRID QUANTUM-CLASSICAL MODEL")
    print("=" * 70)

    quantum_scores, quantum_models = evaluate_hybrid_with_cv(
        X_raw=data["X_raw"],
        y=data["y"],
        folds=data["folds"],
        k_features=data["k_features"],
        epochs=epochs
    )

    # ==========================================================
    # STEP 4: STATISTICAL COMPARISON
    # ==========================================================

    print("\n" + "=" * 70)
    print("STEP 4: STATISTICAL COMPARISON")
    print("=" * 70)

    statistical_results = {}

    qml_auc = quantum_scores["auc"]["folds"]

    for model_name, model_result in classical_results.items():

        classical_auc = model_result["metrics"]["auc"]["folds"]

        print(
            f"\nComparing Hybrid QML vs {model_name}"
        )

        comparison = compare_models(
            qml_auc,
            classical_auc,
            name_a="Hybrid QML",
            name_b=model_name
        )

        statistical_results[model_name] = comparison

    # ==========================================================
    # STEP 5: SUMMARY
    # ==========================================================

    best_classical_name = max(
        classical_results,
        key=lambda name:
        classical_results[name]["metrics"]["auc"]["mean"]
    )

    best_classical_auc = classical_results[
        best_classical_name
    ]["metrics"]["auc"]["mean"]

    qml_auc_mean = quantum_scores["auc"]["mean"]

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"\nBest classical model by mean AUC: "
        f"{best_classical_name}"
    )

    print(
        f"Best classical mean AUC: "
        f"{best_classical_auc:.4f}"
    )

    print(
        f"Hybrid QML mean AUC: "
        f"{qml_auc_mean:.4f}"
    )

    # ==========================================================
    # STEP 6: CREATE JSON REPORT
    # ==========================================================

    report = {
        "project": "SIH26139",

        "dataset": dataset,

        "dataset_display_name":
            data["display_name"],

        "disease_category":
            data["disease_category"],

        "dataset_information": {
            "n_patients":
                data["n_patients"],

            "original_features":
                len(data["feature_names"]),

            "selected_features_per_fold":
                k_features,

            "outer_cv_folds":
                n_splits
        },

        "methodology": {
            "feature_selection":
                "SelectKBest with f_classif",

            "scaling":
                "MinMaxScaler to [0, pi]",

            "cross_validation":
                "StratifiedKFold with shuffle=True and random_state=42",

            "classical_tuning":
                "GridSearchCV inside outer training folds",

            "qml_qubits":
                4,

            "qml_layers":
                2,

            "qml_epochs":
                epochs
        },

        "classical_results":
            classical_results,

        "hybrid_qml_results":
            quantum_scores,

        "statistical_comparisons":
            statistical_results,

        "summary": {
            "best_classical_model_by_mean_auc":
                best_classical_name,

            "best_classical_mean_auc":
                float(best_classical_auc),

            "hybrid_qml_mean_auc":
                float(qml_auc_mean)
        }
    }

    # ==========================================================
    # SAVE REPORT
    # ==========================================================

    output_file = (
        f"results_report_{dataset}.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=2
        )

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    print(
        f"\nFull results saved to: "
        f"{output_file}"
    )

    return report


def main():

    parser = argparse.ArgumentParser(
        description=(
            "SIH26139 Hybrid Quantum-Classical "
            "Disease Detection"
        )
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="breast_cancer",
        choices=[
            "breast_cancer",
            "heart_disease",
            "parkinsons"
        ],
        help="Dataset to evaluate"
    )

    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Optional local dataset file or URL"
    )

    parser.add_argument(
        "--k-features",
        type=int,
        default=4,
        help="Number of features selected per fold"
    )

    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of outer CV folds"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="QML training epochs"
    )

    args = parser.parse_args()

    run_experiment(
        dataset=args.dataset,
        source=args.source,
        k_features=args.k_features,
        n_splits=args.n_splits,
        epochs=args.epochs
    )


if __name__ == "__main__":
    main()