"""
SIH26139 -- End-to-end training orchestrator.

Run this single script to reproduce the entire pipeline:
data loading -> classical baselines -> hybrid QML training ->
SHAP explanations -> statistical comparison -> final report.

Usage
-----
    python train.py
    python train.py --source data/raw/heart_disease.csv   # local file instead of the UCI URL
    python train.py --k-features 6 --n-splits 10
"""

import argparse
import json
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocessing import prepare_dataset, UCI_URL
from classical_baselines import run_all_baselines
from quantum_model import evaluate_hybrid_with_cv
from shap_explain import explain_predictions, explain_single_patient
from statistical_test import compare_models, print_comparison


def main(dataset: str, source: str, k_features: int, n_splits: int, epochs: int):
    print("=" * 60)
    print("SIH26139 -- Hybrid Quantum-Classical Disease Detection")
    print("=" * 60)

    print(f"\n[1/5] Loading and preprocessing data ({dataset})...")
    data = prepare_dataset(dataset=dataset, source=source,
                            k_features=k_features, n_splits=n_splits)
    print(f"  {data['display_name']}  ({data['disease_category']})")
    print(f"  Patients: {data['n_patients']}  |  Features selected: {data['feature_names']}")

    print("\n[2/5] Tuning and evaluating classical baselines...")
    classical_results = run_all_baselines(data["X"], data["y"], data["folds"])

    print("\n[3/5] Training hybrid quantum-classical model across folds...")
    quantum_scores, quantum_models = evaluate_hybrid_with_cv(
        data["X"], data["y"], data["folds"], epochs=epochs
    )
    print(f"  Hybrid QML: AUC = {quantum_scores['auc']['mean']:.3f} +/- {quantum_scores['auc']['std']:.3f}")

    print("\n[4/5] Computing SHAP explanations for sample patients...")
    best_fold = int(np.argmax(quantum_scores["auc"]["folds"]))
    best_model = quantum_models[best_fold]
    train_idx, test_idx = data["folds"][best_fold]
    n_explain = min(3, len(test_idx))
    shap_values, _ = explain_predictions(
        best_model,
        X_background=data["X"][train_idx],
        X_explain=data["X"][test_idx][:n_explain],
        feature_names=data["feature_names"],
    )
    for i, row in enumerate(shap_values):
        readable = explain_single_patient(row, data["feature_names"])
        print(f"  Patient {i + 1}: " + ", ".join(f"{f}={v:+.3f}" for f, v in readable.items()))

    print("\n[5/5] Statistical comparison -- Hybrid QML vs best classical baseline...")
    best_classical_name = max(
        classical_results,
        key=lambda k: classical_results[k]["metrics"]["auc"]["mean"],
    )
    comparison = compare_models(
        quantum_scores["auc"]["folds"],
        classical_results[best_classical_name]["metrics"]["auc"]["folds"],
        name_a="Hybrid QML",
        name_b=best_classical_name,
    )
    print_comparison(comparison)

    report = {
        "dataset": data["dataset"],
        "disease_category": data["disease_category"],
        "display_name": data["display_name"],
        "n_patients": data["n_patients"],
        "features_used": data["feature_names"],
        "classical_results": {
            name: {
                "auc_mean": r["metrics"]["auc"]["mean"],
                "auc_std": r["metrics"]["auc"]["std"],
                "f1_mean": r["metrics"]["f1"]["mean"],
                "sensitivity_mean": r["metrics"]["sensitivity"]["mean"],
                "best_params": r["best_params"],
            }
            for name, r in classical_results.items()
        },
        "hybrid_qml_results": {
            "auc_mean": quantum_scores["auc"]["mean"],
            "auc_std": quantum_scores["auc"]["std"],
            "f1_mean": quantum_scores["f1"]["mean"],
            "sensitivity_mean": quantum_scores["sensitivity"]["mean"],
        },
        "statistical_comparison": comparison,
    }

    out_path = os.path.join(os.path.dirname(__file__), f"results_report_{dataset}.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Done. Full results saved to {out_path}")
    print("=" * 60)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="heart_disease",
                         choices=["heart_disease", "breast_cancer", "parkinsons"],
                         help="Which disease to train on")
    parser.add_argument("--source", default=None,
                         help="Override the default URL/location for the chosen dataset "
                              "(ignored for breast_cancer, which needs no download)")
    parser.add_argument("--k-features", type=int, default=4)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    main(args.dataset, args.source, args.k_features, args.n_splits, args.epochs)
