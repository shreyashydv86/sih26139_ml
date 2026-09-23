"""
SIH26139 -- SHAP explainability module.

Wraps the trained hybrid model in a SHAP KernelExplainer, using the
ORIGINAL clinical feature names (age, cholesterol, chest pain type...)
rather than PCA components, so every explanation is something a
doctor can actually act on.
"""

import numpy as np
import torch
import shap


def make_predict_fn(model):
    """Wrap the PyTorch model so SHAP can call it like a plain function."""
    def predict_fn(X_numpy: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_numpy, dtype=torch.float32)
            return model(X_t).numpy()
    return predict_fn


def explain_predictions(model, X_background: np.ndarray, X_explain: np.ndarray,
                         feature_names: list, n_background: int = 10,
                         nsamples: int = 200):
    """
    Compute SHAP values for a small batch of patients.

    Parameters kept deliberately small (n_background=10, nsamples=200):
    SHAP's KernelExplainer calls the model nsamples times PER sample,
    and each call runs a full quantum circuit simulation. Using the
    library's defaults (nsamples=2048) here would take hours; these
    reduced settings make the explainability step practical without
    materially changing the ranking of which features matter most.
    """
    predict_fn = make_predict_fn(model)
    background = X_background[:n_background]

    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(X_explain, nsamples=nsamples)

    return shap_values, explainer


def explain_single_patient(shap_values_row: np.ndarray, feature_names: list) -> dict:
    """
    Turn one patient's raw SHAP array into a sorted, human-readable
    dict of {feature_name: contribution}, largest absolute impact first.
    """
    values = np.asarray(shap_values_row).flatten().tolist()
    pairs = list(zip(feature_names, values))
    pairs.sort(key=lambda p: abs(p[1]), reverse=True)
    return dict(pairs)


if __name__ == "__main__":
    from preprocessing import prepare_dataset
    from quantum_model import evaluate_hybrid_with_cv

    data = prepare_dataset()
    scores, models = evaluate_hybrid_with_cv(data["X"], data["y"], data["folds"])

    best_fold = int(np.argmax(scores["auc"]["folds"]))
    best_model = models[best_fold]
    train_idx, test_idx = data["folds"][best_fold]

    shap_values, _ = explain_predictions(
        best_model,
        X_background=data["X"][train_idx],
        X_explain=data["X"][test_idx][:5],
        feature_names=data["feature_names"],
    )

    for i, row in enumerate(shap_values):
        print(f"\nPatient {i+1}:")
        for feat, val in explain_single_patient(row, data["feature_names"]).items():
            print(f"  {feat:20s}: {val:+.4f}")
