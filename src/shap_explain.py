"""
SIH26139 -- SHAP explainability module.

Provides lightweight SHAP explanations for an already-trained model.

Important:
- This module does NOT retrain the QML model.
- SHAP must receive the same feature representation that the trained
  model expects.
- Feature names should therefore correspond to the model input features.
- For the current 4-feature QML pipeline, these are the 4 selected
  features from the relevant preprocessing fold.
"""

import numpy as np
import torch
import shap


def make_predict_fn(model):
    """
    Wrap a PyTorch model so SHAP can call it like a normal NumPy function.
    """

    def predict_fn(X_numpy: np.ndarray) -> np.ndarray:
        model.eval()

        with torch.no_grad():
            X_t = torch.tensor(
                X_numpy,
                dtype=torch.float32
            )

            predictions = model(X_t)

        return predictions.detach().cpu().numpy()

    return predict_fn


def explain_predictions(
    model,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    feature_names: list,
    n_background: int = 10,
    nsamples: int = 100
):
    """
    Compute SHAP values for a small set of already-preprocessed samples.

    Parameters
    ----------
    model:
        Already-trained PyTorch/QML model.

    X_background:
        Background samples in the SAME feature representation expected
        by the model.

    X_explain:
        Samples to explain, also in the SAME representation.

    feature_names:
        Names corresponding to the columns of X_background/X_explain.

    n_background:
        Number of background samples used by KernelExplainer.

    nsamples:
        Number of SHAP perturbation samples.

    Returns
    -------
    shap_values, explainer
    """

    X_background = np.asarray(X_background, dtype=np.float32)
    X_explain = np.asarray(X_explain, dtype=np.float32)

    if X_background.ndim != 2:
        raise ValueError(
            "X_background must be a 2D array."
        )

    if X_explain.ndim != 2:
        raise ValueError(
            "X_explain must be a 2D array."
        )

    if X_background.shape[1] != len(feature_names):
        raise ValueError(
            "Number of feature names does not match "
            "the number of model input features."
        )

    if X_explain.shape[1] != X_background.shape[1]:
        raise ValueError(
            "X_explain and X_background must have "
            "the same number of features."
        )

    background = X_background[:n_background]

    predict_fn = make_predict_fn(model)

    explainer = shap.KernelExplainer(
        predict_fn,
        background
    )

    shap_values = explainer.shap_values(
        X_explain,
        nsamples=nsamples
    )

    return shap_values, explainer


def explain_single_patient(
    shap_values_row: np.ndarray,
    feature_names: list
) -> dict:
    """
    Convert one patient's SHAP values into a sorted dictionary.

    Features with the largest absolute SHAP impact appear first.
    """

    values = np.asarray(
        shap_values_row
    ).flatten().tolist()

    if len(values) != len(feature_names):
        raise ValueError(
            "Number of SHAP values does not match "
            "number of feature names."
        )

    pairs = list(
        zip(feature_names, values)
    )

    pairs.sort(
        key=lambda pair: abs(pair[1]),
        reverse=True
    )

    return dict(pairs)


def summarize_shap_values(
    shap_values: np.ndarray,
    feature_names: list
) -> dict:
    """
    Calculate mean absolute SHAP importance for each feature.

    This is useful for creating a simple feature-importance table
    without retraining the model.
    """

    values = np.asarray(shap_values)

    if values.ndim == 1:
        values = values.reshape(1, -1)

    mean_abs_values = np.mean(
        np.abs(values),
        axis=0
    )

    pairs = list(
        zip(feature_names, mean_abs_values)
    )

    pairs.sort(
        key=lambda pair: pair[1],
        reverse=True
    )

    return {
        feature: float(importance)
        for feature, importance in pairs
    }


if __name__ == "__main__":
    print("SHAP module loaded successfully.")
    print(
        "No model training is performed by this module."
    )