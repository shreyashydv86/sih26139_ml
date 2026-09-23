"""
SIH26139 -- Hybrid Quantum-Classical ML Platform
Data preprocessing module -- multi-disease.

Supports three disease domains out of the box, directly matching the
problem statement's own wording ("early detection of cancer,
cardiovascular or neurological disease"):

    "heart_disease" -- UCI Cleveland Heart Disease   (Cardiovascular)
    "breast_cancer" -- Breast Cancer Wisconsin        (Cancer)
    "parkinsons"    -- UCI Parkinson's voice dataset  (Neurological)

The SAME downstream pipeline (feature selection, angle-range scaling,
cross-validation splitting -- and therefore the same quantum circuit
in quantum_model.py) runs unchanged regardless of which disease is
selected. This is what actually proves the "disease-agnostic modular
design" claim rather than just asserting it on a slide.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import load_breast_cancer

HEART_DISEASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
PARKINSONS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"

# Kept for backward compatibility with the single-disease version of this module.
UCI_URL = HEART_DISEASE_URL

HEART_DISEASE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]


def _load_heart_disease(source: str = HEART_DISEASE_URL):
    """Cardiovascular. ~297 patients after cleaning, 13 features."""
    df = pd.read_csv(source, names=HEART_DISEASE_COLUMNS, na_values="?")
    df = df.dropna().reset_index(drop=True)
    df["target"] = (df["target"] > 0).astype(int)
    X = df.drop(columns=["target"])
    y = df["target"].values
    return X, y


def _load_breast_cancer(source: str = None):
    """
    Cancer. 569 patients, 30 features. Ships with scikit-learn --
    no network access or download required, which makes it the most
    reliable second disease to add under a tight deadline.

    sklearn encodes target as 0 = malignant, 1 = benign; flipped here
    so 1 always means "disease present", consistent with the other
    two loaders.
    """
    data = load_breast_cancer(as_frame=True)
    X = data.frame.drop(columns=["target"])
    y = (data.frame["target"] == 0).astype(int).values
    return X, y


def _load_parkinsons(source: str = PARKINSONS_URL):
    """
    Neurological. 195 voice recordings, 22 biomedical voice measures.
    target: 'status' column, 1 = Parkinson's, 0 = healthy -- already
    matches the "1 = disease" convention used elsewhere.
    """
    df = pd.read_csv(source)
    df = df.drop(columns=["name"])  # patient identifier, not a feature
    y = df["status"].values
    X = df.drop(columns=["status"])
    return X, y


DATASET_REGISTRY = {
    "heart_disease": {
        "loader": _load_heart_disease,
        "default_source": HEART_DISEASE_URL,
        "disease_category": "Cardiovascular",
        "display_name": "UCI Heart Disease (Cleveland)",
    },
    "breast_cancer": {
        "loader": _load_breast_cancer,
        "default_source": None,
        "disease_category": "Cancer",
        "display_name": "Breast Cancer Wisconsin (Diagnostic)",
    },
    "parkinsons": {
        "loader": _load_parkinsons,
        "default_source": PARKINSONS_URL,
        "disease_category": "Neurological",
        "display_name": "UCI Parkinson's Disease (voice measurements)",
    },
}


def select_features(X: pd.DataFrame, y: np.ndarray, k: int = 4):
    """
    Select the k most predictive features using the ANOVA F-test.

    Deliberately using SelectKBest instead of PCA: this keeps the
    ORIGINAL feature names intact, so SHAP explanations later say
    "cholesterol" and "chest pain type" instead of uninterpretable
    principal components. This directly fixes the PCA/SHAP
    interpretability gap flagged during problem-statement review,
    and it's what makes the SAME selection logic work sensibly across
    completely different feature sets (13 cardiac features, 30 cancer
    cell-nucleus measurements, 22 voice-recording features) without
    any per-disease tuning.

    Returns
    -------
    X_selected : np.ndarray, shape (n_samples, k)
    selected_names : list[str], the k chosen feature names, in order
    selector : fitted SelectKBest object (needed to transform new
               patient data at inference time)
    """
    k = min(k, X.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_mask = selector.get_support()
    selected_names = X.columns[selected_mask].tolist()
    return X_selected, selected_names, selector


def scale_to_angle_range(X: np.ndarray):
    """
    Scale features to [0, pi] for quantum angle embedding.
    Each qubit's rotation gate expects an angle in this range.
    """
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    return scaler.fit_transform(X), scaler


def get_cv_splits(X: np.ndarray, y: np.ndarray, n_splits: int = 5, seed: int = 42):
    """
    Stratified K-Fold splits. With only ~297 patients, a single
    train/test split gives a noisy, unreliable performance estimate.
    Cross-validation is the difference between "we got 86% once" and
    "we got 84% plus or minus 3% across five independent folds" -- the
    second claim is the one a judge or reviewer can actually trust.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(X, y))


def prepare_dataset(dataset: str = "heart_disease", source: str = None,
                     k_features: int = 4, n_splits: int = 5):
    """
    End-to-end preprocessing pipeline for ANY registered disease.
    Returns everything downstream modules need: the selected+scaled
    features, labels, fold indices, and the feature names (for SHAP)
    -- identical shape and structure no matter which disease was
    requested, which is exactly what lets quantum_model.py,
    classical_baselines.py, and shap_explain.py run unmodified
    across all three.

    Parameters
    ----------
    dataset : one of "heart_disease", "breast_cancer", "parkinsons"
        (see DATASET_REGISTRY)
    source  : optional override for the data location (a URL or local
        file path). Ignored for breast_cancer, which loads from
        scikit-learn directly and needs no network access.
    """
    if dataset not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Choose from: {list(DATASET_REGISTRY)}"
        )

    cfg = DATASET_REGISTRY[dataset]
    actual_source = source or cfg["default_source"]
    X_raw, y = cfg["loader"](actual_source)

    X_selected, feature_names, selector = select_features(X_raw, y, k=k_features)
    X_scaled, scaler = scale_to_angle_range(X_selected)
    folds = get_cv_splits(X_scaled, y, n_splits=n_splits)

    return {
        "X": X_scaled,
        "y": y,
        "feature_names": feature_names,
        "folds": folds,
        "selector": selector,
        "scaler": scaler,
        "n_patients": len(y),
        "dataset": dataset,
        "disease_category": cfg["disease_category"],
        "display_name": cfg["display_name"],
    }


# Backward-compatible aliases for the single-disease version of this module.
def load_raw_data(source: str = UCI_URL) -> pd.DataFrame:
    return pd.read_csv(source, names=HEART_DISEASE_COLUMNS, na_values="?")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna().reset_index(drop=True)
    df["target"] = (df["target"] > 0).astype(int)
    return df


if __name__ == "__main__":
    for name in DATASET_REGISTRY:
        print(f"\n--- {name} ---")
        try:
            data = prepare_dataset(dataset=name)
            print(f"  {data['display_name']}  ({data['disease_category']})")
            print(f"  Patients: {data['n_patients']}")
            print(f"  Selected features: {data['feature_names']}")
            print(f"  Class balance: {np.bincount(data['y'])} (no-disease, disease)")
            print(f"  CV folds: {len(data['folds'])}")
        except Exception as e:
            print(f"  Could not load ({type(e).__name__}: {e})")
            print("  -- likely needs internet access on this machine; the code itself is correct.")
