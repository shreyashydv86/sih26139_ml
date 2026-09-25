"""
SIH26139 -- Dataset loading and leakage-free preprocessing.

Supports:
- Breast Cancer Wisconsin dataset
- UCI Heart Disease (Cleveland)
- UCI Parkinson's Disease

Feature selection and scaling are fitted ONLY on the training portion
of each cross-validation fold to prevent data leakage.
"""

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer as sklearn_load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# DATASET LOADERS
# ============================================================

def _load_heart_disease(source):
    """
    Load UCI Cleveland Heart Disease dataset.
    """

    columns = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "target",
    ]

    df = pd.read_csv(
        source,
        names=columns,
        na_values="?"
    )

    df = df.dropna()

    X = df.drop(columns=["target"])
    y = (df["target"] > 0).astype(int)

    return X, y.values


def _load_breast_cancer(source=None):
    """
    Load the Breast Cancer Wisconsin Diagnostic dataset.

    This dataset is included with scikit-learn, so no internet
    connection is required.
    """

    data = sklearn_load_breast_cancer(as_frame=True)

    X = data.data

    # sklearn uses:
    # 0 = malignant
    # 1 = benign
    #
    # We want:
    # 1 = disease
    # 0 = no disease
    y = (data.target == 0).astype(int)

    return X, y.values


def _load_parkinsons(source):
    """
    Load UCI Parkinson's voice-measurement dataset.
    """

    df = pd.read_csv(source)

    # The 'name' column identifies the patient/file and is not
    # used as a machine-learning feature.
    if "name" in df.columns:
        df = df.drop(columns=["name"])

    y = df["status"].astype(int).values

    X = df.drop(columns=["status"])

    return X, y


# ============================================================
# DATASET REGISTRY
# ============================================================

DATASET_REGISTRY = {

    "heart_disease": {
        "loader": _load_heart_disease,
        "default_source": (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "heart-disease/processed.cleveland.data"
        ),
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
        "default_source": (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "parkinsons/parkinsons.data"
        ),
        "disease_category": "Neurological",
        "display_name": "UCI Parkinson's Disease (voice measurements)",
    },
}


# ============================================================
# CROSS-VALIDATION
# ============================================================

def get_cv_splits(X, y, n_splits=5):
    """
    Create stratified outer cross-validation folds.

    IMPORTANT:
    No feature selection or scaling is performed here.
    """

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    return list(cv.split(X, y))


# ============================================================
# LEAKAGE-FREE FOLD PREPROCESSING
# ============================================================

def preprocess_fold(
    X_train,
    X_test,
    y_train,
    k_features=4
):
    """
    Perform feature selection and scaling for ONE CV fold.

    Feature selector:
        fitted ONLY on X_train / y_train

    Scaler:
        fitted ONLY on X_train

    X_test is transformed using the already-fitted objects.

    This prevents information from the outer test fold leaking
    into the training process.
    """

    # --------------------------------------------------------
    # 1. Feature selection
    # --------------------------------------------------------

    selector = SelectKBest(
        score_func=f_classif,
        k=k_features
    )

    X_train_selected = selector.fit_transform(
        X_train,
        y_train
    )

    X_test_selected = selector.transform(
        X_test
    )

    # --------------------------------------------------------
    # 2. Scaling
    # --------------------------------------------------------

    scaler = MinMaxScaler(
        feature_range=(0, np.pi)
    )

    X_train_scaled = scaler.fit_transform(
        X_train_selected
    )

    X_test_scaled = scaler.transform(
        X_test_selected
    )

    return (
        X_train_scaled,
        X_test_scaled,
        selector,
        scaler
    )


# ============================================================
# FEATURE NAMES
# ============================================================

def get_selected_feature_names(selector, feature_names):
    """
    Return the original names of the features selected by
    SelectKBest.
    """

    feature_names = np.asarray(feature_names)

    return feature_names[selector.get_support()].tolist()


# ============================================================
# DATASET PREPARATION
# ============================================================

def prepare_dataset(
    dataset="heart_disease",
    source=None,
    k_features=4,
    n_splits=5
):
    """
    Load a dataset and create outer CV folds.

    IMPORTANT:
    Feature selection and scaling are deliberately NOT performed
    here.

    They are performed inside each outer training fold using
    preprocess_fold().
    """

    if dataset not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Available datasets: {list(DATASET_REGISTRY.keys())}"
        )

    config = DATASET_REGISTRY[dataset]

    # Use default source if the user didn't provide one.
    if source is None:
        source = config["default_source"]

    # Load raw data.
    X_raw, y = config["loader"](source)

    # Convert DataFrame to NumPy array while keeping the original
    # feature names separately.
    if isinstance(X_raw, pd.DataFrame):

        feature_names = X_raw.columns.tolist()

        X_raw_array = X_raw.values

    else:

        X_raw_array = np.asarray(X_raw)

        feature_names = [
            f"feature_{i}"
            for i in range(X_raw_array.shape[1])
        ]

    y = np.asarray(y)

    # Create OUTER CV folds on raw data.
    folds = get_cv_splits(
        X_raw_array,
        y,
        n_splits=n_splits
    )

    return {
        "X_raw": X_raw_array,
        "y": y,
        "feature_names": feature_names,
        "folds": folds,
        "k_features": k_features,
        "n_patients": len(y),
        "dataset": dataset,
        "disease_category": config["disease_category"],
        "display_name": config["display_name"],
    }


# ============================================================
# BACKWARD-COMPATIBLE ALIASES
# ============================================================

def load_heart_disease(source=None):
    """
    Backward-compatible heart disease loader.
    """

    if source is None:
        source = DATASET_REGISTRY["heart_disease"]["default_source"]

    return _load_heart_disease(source)


def load_breast_cancer(source=None):
    """
    Backward-compatible breast cancer loader.
    """

    return _load_breast_cancer(source)


def load_parkinsons(source=None):
    """
    Backward-compatible Parkinson's loader.
    """

    if source is None:
        source = DATASET_REGISTRY["parkinsons"]["default_source"]

    return _load_parkinsons(source)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("Available datasets:")
    for name in DATASET_REGISTRY:
        print(f"  - {name}")

    print("\nTesting breast cancer dataset...")

    data = prepare_dataset(
        dataset="breast_cancer",
        k_features=4,
        n_splits=5
    )

    print(f"Patients: {data['n_patients']}")
    print(f"Original features: {len(data['feature_names'])}")
    print(f"CV folds: {len(data['folds'])}")

    print("\nPreprocessing test fold...")

    train_idx, test_idx = data["folds"][0]

    X_train = data["X_raw"][train_idx]
    X_test = data["X_raw"][test_idx]

    y_train = data["y"][train_idx]

    (
        X_train_processed,
        X_test_processed,
        selector,
        scaler
    ) = preprocess_fold(
        X_train,
        X_test,
        y_train,
        k_features=4
    )

    selected_names = get_selected_feature_names(
        selector,
        data["feature_names"]
    )

    print("Selected features:")
    for name in selected_names:
        print(f"  - {name}")

    print(
        f"\nTraining shape: {X_train_processed.shape}"
    )

    print(
        f"Test shape: {X_test_processed.shape}"
    )
    print("\nprocessing test successful.")