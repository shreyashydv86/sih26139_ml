"""
SIH26139 -- Hybrid Quantum-Classical Model.
"""

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    recall_score,
    accuracy_score,
)

from src.preprocessing import preprocess_fold


N_QUBITS = 4
N_LAYERS = 2


# ============================================================
# Quantum device
# ============================================================

dev = qml.device(
    "default.qubit",
    wires=N_QUBITS
)


# ============================================================
# Quantum circuit
# ============================================================

@qml.qnode(
    dev,
    interface="torch",
    diff_method="backprop"
)
def quantum_circuit(inputs, weights):

    qml.AngleEmbedding(
        inputs,
        wires=range(N_QUBITS),
        rotation="Y"
    )

    qml.BasicEntanglerLayers(
        weights,
        wires=range(N_QUBITS)
    )

    return qml.expval(
        qml.PauliZ(0)
    )


# ============================================================
# Hybrid model
# ============================================================

class HybridQuantumModel(nn.Module):

    def __init__(self, n_features=4):

        super().__init__()

        self.pre = nn.Sequential(
            nn.Linear(n_features, 8),
            nn.ReLU(),
            nn.Linear(8, N_QUBITS),
            nn.Tanh()
        )

        self.q_weights = nn.Parameter(
            torch.randn(
                N_LAYERS,
                N_QUBITS
            ) * 0.05
        )

        self.post = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        x = self.pre(x)

        # Convert [-1, 1] to [0, pi]
        x = (x + 1.0) * (np.pi / 2.0)

        quantum_outputs = []

        for sample in x:

            q_output = quantum_circuit(
                sample,
                self.q_weights
            )

            quantum_outputs.append(q_output)

        quantum_outputs = torch.stack(
            quantum_outputs
        ).float().reshape(-1, 1)

        output = self.post(
            quantum_outputs
        )

        return output.squeeze(1)


# ============================================================
# Training
# ============================================================

def train_model(
    X_train,
    y_train,
    epochs=50,
    learning_rate=0.01
):

    X_train = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    y_train = torch.tensor(
        y_train,
        dtype=torch.float32
    )

    model = HybridQuantumModel(
        n_features=X_train.shape[1]
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    loss_function = nn.BCELoss()

    model.train()

    for epoch in range(epochs):

        optimizer.zero_grad()

        predictions = model(
            X_train
        )

        loss = loss_function(
            predictions,
            y_train
        )

        loss.backward()

        optimizer.step()

        if (epoch + 1) % 10 == 0:

            print(
                f"      Epoch {epoch + 1}/{epochs} "
                f"- Loss: {loss.item():.4f}"
            )

    return model


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test
):

    X_test = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    model.eval()

    with torch.no_grad():

        probabilities = (
            model(X_test)
            .cpu()
            .numpy()
        )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

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

    return {
        "auc": float(auc),
        "f1": float(f1),
        "sensitivity": float(sensitivity),
        "accuracy": float(accuracy),
    }


# ============================================================
# Cross-validation
# ============================================================

def evaluate_hybrid_with_cv(
    X_raw,
    y,
    folds,
    k_features=4,
    epochs=50
):

    metric_values = {
        "auc": [],
        "f1": [],
        "sensitivity": [],
        "accuracy": [],
    }

    models = []

    for fold_number, (train_idx, test_idx) in enumerate(
        folds,
        start=1
    ):

        print("\n" + "=" * 60)
        print(
            f"Hybrid QML - Fold {fold_number}"
        )
        print("=" * 60)

        X_train_raw = X_raw[train_idx]
        X_test_raw = X_raw[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

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

        print(
            f"      Selected features: "
            f"{X_train.shape[1]}"
        )

        model = train_model(
            X_train,
            y_train,
            epochs=epochs
        )

        scores = evaluate_model(
            model,
            X_test,
            y_test
        )

        for metric_name in metric_values:

            metric_values[
                metric_name
            ].append(
                scores[metric_name]
            )

        models.append(model)

        print(
            f"      AUC: "
            f"{scores['auc']:.3f}"
        )

        print(
            f"      F1: "
            f"{scores['f1']:.3f}"
        )

        print(
            f"      Sensitivity: "
            f"{scores['sensitivity']:.3f}"
        )

        print(
            f"      Accuracy: "
            f"{scores['accuracy']:.3f}"
        )

    summary = {}

    for metric_name, values in metric_values.items():

        summary[metric_name] = {
            "mean": float(
                np.mean(values)
            ),

            "std": float(
                np.std(values)
            ),

            "folds": [
                float(v)
                for v in values
            ],
        }

    return summary, models


# ============================================================
# Standalone test
# ============================================================

if __name__ == "__main__":

    from src.preprocessing import prepare_dataset

    print(
        "Testing Hybrid Quantum-Classical Model..."
    )

    data = prepare_dataset(
        dataset="breast_cancer",
        k_features=4,
        n_splits=5
    )

    results, models = evaluate_hybrid_with_cv(
        X_raw=data["X_raw"],
        y=data["y"],
        folds=data["folds"],
        k_features=data["k_features"],
        epochs=10
    )

    print("\nFinal QML Results:")

    for metric_name, scores in results.items():

        print(
            f"{metric_name}: "
            f"{scores['mean']:.4f} "
            f"+/- "
            f"{scores['std']:.4f}"
        )