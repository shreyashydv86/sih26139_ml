"""
SIH26139 -- Hybrid quantum-classical model.

Architecture: classical pre-layer -> 4-qubit Variational Quantum
Circuit (angle embedding + entangling layers) -> classical post-layer.

Trained per cross-validation fold with the Adam optimiser. Barren
plateau mitigations applied throughout: qubit count kept low (4),
circuit depth kept shallow (2 layers), and parameters initialised
close to zero rather than uniformly at random.
"""

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from sklearn.metrics import roc_auc_score, f1_score, recall_score, accuracy_score

N_QUBITS = 4
N_LAYERS = 2

dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    """
    inputs  : tensor of shape (N_QUBITS,) -- already scaled to [0, pi]
    weights : tensor of shape (N_LAYERS, N_QUBITS) -- trainable
    """
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
    qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS))
    return qml.expval(qml.PauliZ(0))


class HybridQML(nn.Module):
    """
    Three-layer hybrid architecture:
      1. Classical pre-layer  : Linear(n_features -> N_QUBITS) + Tanh
      2. Quantum layer        : the VQC above
      3. Classical post-layer : Linear(1 -> 1) + Sigmoid
    """

    def __init__(self, n_features: int):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Linear(n_features, 8),
            nn.ReLU(),
            nn.Linear(8, N_QUBITS),
            nn.Tanh(),
        )
        # Initialise near zero -- a barren-plateau mitigation.
        self.q_weights = nn.Parameter(torch.randn(N_LAYERS, N_QUBITS) * 0.05)
        self.post = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())

    def forward(self, x):
        # Pre-layer output is in [-1, 1] (Tanh range); rescale to [0, pi]
        pre_out = self.pre(x)
        angles = (pre_out + 1) / 2 * np.pi

        q_out = torch.stack([
            quantum_circuit(angles[i], self.q_weights)
            for i in range(angles.shape[0])
        ]).unsqueeze(1).float()

        return self.post(q_out).squeeze(1)


def train_one_fold(X_train, y_train, n_features, epochs=50, lr=0.01):
    """Train a fresh HybridQML instance on one fold's training data."""
    model = HybridQML(n_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)

    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(X_t)
        loss = criterion(preds, y_t)
        loss.backward()
        optimizer.step()

    return model


def evaluate_hybrid_with_cv(X: np.ndarray, y: np.ndarray, folds: list,
                             epochs: int = 50, lr: float = 0.01):
    """
    Train and evaluate the hybrid model across the same outer folds
    used for the classical baselines. This is the direct, fair
    comparison the technical report needs.
    """
    n_features = X.shape[1]
    metrics = {"auc": [], "f1": [], "sensitivity": [], "accuracy": []}
    fold_models = []

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = train_one_fold(X_train, y_train, n_features, epochs, lr)
        fold_models.append(model)

        model.eval()
        with torch.no_grad():
            proba = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
        preds = (proba > 0.5).astype(int)

        metrics["auc"].append(roc_auc_score(y_test, proba))
        metrics["f1"].append(f1_score(y_test, preds))
        metrics["sensitivity"].append(recall_score(y_test, preds))
        metrics["accuracy"].append(accuracy_score(y_test, preds))

        print(f"  Fold {fold_i + 1}/{len(folds)}: AUC = {metrics['auc'][-1]:.3f}")

    summary = {
        k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "folds": v}
        for k, v in metrics.items()
    }
    return summary, fold_models


if __name__ == "__main__":
    from preprocessing import prepare_dataset
    data = prepare_dataset()
    print("Training hybrid QML model across folds...")
    scores, models = evaluate_hybrid_with_cv(data["X"], data["y"], data["folds"])
    print(f"\nHybrid QML: AUC = {scores['auc']['mean']:.3f} +/- {scores['auc']['std']:.3f}")
