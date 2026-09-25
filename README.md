# SIH26139 — Hybrid Quantum-Classical Disease Detection Platform

A disease-agnostic hybrid quantum-classical machine learning pipeline for
early disease detection across three categories:

- Cancer
- Cardiovascular disease
- Neurological disease

The project uses the same classical baseline pipeline and the same
4-qubit hybrid quantum-classical architecture across all three datasets.
Only the dataset loading and preprocessing configuration changes.

---

## Datasets

| Disease | Category | Dataset | Patients | Status |
|---|---|---|---:|---|
| Heart disease | Cardiovascular | UCI Cleveland | 297 | Tested |
| Breast cancer | Cancer | Wisconsin Breast Cancer / scikit-learn | 569 | Tested |
| Parkinson's disease | Neurological | UCI Parkinson's voice dataset | 195 | Tested |

The same model architecture is used across all three datasets:

```text
Raw clinical features
        ↓
Training-fold feature selection
        ↓
Min-Max scaling
        ↓
Classical pre-processing layer
        ↓
4-qubit variational quantum circuit
        ↓
Classical output layer
        ↓
Disease prediction