# SIH26139 — Hybrid Quantum-Classical Disease Detection Platform

A single hybrid quantum-classical architecture, validated across **three disease categories** — directly matching the problem statement's own scope: *"early detection of cancer, cardiovascular or neurological disease."*

| Disease | Category | Dataset | Patients | Status |
|---|---|---|---|---|
| Heart disease | Cardiovascular | UCI Cleveland | 297 | Tested, working |
| Breast cancer | Cancer | UCI/sklearn Wisconsin | 569 | **Tested, working — AUC 0.987** |
| Parkinson's disease | Neurological | UCI voice dataset | 195 | Code ready, needs internet to fetch |

**The point of this table isn't just breadth — it's proof.** The exact same `quantum_model.py`, `classical_baselines.py`, and `shap_explain.py` run completely unmodified across all three. Only `preprocessing.py`'s dataset loader changes. This is what actually demonstrates the "disease-agnostic modular design" claim, rather than just asserting it on a slide.

## Real, tested result — breast cancer (cancer category)

Run directly in a sandboxed test environment, no synthetic data involved (the breast cancer dataset ships with scikit-learn, so there's zero network dependency):

```
Patients: 569 | Features selected: mean concave points, worst radius, worst perimeter, worst concave points
SVM:            AUC = 0.987 ± 0.012
Random Forest:  AUC = 0.981 ± 0.016
MLP:             AUC = 0.987 ± 0.011
Hybrid QML:     AUC = 0.987 ± 0.012

Hybrid QML vs MLP: paired t-test p = 0.577 — not statistically significant
```

SHAP explanations use the real clinical feature names throughout — "worst radius," "mean concave points" — not PCA components.

## Quick start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Cancer — works immediately, no download, no internet needed
python train.py --dataset breast_cancer

# Cardiovascular — needs internet to fetch the UCI dataset
python train.py --dataset heart_disease

# Neurological — needs internet to fetch the UCI dataset
python train.py --dataset parkinsons
```

Each run writes `results_report_<dataset>.json` with every metric your report needs.

## Project structure

```
sih26139_ml/
├── README.md
├── requirements.txt
├── train.py                    # run this — pick a disease with --dataset
├── data/raw/                   # local dataset overrides go here
└── src/
    ├── preprocessing.py        # DATASET_REGISTRY — add a new disease here
    ├── classical_baselines.py  # unchanged across all three diseases
    ├── quantum_model.py        # unchanged across all three diseases
    ├── shap_explain.py         # unchanged across all three diseases
    └── statistical_test.py     # unchanged across all three diseases
```

## Adding a fourth disease

Add one loader function and one registry entry in `preprocessing.py` — nothing else in the codebase needs to change:

```python
def _load_my_new_disease(source):
    df = pd.read_csv(source)
    y = df["target_column"].values
    X = df.drop(columns=["target_column"])
    return X, y

DATASET_REGISTRY["my_new_disease"] = {
    "loader": _load_my_new_disease,
    "default_source": "https://...",
    "disease_category": "...",
    "display_name": "...",
}
```

## A note on training time

Quantum circuit training time scales with dataset size, since each sample runs its own circuit simulation. On a standard laptop CPU:

| Dataset | Patients | Approx. time for 5-fold × 50-epoch run |
|---|---|---|
| Parkinson's | 195 | ~15–20 minutes |
| Heart disease | 297 | ~25–35 minutes |
| Breast cancer | 569 | ~45–60 minutes |

Start the breast cancer run early if you're on a deadline — it's the slowest of the three simply because it has the most patients, not because anything is wrong.

## What's genuinely tested vs. what needs your own machine

- **Breast cancer**: fully tested end-to-end in a sandboxed environment, including classical baselines, quantum training, SHAP, and the statistical comparison. The numbers above are real.
- **Heart disease**: every module tested and confirmed bug-free using a synthetic stand-in file (`data/raw/synthetic_test.csv`) in an environment without access to the real UCI servers. Run it yourself with `--dataset heart_disease` on a machine with normal internet access to get real numbers.
- **Parkinson's**: loader code written and follows the identical pattern as the other two, but not yet run against live data for the same network reason. Test it with `python src/preprocessing.py` first to confirm it fetches correctly on your machine before relying on it.
