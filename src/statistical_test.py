"""
SIH26139 -- Statistical comparison of model performance.

Compares paired outer-CV AUC scores using:
1. Paired t-test
2. Wilcoxon signed-rank test

Both tests are reported separately.
"""

import numpy as np

from scipy.stats import ttest_rel, wilcoxon


def compare_models(
    scores_a,
    scores_b,
    name_a="Model A",
    name_b="Model B"
):
    """
    Compare two models using paired outer-CV scores.
    """

    scores_a = np.asarray(
        scores_a,
        dtype=float
    )

    scores_b = np.asarray(
        scores_b,
        dtype=float
    )

    if len(scores_a) != len(scores_b):
        raise ValueError(
            "Both models must have the same "
            "number of CV fold scores."
        )

    if len(scores_a) < 2:
        raise ValueError(
            "At least two paired fold scores "
            "are required."
        )

    # Paired t-test
    t_statistic, t_p_value = ttest_rel(
        scores_a,
        scores_b
    )

    # Wilcoxon signed-rank test
    try:

        w_statistic, w_p_value = wilcoxon(
            scores_a,
            scores_b,
            zero_method="wilcox",
            alternative="two-sided"
        )

    except ValueError:

        w_statistic = 0.0
        w_p_value = 1.0

    # Mean AUC values
    mean_a = float(
        np.mean(scores_a)
    )

    mean_b = float(
        np.mean(scores_b)
    )

    mean_difference = (
        mean_a - mean_b
    )

    # Return complete result
    return {
        "model_a": name_a,
        "model_b": name_b,

        "mean_auc_a": mean_a,
        "mean_auc_b": mean_b,

        "mean_auc_difference_a_minus_b":
            float(mean_difference),

        "paired_t_test": {
            "statistic":
                float(t_statistic),

            "p_value":
                float(t_p_value),

            "significant_at_0.05":
                bool(t_p_value < 0.05)
        },

        "wilcoxon_test": {
            "statistic":
                float(w_statistic),

            "p_value":
                float(w_p_value),

            "significant_at_0.05":
                bool(w_p_value < 0.05)
        }
    }


if __name__ == "__main__":

    qml_scores = [
        0.90,
        0.88,
        0.91,
        0.89,
        0.90
    ]

    classical_scores = [
        0.92,
        0.90,
        0.93,
        0.91,
        0.92
    ]

    result = compare_models(
        qml_scores,
        classical_scores,
        name_a="Hybrid QML",
        name_b="Classical Model"
    )

    print("Statistical comparison:")

    print(result)