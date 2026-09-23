"""
SIH26139 -- Statistical significance testing.

Compares the hybrid quantum model's per-fold AUC scores against the
best classical baseline's per-fold AUC scores using a paired test.
Reporting this p-value honestly -- whether or not it favours the
quantum model -- is what separates a research-grade comparison from
a single lucky number.
"""

from scipy import stats


def compare_models(fold_scores_a: list, fold_scores_b: list,
                    name_a: str = "Hybrid QML",
                    name_b: str = "Best classical baseline"):
    """
    Runs both a paired t-test and a Wilcoxon signed-rank test (the
    non-parametric alternative, more appropriate with only 5-10 folds)
    and reports both so the comparison is defensible either way.
    """
    t_stat, t_pvalue = stats.ttest_rel(fold_scores_a, fold_scores_b)

    try:
        w_stat, w_pvalue = stats.wilcoxon(fold_scores_a, fold_scores_b)
    except ValueError:
        # Wilcoxon fails if all paired differences are zero, or with
        # too few folds -- fall back gracefully.
        w_stat, w_pvalue = None, None

    result = {
        "model_a": name_a,
        "model_b": name_b,
        "mean_a": sum(fold_scores_a) / len(fold_scores_a),
        "mean_b": sum(fold_scores_b) / len(fold_scores_b),
        "paired_t_test": {"statistic": float(t_stat), "p_value": float(t_pvalue)},
        "wilcoxon_test": {
            "statistic": float(w_stat) if w_stat is not None else None,
            "p_value": float(w_pvalue) if w_pvalue is not None else None,
        },
        "significant_at_0.05": bool(t_pvalue < 0.05),
    }
    return result


def print_comparison(result: dict):
    print(f"\n  {result['model_a']} vs {result['model_b']}")
    print(f"    Mean AUC -- {result['model_a']}: {result['mean_a']:.4f}")
    print(f"    Mean AUC -- {result['model_b']}: {result['mean_b']:.4f}")
    print(f"    Paired t-test p-value:  {result['paired_t_test']['p_value']:.4f}")
    if result["wilcoxon_test"]["p_value"] is not None:
        print(f"    Wilcoxon p-value:       {result['wilcoxon_test']['p_value']:.4f}")
    verdict = "statistically significant" if result["significant_at_0.05"] else "NOT statistically significant"
    print(f"    Verdict (alpha=0.05):   difference is {verdict}")
