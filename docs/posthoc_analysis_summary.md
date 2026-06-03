# Post Hoc Analysis Summary

## Why this analysis

After Phases 1–3 showed consistently negative or weak cross-validated $R^2$ for Ridge models, the next question was whether the result was due to model form rather than genuinely weak signal.

This post hoc check asked two things:

1. Do alternative models perform better on the same wearable features?
2. Do any individual wearable features show clear monotonic relationships with behavior outcomes?

## What was tested

### 1) Alternative model comparison
I compared three models on the **means-only** wearable feature matrix:

- Ridge regression
- Elastic Net
- Random Forest

I also ran a **Random Forest on the augmented feature set** to check whether dynamic features help under a non-linear model.

### 2) Univariate correlation screen
I computed Spearman correlations for every wearable feature / behavior outcome pair using the 40 valid behavior outcomes from behavior.pkl.

## Results

### Model comparison
| Model | Mean $R^2$ | Median $R^2$ | Positive outcomes |
|---|---:|---:|---:|
| Ridge | -21.82 | -10.36 | 0 / 40 |
| Elastic Net | -11.79 | -3.59 | 0 / 40 |
| Random Forest | -0.20 | -0.17 | 0 / 40 |
| Random Forest + dynamic features | -0.21 | -0.17 | 1 / 40 |


- Random Forest is much less negative than Ridge, so the relationship is not strictly ruled out as linear-only
- All mean $R^2$ values were still negative.
- Adding dynamic features does **not** improve the Random Forest result.
- Only one outcome becomes slightly positive under the augmented Random Forest (not enough to support a robust signal)

### Strongest univariate relationships
The largest absolute Spearman correlations were modest, with the strongest around:

- `(vocab learning, delayed, error distance)` vs `mean__floors`: $r \approx 0.55$
- `(naturalistic recall, immediate, proportion correct)` vs `mean__weight`: $r \approx 0.32$
- Several vocab-learning error/correctness outcomes vs `mean__cal` and `mean__cal_bmr`: $r \approx 0.28$–$0.31$

### Interpretation
These are not negligible, but they are scattered and not obviously consistent across related outcomes. The pattern looks more like isolated associations than a stable predictive signal.

## Conclusion

The post hoc analysis does **not** overturn the main conclusion from Phases 1–3:

- Wearable activity levels show weak predictive power for the behavior outcomes.
- Dynamic temporal features do not provide a meaningful improvement.
- Even when the model is made non-linear, generalization remains poor.

The results suggest the problem is not only the linearity of Ridge regression. The signal itself appears limited, noisy, or highly outcome-specific.

## Output files

Generated in `data/posthoc_analysis_outputs/`:

- `model_comparison_all_outcomes.csv`
- `feature_target_spearman_screen.csv`

## Practical takeaway

If the project continues, the most useful next step would be **interpretive** rather than **predictive** work:

- examine why a few outcomes show modest correlations,
- check whether specific wearable domains dominate those associations,
- or conclude the study with a robust null result.
