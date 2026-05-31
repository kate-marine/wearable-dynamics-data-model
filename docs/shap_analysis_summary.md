# SHAP Analysis: Floors Climbed and Vocabulary Learning Error

## Motivation

The post hoc Spearman correlation screen (`src/posthoc_analysis.py`) evaluated every pairwise association between the 14 means-only wearable features and the 40 valid behavioral outcomes in `behavior.pkl`. The strongest observed association was:

> `mean__floors` vs. `('vocab learning', 'delayed', 'error distance')`, Spearman *r* = 0.55

This module (`src/shap_vocab_analysis.py`) investigates that association in depth using a Random Forest model and SHAP attribution.

---

## Methods

### Target variable

**Delayed vocabulary error distance** (`behavior.pkl`, column `('vocab learning', 'delayed', 'error distance')`): a scalar score measuring how far participants' recalled foreign-language vocabulary was from the correct answer on a delayed test. Higher values indicate worse recall (larger errors). 62 of 113 participants had valid (finite, non-missing) scores on this outcome.

### Wearable features and sample selection

The 14 wearable features used throughout this project are participant-level means computed from year-long daily Fitbit data (see Phase 1 for signal selection criteria). Of the 62 participants with valid delayed vocab scores:

- **36 had no floors data** — their Fitbit devices lacked an altimeter, so `mean__floors` was NaN.
- **26 had complete floors data** — these participants form the analysis subset.

This is a device-capability limitation, not random missingness. Participants who own altimeter-equipped Fitbits are a self-selected subset. All analysis is restricted to these **n = 26 participants**.

Within the n = 26 subset, four of the 14 features have no missing values: **floors**, **weight**, **distance**, and **very_act_mins**. These four are used directly without imputation to avoid adding synthetic signal.

### Model

A Random Forest regressor (500 trees, `min_samples_leaf=3`, `max_features="sqrt"`, `random_state=42`) was fit on the four complete features for all 26 participants. SHAP values were computed using `shap.TreeExplainer`.

Predictive generalization was assessed with **leave-one-out cross-validation (LOO-CV)**: in each fold, 25 participants train the model and the 26th is the held-out prediction. R² is computed over all 26 pooled held-out predictions.

### Figures

All figures are saved to `figures/`:

| File | Content |
|---|---|
| `shap_scatter_floors_vs_error_distance.png` | Raw scatter of floors vs. error distance with OLS line and Spearman *r* |
| `shap_correlation_landscape.png` | All-feature Spearman *r* values using pairwise complete observations, with n per feature |
| `shap_beeswarm.png` | SHAP beeswarm across all four features (n=26, in-sample model) |
| `shap_dependence_floors.png` | SHAP value for floors vs. floors level, colored by body weight |

---

## Results

### Correlation landscape

Pairwise Spearman correlations computed against the full n = 62 (using only observations with valid data for each feature):

| Feature | Spearman *r* | *p* | n |
|---|---:|---:|---:|
| floors | +0.548 | 0.004 | 26 |
| light act mins | +0.142 | 0.305 | 54 |
| steps | +0.123 | 0.367 | 56 |
| fair act mins | +0.120 | 0.374 | 57 |
| distance | +0.111 | 0.422 | 55 |
| very act mins | +0.044 | 0.747 | 56 |
| weight | +0.022 | 0.869 | 57 |
| *(remaining 7 features)* | < ±0.09 | > 0.54 | 52–55 |

Floors is the only feature with a statistically significant association (*p* = 0.004). All other features show near-zero correlations. Critically, the floors *r* is based on n = 26 (not n = 62), because only participants with altimeter-equipped devices contribute to that correlation.

### Within the n = 26 subset

Among the 26 participants with floors data, a second significant association appears:

| Feature | Spearman *r* | *p* | n |
|---|---:|---:|---:|
| floors | +0.548 | 0.004 | 26 |
| weight | −0.400 | 0.043 | 26 |
| distance | +0.285 | 0.158 | 26 |
| very act mins | +0.195 | 0.339 | 26 |

`mean__weight` is negatively correlated with error distance (r = −0.40, *p* = 0.043): heavier participants made smaller errors. This association is not present when computed over the full n = 57 with valid weight data (r = +0.02), suggesting it is specific to — or an artifact of — the floors-tracking subsample.

### Predictive performance

| Metric | Value |
|---|---|
| LOO-CV R² (pooled) | 0.008 |
| In-sample R² | 0.543 |

The LOO-CV R² of 0.008 indicates essentially no generalizable predictive signal. The large gap between in-sample and out-of-sample R² reflects overfitting on the small n = 26 sample with 4 features.

### SHAP attribution

With the in-sample model, floors and weight are the dominant contributors to predictions:

| Feature | Mean \|SHAP\| |
|---|---:|
| floors | 0.269 |
| weight | 0.181 |
| very act mins | 0.053 |
| distance | 0.037 |

The beeswarm shows that high floors values (red) consistently push predictions toward higher error distance (positive SHAP), consistent with the positive Spearman *r*. High weight values push predictions in the opposite direction (negative SHAP). The dependence plot for floors shows a roughly monotonic positive relationship between floors level and its SHAP value, with no clear interaction with body weight.

---

## Interpretation

The floors–vocab correlation is the strongest wearable-behavior association observed in this dataset. Its magnitude (r = 0.55) is notable, but several considerations limit its interpretability:

1. **Small n.** The correlation is computed on 26 participants. At this sample size, a correlation of 0.55 has a 95% confidence interval of approximately [0.19, 0.77] (Fisher *z* transformation). The finding is statistically significant but imprecisely estimated.

2. **Device self-selection.** Participants with altimeter-equipped Fitbits are not a random subsample. Systematic differences between device types — or between the kinds of people who own them — could produce spurious correlations. The weight association appearing only within this subset (not in the broader n = 57) is consistent with a self-selection confound rather than a genuine independent relationship.

3. **No multiple-comparison correction.** The Spearman screen evaluated 560 feature-outcome pairs. At nominal α = 0.05, approximately 28 spurious associations are expected by chance. The floors finding would need to survive correction (e.g., Benjamini–Hochberg) to be claimed as robust.

4. **No predictive generalization.** LOO-CV R² = 0.008 confirms the model does not generalize, consistent with the overall null result from Phases 1–3.

**Conclusion:** the floors–vocab error distance association is the strongest signal in the dataset and warrants reporting, but should be framed as a descriptive observation requiring replication in a larger, device-matched sample — not as evidence of a predictive relationship.
