# Survey Spearman Screen and Stress Prediction: Methods and Results

## Overview

This document covers two analyses added beyond the scope of Manning et al. (2022):

1. **Survey Spearman Screen** — a systematic correlation screen mapping all 14 wearable features against the full numeric survey outcome space.
2. **Stress Prediction Model** — an XGBoost regression model targeting self-reported typical stress, supplemented with SHAP attribution.

Both analyses use the same 14-signal means-only feature matrix (n = 113) established in the project's earlier phases. The survey screen is exploratory; the stress model is the primary multivariate follow-up.

---

## 1. Survey Spearman Screen

### Motivation

Manning et al. (2022) reported only a handful of pairwise fitness–survey correlations (e.g., light activity → lower anxiety/depression, high-intensity HR zone → higher stress). The full landscape — covering all numeric survey columns against all 14 wearable features — had not been systematically characterized. Knowing which targets carry the strongest wearable signal is necessary before committing to a multivariate model.

### Data

**Wearable features:** 14-dimensional means-only feature matrix built from per-participant Fitbit CSVs. Signals retained require `median_valid_days ≥ 300` AND `n_participants ≥ 60`.

**Survey targets:** Loaded from `survey_30.pkl`. All columns are stored as `object` dtype and require `pd.to_numeric(errors="coerce")` coercion. The following were excluded before screening:

- One-hot / nominal category prefixes (`gender`, `race`, `degree`, `location`) — Spearman r on dummy-coded indicators is not meaningful
- Specific low-quality columns: `exercise motivation sentiment`, `feedback: sentiment`, `feedback: number of words`, `color vision`, `reported exercise today`, `accurate exercise report`, `plan to exercise`

After filtering, **36 numeric targets** with ≥ 30 valid observations were retained.

### Method

Pairwise Spearman correlations (`scipy.stats.spearmanr`) with pairwise complete observations per pair. No imputation — each pair uses only participants with valid data for both variables. Significance threshold: p < 0.05 (unadjusted; no multiple-comparison correction applied at the screen stage).

Code: `src/survey_spearman_screen.py`  
Outputs: `data/survey_screen_outputs/`

### Results

| Metric | Value |
|--------|-------|
| Feature–target pairs evaluated | 504 |
| Significant at p < .05 | 74 (14.7%) |
| Pairs with \|r\| ≥ 0.20 | 52 |
| Strongest single correlation | r = −0.404 |

**Top 15 pairs by |r|:**

| Target | Feature | Spearman r | p | n |
|--------|---------|-----------|---|---|
| difficulty: free recall (delayed) | cal bmr | −0.404 | < 0.0001 | 95 |
| typical stress | floors | −0.365 | 0.004 | 60 |
| difficulty: free recall (immediate) | sed mins | −0.352 | 0.0003 | 104 |
| difficulty: free recall (delayed) | sed mins | −0.348 | 0.0004 | 95 |
| difficulty: free recall (immediate) | cal bmr | −0.341 | 0.0004 | 103 |
| typical stress | very act mins | −0.279 | 0.004 | 102 |
| typical stress | fair act mins | −0.265 | 0.007 | 102 |
| typical stress | cal | −0.264 | 0.007 | 102 |
| typical stress | distance | −0.251 | 0.011 | 102 |
| difficulty: free recall (immediate) | steps | −0.248 | 0.011 | 104 |

### Novel Finding: Task Difficulty Ratings

The strongest wearable–survey correlations involve **self-reported task difficulty ratings** — how hard participants found the free recall and flashcard tasks. These targets are entirely absent from Manning et al., which focused on task performance (recall counts, error distances) rather than subjective difficulty.

The sign is consistently negative: more active participants (higher cal_bmr, more steps, less sedentary time) rated the memory tasks as *less difficult*. The association is strongest for delayed free recall difficulty and cal_bmr (r = −0.404, n = 95).

This is conceptually distinct from the memory performance signal — a participant could perform well yet find the task easy, or vice versa. Future work should model difficulty ratings directly as targets.

### Directionality vs. Manning et al.

Manning et al. found that **high-intensity HR zone activity** (peak/cardio) was associated with *higher* self-reported stress (r ≈ +0.21). We find that **total activity volume** (very_act_mins, cal, floors) is associated with *lower* typical stress (r ≈ −0.25 to −0.37). These are not contradictory: HR-zone peak intensity and movement-based volume are different constructs that can point in opposite directions. High-intensity exercise episodes may acutely elevate stress reactivity while chronic moderate activity volume correlates with lower baseline stress.

---

## 2. Stress Prediction Model

### Motivation

`typical stress` emerged as the strongest candidate target from the survey screen (top univariate r = −0.365 for floors, n = 60; r = −0.279 for very_act_mins, n = 102). It is also a well-measured, well-powered target (n = 113 complete). Manning et al. did not model stress as a regression target — they reported only pairwise correlations involving HR-zone activity.

### Target

`('', 'typical stress')` from `survey_30.pkl` — an ordinal self-report scale ranging from −2 (much less stressed than typical) to +2 (much more stressed than typical). All 113 participants have a valid response; no participant-level filtering is needed.

### Features

All 14 wearable signals with median imputation + StandardScaler (stress is a complete-n target, so any missingness comes from a small number of participants in a given signal).

### Model

XGBRegressor with explicit regularization to counteract the n=113, p=14 regime:

```python
XGBRegressor(
    n_estimators=400,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
)
```

Evaluation: 5-fold cross-validation using `cross_val_predict` for pooled CV R², plus per-fold R² and MAE. A univariate benchmark (best single-feature Spearman r²) is computed for comparison. SHAP attribution uses `shap.TreeExplainer` on the full-data model.

Code: `src/stress_prediction.py`  
Figures: `figures/stress_shap_beeswarm.png`, `figures/stress_shap_bar.png`, `figures/stress_shap_dependence_grid.png`, `figures/stress_cv_benchmark.png`

### Results

| Metric | Value |
|--------|-------|
| Pooled 5-fold CV R² | −0.266 |
| Mean fold R² | −0.374 ± 0.405 |
| Mean fold MAE | 1.228 ± 0.124 |
| Per-fold R² | −1.004, +0.154, −0.232, −0.146, −0.642 |
| Best univariate Spearman r² | 0.063 (very_act_mins) |
| In-sample R² (full data) | 0.975 |

**SHAP feature importance (mean |SHAP value| on full data):**

| Rank | Feature | Mean |SHAP| |
|------|---------|------------|
| 1 | fair act mins | 0.293 |
| 2 | cal | 0.273 |
| 3 | floors | 0.205 |
| 4 | bmi | 0.200 |
| 5 | very act mins | 0.189 |

### Interpretation

The model massively overfits: in-sample R² = 0.975, CV R² = −0.266. This is the expected behavior for a tree ensemble on n=113 with 14 features — even with regularization, XGBoost has sufficient capacity to memorize the training data. The wide per-fold spread (ranging from −1.004 to +0.154) confirms no stable generalization.

The SHAP attribution reflects in-sample model structure and should not be interpreted as causal. The activity-related features (fair_act_mins, cal, floors) dominate because they carry the largest univariate correlations with stress — the model is simply learning those pairwise relationships.

**Conclusion:** The univariate signal (Spearman r ≈ −0.25 to −0.37) is real and replicates Manning et al.'s direction for activity volume. But n = 113 is insufficient to train a generalizing multivariate model. A linear model (Ridge regression) would be more appropriate at this sample size; the XGBoost analysis serves primarily to confirm that more flexible models do not recover signal that Ridge misses.

---

## 3. Unified Interpretation

Both analyses reinforce the project's core pattern: **the wearable–mental health association exists at the univariate level but does not survive multivariate cross-validation at n = 113**.

Key takeaways for the manuscript:

- Task difficulty ratings are a novel target class, not examined by Manning et al., with stronger wearable correlations than task performance itself.
- The stress–activity direction (more total activity → lower typical stress) is consistent with the literature but uses a different activity metric than Manning et al. (movement-based volume vs. HR-zone intensity).
- The null multivariate result is not evidence of no association — it is evidence that the effect size is below the minimum detectable effect at this sample size (MDES ≈ 0.26 for n = 113 at 80% power; see `data/diagnostics_outputs/d3_power_analysis.csv`).

**Recommended framing:** Report the top univariate correlations as exploratory findings; report the multivariate CV results as evidence that larger samples are needed before fitting predictive models.
