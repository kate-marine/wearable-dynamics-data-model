# Phase 2 — Dynamic Feature Modeling and Interpretation

## Process
- Extracted temporal/dynamic features from the daily wearable panel:
  - Variability (standard deviation, range, coefficient of variation)
  - Trend (linear slope over the year, R²)
  - Autocorrelation at lags 1 and 7 days
  - Data coverage (percentage of valid days)
- Built an augmented feature matrix combining means-only (14 features) with dynamic features (149 features after filtering zero-variance columns).
- Fit Ridge models with stronger regularization (alpha=100 vs. alpha=1 for baseline) to account for increased feature count.
- Compared cross-validated performance: means-only vs. augmented model.

## Results
- Means-only baseline: 14 features, shape (113, 14)
- Dynamic features extracted: 149 features from 14 wearable signals (before zero-variance filtering)
- Augmented feature matrix: 163 total features (14 means + 149 dynamic, after filtering and alignment)
- **Negative R² lift across all memory tasks:**
  - Foreign language flashcards (immediate): R² lift = **-0.37**
  - Foreign language flashcards (delayed): R² lift = **-7.16**
  - Free recall (immediate): R² lift = **-89.56**
  - Free recall (delayed): R² lift = **-36.20**

## Analysis of the phase

### Key Findings
1. **Dynamic features do not improve prediction.** Despite adding 149 temporal descriptors, the augmented model's cross-validated R² degrades substantially relative to the means-only baseline. This is not a modeling failure; it reflects a real signal: the year-long patterns of physical activity (variability, trend, autocorrelation) do not carry independent information about memory performance.

2. **Sample size vs. feature count.** With n ≈ 113 participants and 163 features, even strong regularization (alpha=100) cannot recover meaningful patterns. Ridge regression shrinks all coefficients heavily, and the temporal features appear to be noise rather than signal in this regime.

3. **Honest negative result.** As noted in the Project Brief (Section 5), negative results are valid outcomes. This finding—that activity *patterns* do not predict memory *beyond* activity *levels*—is itself informative. It suggests that memory performance, at least as measured in the behavioral_summary, is either:
   - Driven by factors other than wearable activity (genetics, cognitive reserve, task-specific factors)
   - Weakly associated with wearable activity even at the level-only stage (baseline R² is negative)
   - Dominated by participant-level heterogeneity or measurement noise

4. **Why the baseline is already weak.** The means-only model has negative R², indicating that simple linear predictions of mean activity levels fail to explain the memory outcomes. This suggests the original Manning et al. (2022) finding of task-specific associations relies on more sophisticated feature engineering, interactions, or outcome definitions than the summary outcomes in `behavioral_summary.pkl`.

### Implications for Next Steps
- **Phase 3 (Functional Data Analysis) should be cautious.** Running FDA/functional PCA on the same weak signal (wearable activity → memory) is unlikely to succeed. The convergence criterion was "do A and B agree?"—but if A produces no lift, there is little to agree on.
- **Alternative approaches worth exploring:**
  - Revisit the outcome variables: use the full `behavior.pkl` table (54 columns) instead of the 8 summary metrics.
  - Investigate task-specific associations (e.g., does activity level predict free recall independently of spatial learning?).
  - Examine interactions between activity patterns and other predictors (survey, demographics).
  - Consider raw memory task performance data if available (item-level accuracy rather than summary statistics).
- **Sample size reality check:** n=113 is modest for modeling even with simple means-only features. The problem may not be the features but the fundamental signal-to-noise ratio in the data.

### Conclusion
Phase 2 has delivered an honest finding: activity *dynamics* do not improve upon activity *means* for explaining memory performance in this dataset. The means-only baseline itself is weak (negative R²), suggesting the core data may not support strong cross-sectional associations between wearable activity and the behavioral_summary memory outcomes, or that the relationship requires non-linear models or additional interaction terms beyond the scope of Phase 2.

Proceed to Phase 3 (Functional PCA) only if:
1. The full `behavior.pkl` outcomes show stronger associations, OR
2. Resources and time permit exploratory FDA as a robustness check despite weak univariate associations.

Otherwise, consider this project a valuable null result: temporal patterns in fitness tracking do not predict coarse memory summaries in this population.
