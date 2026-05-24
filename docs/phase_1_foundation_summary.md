# Phase 1 — Shared Foundation

## Process
- Loaded all 113 raw participant CSVs into one long-format table.
- Scanned the full variable vocabulary before pivoting.
- Built a participant-level variable coverage summary to quantify valid-day counts and identify sparse signals.
- Pivoted the wearable data into a participant-by-date panel with explicit missing days.
- Built a means-only feature matrix from the daily wearable signals.
    - In features.py, build_mean_feature_matrix takes the daily panel and for each participant computes the average of each selected wearable signal across all available days (ignoring missing values).

    - Input shape: participant × day × signal
    - Output shape: participant × signal means (columns like mean__steps, mean__sed_mins, etc.)
    - This intentionally removes temporal pattern information, so it’s a pure “activity level” baseline

- Fit a cross-validated Ridge baseline against `behavioral_summary.pkl`.
    - For each behavioral outcome column, the code:
        - aligns participants with available target values,
        - runs a pipeline (median imputation → standardization → Ridge(alpha=1.0)),
        - evaluates with shuffled 5-fold CV (or fewer folds if sample is small),- reports mean/std for R^2, MAE, and RMSE.
- Generated a coverage figure at [figures/phase1_valid_day_histograms.png](figures/phase1_valid_day_histograms.png) and saved tabular outputs under [data/phase1_outputs/](data/phase1_outputs/).

**"Means-only"** means the model gets only average levels of each wearable signal per participant, and no time-pattern features.

So examples are:
- mean__steps
- mean__sed_mins
- mean__distance
- mean__weight

What it excludes:
- day-to-day variability (std, IQR)
- trends/slopes over time
- weekday vs weekend differences
- autocorrelation/regularity/burstiness

It is the **baseline amount-only** model. Then in Phase 2 will add dynamic features and test whether they improve prediction beyond these averages.

## Results
- Raw input size: 803,767 rows across 130 observed variables.
- The full variable vocabulary is highly mixed:
  - 96 variables appear on only one day per participant.
  - 15 wearable signals have essentially full year coverage.
  - 17 more signals are moderately covered.
  - 2 sleep signals are sparse enough to require caution.
- The first-pass wearable feature set was trimmed from 34 signals to the 16 most robust daily signals.
- The resulting means-only feature matrix has shape $(113, 16)$.
- Cross-validated baseline performance remains weak, with negative $R^2$ on the available summary outcomes.

## Analysis of the phase
- The coverage diagnostic is the key result from Phase 1: the raw data are structurally uneven, so feature inclusion must be conservative.
- The new threshold excludes sparsely observed signals such as the sleep measures while retaining the daily activity/weight variables that are most consistently measured.
- The negative baseline $R^2$ is not a failure of the pipeline; it indicates that simple level features are not yet sufficient to explain the memory outcomes well.
- This strengthens the rationale for Phase 2: any improvement will need to come from dynamic features rather than from a means-only summary.
- The next analysis step should use the filtered signal set as the foundation for temporal features, not the full 34-variable set.
