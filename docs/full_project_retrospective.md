# Wearable Dynamics Data Model — Full Project Retrospective

## 1) Project goal and decision criterion

### Core question
Can wearable activity **patterns** predict memory performance **beyond** average activity level?

### Practical criterion used throughout
- Build a means-only baseline first (activity level only).
- Add temporal/dynamic features.
- Evaluate whether cross-validated performance improves.
- Treat negative findings as valid outcomes.

This retrospective documents all major implementation steps, what succeeded, what failed, and why decisions changed over time.

---

## 2) Chronological timeline (high-level)

1. **Phase 0 — Migration/setup:** Repository scaffolding and raw data migration.
2. **Phase 1 — Shared foundation:** Daily panel creation, coverage diagnostics, means-only baseline.
3. **Phase 2 — Dynamic features:** Temporal feature engineering + augmented model comparison.
4. **Exploratory full behavior analysis:** Extend to all fine-grained outcomes in behavior.pkl.
5. **Post hoc analysis:** Alternative models + univariate correlation screen.

---

## 3) Phase 0 — Migration and repository setup

## What was done
- Created analysis structure:
  - `data/raw/`
  - `src/`
  - `notebooks/`
  - `figures/`
- Added `.gitignore` to keep participant data and local artifacts out of version control.
- Copied required inputs into `data/raw/`:
  - 113 Fitbit participant CSVs (`BFM_AMT_0001.csv` ... `BFM_AMT_0113.csv`)
  - `behavior.pkl`
  - `behavioral_summary.pkl`
  - `survey_30.pkl`
- Intentionally excluded upstream code/notebooks, embeddings assets, and the collapsed `fitbit_7_30.pkl` artifact.

## Why this approach
The project brief requires building from raw long-format Fitbit data (not pre-collapsed summaries) so that temporal modeling is actually possible.

## Outcome
Phase 0 succeeded; the repository became self-contained for independent analysis.

---

## 4) Phase 1 — Shared foundation and baseline

## 4.1 Data loading and panel construction
Implemented in:
- `src/loading.py`
- `src/phase1.py`

### Steps performed
1. Loaded all participant CSVs into one long table.
2. Parsed datetimes and numeric values robustly.
3. Discovered full variable vocabulary before pivoting.
4. Built participant-level coverage summary (valid day counts per variable).
5. Pivoted to participant-by-date panel with explicit NaNs for missing days.

### Key numbers observed
- Raw rows loaded: **803,767**
- Variables discovered: **130**
- Participants: **113**

## 4.2 Signal selection logic
Implemented in `src/features.py` via `select_wearable_signals(...)`.

### Rule used
Selected only predefined wearable variables with:
- median_valid_days >= 300
- n_participants >= 60

### Resulting signals (final)
14 retained signals:
- bmi
- bodyfat
- cal
- cal_bmr
- distance
- fair_act_mins
- floors
- food_cal_log
- light_act_mins
- sed_mins
- steps
- very_act_mins
- water_log
- weight

### Note on “16 vs 14”
An earlier narrative reference mentioned 16 signals. Final executed thresholding retained **14** (as confirmed in `participant_mean_features.csv` and runtime logs).

## 4.3 Means-only feature matrix
Implemented by `build_mean_feature_matrix(...)` in `src/features.py`.

### What it does
For each participant and selected signal, computes the average across available days. This intentionally removes temporal structure.

### Final matrix shape
- Means-only features: **(113, 14)**

## 4.4 Baseline modeling
Implemented in `src/modeling.py` and called from `src/phase1.py`.

### Baseline pipeline
- Median imputation
- Standardization
- Ridge regression (alpha = 1.0)
- Shuffled KFold cross-validation (up to 5 folds)

### Targets used
`behavioral_summary.pkl`

### Baseline results (R² means)
- Foreign language flashcards (immediate): **-0.215**
- Foreign language flashcards (delayed): **-0.373**
- Free recall (immediate): **-0.741**
- Free recall (delayed): **-1.281**

## What worked
- End-to-end pipeline from raw long Fitbit CSVs to model-ready participant features.
- Coverage diagnostics correctly identified sparse/non-daily variables.

## What did not work (scientifically)
- Means-only model did not generalize; all baseline R² values were negative.

## Reasoning at end of Phase 1
Negative baseline was treated as a **signal diagnosis** (weak association), not as a coding bug. Proceeded to dynamic features because the project’s key question is incremental value beyond means.

---

## 5) Phase 2 — Dynamic feature extraction and augmented models

Implemented in:
- `src/dynamic_features.py`
- `src/phase2.py`
- `src/modeling.py` (custom-alpha evaluation)

## 5.1 Dynamic features added
Per participant-signal series:
- std, min, max, range
- linear trend slope and trend R²
- autocorrelation at lag 1 and lag 7
- coefficient of variation
- percent valid days

### Shape after extraction/filtering
- Dynamic features: **(110, 149)**
- Augmented features (means + dynamic, participant-aligned): **(113, 163)**

## 5.2 Model setup and reasoning change
### Attempted challenge
High dimensionality relative to sample size (p > n) increased overfitting risk.

### Adjustment made
Added custom-alpha modeling path and used stronger regularization for augmented model:
- Baseline (means-only): Ridge alpha = 1.0
- Augmented: Ridge alpha = 100.0

Rationale: stronger shrinkage is necessary when moving from 14 to 163 predictors.

## 5.3 Results (augmented vs baseline)
From `data/phase2_outputs/augmented_vs_baseline_comparison.csv`:
- Foreign language (immediate) lift: **-0.369**
- Foreign language (delayed) lift: **-7.162**
- Free recall (immediate) lift: **-89.557**
- Free recall (delayed) lift: **-36.203**

## What worked
- Dynamic extraction pipeline was stable and reusable.
- Comparison framework (baseline vs augmented) clearly quantified incremental value.

## What did not work (scientifically)
- Dynamic features did not help; all summary-outcome lifts were negative.

## Reasoning at end of Phase 2
At this point, the question became whether summary outcomes were too coarse. Instead of moving directly to functional PCA/FDA, the next step was to test the full outcome space (`behavior.pkl`).

---

## 6) Exploratory extension — Full behavior.pkl outcomes

Implemented in `src/exploratory_full_behavior.py`.

## 6.1 Initial attempt and failure
### What was attempted
Run the same baseline-vs-augmented Ridge workflow on all 54 behavior outcomes.

### Failure encountered
Modeling failed with:
- `ValueError: Input y contains infinity or a value too large for dtype('float64')`

### Root cause analysis
Inspection showed:
- Some outcomes had `inf` values (notably primacy/recency columns)
- Some outcomes were entirely NaN

## 6.2 Fix implemented
Added target filtering in the exploratory script:
1. Keep only columns with >=30 valid finite observations.
2. Replace ±inf with NaN before evaluation.

This preserved model robustness and avoided biased row-wise deletion across outcomes.

## 6.3 Secondary execution issue and fix
### Issue
Running via direct snippet import failed due package-relative imports:
- `ImportError: attempted relative import with no known parent package`

### Fix
Executed as module from project root with venv active:
- `python -m src.exploratory_full_behavior`

## 6.4 Exploratory results
After filtering:
- Outcomes tested: **40** (from original 54)
- Positive lift outcomes: **6/40**
- Negative lift outcomes: **34/40**

Lift distribution:
- Mean lift: **-789.10**
- Median lift: **-256.44**
- Min: **-7430.65**
- Max: **+74.51**

Important interpretation:
Even the largest positive lift occurred on an extremely negative baseline (for vocab temporal clustering), so absolute predictive quality remained poor.

## What worked
- Data-quality gate made full-outcome modeling executable.
- Confirmed conclusions were not an artifact of only using summary outcomes.

## What did not work (scientifically)
- Fine-grained outcomes still showed predominantly negative or unstable generalization.
- Dynamic features remained broadly unhelpful.

## Reasoning after exploratory run
The null pattern survived outcome-granularity expansion, reducing justification for a computationally heavier FDA step as a primary next move.

---

## 7) Post hoc robustness analysis

Implemented in `src/posthoc_analysis.py`.

## 7.1 Why this was run
After repeated negative Ridge results, tested whether model form (linearity) explained the null finding.

## 7.2 Analyses performed
1. Model comparison on means-only features:
   - Ridge
   - Elastic Net
   - Random Forest
2. Additional non-linear check with dynamic features:
   - Random Forest on augmented features
3. Univariate screen:
   - Spearman feature-target correlations across all valid outcomes

## 7.3 Results
### Model comparison (40 outcomes)
- Ridge: mean R² **-21.82**, positive outcomes **0/40**
- Elastic Net: mean R² **-11.79**, positive outcomes **0/40**
- Random Forest (means): mean R² **-0.199**, positive outcomes **0/40**
- Random Forest (augmented): mean R² **-0.207**, positive outcomes **1/40**

### Correlation screen
- Evaluated pairs: **560**
- Strongest association observed:
  - (`vocab learning`, `delayed`, `error distance`) vs `mean__floors`, Spearman r ≈ **0.548**
- Most strong correlations were isolated rather than coherent across related outcomes.

## What worked
- Robustness checks were straightforward to add using the same cleaned target set.
- Random Forest reduced extremity of negative R² but did not produce reliable positive generalization.

## What did not work (scientifically)
- Non-linear models did not materially change the overall conclusion.
- Dynamic features still failed to provide robust benefit.

## Reasoning after post hoc
The weak signal is unlikely to be only a linear-model failure. Evidence points to limited predictive information in current wearable features for these outcomes.

---

## 8) Consolidated “things tried that did not work”

1. **Means-only Ridge on summary outcomes**
   - Did not generalize (all negative R²).
2. **Means + dynamic Ridge on summary outcomes**
   - Performance degraded further; large negative lifts.
3. **Switching to full behavior outcomes (54) without cleaning**
   - Failed technically due inf/NaN targets.
4. **Dynamic-augmented Ridge on cleaned full outcomes**
   - Mostly negative lift; severe instability for some outcomes.
5. **Alternative models (Elastic Net, Random Forest)**
   - Did not produce robust positive predictive performance.
6. **Dynamic features with non-linear model (RF augmented)**
   - Essentially no improvement (1 weakly positive outcome out of 40).

---

## 9) Why these choices were made (decision logic)

- **Baseline-first discipline:** ensured any “dynamic” improvement was measured against a clear reference.
- **Coverage-based signal filtering:** prevented sparse variables from introducing synthetic noise.
- **Stronger regularization when dimensionality grew:** attempted to control p > n instability.
- **Outcome expansion to behavior.pkl:** tested whether null result was due to outcome coarseness.
- **Data-quality filtering (finite + minimum n):** required for valid model fitting and fair CV.
- **Post hoc model diversity:** tested whether linear assumptions were the limiting factor.

This sequence prioritized validity and interpretability over optimization.

---

## 10) Final interpretation

Across all phases and robustness checks:
- Wearable activity features (means and dynamics) showed **weak predictive utility** for the available memory/behavior outcomes.
- Dynamic temporal features did **not** reliably improve generalization beyond means-only baselines.
- Outcome granularity expansion and non-linear modeling did **not** overturn the null finding.

The project therefore yields a **robust null result** under the implemented design.

---

## 11) Artifacts produced

## Core code
- `src/loading.py`
- `src/features.py`
- `src/modeling.py`
- `src/dynamic_features.py`
- `src/plots.py`
- `src/phase1.py`
- `src/phase2.py`
- `src/exploratory_full_behavior.py`
- `src/posthoc_analysis.py`

## Outputs
- `data/phase1_outputs/`
- `data/phase2_outputs/`
- `data/exploratory_full_behavior_outputs/`
- `data/posthoc_analysis_outputs/`
- `figures/phase1_valid_day_histograms.png`

## Documentation
- `docs/phase_0_migration_summary.md`
- `docs/phase_1_foundation_summary.md`
- `docs/phase_2_dynamic_features_summary.md`
- `docs/phase_3_exploratory_full_behavior_summary.md`
- `docs/posthoc_analysis_summary.md`
- `docs/full_project_retrospective.md` (this file)

---

## 12) Suggested closeout statement

Given the complete sequence of baseline, dynamic, exploratory, and post hoc checks, the most defensible closeout is:

- The primary hypothesis (activity patterns add predictive value beyond levels) was not supported.
- The null result is reproducible within this pipeline.
- Future work should likely focus on richer covariates, different target constructions, or a substantially larger sample before expanding model complexity.
