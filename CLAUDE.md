# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the pipeline

All modules must be run as Python packages from the project root with the venv active. Direct script execution fails due to relative imports.

```bash
# Environment setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run analysis phases
python -m src.phase1                          # means-only baseline (Phase 1)
python -m src.phase2                          # dynamic feature augmentation (Phase 2)
python -m src.exploratory_full_behavior       # full behavior.pkl outcomes (Phase 3)
python -m src.posthoc_analysis                # alternative models + Spearman screen
python -m src.shap_vocab_analysis             # SHAP analysis of the floors/vocab finding

# Run diagnostics
python -m src.diagnostics_d1                  # synthetic-target sanity check (run first if CV harness is suspect)
python -m src.diagnostics_d2_mechanism        # Fold 4 mechanism analysis
python -m src.diagnostics_d3_robust_metrics_detailed  # robust metrics (Pearson + MAE)
python -m src.diagnostics_d3_cleaned_plus_power       # correlations on cleaned data + power analysis
python -m src.data_cleaning                   # data quality inspection and winsorization
python -m src.generate_diagnostic_reference   # final diagnostic reference report
```

## Data layout

Raw data lives in `data/raw/` (gitignored, not committed). Required files:
- `BFM_AMT_0001.csv` … `BFM_AMT_0113.csv` — per-participant long-format Fitbit CSVs (columns: `datetime`, `variable`, `value`)
- `behavior.pkl` — full behavioral outcome table (54 columns)
- `behavioral_summary.pkl` — coarse summary outcomes (8 columns used in Phases 1–2)
- `survey_30.pkl` — mental health / demographics survey

**Do not use `fitbit_7_30.pkl`** — that is the original paper's pre-collapsed 7/30-day summary artifact. Building on it silently reduces the project to the original paper's analysis.

Phase outputs are saved under `data/phase1_outputs/`, `data/phase2_outputs/`, `data/exploratory_full_behavior_outputs/`, `data/posthoc_analysis_outputs/`, `data/diagnostics_outputs/`, `data/cleaned/`, and `data/shap_outputs/`.

## Architecture

The pipeline has a strict layered structure. All phases share the same upstream loading and selection logic; only the feature representation and model evaluation vary.

```
src/loading.py          → raw CSV → long table → daily panel (participant × date × variable)
src/features.py         → panel → means-only feature matrix (113 × 14); signal selection logic
src/dynamic_features.py → panel → dynamic feature matrix (110 × 149); augmented matrix
src/modeling.py         → feature matrix + targets → Ridge CV metrics (R², MAE, RMSE)
src/data_cleaning.py    → feature matrix → winsorized copy + impossible-value audit
src/phase1.py           → orchestrates Phase 1 end-to-end
src/phase2.py           → orchestrates Phase 2 (means + dynamic, baseline vs. augmented)
src/exploratory_full_behavior.py → Phase 3: same pipeline on all 40 valid behavior.pkl outcomes
src/posthoc_analysis.py      → Ridge / ElasticNet / RandomForest comparison + Spearman screen
src/shap_vocab_analysis.py   → focused SHAP analysis on the floors/vocab-error-distance finding
src/diagnostics_*.py         → targeted validity checks (synthetic targets, fold mechanics, power)
src/plots.py            → figure helpers (e.g., valid-day histograms)
src/visualizations.py   → additional visualization utilities
```

**Participant ID convention:** files are sorted lexicographically (`BFM_AMT_0001.csv` → `P0`, …, `BFM_AMT_0113.csv` → `P112`). The same positional indexing is applied to pickle targets. This join is order-dependent — do not change sort order.

**Signal selection thresholds** (in `features.py`): a signal is included only if `median_valid_days >= 300` AND `n_participants >= 60`. This yields 14 retained signals (steps, cal, distance, floors, sed_mins, light_act_mins, fair_act_mins, very_act_mins, cal_bmr, bmi, bodyfat, weight, water_log, food_cal_log).

**Modeling pipeline** (in `modeling.py`): median imputation → StandardScaler → Ridge → shuffled KFold (up to 5 folds). Baseline uses `alpha=1.0`; augmented model uses `alpha=100.0` to account for the jump from 14 to 163 predictors.

**Target filtering rule** (in `exploratory_full_behavior.py` and `posthoc_analysis.py`): keep only outcome columns with ≥ 30 finite observations; replace ±inf with NaN before fitting. This is required — several `behavior.pkl` columns contain inf values that crash `cross_validate`.

## Project status and findings

This project has completed Phases 0–3 plus post hoc analysis. **The overall finding is a robust null result**: wearable activity patterns (dynamic features) do not improve cross-validated prediction of memory outcomes beyond means-only baselines. All Ridge R² values were negative; alternative models (ElasticNet, Random Forest) confirmed the weak signal rather than overturning it. See `docs/full_project_retrospective.md` for the complete decision log.

The planned Approach B (Functional Data Analysis / functional PCA) has not been started and is only warranted if new evidence suggests signal exists.

A focused SHAP analysis (`src/shap_vocab_analysis.py`) examined the dataset's strongest observed association: `mean__floors` vs. delayed vocab error distance (Spearman r = 0.55). Key design decisions: (1) only the n=26 participants with altimeter-equipped Fitbits have floors data within the n=62 with valid vocab scores, so that subset is used without imputation; (2) only the 4 features with complete data in that subset are used (floors, weight, distance, very_act_mins); (3) LOO-CV R² = 0.008, confirming no generalizable predictive signal. See `docs/shap_analysis_summary.md`.
