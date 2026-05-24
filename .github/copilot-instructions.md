# Project Guidelines

## Code Style
- No code style conventions are documented yet in this repo.
- New code should follow the structure implied by the brief: small, explicit modules for loading, feature engineering, modeling, and plotting.
- Keep implementation readable and reproducible; prefer clear transformations over clever one-liners.

## Architecture
- The project asks whether activity patterns predict memory beyond average activity levels.
- Treat the means-only baseline as mandatory; any lift must come from dynamic features, not re-discovering activity level.
- Shared foundation first: load raw long-format Fitbit CSVs, pivot to a participant-by-date panel, assess missingness, build a feature matrix, and fit a means-only baseline.
- Primary path: temporal/dynamic feature extraction plus interpretable ML and SHAP.
- Secondary path: functional PCA / FDA only after the primary path is working.

## Build and Test
- No build/test scripts or automated test suite are documented yet.
- Documented setup commands:
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`
- After the raw CSVs are in place, inspect the full variable vocabulary before pivoting:
  - the Python one-liner in [PROJECT_BRIEF.md](PROJECT_BRIEF.md) that scans `data/raw/*.csv` for the `variable` column.

## Project Conventions
- Raw participant data belongs in `data/raw/` and should be excluded from version control.
- Build the full variable vocabulary across participants before pivoting so columns align.
- Use explicit NaNs for missing days after pivoting.
- Treat `behavior.pkl`, `behavioral_summary.pkl`, and `survey_30.pkl` as separate inputs from the raw Fitbit CSVs.
- Do not use `data/preprocessed/fitbit_7_30.pkl`; it would reduce the project to the original paper’s collapsed summary.
- Do not copy upstream code/notebooks or the `embeddings/opt/` folder into this repo.

## Integration Points
- Source data comes from the upstream repository at https://github.com/ContextLab/brainfit-paper.
- Needed inputs from that source: per-participant CSVs from `raw_formatted/`, plus `behavior.pkl`, `behavioral_summary.pkl`, and `survey_30.pkl`.
- Upstream notebooks `fitness_data.ipynb` and `reverse_correlation_analysis.ipynb` are reference-only for parsing/file-format context.
- Likely analysis tools mentioned in the docs: `catch22`, `tsfresh`, XGBoost or LightGBM, SHAP, and functional PCA/FDA.

## Security
- Do not redistribute participant data in this repo.
- Keep raw Fitbit CSVs and other sensitive inputs local and gitignored.
- Respect the terms of the original study and source repository.
- Avoid committing large derived artifacts unless they are clearly needed for the analysis.

## Current Status
- This repository is still at the documentation / scaffolding stage.
- No source code, notebooks, or project automation are present yet.
- Start with Phase 0 migration and repository setup before analysis work.
