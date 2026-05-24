# Phase 0 — Migration and Repository Setup

## Process
- Created the project scaffold required by the brief: [data/raw/](data/raw/), [src/](src/), [notebooks/](notebooks/), and [figures/](figures/).
- Added [`.gitignore`](.gitignore) to keep all participant data under `data/` out of version control, along with standard Python, notebook, and editor artifacts.
- Copied the required source inputs from the upstream repo into [data/raw/](data/raw/):
  - 113 per-participant Fitbit CSVs (`BFM_AMT_0001.csv` through `BFM_AMT_0113.csv`)
  - `behavior.pkl`
  - `behavioral_summary.pkl`
  - `survey_30.pkl`
- Intentionally did not copy the upstream code/notebooks, `embeddings/opt/`, or `fitbit_7_30.pkl`.

## Results
- Phase 0 is complete and the repo is now self-contained for analysis work.
- The raw data landed in the expected long-format structure with 113 participant files.
- The repository now has the minimal filesystem layout needed for the analysis pipeline.

## Analysis of the phase
- This phase succeeded because the project now starts from the raw daily Fitbit records rather than the collapsed 7/30 summary used in the original paper.
- Keeping the upstream notebooks and embeddings out of the repo preserves the project’s independent implementation and avoids accidental scope drift.
- The `.gitignore` decision is essential: the working analysis can proceed locally without risking redistribution of participant data.
