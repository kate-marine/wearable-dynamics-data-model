# Project Brief: Activity Patterns and Memory

This document is a handoff brief. It describes the project, the data, the analysis
plan, and where things currently stand, so work can continue from here.

---

## 1. The core idea

**Question:** Do *patterns* of physical activity over a year predict memory
performance *beyond* what *average* activity levels capture?

This project builds on the dataset from Manning et al. (2022), "Fitness tracking
reveals task-specific associations between memory, mental health, and physical
activity" (*Scientific Reports* 12:13822). In that study, 113 participants shared
a year of Fitbit data, then completed four memory tasks (free recall, naturalistic
story recall, foreign-language flashcards, spatial learning) plus a mental-health
and demographics survey.

The original paper characterized each participant's activity mostly through
**weekly averages** and a **recent-vs-baseline ratio** (7-day mean / preceding
30-day mean). That collapses a year of behavior into level-based summary statistics.

The gap this project targets: two people with the *same* average activity can have
very *different* patterns (steady vs. bursty; consistent vs. weekend-only). The
paper never tests whether those patterns carry independent information. **We do.**

The whole project hinges on one comparison:
- a **means-only baseline** model (level features, like the original paper used), vs.
- an **augmented** model that adds **dynamic features** (variability, autocorrelation,
  weekday/weekend structure, trend, regularity, etc.).

Does the augmented model predict memory better than the baseline? And which dynamic
features drive any lift, for which tasks?

---

## 2. The two methodological approaches

There are two ways to represent each participant's year-long timeseries. They share
all the upstream work (loading, cleaning, the means-only baseline) and only diverge
at the representation step. **Approach A is the primary; Approach B is a planned,
focused confirmatory add-on.**

### Approach A (PRIMARY): Temporal feature representation + interpretable ML

Represent each participant's year, per signal, as a **vector of scalar descriptors**
of its dynamics. Then predict memory from those features with interpretable models.

- Feature extraction: hand-crafted dynamic features and/or a library like `catch22`
  or `tsfresh` (variability, autocorrelation at various lags, trend slope, entropy,
  burstiness, weekday/weekend ratio, circadian regularity, etc.).
- Modeling: gradient-boosted trees (e.g. XGBoost / LightGBM) or regularized linear
  models, one model per memory-task outcome.
- Interpretation: SHAP values to identify which features matter for which task.

Why primary: this toolkit (feature extraction → boosted trees → SHAP) is what
industry data-science and analytics teams actually use, it degrades gracefully
under the heavy missingness in this data, and it makes the pattern-vs-amount test
direct and legible.

### Approach B (PLANNED, confirmatory): Functional Data Analysis (FDA)

Represent each participant's year, per signal, as a **smooth function**. Use
functional PCA to extract a few interpretable modes of variation (overall level,
seasonal trend, weekday/weekend swing, volatility) and regress memory on the
functional component scores.

Why secondary, not primary: FDA wants reasonably complete, smooth functions over a
common domain, which the ragged/missing wearable data makes harder; and it's a
weaker hiring signal than Approach A. But running it on the **2–3 best-covered
signals only** (likely steps + a heart-rate measure) as a *confirmatory second
lens* is valuable: if hand-crafted features (A) and data-driven functional
components (B) **agree** on what matters, that convergence strengthens the result.
The "do the two representations agree?" comparison is itself a finding.

**Do not start Approach B until Approach A is producing results.** It is upside,
not a dependency.

---

## 3. The data

Raw data is **per-participant CSVs** in `data/raw/` (originally `raw_formatted/` in
the source repo), named `BFM_AMT_0001.csv` ... `BFM_AMT_0113.csv` (BFM = BrainFit
Memory, AMT = Amazon Mechanical Turk). There are 113 participants.

**Format is LONG:** each CSV has three columns — `datetime`, `variable`, `value` —
where each row is one observation of one variable at one timestamp. Daily resolution
(end-of-day timestamps like `2017-07-05 23:59:59`). Participant 1 has ~5,122 rows;
~804K rows total across all files. `value` reads in as a string and needs casting
to float.

Key things to know about the data:
- **Heavy, structural missingness.** Different Fitbit models track different signals,
  so the *union* of variables across all participants is larger than any one
  participant's set. Many participants are sparse; some are nearly all-NaN; wear
  duration varies widely. A wear-time/coverage diagnostic and a defensible inclusion
  threshold are prerequisites, not afterthoughts.
- **Do NOT build on `data/preprocessed/fitbit_7_30.pkl`.** That is the original
  paper's already-collapsed summary (113 x 34, the 7/30 baseline ratios). Using it
  would silently reduce this project to theirs. All dynamic features must be computed
  from the raw daily CSVs.
- **Memory outcomes (prediction targets):** `data/preprocessed/behavior.pkl` (and
  `behavioral_summary.pkl`). Survey/mental-health/demographics: `survey_30.pkl`.
- The big `data/preprocessed/embeddings/opt/` folder (hundreds of
  `10_0.001_*.pkl` files) is the naturalistic-recall text-embedding machinery for
  scoring the story task — **ignore it** for the fitness-feature work.

**Note on the original repo's code:** the source repo's notebooks (especially
`fitness_data.ipynb` and `reverse_correlation_analysis.ipynb`) contain the logic for
loading/parsing the raw fitness data. Worth reading as a reference for the file
format — but write a fresh loader; do not reproduce their analysis.

---

## 4. Analysis plan (phased)

**Phase 0 — migration (NOT DONE YET — this is the actual first step):**
The data has not been copied into this repo yet. Everything still lives in the
forked source repo on the local machine. Before any analysis:
0a. Copy ONLY the needed inputs from the fork into `data/raw/`: the per-participant
    CSVs from the fork's `data/raw_formatted/`, plus `behavior.pkl`,
    `behavioral_summary.pkl`, and `survey_30.pkl` from the fork's `data/preprocessed/`.
0b. Create the repo structure from section 7 (`src/`, `notebooks/`, `figures/`).
0c. Create a `.gitignore` excluding `data/` (participant data is not committed) plus
    standard Python entries.
0d. Do NOT copy: the original paper's code/notebooks, the `embeddings/opt/` folder,
    or `fitbit_7_30.pkl` (the collapsed summary). See section 3 for why.

**Phase 1 — shared foundation (after migration; commits to neither approach):**
1. **Loader.** Read all 113 long-format CSVs, pivot each to wide (one row per date,
   one column per variable), stack into a single clean panel indexed by
   (participant, date), with explicit NaNs for missing days. Cast values to float.
   Build the full variable vocabulary across all participants first so columns align.
2. **Data-quality diagnostic.** For each participant and signal, count valid
   (non-NaN) days. Plot histogram(s) of valid-days-per-participant per signal.
   Decide and document a wear-time inclusion threshold. (This is also the first
   figure for the writeup.)
3. **Feature matrix.** One row per participant; columns = level features (means)
   AND dynamic features. Plus the memory-score target table.
4. **Means-only baseline model.** The comparison point for both approaches.

**Phase 2 — Approach A (the main project; portfolio/preprint-ready on its own):**
5. Temporal/dynamic feature extraction per signal.
6. Boosted (or regularized) model per memory task, with cross-validation designed
   carefully for the small sample (n≈113, fewer after inclusion threshold).
7. SHAP interpretation: which features, which tasks.
8. Report cross-validated lift of augmented model over means-only baseline.

**Phase 3 — Approach B (only if Phase 2 lands and time allows):**
9. FDA / functional PCA on the 2–3 best-covered signals.
10. Compare: do A and B agree on what matters?

Always have a finished, defensible artifact at the end of Phase 2.

---

## 5. Important constraints & cautions

- **Sample size is small (n≈113, fewer after filtering).** This is why Approach A
  (not deep learning) is primary. Cross-validation must be done carefully; watch for
  leakage and over-optimistic estimates. Honest negative results are a valid outcome.
- **Multiple comparisons:** there are four memory tasks (several with immediate +
  delayed variants) and many features. Be disciplined about correction / honest
  reporting so apparent findings aren't noise.
- **The novel claim is specifically "pattern beyond amount."** Keep mean activity in
  the baseline so any lift is attributable to *dynamics*, not just re-discovering
  that active people differ.
- **Causality:** this is correlational (cross-sectional memory testing). Frame
  findings as associations, as the original paper carefully did.

---

## 6. Current status / where we are

- Standalone repo created (separated from the original fork; history should start
  fresh — their data is an input we credit, not inherited commit history). This repo
  is currently empty except for the README and this brief.
- README written (leads with our question, credits Manning et al., marks status as
  in-progress).
- Data confirmed to exist (113 long-format per-participant CSVs at daily resolution),
  but it is **still in the forked source repo — NOT yet copied into this repo.**
- **Next concrete step: Phase 0 migration** (see section 4). The data lives in the
  fork at its local path on this machine; copy the needed inputs into `data/raw/`,
  set up structure, and add `.gitignore`. ONLY after migration, begin Phase 1 by
  inspecting the full variable vocabulary so the pivot aligns columns across all 113
  participants:

  ```bash
  python -c "
  import pandas as pd, glob
  vs=set()
  for f in glob.glob('data/raw/*.csv'):
      vs |= set(pd.read_csv(f, usecols=['variable'])['variable'].unique())
  print(len(vs)); print(sorted(vs))
  "
  ```

  Then map each variable to a feature family (activity / cardiovascular / sleep /
  body) and choose the dynamic features appropriate to each before building the
  feature matrix.

---

## 7. Suggested repo structure

```
.
├── data/
│   └── raw/          # per-participant CSVs (gitignored, not committed)
├── src/              # loader, feature engineering, modeling
├── notebooks/        # exploration, figures
├── figures/
├── requirements.txt
└── README.md
```

---

## 8. Open question to resolve with the original author

An email is going to Jeremy Manning (the dataset's author, and current course
instructor) asking (a) whether this pattern-vs-amount angle has already been
explored in his lab, and (b) whether he sees pitfalls. His answer may affect scope,
whether the data can live in this repo or should be referenced from the source, and
whether the work connects to his lab. Don't block Phase 1 on the reply, but factor
it in when it arrives.