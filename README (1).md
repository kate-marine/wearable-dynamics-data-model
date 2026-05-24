# Activity Patterns and Memory

**Do *patterns* of physical activity predict memory performance beyond *average* activity levels?**

This project asks whether the dynamics of someone's day-to-day physical activity over a year — its variability, weekday/weekend structure, trends, and regularity — carry information about their memory performance that simple averages do not.

> **Status:** Work in progress. The data pipeline and analysis plan are in place; modeling and results are in development. This README will be updated with findings as they come in.

## Background

This project builds on the dataset from:

> Manning, J. R., Notaro, G. M., Chen, E., & Fitzpatrick, P. C. (2022). Fitness tracking reveals task-specific associations between memory, mental health, and physical activity. *Scientific Reports*, 12, 13822. https://doi.org/10.1038/s41598-022-17781-0

In that study, 113 participants shared a year of Fitbit data and then completed a battery of memory tasks (free recall, naturalistic story recall, foreign-language flashcards, and spatial learning) along with a mental-health and demographics survey. The original analysis characterized each participant's activity primarily through weekly averages and a recent-versus-baseline ratio (a 7-day mean divided by the preceding 30-day mean).

That summarization collapses an entire year of behavior into a small number of level-based statistics. Two people with identical average step counts can have very different *patterns* — one steady, one bursty; one consistent across the week, one a weekend-only exerciser. **This project tests whether those patterns matter.**

## Research question

Do dynamic features of year-long activity timeseries predict task-specific memory performance *over and above* mean activity levels?

The analysis is framed as a direct comparison:

- A **means-only baseline** model, using the kind of level-based features the original paper relied on.
- An **augmented** model that adds dynamic features (variability, autocorrelation, weekday/weekend structure, trend, regularity, and related descriptors).

The question is whether the augmented model predicts memory performance better than the baseline, and — using interpretable methods — *which* dynamic features drive any improvement, and for *which* memory tasks.

## Approach

1. **Load** the raw per-participant daily Fitbit data (long-format CSVs) into a clean panel indexed by participant and date.
2. **Assess data quality** — quantify wear-time and missingness per participant and per signal, and define a defensible inclusion threshold.
3. **Engineer features** — for each participant and signal, compute both level features (means) and dynamic features (variability, autocorrelation, trends, weekday/weekend structure, etc.).
4. **Model** — predict each memory-task outcome from (a) the means-only baseline and (b) the augmented feature set, using interpretable models with proper cross-validation given the modest sample size.
5. **Interpret** — use feature-attribution methods to identify which activity patterns relate to which memory tasks.
6. *(Planned)* **Confirm** — as a robustness check, re-examine the richest signals with a complementary functional-data-analysis representation and ask whether the two approaches agree on what matters.

## Repository structure

```
.
├── data/            # raw inputs (not committed — see "Data" below)
│   └── raw/
├── src/             # data loading, feature engineering, modeling code
├── notebooks/       # exploratory analysis and figure generation
├── figures/         # generated figures
├── requirements.txt # Python dependencies
└── README.md
```

## Data

This project uses data from Manning et al. (2022), available in the authors' repository: https://github.com/ContextLab/brainfit-paper

The participant data is **not** redistributed in this repository. To reproduce the analysis, obtain the raw data from the original source and place the per-participant CSVs in `data/raw/`. (See `.gitignore`.)

## Reproducing the analysis

```bash
# clone and set up
git clone <YOUR_REPO_URL>
cd <YOUR_REPO_NAME>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# add the raw data to data/raw/ (see "Data" above), then:
# (commands to be added as the pipeline is built)
```

## Acknowledgments

This project would not be possible without the dataset collected and openly shared by Jeremy R. Manning, Gina M. Notaro, Esme Chen, and Paxton C. Fitzpatrick. All credit for the original data collection and the findings in the 2022 paper belongs to them. The questions, code, and analysis in *this* repository are my own.

## License

Code in this repository is released under the MIT License (see `LICENSE`). The underlying dataset is governed by the terms of the original study and its repository.

## Author

[Your name] — [your email or link]
