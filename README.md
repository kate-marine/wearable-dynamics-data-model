# Fitbit Activity Dynamics and Memory Performance
Author: Kate Marine kate-marine

## Overview

This project tests whether the temporal pattern of someone's Fitbit activity (such how it varies day to day, trends over time, and autocorrelates) predicts memory-task performance beyond what their average activity level already captures. I built two models, one as a baseline using only mean activity, and another with added temporal/dynamic features. I then compared performance with Elastic Net and Random Forest models as well as a univariate screen of 560 feature–outcome pairs. All results concluded that dynamic features add no reliable predictive value over a simple mean-activity baseline.

I confirmed that this null result is not a pipeline bug from a synthetic-signal test which confirmed the cross-validation harness *can* recover a known effect. I then conducted a power analysis which showed that at n = 113 this study could not have reliably detected correlations smaller than |r| ≈ 0.26, so the conclusion is not necessarily proving that no association exists between temporal fitness activity and memory performance but rather that there is no reliable evidence of an association given the limited dataset. 

## Approach

I built two models, one as a baseline using only mean activity, and another with added temporal/dynamic features. These included variability metrics (standard deviation, range, coefficient of variation) linear slope over the year, and autocorrelation at lags 1 and 7 days. For both models I standardized every feature so they were on a common scale and then fit a Ridge regression. I scored everything with shuffled k-fold cross-validation, and then looked at the R² to compare the two models' performance. 

After I got pretty weak cross-validated $R^2$ for both Ridge models, I then starting looking into whether the result was due the kind of model I was using and so I tried out alternative models (Elastic Net and Random Forest) to see if they would perform better on the same features. Finally I looked to see if any individual features showed clear monotonic relationships with behavior outcomes (mainly as motivation for next steps) by computing Spearman correlations for every fitbit feature / behavior outcome pair from the 40 valid behavior outcomes from behavior.pkl.

## Findings

Adding temporal dynamics did not help predicting memory-task performance, and it actually did significantly worst then the baseline model using average activity level. The models are mostly likely overfitting as is common with having more predictors (163) than participants (113). The null result stayed the same even after three stress tests (expanding to 40 fine-grained outcomes, switching to Elastic Net and Random Forest models, and a univariate Spearman screen across 560 feature–target pairs where no dynamic feature appeared among the top correlates).


## Downloading the data

I used 113 participants' Fitbit data along with memory-task outcomes from the study _Manning, J. R., Notaro, G. M., Chen, E., & Fitzpatrick, P. C. (2022)_. Fitness tracking reveals task-specific associations between memory, mental health, and physical activity. *Scientific Reports*, 12, 13822. https://doi.org/10.1038/s41598-022-17781-0

I reshaped the raw Fitbit CSVs into a participant-by-date panel for the temporal modeling. 


## Running the code

Set up:

```bash
git clone https://github.com/kate-marine/wearable-dynamics-data-model.git
cd wearable-dynamics-data-model
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# run
python -m src.phase1                      # load CSVs, build panel, coverage diagnostics, means-only baseline
python -m src.phase2                      # dynamic feature extraction and model comparison
python -m src.exploratory_full_behavior   # rerun comparison across all behavior.pkl outcomes
python -m src.posthoc_analysis            # Elastic Net, Random Forest, Spearman univariate screen

```

## Contributing to the code

### Challenges and potential next steps:
I was a little limited in what I could include in the models since things like sleep and heart-rate/HRV (probably pretty strong ties to cognitive performance) were too sparse in the data. So this could definitely be revisited/replicated if can get more data. As a next step I might look into a different target metric (rather than memory) such as one of the mental health measures like typical stress. From a Spearman screen I ran I might look into the mean__floors vs. vocab learning correlation as well. 

The biggest problem with the apporach I've taken is that the sample size of 113 participants is too small for meaningful modeling and led to significant overfitting. 

## Acknowledgements

_Manning, J. R., Notaro, G. M., Chen, E., & Fitzpatrick, P. C. (2022)_. Fitness tracking reveals task-specific associations between memory, mental health, and physical activity. *Scientific Reports*, 12, 13822. https://doi.org/10.1038/s41598-022-17781-0
