"""
Post hoc analysis for wearable activity vs. behavior outcomes:
check whether the negative Ridge results are likely due to model form rather than actual weak signal.

Two checks:
1. Model comparison on means-only wearable features using Ridge, ElasticNet,
   and Random Forest regression
2. Univariate Spearman correlation screen between each wearable feature and
    each valid behavior outcome
3. Direct means-only vs. augmented Random Forest check to see whether the
    dynamic features help under a non-linear model
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .dynamic_features import build_augmented_feature_matrix, extract_dynamic_features
from .features import build_mean_feature_matrix, select_wearable_signals
from .loading import (
    build_daily_panel,
    discover_variable_vocabulary,
    load_raw_long_table,
    load_target_table,
    summarize_variable_coverage,
)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "data" / "posthoc_analysis_outputs"


def _finite_target_frame(targets: pd.DataFrame, min_valid: int = 30) -> pd.DataFrame:
    """Keep only target columns with enough observations"""

    clean = targets.replace([np.inf, -np.inf], np.nan)
    keep_cols = [col for col in clean.columns if clean[col].notna().sum() >= min_valid]
    return clean[keep_cols].copy()


def _cv_scores_for_model(
    X: pd.DataFrame,
    y: pd.Series,
    make_model: Callable[[], object],
    random_state: int = 42,
) -> dict[str, float]:
    """cross-validated regression and return summary scores"""

    n_splits = min(5, len(X))
    if n_splits < 2:
        raise ValueError("Need at least two samples for cross-validation.")

    model = make_model()
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring={
            "r2": "r2",
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
        },
        return_train_score=False,
    )
    return {
        "r2_mean": float(scores["test_r2"].mean()),
        "r2_std": float(scores["test_r2"].std()),
        "mae_mean": float(-scores["test_mae"].mean()),
        "mae_std": float(scores["test_mae"].std()),
        "rmse_mean": float(-scores["test_rmse"].mean()),
        "rmse_std": float(scores["test_rmse"].std()),
    }


def _evaluate_targets(
    X: pd.DataFrame,
    targets: pd.DataFrame,
    model_name: str,
    make_model: Callable[[], object],
) -> pd.DataFrame:
    """Evaluate one model across all target columns."""

    rows = []
    aligned = targets.reindex(X.index)
    for column in aligned.columns:
        y = aligned[column].dropna()
        if y.empty:
            continue
        x = X.loc[y.index].replace([np.inf, -np.inf], np.nan)
        metrics = _cv_scores_for_model(x, y, make_model=make_model)
        rows.append({"target": column, "model": model_name, **metrics, "n_samples": int(len(y))})

    return pd.DataFrame(rows).sort_values("r2_mean", ascending=False).reset_index(drop=True)


def _build_feature_target_correlations(X: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Compute Spearman correlations for every feature-target pair"""

    rows = []
    aligned = targets.reindex(X.index).replace([np.inf, -np.inf], np.nan)
    for target_col in aligned.columns:
        y = aligned[target_col].dropna()
        if y.empty:
            continue
        x = X.loc[y.index]
        for feature_col in x.columns:
            corr = x[feature_col].corr(y, method="spearman")
            if pd.notna(corr):
                rows.append(
                    {
                        "target": target_col,
                        "feature": feature_col,
                        "spearman_r": float(corr),
                        "abs_spearman_r": float(abs(corr)),
                        "n_samples": int(len(y)),
                    }
                )

    return pd.DataFrame(rows).sort_values("abs_spearman_r", ascending=False).reset_index(drop=True)


def run_posthoc_analysis() -> None:
    """Run a post hoc robustness check on the wearable-to-behavior analysis"""

    print("=" * 80)
    print("POST HOC ANALYSIS: alternative models and univariate screens")
    print("=" * 80)

    print("\n[1/4] Loading wearable data...")
    long_df = load_raw_long_table(RAW_DIR)
    vocabulary = discover_variable_vocabulary(long_df)
    coverage_summary = summarize_variable_coverage(long_df)
    wearable_signals = select_wearable_signals(
        vocabulary,
        coverage_summary=coverage_summary,
        min_median_days=300,
        min_participants=60,
    )
    panel = build_daily_panel(long_df, variables=wearable_signals)
    means_features = build_mean_feature_matrix(panel, wearable_signals)
    dynamic_features = extract_dynamic_features(panel, wearable_signals)
    augmented_features = build_augmented_feature_matrix(means_features, dynamic_features)
    print(f"  Selected wearable signals: {len(wearable_signals)}")
    print(f"  Means-only feature matrix: {means_features.shape}")
    print(f"  Dynamic feature matrix: {dynamic_features.shape}")
    print(f"  Augmented feature matrix: {augmented_features.shape}")

    print("\n[2/4] Loading and filtering behavior outcomes...")
    behavior_full = load_target_table(RAW_DIR, "behavior.pkl")
    behavior_valid = _finite_target_frame(behavior_full, min_valid=30)
    print(f"  Original behavior outcomes: {behavior_full.shape}")
    print(f"  Valid behavior outcomes: {behavior_valid.shape}")

    print("\n[3/4] Comparing alternative models on means-only features...")
    model_specs: list[tuple[str, Callable[[], object]]] = [
        (
            "ridge",
            lambda: Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", Ridge(alpha=1.0)),
                ]
            ),
        ),
        (
            "elastic_net",
            lambda: Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=20000, random_state=42)),
                ]
            ),
        ),
        (
            "random_forest",
            lambda: Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", RandomForestRegressor(n_estimators=400, min_samples_leaf=5, random_state=42)),
                ]
            ),
        ),
    ]

    model_results = []
    for model_name, make_model in model_specs:
        print(f"  Evaluating {model_name}...")
        result = _evaluate_targets(means_features, behavior_valid, model_name=model_name, make_model=make_model)
        model_results.append(result)
        print(
            f"    mean R² = {result['r2_mean'].mean():.4f}, "
            f"positive outcomes = {(result['r2_mean'] > 0).sum()}/{len(result)}"
        )

    all_model_results = pd.concat(model_results, ignore_index=True)

    print("  Evaluating random_forest_augmented...")
    rf_augmented = _evaluate_targets(
        augmented_features,
        behavior_valid,
        model_name="random_forest_augmented",
        make_model=lambda: Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(n_estimators=400, min_samples_leaf=5, random_state=42)),
            ]
        ),
    )
    all_model_results = pd.concat([all_model_results, rf_augmented], ignore_index=True)
    print(
        f"    mean R² = {rf_augmented['r2_mean'].mean():.4f}, "
        f"positive outcomes = {(rf_augmented['r2_mean'] > 0).sum()}/{len(rf_augmented)}"
    )

    print("\n[4/4] Running univariate Spearman correlation screen...")
    corr_results = _build_feature_target_correlations(means_features, behavior_valid)
    print(f"  Feature-target pairs evaluated: {len(corr_results):,}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_model_results.to_csv(OUTPUT_DIR / "model_comparison_all_outcomes.csv", index=False)
    corr_results.to_csv(OUTPUT_DIR / "feature_target_spearman_screen.csv", index=False)

    # Compact summary
    print("\n" + "=" * 80)
    print("POST HOC SUMMARY")
    print("=" * 80)

    print("\nModel comparison on means-only features:")
    summary = (
        all_model_results.groupby("model")
        .agg(
            mean_r2=("r2_mean", "mean"),
            median_r2=("r2_mean", "median"),
            positive_outcomes=("r2_mean", lambda s: int((s > 0).sum())),
        )
        .reset_index()
    )
    print(summary.to_string(index=False))

    print("\nStrongest univariate relationships:")
    print(corr_results.head(10)[["target", "feature", "spearman_r", "abs_spearman_r"]].to_string(index=False))

    print("\nTop target-level max correlations:")
    target_strength = (
        corr_results.groupby("target")["abs_spearman_r"].max().sort_values(ascending=False).reset_index()
    )
    print(target_strength.head(10).to_string(index=False))

    print("\nOutputs saved to:")
    print(f"  {OUTPUT_DIR / 'model_comparison_all_outcomes.csv'}")
    print(f"  {OUTPUT_DIR / 'feature_target_spearman_screen.csv'}")
    print("=" * 80)


if __name__ == "__main__":
    run_posthoc_analysis()
