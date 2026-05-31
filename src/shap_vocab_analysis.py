"""
Focused SHAP analysis: vocab learning (delayed) error distance vs. wearable features.

The post hoc Spearman screen identified the strongest wearable-behavior association
in this dataset: mean__floors vs ('vocab learning', 'delayed', 'error distance'),
Spearman r ≈ 0.55, n=62.

This module:
1. Fits a Random Forest on the 14 means-only wearable features for the 62 participants
   with valid vocab-delayed error-distance scores.
2. Computes cross-validated R² to quantify predictive generalization.
3. Uses SHAP TreeExplainer to attribute model predictions to individual features.
4. Produces four figures saved to figures/:
   - shap_scatter_floors_vs_error_distance.png   — raw data with annotated Spearman r
   - shap_beeswarm.png                           — SHAP summary across all features
   - shap_dependence_floors.png                  — SHAP value for floors vs. floors level
   - shap_bar_importance.png                     — mean |SHAP| feature ranking

Run: python -m src.shap_vocab_analysis
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_validate

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
FIGURES_DIR = ROOT / "figures"
OUTPUT_DIR = ROOT / "data" / "shap_outputs"

TARGET_COL = ("vocab learning", "delayed", "error distance")
TARGET_LABEL = "Vocab learning — delayed error distance"
RANDOM_STATE = 42


def _load_aligned_data() -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) aligned to participants with valid target values."""
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
    X = build_mean_feature_matrix(panel, wearable_signals)

    behavior = load_target_table(RAW_DIR, "behavior.pkl")
    y_raw = behavior[TARGET_COL].replace([np.inf, -np.inf], np.nan)
    y = y_raw.dropna()

    X_aligned = X.loc[y.index]
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X_aligned),
        index=X_aligned.index,
        columns=X_aligned.columns,
    )
    return X_imputed, y


def _cv_performance(X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Cross-validated Random Forest performance."""
    rf = RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=4,
        max_features="sqrt",
        random_state=RANDOM_STATE,
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        rf, X, y,
        cv=cv,
        scoring={"r2": "r2", "mae": "neg_mean_absolute_error"},
    )
    return {
        "r2_mean": float(scores["test_r2"].mean()),
        "r2_std": float(scores["test_r2"].std()),
        "mae_mean": float(-scores["test_mae"].mean()),
        "mae_std": float(scores["test_mae"].std()),
        "r2_folds": scores["test_r2"].tolist(),
    }


def _fit_rf(X: pd.DataFrame, y: pd.Series) -> RandomForestRegressor:
    rf = RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=4,
        max_features="sqrt",
        random_state=RANDOM_STATE,
    )
    rf.fit(X, y)
    return rf


def _pretty_feature_name(col: str) -> str:
    """Strip mean__ prefix and replace underscores for plot labels."""
    return col.removeprefix("mean__").replace("_", " ")


def plot_scatter_floors(X: pd.DataFrame, y: pd.Series, out_path: Path) -> None:
    """Raw scatter: mean__floors vs error distance, annotated with Spearman r."""
    x_vals = X["mean__floors"]
    spearman_r, spearman_p = stats.spearmanr(x_vals, y)
    pearson_r, pearson_p = stats.pearsonr(x_vals, y)

    slope, intercept, *_ = stats.linregress(x_vals, y)
    x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
    y_line = slope * x_line + intercept

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x_vals, y, color="#4C72B0", alpha=0.65, edgecolors="white", linewidths=0.4, s=55, zorder=3)
    ax.plot(x_line, y_line, color="#C44E52", linewidth=1.8, zorder=2)

    ax.set_xlabel("Mean floors climbed per day", fontsize=12)
    ax.set_ylabel("Delayed vocab error distance\n(higher = worse)", fontsize=12)
    ax.set_title("Floors climbed vs. vocabulary learning error", fontsize=13, fontweight="bold")

    annotation = (
        f"Spearman $r$ = {spearman_r:.2f} ($p$ = {spearman_p:.3f})\n"
        f"Pearson $r$ = {pearson_r:.2f} ($p$ = {pearson_p:.3f})\n"
        f"$n$ = {len(y)}"
    )
    ax.text(
        0.97, 0.05, annotation,
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_shap_beeswarm(shap_values: np.ndarray, X: pd.DataFrame, out_path: Path) -> None:
    """SHAP beeswarm (summary) plot across all 14 features."""
    display_names = [_pretty_feature_name(c) for c in X.columns]
    X_display = X.rename(columns=dict(zip(X.columns, display_names)))

    fig, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(
        shap_values,
        X_display,
        show=False,
        plot_size=None,
        color_bar_label="Feature value (normalized)",
    )
    ax = plt.gca()
    ax.set_title(
        f"SHAP feature contributions\n{TARGET_LABEL}  ($n$={len(X)})",
        fontsize=12, fontweight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_shap_dependence_floors(
    shap_values: np.ndarray, X: pd.DataFrame, out_path: Path
) -> None:
    """SHAP dependence plot for mean__floors, colored by mean__cal."""
    floors_idx = list(X.columns).index("mean__floors")
    cal_idx = list(X.columns).index("mean__cal")

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(
        X["mean__floors"],
        shap_values[:, floors_idx],
        c=X["mean__cal"],
        cmap="coolwarm",
        alpha=0.75,
        edgecolors="white",
        linewidths=0.3,
        s=55,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Mean daily calories", fontsize=10)

    ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Mean floors climbed per day", fontsize=12)
    ax.set_ylabel("SHAP value\n(impact on predicted error distance)", fontsize=12)
    ax.set_title(
        "SHAP dependence: floors climbed\n(colored by daily calories)",
        fontsize=12, fontweight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_shap_bar(shap_values: np.ndarray, X: pd.DataFrame, out_path: Path) -> None:
    """Horizontal bar chart of mean |SHAP| values."""
    mean_abs_shap = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=[_pretty_feature_name(c) for c in X.columns],
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#4C72B0" if f != "floors" else "#C44E52" for f in mean_abs_shap.index]
    bars = ax.barh(mean_abs_shap.index, mean_abs_shap.values, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Mean |SHAP value|", fontsize=12)
    ax.set_title(
        f"Feature importance (SHAP)\n{TARGET_LABEL}  ($n$={len(X)})",
        fontsize=12, fontweight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def run_shap_vocab_analysis() -> None:
    print("=" * 70)
    print("SHAP ANALYSIS — VOCAB LEARNING DELAYED ERROR DISTANCE")
    print("=" * 70)

    print("\n[1/5] Loading and aligning data...")
    X, y = _load_aligned_data()
    print(f"  Feature matrix: {X.shape}")
    print(f"  Target (n valid): {len(y)}")
    print(f"  Target range: [{y.min():.2f}, {y.max():.2f}], mean={y.mean():.2f}")

    print("\n[2/5] Cross-validated Random Forest performance...")
    cv_metrics = _cv_performance(X, y)
    print(f"  CV R²: {cv_metrics['r2_mean']:.3f} ± {cv_metrics['r2_std']:.3f}")
    print(f"  CV MAE: {cv_metrics['mae_mean']:.3f} ± {cv_metrics['mae_std']:.3f}")
    print(f"  Per-fold R²: {[f'{v:.3f}' for v in cv_metrics['r2_folds']]}")

    print("\n[3/5] Fitting Random Forest on full aligned data for SHAP...")
    rf = _fit_rf(X, y)
    train_r2 = rf.score(X, y)
    print(f"  Train R² (in-sample): {train_r2:.3f}")

    print("\n[4/5] Computing SHAP values...")
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X)
    print(f"  SHAP values shape: {shap_values.shape}")

    print("\n[5/5] Generating figures...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_scatter_floors(X, y, FIGURES_DIR / "shap_scatter_floors_vs_error_distance.png")
    plot_shap_beeswarm(shap_values, X, FIGURES_DIR / "shap_beeswarm.png")
    plot_shap_dependence_floors(shap_values, X, FIGURES_DIR / "shap_dependence_floors.png")
    plot_shap_bar(shap_values, X, FIGURES_DIR / "shap_bar_importance.png")

    # Save SHAP values as CSV for reproducibility
    shap_df = pd.DataFrame(shap_values, index=X.index, columns=X.columns)
    shap_df.to_csv(OUTPUT_DIR / "shap_values_vocab_delayed_error_distance.csv")

    # Save CV metrics
    cv_summary = pd.DataFrame([{
        "target": str(TARGET_COL),
        "n_participants": len(y),
        "model": "RandomForest(n_estimators=500,min_samples_leaf=4,max_features=sqrt)",
        **{k: v for k, v in cv_metrics.items() if k != "r2_folds"},
    }])
    cv_summary.to_csv(OUTPUT_DIR / "cv_performance_vocab_delayed_error_distance.csv", index=False)

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nTarget: {TARGET_LABEL}")
    print(f"n = {len(y)} participants (of 113 total with valid data)")
    print(f"\nPredictive performance (5-fold CV):")
    print(f"  R²  = {cv_metrics['r2_mean']:.3f} ± {cv_metrics['r2_std']:.3f}")
    print(f"  MAE = {cv_metrics['mae_mean']:.3f} ± {cv_metrics['mae_std']:.3f}")

    spearman_r, spearman_p = stats.spearmanr(X["mean__floors"], y)
    print(f"\nStrongest univariate: mean__floors")
    print(f"  Spearman r = {spearman_r:.3f}, p = {spearman_p:.4f}")

    print("\nTop features by mean |SHAP|:")
    mean_abs_shap = pd.Series(
        np.abs(shap_values).mean(axis=0), index=X.columns
    ).sort_values(ascending=False)
    for feat, val in mean_abs_shap.head(5).items():
        print(f"  {_pretty_feature_name(feat):25s}  {val:.4f}")

    print(f"\nOutputs: figures/ and {OUTPUT_DIR.relative_to(ROOT)}/")
    print("=" * 70)


if __name__ == "__main__":
    run_shap_vocab_analysis()
