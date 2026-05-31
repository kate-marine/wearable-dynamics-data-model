"""
Gradient boosting model: predicting typical stress from wearable features.

The survey Spearman screen found that physical activity features correlate with
self-reported typical stress — notably in the opposite direction from what Manning
et al. (2022) reported for their heart-rate-zone peak activity metric:

  Manning et al.: peak HR activity → HIGHER typical stress (r = +0.21)
  This study:     very_act_mins   → LOWER  typical stress (r = −0.28, n=102)
                  floors          → LOWER  typical stress (r = −0.36, n=60)

The original paper reported only pairwise bootstrap correlations. This module
builds a multivariate XGBoost model that can:
  (a) test whether multiple wearable features together predict stress better
      than any single feature alone, and
  (b) use SHAP to reveal which features drive predictions and whether feature
      interactions are present (e.g., does low-floor AND high-BMI predict stress
      differently than either alone?).

Target: ('', 'typical stress') — 5-point Likert (−2 to +2), n=113, complete.
Features: 14 means-only wearable features (median imputed; NaN rate < 10% for
          most features, floors being the exception at ~32% NaN).

Outputs (data/stress_prediction_outputs/):
  cv_performance.csv                 — per-fold and mean R², MAE
  shap_values_typical_stress.csv     — SHAP matrix (113 × 14)
  univariate_benchmark.csv           — Spearman r for each feature individually

Figures (figures/):
  stress_shap_beeswarm.png           — SHAP summary across all 14 features
  stress_shap_bar.png                — mean |SHAP| feature ranking
  stress_shap_dependence_grid.png    — dependence plots for top 4 features
  stress_univariate_benchmark.png    — CV R² vs. best single-feature benchmark

Run: python -m src.stress_prediction
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_predict
import xgboost as xgb

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
OUTPUT_DIR = ROOT / "data" / "stress_prediction_outputs"

TARGET_COL = ("", "typical stress")
TARGET_LABEL = "Typical stress (self-reported)"
RANDOM_STATE = 42
N_SPLITS = 5


def _pretty(col: str) -> str:
    return col.removeprefix("mean__").replace("_", " ")


def _load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Return imputed feature matrix X and target y (typical stress)."""
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
    X_raw = build_mean_feature_matrix(panel, wearable_signals)

    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(
        imputer.fit_transform(X_raw),
        index=X_raw.index,
        columns=X_raw.columns,
    )

    survey = load_target_table(RAW_DIR, "survey_30.pkl")
    y = pd.to_numeric(survey[TARGET_COL], errors="coerce").dropna()
    y = y.reindex(X.index)  # align — all 113 have valid stress
    return X, y


def _make_model() -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        verbosity=0,
    )


def _cv_performance(X: pd.DataFrame, y: pd.Series) -> dict:
    """5-fold CV: per-fold R² and pooled metrics."""
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    y_pred = cross_val_predict(_make_model(), X, y, cv=cv)
    fold_r2 = []
    fold_mae = []
    for train_idx, test_idx in cv.split(X):
        m = _make_model()
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        yp = m.predict(X.iloc[test_idx])
        fold_r2.append(float(r2_score(y.iloc[test_idx], yp)))
        fold_mae.append(float(mean_absolute_error(y.iloc[test_idx], yp)))
    return {
        "r2_pooled": float(r2_score(y, y_pred)),
        "r2_mean": float(np.mean(fold_r2)),
        "r2_std": float(np.std(fold_r2)),
        "mae_mean": float(np.mean(fold_mae)),
        "mae_std": float(np.std(fold_mae)),
        "fold_r2": fold_r2,
        "y_pred_loo": y_pred,
    }


def _univariate_benchmark(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Spearman r for each feature vs typical stress."""
    rows = []
    for col in X.columns:
        r, p = stats.spearmanr(X[col], y)
        rows.append({"feature": _pretty(col), "spearman_r": r, "abs_r": abs(r), "p_value": p})
    return pd.DataFrame(rows).sort_values("abs_r", ascending=False).reset_index(drop=True)


def plot_shap_beeswarm(shap_values: np.ndarray, X: pd.DataFrame, out_path: Path) -> None:
    X_display = X.rename(columns={c: _pretty(c) for c in X.columns})
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(shap_values, X_display, show=False, plot_size=None)
    ax = plt.gca()
    ax.set_title(
        f"SHAP feature contributions — {TARGET_LABEL}\n"
        f"(XGBoost, 14 wearable features, $n$=113)",
        fontsize=11, fontweight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_shap_bar(shap_values: np.ndarray, X: pd.DataFrame, out_path: Path) -> None:
    mean_abs = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=[_pretty(c) for c in X.columns],
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#4C72B0"] * len(mean_abs)
    ax.barh(mean_abs.index, mean_abs.values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title(
        f"Feature importance (SHAP)\n{TARGET_LABEL}",
        fontsize=11, fontweight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_dependence_grid(
    shap_values: np.ndarray, X: pd.DataFrame, top_features: list[str], out_path: Path
) -> None:
    """2×2 grid of SHAP dependence plots for the top 4 features."""
    n = min(4, len(top_features))
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for i, feat_raw in enumerate(top_features[:n]):
        feat_label = _pretty(feat_raw)
        feat_idx = list(X.columns).index(feat_raw)
        ax = axes[i]

        r, p = stats.spearmanr(X[feat_raw], shap_values[:, feat_idx])
        sc = ax.scatter(
            X[feat_raw], shap_values[:, feat_idx],
            c=X[feat_raw], cmap="RdBu_r",
            alpha=0.70, edgecolors="white", linewidths=0.3, s=45,
        )
        ax.axhline(0, color="#888", linewidth=0.8, linestyle="--")
        ax.set_xlabel(f"Mean {feat_label}", fontsize=10)
        ax.set_ylabel("SHAP value", fontsize=10)
        ax.set_title(
            f"{feat_label}  (Spearman $r$ = {r:+.2f})",
            fontsize=10, fontweight="bold",
        )
        ax.spines[["top", "right"]].set_visible(False)

    for j in range(n, 4):
        axes[j].set_visible(False)

    fig.suptitle(
        f"SHAP dependence — top features predicting {TARGET_LABEL}\n($n$=113)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_cv_benchmark(cv_results: dict, univariate: pd.DataFrame, out_path: Path) -> None:
    """Bar chart comparing CV R² to a null model and Spearman r² benchmark."""
    best_r2 = float(univariate["spearman_r"].abs().max() ** 2)  # r² of best single feature

    labels = ["Null model\n(predict mean)", "Best single\nfeature (r²)", "XGBoost\n(5-fold CV R²)"]
    values = [0.0, best_r2, cv_results["r2_pooled"]]
    colors = ["#AAAAAA", "#4C72B0", "#2ca02c" if cv_results["r2_pooled"] > 0 else "#C44E52"]

    fig, ax = plt.subplots(figsize=(5, 4.5))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.5, width=0.55)
    if cv_results["r2_pooled"] > 0:
        ax.errorbar(
            2, cv_results["r2_pooled"],
            yerr=cv_results["r2_std"],
            fmt="none", color="black", capsize=5, linewidth=1.5,
        )
    ax.axhline(0, color="#555", linewidth=0.8)
    ax.set_ylabel("$R^2$", fontsize=12)
    ax.set_title(
        f"Predictive performance — {TARGET_LABEL}",
        fontsize=11, fontweight="bold",
    )
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(val, 0) + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=10,
        )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def run_stress_prediction() -> None:
    print("=" * 70)
    print("STRESS PREDICTION — XGBOOST + SHAP")
    print("=" * 70)

    print("\n[1/5] Loading data...")
    X, y = _load_data()
    print(f"  Feature matrix: {X.shape}")
    print(f"  Target n: {y.notna().sum()}, range [{y.min():.0f}, {y.max():.0f}], mean={y.mean():.2f}")

    print("\n[2/5] Univariate Spearman benchmark...")
    univariate = _univariate_benchmark(X, y)
    print(univariate[["feature", "spearman_r", "p_value"]].head(8).to_string(index=False))

    print("\n[3/5] 5-fold cross-validated XGBoost...")
    cv_results = _cv_performance(X, y)
    print(f"  Pooled CV R²: {cv_results['r2_pooled']:.3f}")
    print(f"  Mean fold R²: {cv_results['r2_mean']:.3f} ± {cv_results['r2_std']:.3f}")
    print(f"  Mean fold MAE: {cv_results['mae_mean']:.3f} ± {cv_results['mae_std']:.3f}")
    print(f"  Per-fold R²: {[f'{v:.3f}' for v in cv_results['fold_r2']]}")
    best_univariate_r2 = float(univariate["spearman_r"].abs().max() ** 2)
    print(f"  Best single-feature r² (Spearman): {best_univariate_r2:.3f} "
          f"({univariate.iloc[0]['feature']})")

    print("\n[4/5] Fitting full model for SHAP...")
    model = _make_model()
    model.fit(X, y)
    print(f"  In-sample R²: {model.score(X, y):.3f}")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    print(f"  SHAP values shape: {shap_values.shape}")

    mean_abs_shap = pd.Series(
        np.abs(shap_values).mean(axis=0), index=X.columns
    ).sort_values(ascending=False)
    top_features = mean_abs_shap.index.tolist()

    print("\n[5/5] Generating figures and saving outputs...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_shap_beeswarm(shap_values, X, FIGURES_DIR / "stress_shap_beeswarm.png")
    plot_shap_bar(shap_values, X, FIGURES_DIR / "stress_shap_bar.png")
    plot_dependence_grid(shap_values, X, top_features, FIGURES_DIR / "stress_shap_dependence_grid.png")
    plot_cv_benchmark(cv_results, univariate, FIGURES_DIR / "stress_cv_benchmark.png")

    # Save outputs
    shap_df = pd.DataFrame(shap_values, index=X.index, columns=X.columns)
    shap_df.to_csv(OUTPUT_DIR / "shap_values_typical_stress.csv")
    univariate.to_csv(OUTPUT_DIR / "univariate_benchmark.csv", index=False)

    cv_summary = pd.DataFrame([{
        "target": str(TARGET_COL),
        "model": "XGBoost(n_estimators=400,max_depth=3,lr=0.05)",
        "n": len(y),
        "cv_folds": N_SPLITS,
        "r2_pooled": cv_results["r2_pooled"],
        "r2_mean": cv_results["r2_mean"],
        "r2_std": cv_results["r2_std"],
        "mae_mean": cv_results["mae_mean"],
        "mae_std": cv_results["mae_std"],
    }])
    cv_summary.to_csv(OUTPUT_DIR / "cv_performance.csv", index=False)

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nTarget: {TARGET_LABEL}  (n=113, 5-point Likert)")
    print(f"\nCross-validated performance (5-fold XGBoost):")
    print(f"  Pooled R²   = {cv_results['r2_pooled']:.3f}")
    print(f"  Mean fold R² = {cv_results['r2_mean']:.3f} ± {cv_results['r2_std']:.3f}")
    print(f"  MAE          = {cv_results['mae_mean']:.3f} ± {cv_results['mae_std']:.3f}")
    print(f"\nComparison:")
    print(f"  Best univariate Spearman r = {univariate.iloc[0]['spearman_r']:+.3f} "
          f"({univariate.iloc[0]['feature']}, p={univariate.iloc[0]['p_value']:.4f})")
    print(f"  Best univariate r²         = {best_univariate_r2:.3f}")
    print(f"\nTop SHAP features (mean |SHAP|):")
    for feat, val in mean_abs_shap.head(6).items():
        r_uni = float(univariate.loc[univariate["feature"] == _pretty(feat), "spearman_r"].values[0])
        print(f"  {_pretty(feat):20s}  SHAP={val:.4f}  univariate r={r_uni:+.3f}")
    print(f"\nNote on directionality:")
    print(f"  Manning et al. (2022) found: high-intensity HR zone → higher stress (r=+0.21)")
    print(f"  This study finds: very_act_mins → lower stress (r={univariate.loc[univariate['feature']=='very act mins','spearman_r'].values[0]:+.3f})")
    print(f"  These measures differ: HR peak zone (device-specific) vs. movement-based activity level.")
    print(f"\nOutputs: {OUTPUT_DIR.relative_to(ROOT)}/  and  figures/")
    print("=" * 70)


if __name__ == "__main__":
    run_stress_prediction()
