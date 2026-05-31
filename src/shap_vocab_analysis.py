"""
Focused SHAP analysis: vocab learning (delayed) error distance vs. wearable features.

The post hoc Spearman screen found the dataset's strongest wearable-behavior association:
  mean__floors vs ('vocab learning', 'delayed', 'error distance'), r ≈ 0.55

However, floors tracking is device-specific: only 26 of the 62 participants with
valid vocab scores owned a Fitbit with an altimeter. The analysis therefore uses
that n=26 complete-case subset. A second association (mean__weight, r=-0.40) also
emerges in the same subset.

Four wearable features have no missing values for the n=26 participants:
  floors, weight, distance, very_act_mins.
These are used directly without imputation.

This module produces four figures (saved to figures/):
  shap_scatter_floors_vs_error_distance.png  — raw scatter, annotated Spearman r
  shap_correlation_landscape.png             — all-feature Spearman r with varying n
  shap_beeswarm.png                          — SHAP summary, 4 complete features
  shap_dependence_floors.png                 — SHAP dependence for floors, colored by weight

Run: python -m src.shap_vocab_analysis
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict

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
TARGET_LABEL = "Delayed vocab error distance"

# These four features have no missing values for the 26 floors-tracking participants.
COMPLETE_FEATURES = ["mean__floors", "mean__weight", "mean__distance", "mean__very_act_mins"]

RANDOM_STATE = 42
ALPHA = 0.05


def _pretty(col: str) -> str:
    return col.removeprefix("mean__").replace("_", " ")


def _load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Return means-only X (113×14) and target y (62 valid), unimputed."""
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
    y = behavior[TARGET_COL].replace([np.inf, -np.inf], np.nan).dropna()
    return X, y


def _build_subsets(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Return (X_62, y_62) all-target participants and (X_26, y_26) floors-complete subset."""
    X_62 = X.loc[y.index]
    floors_valid = X_62["mean__floors"].notna()
    X_26 = X_62.loc[floors_valid, COMPLETE_FEATURES]
    y_26 = y[floors_valid]
    return X_62, y, X_26, y_26


def _pairwise_correlations(X: pd.DataFrame, y: pd.Series, label: str = "") -> pd.DataFrame:
    """Spearman r for each feature vs target using pairwise complete observations."""
    rows = []
    for col in X.columns:
        valid = X[col].notna()
        n = int(valid.sum())
        if n < 10:
            continue
        r, p = stats.spearmanr(X.loc[valid, col], y[valid])
        rows.append({"feature": col, "spearman_r": r, "p_value": p, "n": n, "subset": label})
    return pd.DataFrame(rows).sort_values("spearman_r", ascending=False)


def _loo_cv_r2(X: pd.DataFrame, y: pd.Series) -> float:
    """Leave-one-out CV R², computed over all held-out predictions pooled."""
    rf = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=3, max_features="sqrt", random_state=RANDOM_STATE
    )
    y_pred = cross_val_predict(rf, X, y, cv=LeaveOneOut())
    return float(r2_score(y, y_pred))


def plot_scatter(X_26: pd.DataFrame, y_26: pd.Series, out_path: Path) -> None:
    """Raw scatter: floors vs error distance (n=26)."""
    x_vals = X_26["mean__floors"]
    r, p = stats.spearmanr(x_vals, y_26)
    slope, intercept, *_ = stats.linregress(x_vals, y_26)
    x_line = np.linspace(x_vals.min(), x_vals.max(), 200)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        x_vals, y_26,
        color="#4C72B0", alpha=0.70, edgecolors="white", linewidths=0.4, s=60, zorder=3,
    )
    ax.plot(x_line, slope * x_line + intercept, color="#C44E52", linewidth=1.8, zorder=2)

    ax.set_xlabel("Mean floors climbed per day", fontsize=12)
    ax.set_ylabel("Delayed vocab error distance\n(higher = worse recall)", fontsize=12)
    ax.set_title("Floors climbed vs. vocabulary learning error", fontsize=13, fontweight="bold")
    ax.text(
        0.97, 0.05,
        f"Spearman $r$ = {r:.2f}  ($p$ = {p:.3f})\n$n$ = {len(y_26)} (floors-tracking devices only)",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_correlation_landscape(corr_df: pd.DataFrame, out_path: Path) -> None:
    """Horizontal bar chart of all-feature Spearman r values, annotated with n."""
    df = corr_df.copy().sort_values("spearman_r")
    colors = []
    for _, row in df.iterrows():
        if row["p_value"] < ALPHA:
            colors.append("#C44E52" if row["spearman_r"] > 0 else "#4C72B0")
        else:
            colors.append("#AAAAAA")

    labels = [f"{_pretty(f)}  (n={int(n)})" for f, n in zip(df["feature"], df["n"])]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.barh(labels, df["spearman_r"].values, color=colors, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.set_xlabel("Spearman $r$ with delayed vocab error distance", fontsize=11)
    ax.set_title(
        "Wearable feature correlations with vocab learning error\n"
        "(pairwise complete observations; varying n per feature)",
        fontsize=11, fontweight="bold",
    )
    sig_patch = mpatches.Patch(color="#C44E52", label="Significant ($p$ < .05, positive $r$)")
    ns_patch = mpatches.Patch(color="#AAAAAA", label="Not significant")
    neg_patch = mpatches.Patch(color="#4C72B0", label="Significant ($p$ < .05, negative $r$)")
    ax.legend(handles=[sig_patch, neg_patch, ns_patch], fontsize=9, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_shap_beeswarm(
    shap_values: np.ndarray, X_26: pd.DataFrame, out_path: Path
) -> None:
    """SHAP beeswarm for the 4 complete features, n=26."""
    X_display = X_26.rename(columns={c: _pretty(c) for c in X_26.columns})
    fig, ax = plt.subplots(figsize=(7, 4))
    shap.summary_plot(shap_values, X_display, show=False, plot_size=None)
    ax = plt.gca()
    ax.set_title(
        f"SHAP feature contributions — {TARGET_LABEL}\n"
        f"(Random Forest on 4 complete features, $n$=26)",
        fontsize=11, fontweight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_shap_dependence(
    shap_values: np.ndarray, X_26: pd.DataFrame, out_path: Path
) -> None:
    """SHAP dependence for floors, colored by weight."""
    floors_idx = list(X_26.columns).index("mean__floors")
    r_floors, p_floors = stats.spearmanr(X_26["mean__floors"], X_26["mean__weight"])

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(
        X_26["mean__floors"],
        shap_values[:, floors_idx],
        c=X_26["mean__weight"],
        cmap="coolwarm_r",
        alpha=0.80,
        edgecolors="white",
        linewidths=0.3,
        s=60,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Mean daily weight (lbs)", fontsize=10)

    ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Mean floors climbed per day", fontsize=12)
    ax.set_ylabel("SHAP value for floors\n(contribution to predicted error distance)", fontsize=12)
    ax.set_title(
        "SHAP dependence: floors climbed\n(colored by body weight, $n$=26)",
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

    print("\n[1/5] Loading data...")
    X, y = _load_data()
    X_62, y_62, X_26, y_26 = _build_subsets(X, y)
    print(f"  Participants with valid target: {len(y_62)}")
    print(f"  Participants with floors data (analysis subset): {len(y_26)}")

    print("\n[2/5] Pairwise Spearman correlations...")
    corr_62 = _pairwise_correlations(X_62, y_62, label="n=62 (all target-valid)")
    corr_26 = _pairwise_correlations(X_26, y_26, label="n=26 (floors-tracking subset)")
    print("  All target-valid participants (n=62 base):")
    print(corr_62[["feature", "spearman_r", "p_value", "n"]].to_string(index=False))
    print("\n  Floors-tracking subset (n=26):")
    print(corr_26[["feature", "spearman_r", "p_value", "n"]].to_string(index=False))

    print("\n[3/5] Leave-one-out CV performance (n=26, 4 complete features)...")
    loo_r2 = _loo_cv_r2(X_26, y_26)
    print(f"  LOO-CV R² (pooled predictions): {loo_r2:.3f}")

    print("\n[4/5] Fitting Random Forest on n=26 for SHAP...")
    rf = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=3, max_features="sqrt", random_state=RANDOM_STATE
    )
    rf.fit(X_26, y_26)
    print(f"  In-sample R²: {rf.score(X_26, y_26):.3f}")

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_26)

    print("\n[5/5] Generating figures...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_scatter(X_26, y_26, FIGURES_DIR / "shap_scatter_floors_vs_error_distance.png")
    plot_correlation_landscape(corr_62, FIGURES_DIR / "shap_correlation_landscape.png")
    plot_shap_beeswarm(shap_values, X_26, FIGURES_DIR / "shap_beeswarm.png")
    plot_shap_dependence(shap_values, X_26, FIGURES_DIR / "shap_dependence_floors.png")

    # Save outputs
    shap_df = pd.DataFrame(shap_values, index=X_26.index, columns=X_26.columns)
    shap_df.to_csv(OUTPUT_DIR / "shap_values_vocab_delayed_error_distance.csv")
    pd.concat([corr_62, corr_26], ignore_index=True).to_csv(
        OUTPUT_DIR / "pairwise_correlations_vocab_delayed_error_distance.csv", index=False
    )
    summary = pd.DataFrame([{
        "target": str(TARGET_COL),
        "n_target_valid": len(y_62),
        "n_floors_complete": len(y_26),
        "features_used": ", ".join(COMPLETE_FEATURES),
        "loo_cv_r2": loo_r2,
        "floors_spearman_r_in_62": float(corr_62.loc[corr_62["feature"] == "mean__floors", "spearman_r"].values[0]),
        "floors_spearman_p_in_62": float(corr_62.loc[corr_62["feature"] == "mean__floors", "p_value"].values[0]),
        "floors_spearman_r_in_26": float(corr_26.loc[corr_26["feature"] == "mean__floors", "spearman_r"].values[0]),
        "floors_spearman_p_in_26": float(corr_26.loc[corr_26["feature"] == "mean__floors", "p_value"].values[0]),
        "weight_spearman_r_in_26": float(corr_26.loc[corr_26["feature"] == "mean__weight", "spearman_r"].values[0]),
        "weight_spearman_p_in_26": float(corr_26.loc[corr_26["feature"] == "mean__weight", "p_value"].values[0]),
    }])
    summary.to_csv(OUTPUT_DIR / "analysis_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nTarget: {TARGET_LABEL}")
    print(f"  n = {len(y_62)} with valid scores; {len(y_26)} with floors tracking data")
    print(f"\nSignificant associations in floors-tracking subset (n=26):")
    sig = corr_26[corr_26["p_value"] < ALPHA].copy()
    for _, row in sig.iterrows():
        print(f"  {_pretty(row['feature']):20s}  r={row['spearman_r']:+.3f}  p={row['p_value']:.4f}  n={int(row['n'])}")
    print(f"\nPredictive performance (LOO-CV pooled, 4 complete features, n=26):")
    print(f"  R² = {loo_r2:.3f}")
    print(f"\nNote: floors tracking is altimeter-specific (Fitbit model dependent).")
    print(f"  The n=26 subset is self-selected by device type — interpret with caution.")
    print(f"\nTop SHAP contributors (mean |SHAP|):")
    mean_abs = pd.Series(np.abs(shap_values).mean(axis=0), index=X_26.columns)
    for feat, val in mean_abs.sort_values(ascending=False).items():
        print(f"  {_pretty(feat):20s}  {val:.4f}")
    print(f"\nOutputs: figures/ and {OUTPUT_DIR.relative_to(ROOT)}/")
    print("=" * 70)


if __name__ == "__main__":
    run_shap_vocab_analysis()
