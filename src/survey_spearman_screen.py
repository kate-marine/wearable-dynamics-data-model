"""
Spearman correlation screen: wearable features vs. all survey outcomes.

Mirrors the approach of posthoc_analysis.py (which screened behavior.pkl) but
applied to survey_30.pkl. The original Manning et al. (2022) paper reported only
a handful of pairwise fitness-survey correlations; this screen characterises the
full landscape across all numeric survey columns.

Targets included: all numeric columns with >= 30 valid observations. Columns with
fewer than 5 unique values are flagged as ordinal/Likert. Heavily imbalanced binary
columns (< 5 positive cases) are flagged but retained so the results table is
complete; their correlations should be interpreted with caution.

Outputs (data/survey_screen_outputs/):
  survey_spearman_screen.csv          — all feature × target pairs, ranked by |r|
  survey_target_summary.csv           — per-target metadata (n, n_unique, balance)
  survey_top_correlations.csv         — pairs with |r| >= 0.20

Figures (figures/):
  survey_correlation_heatmap.png      — features × targets heatmap of Spearman r
  survey_top_bar.png                  — bar chart of the 20 strongest |r| pairs

Run: python -m src.survey_spearman_screen
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

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
OUTPUT_DIR = ROOT / "data" / "survey_screen_outputs"

MIN_VALID = 30
ALPHA = 0.05

# One-hot / nominal category prefixes — excluded because Spearman r on
# dummy-coded categories is not a meaningful ordinal association test.
EXCLUDE_PREFIXES = {"gender", "race", "degree", "location"}

# Columns excluded individually: binary with near-zero variance or NLP artifacts
# with opaque semantics (exercise_motivation_sentiment has only 5 values that are
# all > 0.93 — it's not a meaningful scale).
EXCLUDE_COLS = {
    "exercise motivation sentiment",
    "feedback: sentiment",        # n=34 only
    "feedback: number of words",  # text count, sparse
    "color vision",               # almost all identical (111/113)
    "reported exercise today",    # binary event flag
    "accurate exercise report",   # binary
    "plan to exercise",           # binary
    "reported exercise accuracy", # check at runtime
}


def _pretty_feature(col: str) -> str:
    return col.removeprefix("mean__").replace("_", " ")


def _pretty_target(col: object) -> str:
    if isinstance(col, tuple):
        parts = [str(p) for p in col if str(p).strip()]
        return ": ".join(parts)
    return str(col)


def _select_numeric_targets(survey: pd.DataFrame) -> pd.DataFrame:
    """Coerce all columns to numeric, drop one-hot categories and sparse cols."""
    rows = {}
    for col in survey.columns:
        prefix = col[0] if isinstance(col, tuple) else ""
        label = _pretty_target(col)

        if prefix in EXCLUDE_PREFIXES:
            continue
        if any(ex in label for ex in EXCLUDE_COLS):
            continue

        s = pd.to_numeric(survey[col], errors="coerce")
        n_valid = s.notna().sum()
        if n_valid >= MIN_VALID:
            rows[col] = s

    return pd.DataFrame(rows, index=survey.index)


def _target_summary(targets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in targets.columns:
        s = targets[col].dropna()
        n_unique = s.nunique()
        is_binary = n_unique == 2
        n_positive = int((s == 1).sum()) if is_binary else None
        rows.append({
            "target": _pretty_target(col),
            "raw_col": str(col),
            "n_valid": int(s.notna().sum()),
            "n_unique": n_unique,
            "is_binary": is_binary,
            "n_positive": n_positive,
            "low_positive_flag": (is_binary and n_positive is not None and n_positive < 5),
            "min": float(s.min()),
            "max": float(s.max()),
            "mean": float(s.mean()),
        })
    return pd.DataFrame(rows)


def _run_screen(X: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise Spearman r for every feature × target combination."""
    rows = []
    for target_col in targets.columns:
        y = targets[target_col].dropna()
        if len(y) < MIN_VALID:
            continue
        x_aligned = X.reindex(y.index)
        for feature_col in x_aligned.columns:
            valid = x_aligned[feature_col].notna()
            n = int(valid.sum())
            if n < MIN_VALID:
                continue
            r, p = stats.spearmanr(x_aligned.loc[valid, feature_col], y[valid])
            if pd.notna(r):
                rows.append({
                    "target": _pretty_target(target_col),
                    "feature": _pretty_feature(feature_col),
                    "spearman_r": float(r),
                    "abs_r": float(abs(r)),
                    "p_value": float(p),
                    "n": n,
                    "significant": p < ALPHA,
                })
    return (
        pd.DataFrame(rows)
        .sort_values("abs_r", ascending=False)
        .reset_index(drop=True)
    )


def plot_heatmap(screen: pd.DataFrame, out_path: Path) -> None:
    """Heatmap of Spearman r: features (rows) × survey targets (cols)."""
    pivot = screen.pivot_table(
        index="feature", columns="target", values="spearman_r", aggfunc="first"
    )
    # Cluster by absolute row-sum for visual ordering
    pivot = pivot.loc[pivot.abs().sum(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 0.55), max(5, len(pivot) * 0.45)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", vmin=-0.6, vmax=0.6)
    fig.colorbar(im, ax=ax, label="Spearman $r$", shrink=0.7)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=60, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_title(
        "Wearable features vs. survey outcomes — Spearman $r$\n"
        "(pairwise complete observations)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_top_pairs(screen: pd.DataFrame, out_path: Path, n_top: int = 20) -> None:
    """Horizontal bar chart of the top-N feature × target pairs by |r|."""
    top = screen.head(n_top).copy()
    top["label"] = top["feature"] + "  →  " + top["target"]
    colors = ["#C44E52" if r > 0 else "#4C72B0" for r in top["spearman_r"]]
    sig_markers = ["*" if s else "" for s in top["significant"]]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(
        range(len(top)), top["abs_r"].values,
        color=colors, edgecolor="white", linewidth=0.5,
    )
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(
        [f"{lbl}  {m}" for lbl, m in zip(top["label"], sig_markers)],
        fontsize=8.5,
    )
    ax.invert_yaxis()
    ax.set_xlabel("| Spearman $r$ |", fontsize=11)
    ax.set_title(
        f"Top {n_top} wearable–survey associations\n"
        "Red = positive $r$, Blue = negative $r$, * = $p$ < .05",
        fontsize=11, fontweight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def run_survey_spearman_screen() -> pd.DataFrame:
    print("=" * 70)
    print("SURVEY SPEARMAN SCREEN")
    print("=" * 70)

    print("\n[1/4] Loading wearable features...")
    long_df = load_raw_long_table(RAW_DIR)
    vocabulary = discover_variable_vocabulary(long_df)
    coverage_summary = summarize_variable_coverage(long_df)
    wearable_signals = select_wearable_signals(
        vocabulary, coverage_summary=coverage_summary,
        min_median_days=300, min_participants=60,
    )
    panel = build_daily_panel(long_df, variables=wearable_signals)
    X = build_mean_feature_matrix(panel, wearable_signals)
    print(f"  Feature matrix: {X.shape}")

    print("\n[2/4] Loading and profiling survey targets...")
    survey = load_target_table(RAW_DIR, "survey_30.pkl")
    targets = _select_numeric_targets(survey)
    summary = _target_summary(targets)
    print(f"  Total numeric columns: {len(targets.columns)}")
    print(f"  Columns with n >= {MIN_VALID}: {len(targets.columns)}")
    flagged = summary[summary["low_positive_flag"]]
    if len(flagged):
        print(f"  ⚠ Low-positive binary columns (< 5 cases, noisy): {flagged['target'].tolist()}")

    print("\n[3/4] Running pairwise Spearman screen...")
    screen = _run_screen(X, targets)
    n_sig = screen["significant"].sum()
    print(f"  Feature–target pairs evaluated: {len(screen):,}")
    print(f"  Significant at p < .05: {n_sig} ({100*n_sig/len(screen):.1f}%)")

    print("\n[4/4] Generating outputs...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    screen.to_csv(OUTPUT_DIR / "survey_spearman_screen.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "survey_target_summary.csv", index=False)
    screen[screen["abs_r"] >= 0.20].to_csv(
        OUTPUT_DIR / "survey_top_correlations.csv", index=False
    )
    plot_heatmap(screen, FIGURES_DIR / "survey_correlation_heatmap.png")
    plot_top_pairs(screen, FIGURES_DIR / "survey_top_bar.png")

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nTop 25 feature–survey associations by |Spearman r|:")
    print(
        screen.head(25)[["target", "feature", "spearman_r", "p_value", "n", "significant"]]
        .to_string(index=False)
    )
    print(f"\nPer-target maximum |r| (top 15):")
    top_by_target = (
        screen.groupby("target")
        .apply(lambda df: df.loc[df["abs_r"].idxmax()])
        .sort_values("abs_r", ascending=False)
        .head(15)[["target", "feature", "spearman_r", "p_value", "n"]]
    )
    print(top_by_target.to_string(index=False))
    print(f"\nOutputs: {OUTPUT_DIR.relative_to(ROOT)}/  and  figures/")
    print("=" * 70)

    return screen


if __name__ == "__main__":
    run_survey_spearman_screen()
