"""
Generate figures for the project analyses.
Produces:
- coverage_by_variable.png
- phase2_r2_comparison.png
- exploratory_r2_lift_histogram.png
- top_spearman_correlations.png
- top_correlation_scatter_1.png, _2.png, _3.png

Run as: python -m src.visualizations
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT / "data"

sns.set(style="whitegrid")


def plot_coverage():
    cov_file = DATA_DIR / "phase1_outputs" / "variable_coverage_summary.csv"
    if not cov_file.exists():
        print("coverage file not found:", cov_file)
        return
    cov = pd.read_csv(cov_file)
    # keep top 20 by median_valid_days then by n_participants
    cov_sort = cov.sort_values(["median_valid_days", "n_participants"], ascending=False).head(20)
    plt.figure(figsize=(8,6))
    sns.barplot(x="median_valid_days", y="variable", data=cov_sort, palette="viridis")
    plt.title("Top 20 variables by median valid days")
    plt.xlabel("Median valid days")
    plt.ylabel("")
    plt.tight_layout()
    out = FIG_DIR / "coverage_by_variable.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print("Wrote", out)


def plot_phase2_r2_comparison():
    f = DATA_DIR / "phase2_outputs" / "augmented_vs_baseline_comparison.csv"
    if not f.exists():
        print("phase2 file not found:", f)
        return
    df = pd.read_csv(f)
    # keep top 8 for readability
    df_plot = df.copy()
    df_plot = df_plot.sort_values("baseline_r2_mean")
    plt.figure(figsize=(9,6))
    x = range(len(df_plot))
    plt.plot(x, df_plot["baseline_r2_mean"], marker="o", label="Baseline R²")
    plt.plot(x, df_plot["augmented_r2_mean"], marker="o", label="Augmented R²")
    plt.xticks(x, df_plot["target"], rotation=90, fontsize=8)
    plt.ylabel("R²")
    plt.title("Phase 2: Baseline vs Augmented R² (all targets)")
    plt.legend()
    plt.tight_layout()
    out = FIG_DIR / "phase2_r2_comparison.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print("Wrote", out)


def plot_exploratory_lift_hist():
    f = DATA_DIR / "exploratory_full_behavior_outputs" / "lift_comparison_all_outcomes.csv"
    if not f.exists():
        print("exploratory lift file not found:", f)
        return
    df = pd.read_csv(f)
    # r2_lift column may be named differently; try common names
    if "r2_lift" in df.columns:
        lifts = df["r2_lift"]
    elif "r2_mean" in df.columns and "baseline_r2_mean" in df.columns:
        lifts = df["augmented_r2_mean"] - df["baseline_r2_mean"]
    else:
        # fallback: if file only contains baseline r2, plot distribution
        lifts = df["r2_mean"]
    plt.figure(figsize=(6,4))
    sns.histplot(lifts.dropna(), bins=30, kde=False, color="#4c78a8")
    plt.axvline(0, color="k", linestyle="--")
    plt.xlabel("R² lift (augmented - baseline)")
    plt.title("Distribution of R² lifts (exploratory outcomes)")
    plt.tight_layout()
    out = FIG_DIR / "exploratory_r2_lift_histogram.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print("Wrote", out)


def plot_top_spearman():
    f = DATA_DIR / "posthoc_analysis_outputs" / "feature_target_spearman_screen.csv"
    if not f.exists():
        print("spearman file not found:", f)
        return
    df = pd.read_csv(f)
    top = df.sort_values("abs_spearman_r", ascending=False).head(10)
    plt.figure(figsize=(8,5))
    sns.barplot(x="abs_spearman_r", y="feature", data=top, palette="magma")
    plt.xlabel("|Spearman r|")
    plt.title("Top 10 feature-target Spearman correlations (absolute)")
    plt.tight_layout()
    out = FIG_DIR / "top_spearman_correlations.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print("Wrote", out)
    # also create scatter for top 3
    for i, row in top.head(3).iterrows():
        target = row["target"]
        feature = row["feature"]
        # read means features and targets
        means_f = DATA_DIR / "phase1_outputs" / "participant_mean_features.csv"
        targets_f = DATA_DIR / "raw" / "behavior.pkl"
        try:
            means = pd.read_csv(means_f, index_col=0)
        except Exception:
            print("cannot read means features for scatter")
            return
        # load behavior.pkl via pandas (pickle)
        try:
            import pickle
            with open(DATA_DIR / "raw" / "behavior.pkl", "rb") as fh:
                beh = pickle.load(fh)
        except Exception:
            beh = None
        if beh is None:
            print("behavior.pkl not loadable for scatter")
            return
        # ensure index alignment
        if feature not in means.columns:
            print(f"feature {feature} not found in means; skipping scatter")
            continue
        if target not in beh.columns:
            print(f"target {target} not found in behavior.pkl; skipping scatter")
            continue
        df_sc = pd.concat([means[feature], beh[target]], axis=1, join="inner").dropna()
        if df_sc.empty:
            print("no overlapping data for scatter", feature, target)
            continue
        plt.figure(figsize=(5,4))
        sns.regplot(x=feature, y=target, data=df_sc, scatter_kws={"s":20})
        plt.title(f"{feature} vs {target}")
        plt.tight_layout()
        out = FIG_DIR / f"top_correlation_scatter_{i}.png"
        plt.savefig(out, dpi=200)
        plt.close()
        print("Wrote", out)


def main():
    plot_coverage()
    plot_phase2_r2_comparison()
    plot_exploratory_lift_hist()
    plot_top_spearman()


if __name__ == "__main__":
    main()
