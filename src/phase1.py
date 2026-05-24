from __future__ import annotations

from pathlib import Path

from .features import build_mean_feature_matrix, build_valid_day_matrix, select_wearable_signals
from .loading import (
    build_daily_panel,
    discover_variable_vocabulary,
    load_raw_long_table,
    load_target_table,
    summarize_variable_coverage,
)
from .modeling import evaluate_targets
from .plots import plot_valid_day_histograms


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
FIGURES_DIR = ROOT / "figures"
OUTPUT_DIR = ROOT / "data" / "phase1_outputs"


def run_phase1() -> None:
    long_df = load_raw_long_table(RAW_DIR)
    vocabulary = discover_variable_vocabulary(long_df)
    coverage_summary = summarize_variable_coverage(long_df)
    wearable_signals = select_wearable_signals(vocabulary, coverage_summary=coverage_summary, min_median_days=30)
    panel = build_daily_panel(long_df, variables=wearable_signals)
    valid_days = build_valid_day_matrix(panel, wearable_signals)
    features = build_mean_feature_matrix(panel, wearable_signals)

    behavior_summary = load_target_table(RAW_DIR, "behavioral_summary.pkl")
    baseline_results = evaluate_targets(features, behavior_summary)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    coverage_summary.to_csv(OUTPUT_DIR / "variable_coverage_summary.csv")
    valid_days.to_csv(OUTPUT_DIR / "participant_valid_days.csv")
    features.to_csv(OUTPUT_DIR / "participant_mean_features.csv")
    baseline_results.to_csv(OUTPUT_DIR / "means_only_baseline_results.csv", index=False)

    plot_valid_day_histograms(valid_days, FIGURES_DIR / "phase1_valid_day_histograms.png")

    print(f"Loaded {len(long_df):,} raw rows across {len(vocabulary)} variables.")
    print(f"Selected {len(wearable_signals)} wearable signals for Phase 1.")
    print(f"Built feature matrix with shape {features.shape}.")
    print("Top baseline results:")
    print(baseline_results.head().to_string(index=False))


if __name__ == "__main__":
    run_phase1()
