from __future__ import annotations

from pathlib import Path

import pandas as pd
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
from .modeling import evaluate_targets, evaluate_targets_with_alpha


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "data" / "phase2_outputs"


def run_phase2() -> None:
    # Load and prepare data 
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

    #  means-only baseline
    means_features = build_mean_feature_matrix(panel, wearable_signals)

    # extract dynamic features and build augmented matrix
    dynamic_features = extract_dynamic_features(panel, wearable_signals)
    augmented_features = build_augmented_feature_matrix(means_features, dynamic_features)

    # Standardize for better interpretability
    scaler = StandardScaler()
    augmented_scaled = pd.DataFrame(
        scaler.fit_transform(augmented_features),
        index=augmented_features.index,
        columns=augmented_features.columns,
    )

    # Load targets
    behavior_summary = load_target_table(RAW_DIR, "behavioral_summary.pkl")

    # Evaluate both baselines with stronger alpha 
    baseline_results = evaluate_targets(means_features, behavior_summary)
    augmented_results = evaluate_targets_with_alpha(augmented_scaled, behavior_summary, alpha=100.0)

    # Compute lift
    lift_results = pd.merge(
        baseline_results.rename(columns={col: f"baseline_{col}" for col in baseline_results.columns if col != "target"}),
        augmented_results.rename(columns={col: f"augmented_{col}" for col in augmented_results.columns if col != "target"}),
        on="target",
        how="inner",
    )
    lift_results["r2_lift"] = lift_results["augmented_r2_mean"] - lift_results["baseline_r2_mean"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline_results.to_csv(OUTPUT_DIR / "means_only_baseline_results.csv", index=False)
    augmented_results.to_csv(OUTPUT_DIR / "augmented_model_results.csv", index=False)
    dynamic_features.to_csv(OUTPUT_DIR / "dynamic_features_matrix.csv")
    augmented_features.to_csv(OUTPUT_DIR / "augmented_features_matrix.csv")
    lift_results.to_csv(OUTPUT_DIR / "augmented_vs_baseline_comparison.csv", index=False)

    print(f"Selected {len(wearable_signals)} wearable signals for Phase 2.")
    print(f"Means-only feature matrix shape: {means_features.shape}")
    print(f"Dynamic features extracted: {dynamic_features.shape}")
    print(f"Augmented feature matrix shape: {augmented_features.shape}")
    print("\n--- Means-only baseline ---")
    print(baseline_results.head().to_string(index=False))
    print("\n--- Augmented model results ---")
    print(augmented_results.head().to_string(index=False))
    print("\n--- R² lift (augmented - baseline) ---")
    print(lift_results[["target", "baseline_r2_mean", "augmented_r2_mean", "r2_lift"]].to_string(index=False))


if __name__ == "__main__":
    run_phase2()
