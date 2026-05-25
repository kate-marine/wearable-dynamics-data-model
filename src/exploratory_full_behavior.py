"""
Exploratory Phase: Full behavior.pkl outcomes analysis

This phase reuses the Phase 1 and Phase 2 pipelines but applies them to all 54
fine-grained behavioral outcomes in behavior.pkl instead of the 8 summary metrics.

Goal: Identify whether any task-specific or fine-grained metrics (primacy, recency,
clustering strategies, error measures) show positive R² lift from dynamic wearable features.

Process:
1. Run Phase 1 baseline (means-only) on all 54 behavior outcomes
2. Run Phase 2 augmented (means + dynamics) on all 54 outcomes
3. Compute R² lift for each outcome
4. Identify which (if any) outcomes show positive lift
5. Summarize findings for a possible follow-up deep dive

This is not a formally scheduled phase but a pragmatic exploration before Phase 3 (FDA).
"""

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
OUTPUT_DIR = ROOT / "data" / "exploratory_full_behavior_outputs"


def run_exploratory_full_behavior() -> None:
    """Run Phase 1 & 2 analysis on all 54 fine-grained behavior outcomes."""

    print("=" * 80)
    print("EXPLORATORY ANALYSIS: Full behavior.pkl outcomes (54 metrics)")
    print("=" * 80)

    # Load and prepare data (same as Phase 1 & 2)
    print("\n[1/5] Loading raw wearable data...")
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

    print(f"  Loaded {len(long_df):,} raw rows")
    print(f"  Selected {len(wearable_signals)} wearable signals")
    print(f"  Panel shape: {panel.shape}")

    # Build feature matrices
    print("\n[2/5] Building feature matrices...")
    means_features = build_mean_feature_matrix(panel, wearable_signals)
    print(f"  Means-only features: {means_features.shape}")

    print("\n[3/5] Extracting dynamic features...")
    dynamic_features = extract_dynamic_features(panel, wearable_signals)
    augmented_features = build_augmented_feature_matrix(means_features, dynamic_features)
    scaler = StandardScaler()
    augmented_scaled = pd.DataFrame(
        scaler.fit_transform(augmented_features),
        index=augmented_features.index,
        columns=augmented_features.columns,
    )
    print(f"  Dynamic features: {dynamic_features.shape}")
    print(f"  Augmented features: {augmented_scaled.shape}")

    # Load full behavior outcomes
    print("\n[4/5] Loading full behavior.pkl (54 outcomes)...")
    behavior_full = load_target_table(RAW_DIR, "behavior.pkl")
    print(f"  Behavior shape: {behavior_full.shape}")
    print(f"  Outcomes: {list(behavior_full.columns)[:5]}... (showing first 5)")
    
    # Filter to only outcomes with sufficient valid data
    import numpy as np
    valid_cols = [col for col in behavior_full.columns 
                  if (pd.notna(behavior_full[col]).sum() - np.isinf(behavior_full[col]).sum()) >= 30]
    
    behavior_full = behavior_full[valid_cols].copy()
    # Replace infinities with NaN (will be imputed by Ridge pipeline)
    behavior_full = behavior_full.replace([np.inf, -np.inf], np.nan)
    print(f"  After filtering: {behavior_full.shape} (kept {len(valid_cols)} valid outcomes, removed {54 - len(valid_cols)})")

    # Evaluate both baselines on all outcomes
    print("\n[5/5] Evaluating means-only and augmented models on all 54 outcomes...")
    baseline_results = evaluate_targets(means_features, behavior_full)
    augmented_results = evaluate_targets_with_alpha(augmented_scaled, behavior_full, alpha=100.0)

    # Compute lift
    lift_results = pd.merge(
        baseline_results.rename(
            columns={col: f"baseline_{col}" for col in baseline_results.columns if col != "target"}
        ),
        augmented_results.rename(
            columns={col: f"augmented_{col}" for col in augmented_results.columns if col != "target"}
        ),
        on="target",
        how="inner",
    )
    lift_results["r2_lift"] = (
        lift_results["augmented_r2_mean"] - lift_results["baseline_r2_mean"]
    )

    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline_results.to_csv(OUTPUT_DIR / "baseline_all_outcomes.csv", index=False)
    augmented_results.to_csv(OUTPUT_DIR / "augmented_all_outcomes.csv", index=False)
    lift_results.to_csv(OUTPUT_DIR / "lift_comparison_all_outcomes.csv", index=False)

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    print(f"\nTotal outcomes tested: {len(lift_results)}")
    print(f"Outcomes with positive R² lift: {(lift_results['r2_lift'] > 0).sum()}")
    print(f"Outcomes with negative R² lift: {(lift_results['r2_lift'] < 0).sum()}")

    print("\n--- TOP 10 OUTCOMES (by R² lift) ---")
    print(lift_results[["target", "baseline_r2_mean", "augmented_r2_mean", "r2_lift"]].head(10).to_string(index=False))

    print("\n--- BOTTOM 10 OUTCOMES (by R² lift) ---")
    print(lift_results[["target", "baseline_r2_mean", "augmented_r2_mean", "r2_lift"]].tail(10).to_string(index=False))

    positive_lift = lift_results[lift_results["r2_lift"] > 0]
    if not positive_lift.empty:
        print("\n--- OUTCOMES WITH POSITIVE R² LIFT ---")
        print(positive_lift[["target", "baseline_r2_mean", "augmented_r2_mean", "r2_lift"]].to_string(index=False))
    else:
        print("\n--- NO OUTCOMES WITH POSITIVE R² LIFT ---")
        print("The augmented model does not improve upon the means-only baseline for any outcome.")

    # Distribution stats
    print(f"\n--- DISTRIBUTION OF R² LIFTS ---")
    print(f"  Mean: {lift_results['r2_lift'].mean():.4f}")
    print(f"  Median: {lift_results['r2_lift'].median():.4f}")
    print(f"  Std: {lift_results['r2_lift'].std():.4f}")
    print(f"  Min: {lift_results['r2_lift'].min():.4f}")
    print(f"  Max: {lift_results['r2_lift'].max():.4f}")

    print("\n" + "=" * 80)
    print(f"Outputs saved to: {OUTPUT_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    run_exploratory_full_behavior()
