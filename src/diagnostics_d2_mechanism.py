"""
Diagnostic 2 + mechanism investigation:
- Inspect real target distributions (describe, non-finite, outliers)
- Identify Fold 4 participants
- Compare Fold 4 features/targets vs rest of cohort
- Look for low-variance or extreme-valued feature columns driving instability

Run: python -m src.diagnostics_d2_mechanism
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

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
OUTPUT_DIR = ROOT / "data" / "diagnostics_outputs"
FIG_DIR = ROOT / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def run_diagnostic_2():
    print("=" * 80)
    print("DIAGNOSTIC 2: TARGET INSPECTION & FOLD 4 MECHANISM")
    print("=" * 80)
    
    # Load data
    print("\n[Step 1] Loading features and targets...")
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
    behavior_summary = load_target_table(RAW_DIR, "behavioral_summary.pkl")
    
    print(f"✓ Features: {means_features.shape}, Targets: {behavior_summary.shape}")
    
    # Identify Fold 4 participants
    print("\n[Step 2] Identifying Fold 4 participants...")
    np.random.seed(42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_indices = list(cv.split(means_features))
    train_4, val_4 = fold_indices[4]
    
    val_participants = means_features.index[val_4]
    train_participants = means_features.index[train_4]
    
    print(f"  Fold 4 validation set: {len(val_4)} participants: {list(val_participants)}")
    print(f"  Fold 4 training set: {len(train_4)} participants")
    
    # Step 3: Inspect real targets
    print("\n[Step 3] TARGET DISTRIBUTION STATISTICS")
    print("-" * 80)
    
    targets_report = []
    for col in behavior_summary.columns:
        y = behavior_summary[col]
        # Convert to numeric, handling non-numeric types
        y_numeric = pd.to_numeric(y, errors='coerce')
        y_finite = y_numeric.dropna()
        # Count infinities after conversion
        n_inf = np.isinf(y_numeric).sum() if y_numeric.notna().any() else 0
        
        report = {
            "target": col,
            "n_total": len(y),
            "n_valid": y_numeric.notna().sum(),
            "n_inf": n_inf,
            "n_finite": len(y_finite),
            "mean": y_finite.mean() if len(y_finite) > 0 else np.nan,
            "std": y_finite.std() if len(y_finite) > 0 else np.nan,
            "min": y_finite.min() if len(y_finite) > 0 else np.nan,
            "max": y_finite.max() if len(y_finite) > 0 else np.nan,
            "range": (y_finite.max() - y_finite.min()) if len(y_finite) > 0 else np.nan,
        }
        targets_report.append(report)
        
        print(f"\n{col}:")
        print(f"  n={report['n_total']}, valid={report['n_valid']}, inf={report['n_inf']}, finite={report['n_finite']}")
        print(f"  mean={report['mean']:.4f}, std={report['std']:.4f}")
        print(f"  min={report['min']:.4f}, max={report['max']:.4f}, range={report['range']:.4f}")
        
        # Flag issues
        if report['n_finite'] < 80:
            print(f"  ⚠ FLAG: Only {report['n_finite']} valid samples (missing data)")
        if report['std'] == 0:
            print(f"  ⚠ FLAG: Zero variance!")
        if report['n_inf'] > 0:
            print(f"  ⚠ FLAG: {report['n_inf']} infinite values")
    
    targets_df = pd.DataFrame(targets_report)
    targets_df.to_csv(OUTPUT_DIR / "d2_target_statistics.csv", index=False)
    
    # Step 4: Feature inspection (general + Fold 4 specific)
    print("\n" + "=" * 80)
    print("[Step 4] FEATURE INSPECTION")
    print("-" * 80)
    
    print("\n[4a] All features:")
    feature_report = []
    for col in means_features.columns:
        x = means_features[col]
        x_finite = x.dropna()
        
        report = {
            "feature": col,
            "n": len(x),
            "n_valid": x_finite.shape[0],
            "n_nan": x.isna().sum(),
            "mean": x_finite.mean(),
            "std": x_finite.std(),
            "min": x_finite.min(),
            "max": x_finite.max(),
            "cv": x_finite.std() / abs(x_finite.mean()) if x_finite.mean() != 0 else np.inf,
        }
        feature_report.append(report)
        
        print(f"  {col}: mean={report['mean']:.4e}, std={report['std']:.4e}, CV={report['cv']:.4f}, n_nan={report['n_nan']}")
        if report['std'] < 0.1:
            print(f"    ⚠ FLAG: Very low std!")
        if report['n_nan'] > 0:
            print(f"    ⚠ FLAG: {report['n_nan']} NaN values")
    
    feature_df = pd.DataFrame(feature_report)
    feature_df.to_csv(OUTPUT_DIR / "d2_feature_statistics.csv", index=False)
    
    # Step 5: Compare Fold 4 vs rest
    print("\n" + "=" * 80)
    print("[Step 5] FOLD 4 vs REST OF COHORT")
    print("-" * 80)
    
    X_fold4_val = means_features.loc[val_participants]
    X_rest = means_features.loc[train_participants]
    
    fold4_comparison = []
    
    print("\nFeature distributions:")
    for col in means_features.columns:
        x_f4 = X_fold4_val[col].dropna()
        x_rest = X_rest[col].dropna()
        
        # Comparison
        report = {
            "feature": col,
            "fold4_n": len(x_f4),
            "fold4_mean": x_f4.mean(),
            "fold4_std": x_f4.std(),
            "fold4_min": x_f4.min(),
            "fold4_max": x_f4.max(),
            "rest_n": len(x_rest),
            "rest_mean": x_rest.mean(),
            "rest_std": x_rest.std(),
            "rest_min": x_rest.min(),
            "rest_max": x_rest.max(),
            "mean_diff": x_f4.mean() - x_rest.mean(),
            "mean_diff_pct": 100 * (x_f4.mean() - x_rest.mean()) / (abs(x_rest.mean()) + 1e-10),
        }
        fold4_comparison.append(report)
        
        print(f"\n  {col}:")
        print(f"    Fold 4: mean={report['fold4_mean']:.4e}, std={report['fold4_std']:.4e}, "
              f"range=[{report['fold4_min']:.4e}, {report['fold4_max']:.4e}]")
        print(f"    Rest:   mean={report['rest_mean']:.4e}, std={report['rest_std']:.4e}, "
              f"range=[{report['rest_min']:.4e}, {report['rest_max']:.4e}]")
        print(f"    Δ mean: {report['mean_diff']:.4e} ({report['mean_diff_pct']:+.1f}%)")
        
        # Flag extreme differences
        if abs(report['mean_diff_pct']) > 50:
            print(f"    ⚠ FLAG: >50% difference in mean!")
        if report['fold4_std'] < report['rest_std'] * 0.1:
            print(f"    ⚠ FLAG: Fold 4 std is <10% of rest (low variance in this fold)")
        if report['fold4_std'] > report['rest_std'] * 10:
            print(f"    ⚠ FLAG: Fold 4 std is >10x rest (high variance in this fold)")
    
    fold4_df = pd.DataFrame(fold4_comparison)
    fold4_df.to_csv(OUTPUT_DIR / "d2_fold4_vs_rest.csv", index=False)
    
    # Step 6: Detailed participant inspection in Fold 4
    print("\n" + "=" * 80)
    print("[Step 6] FOLD 4 INDIVIDUAL PARTICIPANTS")
    print("-" * 80)
    
    print("\nFold 4 validation participants:")
    for pid in val_participants:
        print(f"\n  {pid}:")
        for col in means_features.columns:
            val = means_features.loc[pid, col]
            all_vals = means_features[col].dropna()
            z_score = (val - all_vals.mean()) / (all_vals.std() + 1e-10) if all_vals.std() > 0 else 0
            
            if abs(z_score) > 3:
                print(f"    {col}: {val:.4e} (z={z_score:.2f}) ⚠ EXTREME")
            else:
                print(f"    {col}: {val:.4e} (z={z_score:.2f})")
    
    # Step 7: Plot target distributions
    print("\n" + "=" * 80)
    print("[Step 7] Generating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    
    for idx, col in enumerate(behavior_summary.columns[:4]):
        y = pd.to_numeric(behavior_summary[col], errors='coerce').dropna()
        if len(y) > 1:
            axes[idx].hist(y, bins=15, edgecolor="black", alpha=0.7)
            axes[idx].set_title(col, fontsize=9)
            axes[idx].set_xlabel("Value")
            axes[idx].set_ylabel("Count")
        else:
            axes[idx].text(0.5, 0.5, "No valid data", ha="center", va="center",
                          transform=axes[idx].transAxes)
            axes[idx].set_title(col, fontsize=9)
    
    fig.tight_layout()
    out = FIG_DIR / "diagnostics_target_hists.png"
    fig.savefig(out, dpi=200)
    plt.close()
    print(f"  Wrote: {out}")
    
    # Summary
    print("\n" + "=" * 80)
    print("DIAGNOSTIC 2 SUMMARY")
    print("=" * 80)
    print(f"\nTargets report saved to: {OUTPUT_DIR / 'd2_target_statistics.csv'}")
    print(f"Features report saved to: {OUTPUT_DIR / 'd2_feature_statistics.csv'}")
    print(f"Fold 4 comparison saved to: {OUTPUT_DIR / 'd2_fold4_vs_rest.csv'}")
    print(f"Target histograms saved to: {out}")
    
    # Interpretation
    extreme_features = fold4_df[fold4_df['mean_diff_pct'].abs() > 50]
    low_var_features = fold4_df[fold4_df['fold4_std'] < fold4_df['rest_std'] * 0.1]
    high_var_features = fold4_df[fold4_df['fold4_std'] > fold4_df['rest_std'] * 10]
    
    print(f"\nExtreme differences (>50% mean shift): {len(extreme_features)} features")
    if len(extreme_features) > 0:
        print(extreme_features[['feature', 'mean_diff_pct']].to_string(index=False))
    
    print(f"\nLow variance in Fold 4 (<10% of rest): {len(low_var_features)} features")
    if len(low_var_features) > 0:
        print(low_var_features[['feature', 'fold4_std', 'rest_std']].to_string(index=False))
    
    print(f"\nHigh variance in Fold 4 (>10x rest): {len(high_var_features)} features")
    if len(high_var_features) > 0:
        print(high_var_features[['feature', 'fold4_std', 'rest_std']].to_string(index=False))


if __name__ == "__main__":
    run_diagnostic_2()
