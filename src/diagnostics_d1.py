"""
Diagnostic 1: Synthetic-target sanity check (MOST IMPORTANT)

Tests whether the CV harness can recover a signal we KNOW is present.
If this fails, the harness is broken and prior results are invalid.

Run: python -m src.diagnostics_d1
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .features import build_mean_feature_matrix, select_wearable_signals
from .loading import (
    build_daily_panel,
    discover_variable_vocabulary,
    load_raw_long_table,
    summarize_variable_coverage,
)
from .modeling import ridge_baseline_cv

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "data" / "diagnostics_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_diagnostic_1():
    """Run Diagnostic 1: synthetic-target sanity check."""
    
    print("=" * 80)
    print("DIAGNOSTIC 1: SYNTHETIC-TARGET SANITY CHECK")
    print("=" * 80)
    print("\nThis test checks whether the CV harness can recover a signal we know is present.")
    print("If this fails, the harness is broken and prior results are invalid.\n")
    
    # Step 1: Load the means-only feature matrix (exactly as Phase 1 does)
    print("[Step 1] Loading means-only feature matrix (113 × 14)...")
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
    
    print(f"  ✓ Loaded means-only matrix: {means_features.shape}")
    print(f"  Selected signals: {wearable_signals}")
    
    # Check which features are present for synthetic construction
    print(f"\n[Step 2] Checking feature columns for synthetic target construction...")
    required_features = [f"mean__{sig}" for sig in ["steps", "weight"]]
    available = [f for f in required_features if f in means_features.columns]
    if len(available) < 2:
        print(f"  Warning: only found {len(available)}/2 required features: {available}")
        print(f"  Available columns: {list(means_features.columns)}")
        # Use whatever is available
        if len(means_features.columns) >= 2:
            feat_a = means_features.columns[0]
            feat_b = means_features.columns[1]
            print(f"  Using fallback: {feat_a} and {feat_b}")
        else:
            raise ValueError("Not enough features to construct synthetic target")
    else:
        feat_a, feat_b = available[0], available[1]
    
    print(f"  ✓ Using features: {feat_a}, {feat_b}")
    
    # Step 3: Construct synthetic targets
    print(f"\n[Step 3] Constructing synthetic targets...")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(means_features.fillna(means_features.mean())),
        index=means_features.index,
        columns=means_features.columns,
    )
    
    # Synthetic target with KNOWN SIGNAL
    # y_synth = 3.0 * z(steps) - 2.0 * z(weight) + noise
    np.random.seed(42)
    noise = np.random.normal(0, 0.3, len(X_scaled))
    y_synth_signal = (
        3.0 * X_scaled[feat_a].values
        - 2.0 * X_scaled[feat_b].values
        + noise
    )
    y_synth_signal = pd.Series(y_synth_signal, index=means_features.index)
    
    print(f"  ✓ Synthetic signal target constructed: y = 3*z({feat_a}) - 2*z({feat_b}) + noise(std=0.3)")
    print(f"    Signal mean: {y_synth_signal.mean():.4f}, std: {y_synth_signal.std():.4f}")
    
    # Pure noise target (sanity control)
    y_pure_noise = pd.Series(
        np.random.normal(0, 1.0, len(X_scaled)),
        index=means_features.index,
    )
    print(f"  ✓ Pure noise target constructed (sanity control)")
    print(f"    Noise mean: {y_pure_noise.mean():.4f}, std: {y_pure_noise.std():.4f}")
    
    # Step 4: Run through the EXACT SAME baseline pipeline
    print(f"\n[Step 4] Running targets through baseline pipeline (same as Phase 1)...")
    
    # Test synthetic signal
    print(f"\n  [4a] Synthetic signal target:")
    try:
        metrics_signal = ridge_baseline_cv(means_features, y_synth_signal)
        print(f"    ✓ Cross-validated R²: {metrics_signal['r2_mean']:.6f} ± {metrics_signal['r2_std']:.6f}")
        print(f"    MAE: {metrics_signal['mae_mean']:.6f} ± {metrics_signal['mae_std']:.6f}")
        print(f"    RMSE: {metrics_signal['rmse_mean']:.6f} ± {metrics_signal['rmse_std']:.6f}")
        
        signal_r2 = metrics_signal['r2_mean']
    except Exception as e:
        print(f"    ✗ FAILED: {e}")
        signal_r2 = None
    
    # Test pure noise
    print(f"\n  [4b] Pure noise target (control):")
    try:
        metrics_noise = ridge_baseline_cv(means_features, y_pure_noise)
        print(f"    ✓ Cross-validated R²: {metrics_noise['r2_mean']:.6f} ± {metrics_noise['r2_std']:.6f}")
        print(f"    MAE: {metrics_noise['mae_mean']:.6f} ± {metrics_noise['mae_std']:.6f}")
        print(f"    RMSE: {metrics_noise['rmse_mean']:.6f} ± {metrics_noise['rmse_std']:.6f}")
        
        noise_r2 = metrics_noise['r2_mean']
    except Exception as e:
        print(f"    ✗ FAILED: {e}")
        noise_r2 = None
    
    # Step 5: Interpretation
    print(f"\n" + "=" * 80)
    print("INTERPRETATION:")
    print("=" * 80)
    
    if signal_r2 is None or noise_r2 is None:
        print("\n✗ DIAGNOSTIC 1 INCOMPLETE: harness failed on one or both targets.")
        print("  Cannot proceed; check error messages above.")
        return None
    
    # Key thresholds
    signal_ok = signal_r2 > 0.5
    noise_ok = -0.3 <= noise_r2 <= 0.3
    
    print(f"\nSignal R²: {signal_r2:.6f}")
    print(f"  Expected: > 0.5 (strong recovery of known signal)")
    print(f"  Status: {'✓ PASS' if signal_ok else '✗ FAIL'}")
    
    print(f"\nNoise R²: {noise_r2:.6f}")
    print(f"  Expected: in range [-0.3, 0.3] (near-zero for random target)")
    print(f"  Status: {'✓ PASS' if noise_ok else '✗ FAIL'}")
    
    print(f"\n" + "-" * 80)
    if signal_ok and noise_ok:
        print("RESULT: ✓ HARNESS IS WORKING CORRECTLY")
        print("  The CV pipeline recovers known signal and rejects noise.")
        print("  → Proceed to Diagnostics 2 & 3 to inspect the real targets.")
        print("  → The negative R² on real data is likely a target-data problem or true null.")
    elif not signal_ok:
        print("RESULT: ✗ HARNESS IS BROKEN")
        print("  The CV pipeline FAILED to recover a strong synthetic signal.")
        print("  → This is a critical bug in the harness (preprocessing leakage, misalignment, or NaN handling).")
        print("  → Prior results are INVALID. Proceed to Diagnostic 4 to localize the bug.")
    elif not noise_ok:
        print("RESULT: ✗ HARNESS PRODUCES BIASED R² ON RANDOM DATA")
        print("  Pure noise returned R² = {:.6f}, outside the [-0.3, 0.3] band.".format(noise_r2))
        print("  This suggests systematic bias or instability in the harness.")
    
    print("=" * 80)
    
    # Save results
    results = {
        "diagnostic": "D1_synthetic_target",
        "signal_r2_mean": float(signal_r2),
        "signal_r2_std": float(metrics_signal['r2_std']),
        "noise_r2_mean": float(noise_r2),
        "noise_r2_std": float(metrics_noise['r2_std']),
        "signal_pass": bool(signal_ok),
        "noise_pass": bool(noise_ok),
        "overall_pass": bool(signal_ok and noise_ok),
    }
    
    import json
    out_file = OUTPUT_DIR / "d1_synthetic_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_file}")
    
    return results


if __name__ == "__main__":
    run_diagnostic_1()
