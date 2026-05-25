"""
Data quality inspection and cleaning:
- Identify impossible/extreme values (e.g., water_log > 10000, zero BMI/weight, etc.)
- Winsorize or remove based on domain knowledge
- Return cleaned X and y matrices

Run: python -m src.data_cleaning
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import iqr

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
CLEANED_DIR = ROOT / "data" / "cleaned"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CLEANED_DIR.mkdir(parents=True, exist_ok=True)


# Domain knowledge: reasonable ranges for wearable features
FEATURE_BOUNDS = {
    'mean__bmi': (15, 60),              # Realistic BMI range
    'mean__bodyfat': (0, 60),           # Percent body fat 0-60%
    'mean__cal': (800, 4000),           # Daily calories
    'mean__cal_bmr': (800, 3500),       # BMR calories
    'mean__distance': (0, 50),          # Miles per day
    'mean__fair_act_mins': (0, 500),    # Fair activity minutes
    'mean__floors': (0, 100),           # Floors climbed per day
    'mean__food_cal_log': (0, 5000),    # Logged food calories (sparse)
    'mean__light_act_mins': (0, 1000),  # Light activity mins
    'mean__sed_mins': (0, 1440),        # Sedentary mins (max 1440 in a day)
    'mean__steps': (0, 50000),          # Steps per day (reasonable upper bound)
    'mean__very_act_mins': (0, 500),    # Very active mins
    'mean__water_log': (0, 5000),       # Water logged (cups/ml, reasonable upper)
    'mean__weight': (40, 400),          # Weight in lbs
}

TARGET_BOUNDS = {
    'Free recall (immediate)': (0, 1),
    'Free recall (delayed)': (0, 1),
    'Foreign language flashcards (immediate)': (0, 1),
    'Foreign language flashcards (delayed)': (0, 1),
}


def detect_impossible_values(X, y):
    """Find values outside domain bounds and suspicious patterns."""
    print("=" * 80)
    print("INVALID VALUE DETECTION")
    print("=" * 80)
    
    X_issues = {}
    for col in X.columns:
        if col not in FEATURE_BOUNDS:
            continue
        
        lower, upper = FEATURE_BOUNDS[col]
        x = X[col]
        
        out_of_bounds = (x < lower) | (x > upper)
        n_invalid = out_of_bounds.sum()
        
        if n_invalid > 0:
            X_issues[col] = {
                'n_invalid': n_invalid,
                'invalid_particles': X[out_of_bounds].index.tolist(),
                'values': x[out_of_bounds].values.tolist(),
            }
            print(f"\n{col}: {n_invalid} values outside [{lower}, {upper}]")
            for pid in X[out_of_bounds].index.unique():
                val = X.loc[pid, col]
                print(f"  {pid}: {val:.2e}")
    
    # Check for suspicious patterns (e.g., all zeros indicating missing)
    print("\n" + "-" * 80)
    print("SUSPICIOUS PATTERNS (likely missing data codes):")
    for col in X.columns:
        x = X[col].dropna()
        if len(x) > 0 and (x == 0).sum() > len(x) * 0.5:
            n_zeros = (x == 0).sum()
            print(f"  {col}: {n_zeros}/{len(x)} values are zero (likely missing code)")
    
    return X_issues


def clean_data(X, y, strategy='winsorize_extreme'):
    """
    Clean data by:
    - Winsorizing values beyond domain bounds to bounds
    - Optionally removing rows with too many invalid valuesv
    
    strategy: 'winsorize_extreme' | 'remove_invalid_rows'
    """
    print("\n" + "=" * 80)
    print(f"DATA CLEANING: {strategy}")
    print("=" * 80)
    
    X_clean = X.copy()
    
    for col in X_clean.columns:
        if col not in FEATURE_BOUNDS:
            continue
        
        lower, upper = FEATURE_BOUNDS[col]
        
        # Count out-of-bounds before
        n_before = ((X_clean[col] < lower) | (X_clean[col] > upper)).sum()
        
        if n_before > 0:
            # Winsorize: cap values at bounds
            X_clean[col] = X_clean[col].clip(lower=lower, upper=upper)
            print(f"  {col}: winsorized {n_before} values to [{lower}, {upper}]")
    
    # Optionally remove rows with many NaNs
    if strategy == 'remove_invalid_rows':
        # Mark rows where all/most wearable data is missing
        feature_cols = list(FEATURE_BOUNDS.keys())
        nan_frac = X_clean[feature_cols].isna().sum(axis=1) / len(feature_cols)
        to_remove = nan_frac > 0.5
        n_remove = to_remove.sum()
        if n_remove > 0:
            print(f"  Removing {n_remove} rows with >50% missing features")
            X_clean = X_clean[~to_remove]
            y_clean = y[~to_remove]
        else:
            y_clean = y.copy()
    else:
        y_clean = y.copy()
    
    print(f"\n✓ Cleaned X: {X_clean.shape}, y: {y_clean.shape}")
    
    return X_clean, y_clean


def run_cleaning_diagnostic():
    print("=" * 80)
    print("DATA QUALITY & CLEANING DIAGNOSTIC")
    print("=" * 80)
    
    # Load data
    print("\n[Step 1] Loading data...")
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
    behavior_summary = load_target_table(RAW_DIR, "behavioral_summary.pkl")
    
    print(f"✓ Original X: {X.shape}, y: {behavior_summary.shape}")
    
    # Detect issues
    print("\n[Step 2] Detecting invalid values...")
    X_issues = detect_impossible_values(X, behavior_summary)
    
    # Clean
    print("\n[Step 3] Applying winsorization...")
    X_clean, y_clean = clean_data(X, behavior_summary, strategy='winsorize_extreme')
    
    # Save cleaned data
    X_clean.to_csv(CLEANED_DIR / "X_features_cleaned.csv")
    y_clean.to_csv(CLEANED_DIR / "y_targets_cleaned.csv")
    
    print(f"\n✓ Saved cleaned data to {CLEANED_DIR}")
    
    # Report before/after on key extreme values
    print("\n" + "=" * 80)
    print("BEFORE/AFTER COMPARISON (select extreme values)")
    print("=" * 80)
    
    comparison = []
    for col in ['mean__water_log', 'mean__steps', 'mean__weight', 'mean__bmi']:
        if col in X.columns:
            before_max = X[col].max()
            after_max = X_clean[col].max()
            before_min = X[col].min()
            after_min = X_clean[col].min()
            
            print(f"\n{col}:")
            print(f"  Before: min={before_min:.2e}, max={before_max:.2e}")
            print(f"  After:  min={after_min:.2e}, max={after_max:.2e}")
            
            comparison.append({
                'feature': col,
                'before_min': before_min,
                'after_min': after_min,
                'before_max': before_max,
                'after_max': after_max,
            })
    
    comparison_df = pd.DataFrame(comparison)
    comparison_df.to_csv(OUTPUT_DIR / "d2_cleaning_before_after.csv", index=False)
    
    print(f"\n✓ Comparison saved to: {OUTPUT_DIR / 'd2_cleaning_before_after.csv'}")


if __name__ == "__main__":
    run_cleaning_diagnostic()
