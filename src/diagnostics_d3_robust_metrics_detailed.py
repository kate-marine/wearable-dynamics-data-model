"""
Diagnostic 3 Extended: Do robust metrics (Pearson, MAE) tell the true story?
Check fold-by-fold correlation and MAE on REAL behavioral targets, 
including inspection of Fold 4 specifically.

Run: python -m src.diagnostics_d3_robust_metrics_detailed
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

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
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_diagnostic_d3_detailed():
    print("=" * 80)
    print("DIAGNOSTIC 3 EXTENDED: ROBUST METRICS (PEARSON, MAE) PER FOLD")
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
    
    print(f"✓ Features: {X.shape}, Targets: {behavior_summary.shape}")
    
    # Select only valid targets (not all-NaN)
    valid_targets = behavior_summary.columns[behavior_summary.notna().sum() > 60].tolist()
    print(f"✓ Valid targets (n>60): {valid_targets}")
    
    # Setup CV
    np.random.seed(42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Results storage
    all_results = []
    
    for target_col in valid_targets:
        print(f"\n{'='*80}")
        print(f"TARGET: {target_col}")
        print(f"{'='*80}")
        
        y = pd.to_numeric(behavior_summary[target_col], errors='coerce')
        
        # Remove cases with NaN target
        valid_idx = y.notna()
        X_valid = X[valid_idx]
        y_valid = y[valid_idx]
        
        print(f"Valid samples: {valid_idx.sum()} / {len(y)}")
        
        # Fold iteration
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_valid)):
            X_train = X_valid.iloc[train_idx]
            X_val = X_valid.iloc[val_idx]
            y_train = y_valid.iloc[train_idx]
            y_val = y_valid.iloc[val_idx]
            
            # Build pipeline: impute, scale, ridge
            pipeline = Pipeline([
                ('impute', SimpleImputer(strategy='median')),
                ('scale', StandardScaler()),
                ('ridge', Ridge(alpha=1.0))
            ])
            
            # Fit and predict
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_val)
            
            # Metrics
            mae = np.mean(np.abs(y_pred - y_val))
            rmse = np.sqrt(np.mean((y_pred - y_val) ** 2))
            ss_res = np.sum((y_pred - y_val) ** 2)
            ss_tot = np.sum((y_val - y_val.mean()) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
            
            # Correlation (robust)
            if len(y_val) > 2:
                try:
                    pearson_r, _ = pearsonr(y_pred, y_val)
                except:
                    pearson_r = np.nan
                try:
                    spearman_rho, _ = spearmanr(y_pred, y_val)
                except:
                    spearman_rho = np.nan
            else:
                pearson_r = np.nan
                spearman_rho = np.nan
            
            result = {
                'target': target_col,
                'fold': fold_idx,
                'n_samples': len(y_val),
                'r2': r2,
                'mae': mae,
                'rmse': rmse,
                'pearson_r': pearson_r,
                'spearman_rho': spearman_rho,
                'y_val_mean': y_val.mean(),
                'y_val_std': y_val.std(),
                'y_pred_mean': y_pred.mean(),
                'y_pred_std': y_pred.std(),
            }
            all_results.append(result)
            
            # Print
            fold_label = "FOLD 4" if fold_idx == 4 else f"Fold {fold_idx}"
            print(f"\n  {fold_label}:")
            print(f"    R²={r2:+.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")
            print(f"    Pearson r={pearson_r:+.4f}, Spearman ρ={spearman_rho:+.4f}")
            print(f"    y_val:  mean={y_val.mean():.4f}, std={y_val.std():.4f}")
            print(f"    y_pred: mean={y_pred.mean():.4f}, std={y_pred.std():.4f}")
            
            if fold_idx == 4:
                if abs(y_pred.std() / (y_val.std() + 1e-10)) > 5:
                    print(f"    ⚠⚠⚠ FOLD 4 INSTABILITY: pred_std is {y_pred.std() / (y_val.std() + 1e-10):.1f}x validation std")
    
    # Summary statistics across folds
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "d3_detailed_metrics_per_fold.csv", index=False)
    
    print(f"\n{'='*80}")
    print("ROBUST METRICS SUMMARY (ALL FOLDS)")
    print(f"{'='*80}")
    
    for target_col in valid_targets:
        subset = results_df[results_df['target'] == target_col]
        print(f"\n{target_col}:")
        print(f"  R² (mean±std): {subset['r2'].mean():+.4f} ± {subset['r2'].std():.4f}")
        print(f"  MAE (mean±std): {subset['mae'].mean():.4f} ± {subset['mae'].std():.4f}")
        print(f"  Pearson r (mean±std): {subset['pearson_r'].mean():+.4f} ± {subset['pearson_r'].std():.4f}")
        print(f"  Spearman ρ (mean±std): {subset['spearman_rho'].mean():+.4f} ± {subset['spearman_rho'].std():.4f}")
        
        # Fold 4 specific
        fold4_subset = subset[subset['fold'] == 4].iloc[0]
        print(f"\n  Fold 4 specifically:")
        print(f"    R²={fold4_subset['r2']:+.4f}, MAE={fold4_subset['mae']:.4f}, Pearson r={fold4_subset['pearson_r']:+.4f}")
        print(f"    y_pred_std / y_val_std = {fold4_subset['y_pred_std'] / (fold4_subset['y_val_std'] + 1e-10):.2f}x")
    
    print(f"\n✓ Results saved to: {OUTPUT_DIR / 'd3_detailed_metrics_per_fold.csv'}")


if __name__ == "__main__":
    run_diagnostic_d3_detailed()
