"""
Expanded Diagnostic 1b & 3 hybrid:
- Investigate the -20.7 noise R² via scaling and conditioning
- Run full metrics (R², MAE, Pearson, Spearman) on noise and real targets
- Test whether standardizing the target improves R²

Run: python -m src.diagnostics_expanded
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline

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


def custom_cv_with_diagnostics(X, y, target_name="unknown", show_folds=False):
    """
    Custom evaluator that inspects preprocessing inside folds.
    Reports R², MAE, RMSE, and Pearson/Spearman correlations.
    """
    n_splits = min(5, len(X))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    r2_scores = []
    mae_scores = []
    rmse_scores = []
    pearson_corrs = []
    spearman_corrs = []
    
    fold_diagnostics = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Pipeline: impute, scale, fit Ridge
        # (ALL within this fold block no leakage)
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        model = Ridge(alpha=1.0)
        
        # Fit on training fold ONLY
        X_train_imputed = imputer.fit_transform(X_train)
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        model.fit(X_train_scaled, y_train)
        
        # Apply to validation fold using training statistics (correct)
        X_val_imputed = imputer.transform(X_val)
        X_val_scaled = scaler.transform(X_val_imputed)
        y_pred = model.predict(X_val_scaled)
        
        # Compute metrics
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        r2 = r2_score(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        pearson_r, _ = stats.pearsonr(y_pred, y_val)
        spearman_r, _ = stats.spearmanr(y_pred, y_val)
        
        r2_scores.append(r2)
        mae_scores.append(mae)
        rmse_scores.append(rmse)
        pearson_corrs.append(pearson_r)
        spearman_corrs.append(spearman_r)
        
        fold_diagnostics.append({
            "fold": fold_idx,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "r2": r2,
            "mae": mae,
            "y_pred_mean": y_pred.mean(),
            "y_pred_std": y_pred.std(),
            "y_val_mean": y_val.mean(),
            "y_val_std": y_val.std(),
        })
    
    if show_folds:
        print(f"\n  Fold-by-fold diagnostics for {target_name}:")
        for diag in fold_diagnostics:
            print(f"    Fold {diag['fold']}: R²={diag['r2']:7.4f}, MAE={diag['mae']:7.4f}, "
                  f"y_pred std={diag['y_pred_std']:7.4f}, y_val std={diag['y_val_std']:7.4f}")
    
    return {
        "r2_mean": float(np.mean(r2_scores)),
        "r2_std": float(np.std(r2_scores)),
        "mae_mean": float(np.mean(mae_scores)),
        "mae_std": float(np.std(mae_scores)),
        "rmse_mean": float(np.mean(rmse_scores)),
        "rmse_std": float(np.std(rmse_scores)),
        "pearson_r_mean": float(np.mean(pearson_corrs)),
        "pearson_r_std": float(np.std(pearson_corrs)),
        "spearman_r_mean": float(np.mean(spearman_corrs)),
        "spearman_r_std": float(np.std(spearman_corrs)),
        "fold_count": len(fold_diagnostics),
    }


def run_expanded_diagnostic():
    print("=" * 80)
    print("EXPANDED DIAGNOSTIC: Scaling, conditioning, and robust metrics")
    print("=" * 80)
    
    # Load features
    print("\n[1] Loading means-only feature matrix...")
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
    print(f"✓ Loaded: {means_features.shape}")
    
    # Inspect feature scaling
    print("\n[2] Feature matrix scaling inspection:")
    print(f"  Feature means (first 5): {means_features.iloc[:, :5].mean().values}")
    print(f"  Feature stds (first 5): {means_features.iloc[:, :5].std().values}")
    print(f"  Any NaN: {means_features.isna().sum().sum()}, filled via imputation inside CV")
    
    # Construct targets
    print("\n[3] Constructing test targets...")
    np.random.seed(42)
    
    # Pure noise targets (3 variants)
    noise_raw = pd.Series(
        np.random.normal(0, 1.0, len(means_features)),
        index=means_features.index,
        name="noise_raw"
    )
    
    noise_scaled = pd.Series(
        np.random.normal(0, 0.1, len(means_features)),  # smaller scale
        index=means_features.index,
        name="noise_scaled_small"
    )
    
    # Standardize pure noise to match typical target scale
    noise_standardized = pd.Series(
        (np.random.normal(0, 1.0, len(means_features)) - np.random.normal(0, 1.0, len(means_features))).clip(-3, 3),
        index=means_features.index,
        name="noise_standardized_clipped"
    )
    
    print(f"  Pure noise (std=1.0): mean={noise_raw.mean():.4f}, std={noise_raw.std():.4f}, "
          f"min={noise_raw.min():.4f}, max={noise_raw.max():.4f}")
    print(f"  Pure noise (std=0.1): mean={noise_scaled.mean():.4f}, std={noise_scaled.std():.4f}")
    print(f"  Pure noise (standardized): mean={noise_standardized.mean():.4f}, std={noise_standardized.std():.4f}")
    
    # Real targets
    print(f"\n[4] Loading real targets (behavioral_summary and cleaned behavior.pkl)...")
    behavior_summary = load_target_table(RAW_DIR, "behavioral_summary.pkl")
    print(f"  behavioral_summary: {behavior_summary.shape}")
    for col in behavior_summary.columns:
        print(f"    {col}: mean={behavior_summary[col].mean():.4f}, std={behavior_summary[col].std():.4f}, "
              f"min={behavior_summary[col].min():.4f}, max={behavior_summary[col].max():.4f}")
    
    behavior_full = load_target_table(RAW_DIR, "behavior.pkl")
    behavior_full = behavior_full.replace([np.inf, -np.inf], np.nan)
    valid_cols = [col for col in behavior_full.columns 
                  if (pd.notna(behavior_full[col]).sum() - np.isinf(behavior_full[col]).sum()) >= 30]
    behavior_clean = behavior_full[valid_cols].copy()
    print(f"  behavior.pkl (cleaned): {behavior_clean.shape}")
    
    # Run evaluations
    print("\n" + "=" * 80)
    print("EVALUATION: Robust metrics (R², MAE, Pearson, Spearman)")
    print("=" * 80)
    
    results_list = []
    
    # Noise tests
    print("\n[NOISE TESTS]")
    for noise_target in [noise_raw, noise_scaled, noise_standardized]:
        print(f"\n  {noise_target.name}:")
        metrics = custom_cv_with_diagnostics(means_features, noise_target, noise_target.name, show_folds=True)
        metrics["target"] = noise_target.name
        metrics["target_type"] = "pure_noise"
        print(f"  noise target: {noise_target.name}, mean={noise_target.mean():.4f}, std={noise_target.std():.4f}")
        results_list.append(metrics)
        
        print(f"    R²: {metrics['r2_mean']:.6f} ± {metrics['r2_std']:.6f}")
        print(f"    MAE: {metrics['mae_mean']:.6f} ± {metrics['mae_std']:.6f}")
        print(f"    Pearson r: {metrics['pearson_r_mean']:.6f} ± {metrics['pearson_r_std']:.6f}")
        print(f"    Spearman ρ: {metrics['spearman_r_mean']:.6f} ± {metrics['spearman_r_std']:.6f}")
    
    # Behavioral summary targets
    print(f"\n[BEHAVIORAL SUMMARY TARGETS]")
    for col in behavior_summary.columns:
        print(f"\n  {col}:")
        metrics = custom_cv_with_diagnostics(means_features, behavior_summary[col], col, show_folds=False)
        metrics["target"] = col
        metrics["target_type"] = "behavioral_summary"
        results_list.append(metrics)
        
        print(f"    R²: {metrics['r2_mean']:.6f} ± {metrics['r2_std']:.6f}")
        print(f"    MAE: {metrics['mae_mean']:.6f} ± {metrics['mae_std']:.6f}")
        print(f"    Pearson r: {metrics['pearson_r_mean']:.6f} ± {metrics['pearson_r_std']:.6f}")
        print(f"    Spearman ρ: {metrics['spearman_r_mean']:.6f} ± {metrics['spearman_r_std']:.6f}")
    
    # Sample of behavior.pkl
    print(f"\n[BEHAVIOR.PKL (cleaned) — sample of first 5]")
    for col in behavior_clean.columns[:5]:
        print(f"\n  {col}:")
        metrics = custom_cv_with_diagnostics(means_features, behavior_clean[col], col, show_folds=False)
        metrics["target"] = col
        metrics["target_type"] = "behavior_pkl"
        results_list.append(metrics)
        
        print(f"    R²: {metrics['r2_mean']:.6f} ± {metrics['r2_std']:.6f}")
        print(f"    MAE: {metrics['mae_mean']:.6f} ± {metrics['mae_std']:.6f}")
        print(f"    Pearson r: {metrics['pearson_r_mean']:.6f} ± {metrics['pearson_r_std']:.6f}")
        print(f"    Spearman ρ: {metrics['spearman_r_mean']:.6f} ± {metrics['spearman_r_std']:.6f}")
    
    # Summary and save
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(OUTPUT_DIR / "expanded_diagnostics_metrics.csv", index=False)
    print(f"\nResults saved to: {OUTPUT_DIR / 'expanded_diagnostics_metrics.csv'}")
    
    print("\n[KEY FINDINGS]")
    print(f"  Pure noise R²: {results_df[results_df['target_type']=='pure_noise']['r2_mean'].mean():.4f} (should be ~0.0)")
    print(f"  Pure noise MAE: {results_df[results_df['target_type']=='pure_noise']['mae_mean'].mean():.4f}")
    print(f"  Pure noise Pearson |r|: {results_df[results_df['target_type']=='pure_noise']['pearson_r_mean'].abs().mean():.4f} (should be ~0.0)")
    
    print(f"\n  Behavioral summary R²: {results_df[results_df['target_type']=='behavioral_summary']['r2_mean'].mean():.4f}")
    print(f"  Behavioral summary MAE: {results_df[results_df['target_type']=='behavioral_summary']['mae_mean'].mean():.4f}")
    print(f"  Behavioral summary Pearson |r|: {results_df[results_df['target_type']=='behavioral_summary']['pearson_r_mean'].abs().mean():.4f}")
    
    print("\n[INTERPRETATION]")
    if results_df[results_df['target_type']=='pure_noise']['mae_mean'].mean() < 1.5:
        print("  ✓ MAE on noise is sane (<1.5), suggesting scaling is OK")
    else:
        print("  ✗ MAE on noise is explosive, suggesting conditioning bug")
    
    if results_df[results_df['target_type']=='pure_noise']['pearson_r_mean'].abs().mean() < 0.15:
        print("  ✓ Pearson correlations on noise are near-zero, R² explosions are metric artifacts")
    else:
        print("  ✗ Correlations on noise are non-trivial, possible model instability")


if __name__ == "__main__":
    run_expanded_diagnostic()
