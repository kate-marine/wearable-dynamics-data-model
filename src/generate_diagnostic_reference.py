"""
Generate a final diagnostic reference report showing:
1. Key findings summary
2. Before/after correlation comparison (raw vs cleaned)
3. Power analysis at a glance
"""
from pathlib import Path
import pandas as pd
import json

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "diagnostics_outputs"


def create_final_reference_report():
    print("=" * 80)
    print("FINAL DIAGNOSTIC REFERENCE REPORT")
    print("=" * 80)
    
    # Load cleaned metrics
    cleaned_metrics = pd.read_csv(OUTPUT_DIR / "d3_cleaned_metrics_per_fold.csv")
    power_df = pd.read_csv(OUTPUT_DIR / "d3_power_analysis.csv")
    
    # Generate summary
    summary = {
        "study_metadata": {
            "n_participants": 113,
            "n_folds": 5,
            "n_per_fold": 22.6,
            "wearable_features": 14,
            "targets_valid": 3,
            "total_outcomes": 8,
        },
        "data_quality": {
            "extreme_values_found": 5,
            "values_winsorized": [
                {"participant": "P87", "feature": "mean__water_log", "before": 20633, "after": 5000},
                {"participant": "P1", "feature": "mean__bmi", "before": 0, "after": 15},
                {"participant": "P95", "feature": "mean__bmi", "before": 0, "after": 15},
                {"participant": "P1", "feature": "mean__weight", "before": 0, "after": 40},
                {"participant": "P40", "feature": "mean__bodyfat", "before": 70, "after": 60},
            ]
        },
        "key_findings": {
            "harness_validity": "✅ PASSED (synthetic signal r²=0.954)",
            "fold4_instability": "❌ IDENTIFIED (outlier participants), ✅ EXPLAINED",
            "robust_metrics_stable": "✅ YES (Pearson, MAE consistent across folds)",
            "correlation_strength": "NOISY (~±0.06 to ±0.19), NOT stably near-zero",
        },
        "correlations_cleaned_data": {}
    }
    
    # Add per-target summaries
    for target_col in ['Free recall (immediate)', 'Free recall (delayed)', 'Foreign language flashcards (delayed)']:
        subset = cleaned_metrics[cleaned_metrics['target'] == target_col]
        
        r_values = subset['pearson_r'].values
        mae_values = subset['mae'].values
        
        # Convert numpy types to Python natives
        fold_dict = {str(int(f)): float(r) for f, r in zip(subset['fold'].values, r_values)}
        p_values_list = [float(p) for p in subset['pearson_p'].values]
        
        summary["correlations_cleaned_data"][target_col] = {
            "n_valid_samples": int(subset['n_samples'].iloc[0]),
            "pearson_r_by_fold": fold_dict,
            "pearson_r_mean": float(r_values.mean()),
            "pearson_r_std": float(r_values.std()),
            "pearson_r_min": float(r_values.min()),
            "pearson_r_max": float(r_values.max()),
            "pearson_r_ci": float(1.96 * r_values.std() / (len(r_values) ** 0.5)),
            "mae_mean": float(mae_values.mean()),
            "mae_std": float(mae_values.std()),
            "p_values": p_values_list,
            "interpretation": "No significant folds (all p > 0.05 after Bonferroni * 5 folds ≈ 0.01)" if all(subset['pearson_p'] > 0.01) else "Some within-fold p < 0.05 but unstable across folds"
        }
    
    # Power analysis summary
    summary["power_analysis"] = {
        "mdes_full_n113": 0.261,
        "mdes_per_fold_n22": 0.567,
        "power_for_r015": "35% at n=113, 10% at n=22/fold",
        "power_for_r020": "57% at n=113, 14% at n=22/fold",
        "power_for_r025": "76% at n=113, 20% at n=22/fold",
        "required_n_for_r020_80pct": 445,
        "required_n_for_r015_80pct": 790,
        "conclusion": "Study is underpowered; cannot reliably detect r < 0.26"
    }
    
    # Save JSON
    summary_json = json.dumps(summary, indent=2)
    summary_file = OUTPUT_DIR / "DIAGNOSTIC_REFERENCE.json"
    summary_file.write_text(summary_json)
    
    print("\n" + "=" * 80)
    print("CORRELATION RESULTS (CLEANED DATA)")
    print("=" * 80)
    
    for target_col, data in summary["correlations_cleaned_data"].items():
        print(f"\n{target_col}:")
        print(f"  Pearson r (mean ± SD): {data['pearson_r_mean']:+.4f} ± {data['pearson_r_std']:.4f}")
        print(f"  95% CI from folds: ±{data['pearson_r_ci']:.4f}")
        print(f"  Range: [{data['pearson_r_min']:+.4f}, {data['pearson_r_max']:+.4f}]")
        print(f"  MAE: {data['mae_mean']:.4f} ± {data['mae_std']:.4f}")
        print(f"  Per-fold p-values: {[f'{p:.3f}' for p in data['p_values']]}")
        print(f"  Interpretation: {data['interpretation']}")
    
    print("\n" + "=" * 80)
    print("POWER ANALYSIS SUMMARY")
    print("=" * 80)
    
    print(f"\nMinimum Detectable Effect Size (80% power):")
    print(f"  At full n=113: |r| ≥ {summary['power_analysis']['mdes_full_n113']}")
    print(f"  At per-fold n≈22: |r| ≥ {summary['power_analysis']['mdes_per_fold_n22']}")
    
    print(f"\nTo detect Pearson r = 0.20 at 80% power: need n ≈ {summary['power_analysis']['required_n_for_r020_80pct']}")
    print(f"To detect Pearson r = 0.15 at 80% power: need n ≈ {summary['power_analysis']['required_n_for_r015_80pct']}")
    
    print(f"\n→ Observed correlations (|r| ≈ 0.07-0.19) are BELOW MDES")
    print(f"→ {summary['power_analysis']['conclusion']}")
    
    print("\n" + "=" * 80)
    print("STATISTICAL FRAMING FOR MANUSCRIPT")
    print("=" * 80)
    
    print("""
RECOMMENDED ABSTRACT/CONCLUSIONS WORDING:

"We found no reliable evidence of linear association between mean wearable activity
levels and cognitive task performance (Free recall immediate: r = -0.065 ± 0.241;
Free recall delayed: r = -0.097 ± 0.157; Foreign language flashcards: r = +0.186 ± 0.151,
all p > 0.05). The study was statistically underpowered to detect modest effect sizes
(MDES = 0.261 for 80% power); correlations of r ≈ 0.20 would require approximately 445
participants. The observed null is consistent with either a true absence of association
or a weak signal (r < 0.20) indistinguishable from noise at current sample size.
Future work should pre-register effect size targets and collect data accordingly."

KEY POINTS TO EMPHASIZE:
✓ Data quality verified and cleaned (5 extreme values winsorized)
✓ Robust metrics (Pearson, MAE) used instead of unreliable R²
✓ Fold-wise correlations reported WITH confidence intervals
✓ Power analysis contexualizes the null (not just "we found nothing")
✓ Honest about limitations (n, sparse features, fold instability in extremes)

WHAT NOT TO SAY:
✗ "Strong evidence of no association" (we're underpowered)
✗ "Correlations are stably near zero" (fold-wise variation is large)
✗ "Activity patterns do not predict memory" (could, but undetected at n=113)
    """)
    
    print(f"\n\n✓ Reference saved: {summary_file}")
    print(f"✓ Full diagnostic summary: {OUTPUT_DIR / 'DIAGNOSTIC_SUMMARY.md'}")


if __name__ == "__main__":
    create_final_reference_report()
