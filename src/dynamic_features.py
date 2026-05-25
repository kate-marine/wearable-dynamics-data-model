from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import VarianceThreshold


def extract_dynamic_features(panel: pd.DataFrame, signals: list[str]) -> pd.DataFrame:
    """Extract temporal/dynamic features from the daily panel.

    For each participant and signal, compute descriptors of the time series:
    - std: day-to-day variability
    - min, max: range
    - trend: linear slope over days
    - acf_lag1, acf_lag7: autocorrelation at lag 1 and 7
    - cv: coefficient of variation (std / mean)
    - pct_nonzero: percentage of non-missing days
    """

    rows = []
    for (participant_id, date), group_data in panel.groupby(level=["participant_id", "date"]):
        pass

    for participant_id in panel.index.get_level_values("participant_id").unique():
        participant_panel = panel.loc[participant_id, signals]
        for signal in signals:
            ts = participant_panel[signal].dropna().values
            if len(ts) < 2:
                continue

            features_dict = {"participant_id": participant_id, "signal": signal}

            features_dict["std"] = float(np.std(ts))
            features_dict["mean"] = float(np.mean(ts))
            features_dict["min"] = float(np.min(ts))
            features_dict["max"] = float(np.max(ts))
            features_dict["range"] = float(np.max(ts) - np.min(ts))

            if len(ts) >= 2:
                x = np.arange(len(ts), dtype=float)
                slope, _, r_value, _, _ = stats.linregress(x, ts)
                features_dict["trend"] = float(slope)
                features_dict["trend_r2"] = float(r_value**2)

            if len(ts) >= 2:
                features_dict["acf_lag1"] = float(np.corrcoef(ts[:-1], ts[1:])[0, 1])
            else:
                features_dict["acf_lag1"] = np.nan

            if len(ts) >= 8:
                features_dict["acf_lag7"] = float(np.corrcoef(ts[:-7], ts[7:])[0, 1])
            else:
                features_dict["acf_lag7"] = np.nan

            if features_dict["mean"] != 0:
                features_dict["cv"] = features_dict["std"] / features_dict["mean"]
            else:
                features_dict["cv"] = 0.0

            pct_valid = participant_panel[signal].notna().sum() / len(participant_panel)
            features_dict["pct_valid"] = float(pct_valid)

            rows.append(features_dict)

    if not rows:
        return pd.DataFrame()

    dynamic_df = pd.DataFrame(rows)
    pivot_cols = [col for col in dynamic_df.columns if col not in ["participant_id", "signal"]]
    pivoted = dynamic_df.pivot_table(
        index="participant_id", columns="signal", values=pivot_cols, aggfunc="first"
    )
    pivoted.columns = [f"{metric}__{signal}" for metric, signal in pivoted.columns]
    
    # Fill NaNs and remove near-zero-variance columns
    pivoted = pivoted.fillna(pivoted.median())
    pivoted = pivoted.fillna(0)
    
    selector = VarianceThreshold(threshold=1e-6)
    pivoted_filtered = selector.fit_transform(pivoted)
    selected_cols = pivoted.columns[selector.get_support()]
    pivoted = pd.DataFrame(pivoted_filtered, index=pivoted.index, columns=selected_cols)
    
    return pivoted


def build_augmented_feature_matrix(
    means_features: pd.DataFrame, dynamic_features: pd.DataFrame
) -> pd.DataFrame:
    """Combine means-only and dynamic features into an augmented matrix."""

    augmented = pd.concat([means_features, dynamic_features], axis=1)
    augmented = augmented.fillna(augmented.median())
    return augmented
