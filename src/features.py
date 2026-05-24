from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


WEARABLE_SIGNAL_NAMES = [
    "bmi",
    "bodyfat",
    "cal",
    "cal_bmr",
    "distance",
    "elevation",
    "fair_act_mins",
    "floors",
    "food_cal_log",
    "light_act_mins",
    "sed_mins",
    "steps",
    "very_act_mins",
    "water_log",
    "weight",
    "cardio_cals",
    "cardio_maxval",
    "cardio_mins",
    "cardio_minval",
    "fb_cals",
    "fb_maxval",
    "fb_mins",
    "fb_minval",
    "oor_cals",
    "oor_maxval",
    "oor_mins",
    "oor_minval",
    "peak_cals",
    "peak_maxval",
    "peak_mins",
    "peak_minval",
    "resting_HR",
    "sleep_duration",
    "sleep_efficiency",
]


def select_wearable_signals(
    vocabulary: Iterable[str],
    coverage_summary: pd.DataFrame | None = None,
    min_median_days: float = 30.0,
) -> list[str]:
    """Select the daily wearable signals used for Phase 1.

    Signals are included only if they are in the project's wearable list and
    their median participant coverage is sufficiently dense.
    """

    vocab = set(vocabulary)
    selected = [name for name in WEARABLE_SIGNAL_NAMES if name in vocab]

    if coverage_summary is None:
        return selected

    eligible = coverage_summary.index[coverage_summary["median_valid_days"] >= min_median_days]
    eligible = set(eligible)
    return [name for name in selected if name in eligible]


def build_mean_feature_matrix(panel: pd.DataFrame, signals: list[str]) -> pd.DataFrame:
    """Create participant-level mean features from the daily panel."""

    feature_frame = panel[signals].groupby(level="participant_id").mean()
    feature_frame = feature_frame.add_prefix("mean__")
    return feature_frame


def build_valid_day_matrix(panel: pd.DataFrame, signals: list[str]) -> pd.DataFrame:
    """Count non-missing days per participant and signal."""

    return panel[signals].notna().groupby(level="participant_id").sum().astype(int)
