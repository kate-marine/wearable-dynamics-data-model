from __future__ import annotations

from pathlib import Path

import pandas as pd


RAW_CSV_GLOB = "BFM_AMT_*.csv"


def list_raw_csv_files(raw_dir: Path) -> list[Path]:
    """Return participant CSV files in stable filename order."""

    return sorted(raw_dir.glob(RAW_CSV_GLOB))


def participant_ids_for_files(files: list[Path]) -> dict[Path, str]:
    """Map sorted raw files to P0-style participant IDs.

    The raw wearable CSVs and the pickle targets are both ordered by participant,
    so the file order provides a stable join key.
    """

    return {path: f"P{i}" for i, path in enumerate(files)}


def load_raw_csv(path: Path, participant_id: str) -> pd.DataFrame:
    """Load one long-format participant CSV and normalize its date column."""

    frame = pd.read_csv(path)
    frame = frame.copy()
    frame["participant_id"] = participant_id
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    frame["date"] = frame["datetime"].dt.normalize()
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame[["participant_id", "date", "variable", "value"]]


def load_raw_long_table(raw_dir: Path) -> pd.DataFrame:
    """Load every participant CSV into one long-format table."""

    files = list_raw_csv_files(raw_dir)
    id_map = participant_ids_for_files(files)
    frames = [load_raw_csv(path, id_map[path]) for path in files]
    long_df = pd.concat(frames, ignore_index=True)
    return long_df


def discover_variable_vocabulary(long_df: pd.DataFrame) -> list[str]:
    """Return all observed variable names in sorted order."""

    return sorted(long_df["variable"].dropna().unique().tolist())


def summarize_variable_coverage(long_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize valid-day coverage per variable and participant."""

    coverage = (
        long_df.groupby(["variable", "participant_id"])["date"]
        .nunique()
        .rename("valid_days")
        .reset_index()
    )
    summary = (
        coverage.groupby("variable")["valid_days"]
        .agg(
            n_participants="count",
            median_valid_days="median",
            min_valid_days="min",
            max_valid_days="max",
            mean_valid_days="mean",
        )
        .sort_values(["median_valid_days", "n_participants"], ascending=[False, False])
    )
    return summary


def build_daily_panel(long_df: pd.DataFrame, variables: list[str] | None = None) -> pd.DataFrame:
    """Pivot the long table to a participant-by-date panel with daily rows.

    Missing days are represented explicitly as NaNs.
    """

    frame = long_df.copy()
    if variables is not None:
        frame = frame[frame["variable"].isin(variables)]

    frames: list[pd.DataFrame] = []
    all_vars = variables or discover_variable_vocabulary(frame)
    participant_ranges = (
        long_df.groupby("participant_id")
        .agg(start_date=("date", "min"), end_date=("date", "max"))
        .sort_index()
    )

    for participant_id, date_bounds in participant_ranges.iterrows():
        group = frame[frame["participant_id"] == participant_id]
        if not group.empty:
            daily = (
                group.groupby(["date", "variable"], as_index=False)["value"]
                .mean()
                .pivot(index="date", columns="variable", values="value")
            )
        else:
            daily = pd.DataFrame()

        full_index = pd.date_range(date_bounds["start_date"], date_bounds["end_date"], freq="D")
        daily = daily.reindex(full_index)
        daily.index.name = "date"
        daily = daily.reindex(columns=all_vars)
        daily.insert(0, "participant_id", participant_id)
        daily = daily.reset_index().set_index(["participant_id", "date"])
        frames.append(daily)

    panel = pd.concat(frames).sort_index()
    return panel


def load_target_table(raw_dir: Path, filename: str) -> pd.DataFrame:
    """Load a participant-indexed pickle target table and normalize its index."""

    obj = pd.read_pickle(raw_dir / filename)
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"Expected DataFrame in {filename}, got {type(obj)!r}")
    target = obj.copy()
    target.index = [f"P{i}" for i in range(len(target))]
    target.index.name = "participant_id"
    return target
