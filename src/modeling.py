from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def ridge_baseline_cv(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> dict[str, float]:
    """Cross-validated Ridge baseline for one target column."""

    n_splits = min(5, len(X))
    if n_splits < 2:
        raise ValueError("Need at least two samples for cross-validation.")

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring={
            "r2": "r2",
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
        },
        return_train_score=False,
    )
    return {
        "r2_mean": float(scores["test_r2"].mean()),
        "r2_std": float(scores["test_r2"].std()),
        "mae_mean": float(-scores["test_mae"].mean()),
        "mae_std": float(scores["test_mae"].std()),
        "rmse_mean": float(-scores["test_rmse"].mean()),
        "rmse_std": float(scores["test_rmse"].std()),
    }


def evaluate_targets(X: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Run the baseline on each target column and return a tidy summary."""

    rows = []
    aligned = targets.reindex(X.index)
    for column in aligned.columns:
        y = aligned[column].dropna()
        if y.empty:
            continue
        x = X.loc[y.index]
        metrics = ridge_baseline_cv(x, y)
        rows.append({"target": column, **metrics, "n_samples": int(len(y))})
    return pd.DataFrame(rows).sort_values("r2_mean", ascending=False).reset_index(drop=True)


def ridge_baseline_cv_alpha(X: pd.DataFrame, y: pd.Series, alpha: float = 10.0, random_state: int = 42) -> dict[str, float]:
    """Cross-validated Ridge baseline with custom alpha."""

    n_splits = min(5, len(X))
    if n_splits < 2:
        raise ValueError("Need at least two samples for cross-validation.")

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring={
            "r2": "r2",
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
        },
        return_train_score=False,
    )
    return {
        "r2_mean": float(scores["test_r2"].mean()),
        "r2_std": float(scores["test_r2"].std()),
        "mae_mean": float(-scores["test_mae"].mean()),
        "mae_std": float(scores["test_mae"].std()),
        "rmse_mean": float(-scores["test_rmse"].mean()),
        "rmse_std": float(scores["test_rmse"].std()),
    }


def evaluate_targets_with_alpha(X: pd.DataFrame, targets: pd.DataFrame, alpha: float = 10.0) -> pd.DataFrame:
    """Run the baseline on each target column with custom alpha and return a tidy summary."""

    rows = []
    aligned = targets.reindex(X.index)
    for column in aligned.columns:
        y = aligned[column].dropna()
        if y.empty:
            continue
        x = X.loc[y.index]
        metrics = ridge_baseline_cv_alpha(x, y, alpha=alpha)
        rows.append({"target": column, **metrics, "n_samples": int(len(y))})
    return pd.DataFrame(rows).sort_values("r2_mean", ascending=False).reset_index(drop=True)
