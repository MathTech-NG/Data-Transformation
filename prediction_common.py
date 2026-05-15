"""
prediction_common.py
──────────────────────────────────────────────────────────────────────────────
Shared helpers for out-of-sample OLS evaluation (predict.py), operational
scoring (score.py), and the Streamlit app. Keeps one definition of the
model matrix and Previous_GPA derivation aligned with enrich.py / app.py.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

PREDICTORS = ["Previous_GPA", "Attendance_Rate", "Study_Hours_Per_Week", "Course_Load"]
LEVEL_FOR_PREVIOUS = ["CGPA100", "CGPA200", "CGPA300"]
BEHAVIOURAL = ["Attendance_Rate", "Study_Hours_Per_Week", "Course_Load"]


def load_enriched_deduped(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "ID No" in df.columns:
        df = df.drop_duplicates(subset="ID No", keep="first").reset_index(drop=True)
    return df


def derive_previous_gpa(df: pd.DataFrame) -> pd.Series:
    """Same definition as enrich.py: mean(CGPA100, CGPA200, CGPA300), rounded to 4 dp."""
    return ((df["CGPA100"] + df["CGPA200"] + df["CGPA300"]) / 3).round(4)


def fit_reference_ols(df: pd.DataFrame, target: str = "CGPA"):
    """Full-sample OLS used for operational scoring (matches app.py)."""
    X = sm.add_constant(df[PREDICTORS])
    y = df[target]
    return sm.OLS(y, X).fit()


def cross_val_ols_metrics(
    df: pd.DataFrame,
    target_col: str,
    k: int,
    seed: int,
) -> dict[str, Any]:
    """
    K-fold CV with the same four-predictor OLS specification.
    Returns stacked out-of-fold predictions for MAE / RMSE / R² and baseline metrics.
    """
    if target_col not in df.columns:
        raise KeyError(f"Missing target column: {target_col}")
    n = len(df)
    if k < 2 or k > n:
        raise ValueError(f"k must be in [2, n]; got k={k}, n={n}")

    y = df[target_col].values.astype(float)
    oof_pred = np.zeros(n, dtype=float)
    oof_base = np.zeros(n, dtype=float)

    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    for train_idx, test_idx in kf.split(np.arange(n)):
        d_train = df.iloc[train_idx]
        d_test = df.iloc[test_idx]
        X_tr = sm.add_constant(d_train[PREDICTORS])
        y_tr = d_train[target_col]
        fit = sm.OLS(y_tr, X_tr).fit()
        X_te = sm.add_constant(d_test[PREDICTORS], has_constant="add")
        oof_pred[test_idx] = fit.predict(X_te)
        oof_base[test_idx] = float(y_tr.mean())

    mae = mean_absolute_error(y, oof_pred)
    rmse = float(np.sqrt(mean_squared_error(y, oof_pred)))
    r2 = r2_score(y, oof_pred)
    base_mae = mean_absolute_error(y, oof_base)
    base_rmse = float(np.sqrt(mean_squared_error(y, oof_base)))

    return {
        "n": n,
        "k": k,
        "target": target_col,
        "mae": mae,
        "rmse": rmse,
        "r2_oof": r2,
        "baseline_mean_mae": base_mae,
        "baseline_mean_rmse": base_rmse,
    }


def prepare_scoring_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Build the four-predictor matrix from user rows.

    Either supply Previous_GPA + three behavioural columns, or CGPA100–300 + behavioural.
    If both level GPAs and Previous_GPA are present, Previous_GPA is taken as authoritative
    but a warning is emitted when it disagrees with mean(levels) by more than 0.01.
    """
    msgs: list[str] = []
    used = set(PREDICTORS) | set(LEVEL_FOR_PREVIOUS)
    extras = [c for c in raw.columns if c not in used]
    if extras:
        msgs.append(
            "Ignoring columns not used by the scorer: " + ", ".join(sorted(extras))
        )

    has_prev = "Previous_GPA" in raw.columns
    has_levels = all(c in raw.columns for c in LEVEL_FOR_PREVIOUS)
    for col in BEHAVIOURAL:
        if col not in raw.columns:
            raise ValueError(
                f"Missing required column '{col}'. "
                "Need Attendance_Rate, Study_Hours_Per_Week, Course_Load, plus either "
                "Previous_GPA or CGPA100, CGPA200, CGPA300."
            )

    out = raw.copy()
    if has_levels and has_prev:
        derived = derive_previous_gpa(out)
        diff = (out["Previous_GPA"].astype(float) - derived.astype(float)).abs()
        if diff.max() > 0.01:
            msgs.append(
                "Previous_GPA differs from mean(CGPA100, CGPA200, CGPA300) by >0.01 "
                "on at least one row — using explicit Previous_GPA values."
            )
    elif has_levels:
        out["Previous_GPA"] = derive_previous_gpa(out)
    elif has_prev:
        pass
    else:
        raise ValueError(
            "Provide either 'Previous_GPA' or all of 'CGPA100', 'CGPA200', 'CGPA300' "
            "(plus the three behavioural columns)."
        )

    X = out[PREDICTORS].copy()
    return X, msgs


def soft_validate_predictors(X: pd.DataFrame) -> list[str]:
    """Return warning strings for values outside training-plausible ranges."""
    w: list[str] = []
    if (X["Attendance_Rate"] < 40).any() or (X["Attendance_Rate"] > 100).any():
        w.append("Some Attendance_Rate values are outside [40, 100].")
    if (X["Study_Hours_Per_Week"] < 2).any() or (X["Study_Hours_Per_Week"] > 20).any():
        w.append("Some Study_Hours_Per_Week values are outside [2, 20].")
    if (X["Course_Load"] < 8).any() or (X["Course_Load"] > 20).any():
        w.append("Some Course_Load values are outside [8, 20].")
    if (X["Previous_GPA"] < 0).any() or (X["Previous_GPA"] > 5).any():
        w.append("Some Previous_GPA values are outside [0, 5].")
    return w


def score_dataframe(model: Any, X: pd.DataFrame) -> pd.Series:
    """Vector of predicted CGPA (same target the reference model was fit on)."""
    X_const = sm.add_constant(X[PREDICTORS], has_constant="add")
    return pd.Series(model.predict(X_const), name="predicted_CGPA")


def print_limitation_banner() -> None:
    print(
        "\nNote (L1–L3): Attendance and study hours are partly synthesized from CGPA; "
        "Previous_GPA overlaps arithmetically with final CGPA. "
        "Metrics and scores are illustrative — not empirical proof of causation.\n"
    )
