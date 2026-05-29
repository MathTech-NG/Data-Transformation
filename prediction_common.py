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

DEFAULT_TARGET = "CGPA400"
TRAJECTORY_PRIOR_THRESHOLD = 0.05

CONTINUOUS_PREDICTORS = [
    "Previous_GPA",
    "Trajectory_Slope_Prior",
    "Attendance_Rate",
    "Study_Hours_Per_Week",
    "Course_Load",
]
GENOTYPE_DUMMY_COLS = ["Genotype_AS", "Genotype_SS"]
INTERACTION_COLS = ["Genotype_SS_x_Attendance"]
DESIGN_COLUMNS = CONTINUOUS_PREDICTORS + GENOTYPE_DUMMY_COLS + INTERACTION_COLS
COEF_LABELS = [
    "Previous_GPA",
    "Trajectory_Slope_Prior",
    "Attendance_Rate",
    "Study_Hours_Per_Week",
    "Course_Load",
    "Genotype (AS vs AA)",
    "Genotype (SS vs AA)",
    "SS x Attendance",
]
LEVEL_FOR_PREVIOUS = ["CGPA100", "CGPA200", "CGPA300"]
BEHAVIOURAL = ["Attendance_Rate", "Study_Hours_Per_Week", "Course_Load"]
VALID_GENOTYPES = frozenset({"AA", "AS", "SS"})

PREDICTORS = CONTINUOUS_PREDICTORS


def load_enriched_deduped(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "ID No" in df.columns:
        df = df.drop_duplicates(subset="ID No", keep="first").reset_index(drop=True)
    return df


def derive_previous_gpa(df: pd.DataFrame) -> pd.Series:
    """Same definition as enrich.py: mean(CGPA100, CGPA200, CGPA300), rounded to 4 dp."""
    return ((df["CGPA100"] + df["CGPA200"] + df["CGPA300"]) / 3).round(4)


def derive_trajectory_slope_prior(df: pd.DataFrame) -> pd.Series:
    """
    OLS slope of CGPA100–300 vs t = 1, 2, 3 (excludes CGPA400 — safe for predicting CGPA400).
    Same definition as enrich.py Trajectory_Slope_Prior.
    """
    t = np.array([1.0, 2.0, 3.0])
    t_mean = t.mean()
    denom = float(np.sum((t - t_mean) ** 2))
    y = df[LEVEL_FOR_PREVIOUS].astype(float).values
    y_mean = y.mean(axis=1, keepdims=True)
    num = np.sum((t - t_mean) * (y - y_mean), axis=1)
    slopes = np.where(denom == 0, 0.0, num / denom)
    return pd.Series(slopes, index=df.index).round(4)


def classify_trajectory_prior(slope: float) -> str:
    if slope > TRAJECTORY_PRIOR_THRESHOLD:
        return "Improving"
    if slope < -TRAJECTORY_PRIOR_THRESHOLD:
        return "Declining"
    return "Stable"


def build_design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    OLS design matrix: continuous predictors + genotype dummies (AA reference)
    + Genotype_SS x Attendance_Rate interaction.
    """
    if "Genotype" not in df.columns:
        raise KeyError("Column 'Genotype' is required (AA, AS, or SS).")
    bad = set(df["Genotype"].astype(str).unique()) - VALID_GENOTYPES
    if bad:
        raise ValueError(f"Invalid Genotype value(s): {bad}")

    cont = df[CONTINUOUS_PREDICTORS].astype(float)
    gt = df["Genotype"].astype(str)
    dums = pd.DataFrame(
        {
            "Genotype_AS": (gt == "AS").astype(float),
            "Genotype_SS": (gt == "SS").astype(float),
        },
        index=df.index,
    )
    interaction = pd.DataFrame(
        {
            "Genotype_SS_x_Attendance": (
                dums["Genotype_SS"] * df["Attendance_Rate"].astype(float)
            ),
        },
        index=df.index,
    )
    return pd.concat([cont, dums, interaction], axis=1)


def fit_reference_ols(df: pd.DataFrame, target: str = DEFAULT_TARGET):
    """Full-sample OLS used for operational scoring (matches app.py)."""
    X = sm.add_constant(build_design_matrix(df))
    y = df[target]
    return sm.OLS(y, X).fit()


def cross_val_ols_metrics(
    df: pd.DataFrame,
    target_col: str,
    k: int,
    seed: int,
) -> dict[str, Any]:
    """K-fold CV with the full OLS design matrix."""
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
        X_tr = sm.add_constant(build_design_matrix(d_train))
        y_tr = d_train[target_col]
        fit = sm.OLS(y_tr, X_tr).fit()
        X_te = sm.add_constant(build_design_matrix(d_test), has_constant="add")
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
    """Build rows with continuous predictors + Genotype for build_design_matrix."""
    msgs: list[str] = []
    used = (
        set(CONTINUOUS_PREDICTORS)
        | set(LEVEL_FOR_PREVIOUS)
        | {"Genotype", "Trajectory_Class_Prior"}
        | set(GENOTYPE_DUMMY_COLS)
        | set(INTERACTION_COLS)
    )
    extras = [c for c in raw.columns if c not in used]
    if extras:
        msgs.append(
            "Ignoring columns not used by the scorer: " + ", ".join(sorted(extras))
        )

    if "Genotype" not in raw.columns:
        raise ValueError("Missing required column 'Genotype'. Use AA, AS, or SS.")

    has_prev = "Previous_GPA" in raw.columns
    has_levels = all(c in raw.columns for c in LEVEL_FOR_PREVIOUS)
    for col in BEHAVIOURAL:
        if col not in raw.columns:
            raise ValueError(
                f"Missing required column '{col}'. "
                "Need Attendance_Rate, Study_Hours_Per_Week, Course_Load, Genotype, "
                "plus either Previous_GPA or CGPA100, CGPA200, CGPA300."
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
            "(plus behavioural columns and Genotype)."
        )

    if has_levels:
        out["Trajectory_Slope_Prior"] = derive_trajectory_slope_prior(out)
    elif "Trajectory_Slope_Prior" in raw.columns:
        out["Trajectory_Slope_Prior"] = raw["Trajectory_Slope_Prior"].astype(float)
    else:
        out["Trajectory_Slope_Prior"] = 0.0
        msgs.append(
            "Trajectory_Slope_Prior set to 0 (CGPA100–300 not provided); "
            "prior trend is not applied to this score."
        )

    return out[CONTINUOUS_PREDICTORS + ["Genotype"]].copy(), msgs


def soft_validate_predictors(df: pd.DataFrame) -> list[str]:
    """Return warning strings for values outside training-plausible ranges."""
    w: list[str] = []
    if (df["Attendance_Rate"] < 40).any() or (df["Attendance_Rate"] > 100).any():
        w.append("Some Attendance_Rate values are outside [40, 100].")
    if (df["Study_Hours_Per_Week"] < 2).any() or (df["Study_Hours_Per_Week"] > 20).any():
        w.append("Some Study_Hours_Per_Week values are outside [2, 20].")
    if (df["Course_Load"] < 8).any() or (df["Course_Load"] > 20).any():
        w.append("Some Course_Load values are outside [8, 20].")
    if (df["Previous_GPA"] < 0).any() or (df["Previous_GPA"] > 5).any():
        w.append("Some Previous_GPA values are outside [0, 5].")
    if (df["Trajectory_Slope_Prior"] < -1.5).any() or (df["Trajectory_Slope_Prior"] > 1.5).any():
        w.append("Some Trajectory_Slope_Prior values are outside [-1.5, 1.5].")
    bad_gt = set(df["Genotype"].astype(str).unique()) - VALID_GENOTYPES
    if bad_gt:
        w.append(f"Invalid Genotype value(s): {sorted(bad_gt)}.")
    return w


def score_dataframe(model: Any, feature_rows: pd.DataFrame) -> pd.Series:
    """Vector of predicted CGPA400."""
    X_const = sm.add_constant(build_design_matrix(feature_rows), has_constant="add")
    return pd.Series(model.predict(X_const), name="predicted_CGPA400")


def prediction_decomposition(model: Any, feature_rows: pd.DataFrame) -> pd.DataFrame:
    """Per-term contribution (coefficient × value) for one or more scoring rows."""
    X_const = sm.add_constant(build_design_matrix(feature_rows), has_constant="add")
    rows = []
    for i in range(len(feature_rows)):
        aligned = X_const.iloc[i] * model.params
        rows.append(
            pd.DataFrame(
                {"term": aligned.index, "contribution": aligned.values.round(4)},
            )
        )
    if len(rows) == 1:
        return rows[0]
    return pd.concat(rows, keys=feature_rows.index, names=["row"])


def print_limitation_banner() -> None:
    print(
        "\nNote (L1, L3, L7): Attendance and study hours are partly synthesized; "
        "genotype affects attendance via an assumed health pathway. "
        "Previous_GPA is arithmetically prior to CGPA400 (no overlap). "
        "Trajectory_Slope_Prior uses CGPA100–300 only (not CGPA400). "
        "Metrics and scores are illustrative — not empirical proof of causation.\n"
    )
