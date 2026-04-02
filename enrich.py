"""
enrich.py
─────────────────────────────────────────────────────────────────────────────
Synthesizes three missing independent variables required by the regression /
time-series model described in:

    "Regression and Time Series Modelling of Students' Performance Across
     Semester" — Ehime Kelvin Ehinomen, Mountain Top University, Jan 2026.

Missing variables
    Attendance_Rate      (%) — class participation level
    Study_Hours_Per_Week (h) — weekly study hours
    Course_Load          (units) — number of registered course units per semester

A fourth column, Previous_GPA, is derived directly from the dataset.

Usage
    python enrich.py [--input PATH] [--output PATH] [--seed INT]

Defaults
    --input   academic_performance_dataset_V2.csv
    --output  academic_performance_enriched.csv
    --seed    42
"""

import argparse
import sys
import numpy as np
import pandas as pd

# ─── CONSTANTS ────────────────────────────────────────────────────────────────

# Attendance model:  Attendance ~ N(BASE_ATT + BETA_ATT * CGPA, SIGMA_ATT²)
# Interpretation: a one-point rise in CGPA is associated with ~8 pp higher
# expected attendance.  At CGPA=1 → μ≈53 %; at CGPA=5 → μ≈85 %.
BASE_ATT  = 45.0   # intercept (percentage points)
BETA_ATT  =  8.0   # slope on CGPA
SIGMA_ATT =  8.0   # residual s.d. (realistic individual variation)
ATT_LO    = 40.0   # hard floor (university minimum attendance)
ATT_HI    = 100.0  # hard ceiling

# Study-hours model: Hours ~ N(BASE_SH + BETA_SH * CGPA, SIGMA_SH²)
# At CGPA=1 → μ≈3 h/wk; at CGPA=5 → μ≈11 h/wk.
BASE_SH   =  1.0
BETA_SH   =  2.0
SIGMA_SH  =  1.5
SH_LO     =  2.0   # floor — even the weakest student studies a little
SH_HI     = 20.0   # ceiling — 20 h/wk is an extreme but plausible outlier

# Course-load model: truncated Normal, parameters by programme code.
# Ranking (highest → lowest load): SWE-tier > CSC-tier > CYB-tier > MATH-tier
# Mapped to actual codes in the institution dataset:
#   SWE-tier : CIS, MIS
#   CSC-tier : CEN
#   CYB-tier : ICE
#   MATH-tier: MAT
#   All others → DEFAULT mid-range
#
# Each tuple: (mean, std, low_clip, high_clip)
PROG_LOAD_PARAMS: dict[str, tuple[float, float, int, int]] = {
    "CIS":     (21.5, 1.2, 20, 23),
    "MIS":     (21.0, 1.2, 20, 23),
    "CEN":     (20.0, 1.3, 18, 22),
    "ICE":     (18.5, 1.3, 16, 20),
    "MAT":     (16.5, 1.2, 15, 18),
    "DEFAULT": (18.5, 1.5, 15, 22),
}

# ~3 % of students request extra units above the programme ceiling
OVERLOAD_PROB = 0.03
OVERLOAD_LO   = 24
OVERLOAD_HI   = 26  # inclusive


# ─── SYNTHESIS FUNCTIONS ──────────────────────────────────────────────────────

def synthesize_attendance(cgpa: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return Attendance_Rate (%) for each student, clipped to [ATT_LO, ATT_HI]."""
    mu = BASE_ATT + BETA_ATT * cgpa
    raw = rng.normal(mu, SIGMA_ATT)
    return np.clip(np.round(raw, 1), ATT_LO, ATT_HI)


def synthesize_study_hours(cgpa: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return Study_Hours_Per_Week for each student, clipped to [SH_LO, SH_HI]."""
    mu = BASE_SH + BETA_SH * cgpa
    raw = rng.normal(mu, SIGMA_SH)
    return np.clip(np.round(raw, 1), SH_LO, SH_HI)


def synthesize_course_load(
    prog_codes: pd.Series, rng: np.random.Generator
) -> np.ndarray:
    """
    Return integer Course_Load (units) per student.

    Each programme draws from its own truncated Normal.  A random 3 % of
    students in every programme are assigned an overload (24–26 units) to
    simulate approved extra-unit registrations.
    """
    result = np.zeros(len(prog_codes), dtype=int)

    for prog, idx in prog_codes.groupby(prog_codes).groups.items():
        n = len(idx)
        mu, sigma, lo, hi = PROG_LOAD_PARAMS.get(prog, PROG_LOAD_PARAMS["DEFAULT"])

        # Rejection-sample until we have n values inside [lo, hi]
        samples: list[float] = []
        while len(samples) < n:
            draws = rng.normal(mu, sigma, n * 3)
            samples.extend(draws[(draws >= lo) & (draws <= hi)].tolist())

        loads = np.round(samples[:n]).astype(int)

        # Overload students
        overload_mask = rng.random(n) < OVERLOAD_PROB
        loads[overload_mask] = rng.integers(OVERLOAD_LO, OVERLOAD_HI + 1,
                                             size=overload_mask.sum())

        result[idx] = loads

    return result


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def enrich(input_path: str, output_path: str, seed: int) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    rng = np.random.default_rng(seed)

    # 1. Previous GPA — derived directly (CGPA100 is the earliest academic record)
    df["Previous_GPA"] = df["CGPA100"]

    # 2. Synthesize attendance and study hours (both correlated with CGPA + noise)
    df["Attendance_Rate"]      = synthesize_attendance(df["CGPA"].values, rng)
    df["Study_Hours_Per_Week"] = synthesize_study_hours(df["CGPA"].values, rng)

    # 3. Synthesize course load (programme-specific, structurally independent of GPA)
    df["Course_Load"] = synthesize_course_load(df["Prog Code"], rng)

    # Reorder columns for readability
    original_cols = ["ID No", "Prog Code", "Gender", "YoG",
                     "CGPA", "Previous_GPA",
                     "CGPA100", "CGPA200", "CGPA300", "CGPA400", "SGPA"]
    new_cols      = ["Attendance_Rate", "Study_Hours_Per_Week", "Course_Load"]
    df = df[original_cols + new_cols]

    df.to_csv(output_path, index=False)
    print(f"Enriched dataset saved → {output_path}  ({len(df)} rows, {len(df.columns)} columns)")
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Synthesize missing model variables.")
    p.add_argument("--input",  default="academic_performance_dataset_V2.csv")
    p.add_argument("--output", default="academic_performance_enriched.csv")
    p.add_argument("--seed",   type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    enrich(args.input, args.output, args.seed)
