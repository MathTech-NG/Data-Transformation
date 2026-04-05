"""
enrich.py
─────────────────────────────────────────────────────────────────────────────
Synthesizes three missing independent variables required by the regression /
time-series model described in:

    "Regression and Time Series Modelling of Students' Performance Across
     Semester" — Ehime Kelvin Ehinomen, Mountain Top University, Jan 2026.

Missing variables synthesized
    Attendance_Rate      (%) — class participation level
    Study_Hours_Per_Week (h) — weekly study hours
    Course_Load          (units) — registered course units per semester

Derived variable
    Previous_GPA — lagged cumulative GPA: mean(CGPA100, CGPA200, CGPA300)

─── SCIENTIFIC DESIGN DECISIONS ─────────────────────────────────────────────

Fix 1 — Inverted causal direction (endogeneity mitigation)
    Earlier versions synthesized Attendance and Study Hours as direct functions
    of CGPA: X = f(CGPA) + noise.  This made the regression partially tautological
    because the independent variables were derived from the dependent variable.

    This version uses a BLENDED generative model:
        X = BLEND_ALPHA * base_draw  +  (1 - BLEND_ALPHA) * correlated_component

    where base_draw is drawn from a CGPA-independent population distribution.
    BLEND_ALPHA = 0.60 was chosen to produce r(X, CGPA) ≈ 0.25–0.30, reflecting
    a genuine but weak behavioural correlation rather than a near-deterministic one.
    The correlated component is retained at 40% weight to keep the distributions
    realistic (a completely CGPA-independent attendance variable would be implausible).

Fix 2 — Lagged cumulative GPA (arithmetic overlap elimination)
    Previous_GPA was originally set to CGPA100.  CGPA100 is one of four components
    that make up the final CGPA (approximately their mean), so using it as a predictor
    of CGPA introduced arithmetic overlap — not just correlation.

    This version defines:
        Previous_GPA = mean(CGPA100, CGPA200, CGPA300)

    This is the student's cumulative GPA through their first three years — a
    genuinely prior, temporally lagged quantity.  CGPA400 (Level 400 performance)
    is the new information not yet captured in Previous_GPA, reducing the arithmetic
    overlap compared to v1.

─── USAGE ───────────────────────────────────────────────────────────────────
    python enrich.py [--input PATH] [--output PATH] [--seed INT]

Defaults
    --input   academic_performance_dataset_V2.csv
    --output  academic_performance_enriched.csv
    --seed    42
"""

import argparse
import numpy as np
import pandas as pd

# ─── BEHAVIOURAL VARIABLE CONSTANTS ───────────────────────────────────────────

# Blend alpha: fraction drawn from CGPA-independent population distribution.
# alpha=0.60 → r(Attendance, CGPA) ≈ 0.28,  r(StudyHours, CGPA) ≈ 0.24
BLEND_ALPHA = 0.60

# Independent base distributions
ATT_BASE_MU    = 70.0    # population mean attendance (%)
ATT_BASE_SIGMA = 12.0    # population sd

SH_BASE_MU     =  7.0    # population mean study hours/week
SH_BASE_SIGMA  =  2.5

# Correlated component parameters (realistic shape anchor)
ATT_CORR_BASE  = 45.0
ATT_CORR_BETA  =  8.0    # slope on CGPA
SH_CORR_BASE   =  1.0
SH_CORR_BETA   =  2.0

# Residual noise after blending
ATT_BLEND_NOISE = 3.0
SH_BLEND_NOISE  = 1.5

# Hard bounds
ATT_LO, ATT_HI = 40.0, 100.0
SH_LO,  SH_HI  =  2.0,  20.0


# ─── COURSE LOAD CONSTANTS ────────────────────────────────────────────────────

# Course_Load is measured in NUMBER OF COURSES per semester (not credit units).
# Ground truth from a Mountain Top University Mathematics transcript:
#   8–15 courses per semester across Years 1–3 (incl. zero-credit courses).
#   Overload students (approved extra registrations) can reach up to 20.
#
# Programme-stratified truncated Normal: (mean, std, lo, hi)
# Ranking: SWE-tier (CIS, MIS) > CSC-tier (CEN) > CYB-tier (ICE) > MATH-tier (MAT)
PROG_LOAD_PARAMS: dict[str, tuple[float, float, int, int]] = {
    "CIS":     (13.0, 1.5, 11, 15),   # SWE-tier: heaviest curriculum
    "MIS":     (13.0, 1.5, 11, 15),
    "CEN":     (12.0, 1.5, 10, 14),   # CSC-tier
    "ICE":     (11.0, 1.5,  9, 13),   # CYB-tier
    "MAT":     (10.0, 1.2,  8, 12),   # MATH-tier: lightest curriculum
    "DEFAULT": (11.0, 1.5,  8, 14),   # all other programmes
}

OVERLOAD_PROB = 0.05    # ~5% of students register extra courses
OVERLOAD_LO   = 16
OVERLOAD_HI   = 20      # inclusive — max approved overload


# ─── SYNTHESIS FUNCTIONS ──────────────────────────────────────────────────────

def synthesize_attendance(cgpa: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Blended generative model for Attendance_Rate (%).

        X = BLEND_ALPHA * base  +  (1 - BLEND_ALPHA) * correlated  +  noise

    base       ~ N(ATT_BASE_MU, ATT_BASE_SIGMA²)   [CGPA-independent]
    correlated = ATT_CORR_BASE + ATT_CORR_BETA * CGPA
    noise      ~ N(0, ATT_BLEND_NOISE²)
    """
    n           = len(cgpa)
    base        = rng.normal(ATT_BASE_MU, ATT_BASE_SIGMA, n)
    correlated  = ATT_CORR_BASE + ATT_CORR_BETA * cgpa
    noise       = rng.normal(0, ATT_BLEND_NOISE, n)
    blended     = BLEND_ALPHA * base + (1 - BLEND_ALPHA) * correlated + noise
    return np.clip(np.round(blended, 1), ATT_LO, ATT_HI)


def synthesize_study_hours(cgpa: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Blended generative model for Study_Hours_Per_Week.
    Same blend structure as attendance.
    """
    n           = len(cgpa)
    base        = rng.normal(SH_BASE_MU, SH_BASE_SIGMA, n)
    correlated  = SH_CORR_BASE + SH_CORR_BETA * cgpa
    noise       = rng.normal(0, SH_BLEND_NOISE, n)
    blended     = BLEND_ALPHA * base + (1 - BLEND_ALPHA) * correlated + noise
    return np.clip(np.round(blended, 1), SH_LO, SH_HI)


def synthesize_course_load(
    prog_codes: pd.Series, rng: np.random.Generator
) -> np.ndarray:
    """
    Programme-stratified truncated Normal for Course_Load (integer units).
    Structurally independent of GPA — r(Course_Load, CGPA) ≈ 0.
    ~3% of students receive an approved overload (24–26 units).
    """
    result = np.zeros(len(prog_codes), dtype=int)

    for prog, idx in prog_codes.groupby(prog_codes).groups.items():
        n = len(idx)
        mu, sigma, lo, hi = PROG_LOAD_PARAMS.get(prog, PROG_LOAD_PARAMS["DEFAULT"])

        samples: list[float] = []
        while len(samples) < n:
            draws = rng.normal(mu, sigma, n * 3)
            samples.extend(draws[(draws >= lo) & (draws <= hi)].tolist())

        loads = np.round(samples[:n]).astype(int)
        overload_mask = rng.random(n) < OVERLOAD_PROB
        loads[overload_mask] = rng.integers(
            OVERLOAD_LO, OVERLOAD_HI + 1, size=overload_mask.sum()
        )
        result[idx] = loads

    return result


def derive_previous_gpa(df: pd.DataFrame) -> pd.Series:
    """
    Lagged cumulative GPA: mean(CGPA100, CGPA200, CGPA300).

    Represents the student's academic trajectory through the first three years,
    prior to the Level 400 performance that drives the final CGPA update.
    This reduces (but does not fully eliminate) arithmetic overlap with CGPA.
    """
    return ((df["CGPA100"] + df["CGPA200"] + df["CGPA300"]) / 3).round(4)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def enrich(input_path: str, output_path: str, seed: int) -> pd.DataFrame:
    df  = pd.read_csv(input_path)
    rng = np.random.default_rng(seed)

    # Fix 2: lagged cumulative GPA
    df["Previous_GPA"] = derive_previous_gpa(df)

    # Fix 1: blended synthesis
    df["Attendance_Rate"]      = synthesize_attendance(df["CGPA"].values, rng)
    df["Study_Hours_Per_Week"] = synthesize_study_hours(df["CGPA"].values, rng)

    # Unchanged: programme-stratified course load
    df["Course_Load"] = synthesize_course_load(df["Prog Code"], rng)

    cols = [
        "ID No", "Prog Code", "Gender", "YoG",
        "CGPA", "Previous_GPA",
        "CGPA100", "CGPA200", "CGPA300", "CGPA400", "SGPA",
        "Attendance_Rate", "Study_Hours_Per_Week", "Course_Load",
    ]
    df = df[cols]

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
