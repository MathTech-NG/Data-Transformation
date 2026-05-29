# Technical Engineering Brief
## Regression & Time Series Modelling of Students' Academic Performance
### Prepared by: Ehime Kelvin Ehinomen | For: Data Engineer
### Status: Pre-submission revision — three model fixes required

---

## Context

This brief documents three agreed changes to the modelling pipeline before
final documentation is updated and the project is submitted for supervisor
review. The three problems and their solutions were identified during a
technical review of all scripts, datasets, and the LaTeX writeup. Each
section below states the problem, the mathematical justification for the
fix, and the exact code changes required.

The canonical pipeline files affected are:

| File | Change required |
|---|---|
| `enrich.py` | Fix genotype synthesis + update attendance for SS students |
| `prediction_common.py` | Change target variable + add interaction term to design matrix |
| `predict.py` | Update default target to `CGPA400` |
| `verify.py` | Update correlation checks for new target + new interaction column |
| `model.ipynb` | New time series section (AR(1) + trajectory analysis) |
| `app.py` | Update scoring UI to reflect `CGPA400` as DV |

The raw data file `academic_performance_dataset_V2.csv` and the enriched
output `academic_performance_enriched.csv` are also affected by the
genotype and attendance changes.

---

## Fix 1 — Change Dependent Variable from CGPA to CGPA400

### Problem

The current dependent variable is `CGPA` (overall cumulative GPA).
The dominant predictor `Previous_GPA` is defined as:

```
Previous_GPA = mean(CGPA100, CGPA200, CGPA300)
```

And `CGPA` is approximately:

```
CGPA ≈ mean(CGPA100, CGPA200, CGPA300, CGPA400)
```

This means the regression is algebraically predicting a mean-of-four using
three of its four components. Even with zero real behavioural signal, this
structure alone produces R² > 0.90. The reported R² of 0.953 and OOF R²
of 0.953 are therefore **not a model achievement** — they are a mathematical
artefact of the variable definitions. The model is not predicting; it is
recovering a known arithmetic relationship.

### Solution

Change the dependent variable to **`CGPA400`** — the student's Level 400
(final year) semester GPA.

With this change:

- `Previous_GPA = mean(CGPA100, CGPA200, CGPA300)` is a genuinely prior,
  temporally lagged quantity with **zero shared components** with `CGPA400`.
- The model now addresses a real prediction problem: *given a student's
  first three years of performance and their behavioural profile, what is
  their final year GPA?*
- R² will drop (expect approximately 0.70–0.85) but will be honest and
  scientifically defensible.
- The `--also-cgpa400` flag already exists in `predict.py`, confirming this
  option was anticipated in the original design.

### Required Code Changes

**`prediction_common.py`**

Change the default target in `fit_reference_ols`:

```python
# BEFORE
def fit_reference_ols(df: pd.DataFrame, target: str = "CGPA"):

# AFTER
def fit_reference_ols(df: pd.DataFrame, target: str = "CGPA400"):
```

Update `score_dataframe` return column name:

```python
# BEFORE
return pd.Series(model.predict(X_const), name="predicted_CGPA")

# AFTER
return pd.Series(model.predict(X_const), name="predicted_CGPA400")
```

Update `print_limitation_banner` to reference `CGPA400`:

```python
# AFTER
"Note (L1–L3, L7): Attendance, study hours, and genotype are synthesized; "
"Previous_GPA is arithmetically prior to CGPA400 (no overlap). "
"Metrics and scores are illustrative — not empirical proof of causation.\n"
```

**`predict.py`**

Change the default target in `cross_val_ols_metrics` call:

```python
# BEFORE
m = cross_val_ols_metrics(df, "CGPA", k=args.k, seed=args.seed)
print(f"Out-of-sample ..., target=CGPA, ...")

# AFTER
m = cross_val_ols_metrics(df, "CGPA400", k=args.k, seed=args.seed)
print(f"Out-of-sample ..., target=CGPA400, ...")
```

Remove or demote the `--also-cgpa400` exploratory block — it is now the
primary model, not exploratory.

**`verify.py`**

Update `check_correlations` to test against `CGPA400` where relevant
(Previous_GPA should still correlate strongly with `CGPA400`, target r > 0.60):

```python
# In check_correlations, change references from df["CGPA"] to df["CGPA400"]
# Update the Previous_GPA check band:
results.append(check(
    "Previous_GPA correlated with CGPA400 (r > 0.60)",
    r_prev > 0.60,
    f"Pearson r = {r_prev:.3f}  (lagged 3-year mean vs final-year GPA)",
))
```

**`app.py`**

Update all UI labels, column references, and output column names from
`CGPA` / `predicted_CGPA` to `CGPA400` / `predicted_CGPA400`.

---

## Fix 2 — Add Formal Time Series Analysis (AR(1) + Trajectory Modelling)

### Problem

The project title includes "Time Series Modelling" but the current
implementation only computes cross-sectional means of per-level GPAs
plotted as group lines. This is descriptive trajectory visualisation,
not time series modelling. There is no autoregressive specification,
no trend test, and no stationarity assessment.

Each student has four GPA observations across academic levels:
`CGPA100`, `CGPA200`, `CGPA300`, `CGPA400`. This constitutes a short
panel (T=4) across N=2,974 students — sufficient for the following:

### Solution — Four Components to Implement

---

#### Component A: Descriptive Trajectory Analysis

For each student, define the level-GPA series as a vector:

```
GPA_t  for  t = 1 (Level 100), 2 (Level 200), 3 (Level 300), 4 (Level 400)
```

Compute cohort-level mean GPA at each time point and plot the trajectory.
Break this down by:
- Full cohort mean trajectory
- Per-programme mean trajectory (17 programme codes)
- Per-graduation-year mean trajectory (2010–2014)

This gives rich visualisation that is already partly implemented — but
must be framed explicitly as time series descriptive analysis in the notebook.

---

#### Component B: Per-Student Linear Trend (Slope Classification)

For each student, fit a simple OLS line through their four level-GPAs:

```
GPA_t = a_i + b_i * t + e_t,   t = 1, 2, 3, 4
```

The slope `b_i` is the student's **academic trajectory**:
- `b_i > threshold` → Improving student
- `b_i < -threshold` → Declining student
- Otherwise → Stable student

Suggested threshold: 0.05 GPA points per level (tunable).

This produces a new column `Trajectory_Slope` and a categorical column
`Trajectory_Class` (Improving / Stable / Declining) for every student.
These can then be used as outcome variables or grouping variables in
subsequent analysis.

**Python implementation sketch:**

```python
import numpy as np
import pandas as pd

level_cols = ["CGPA100", "CGPA200", "CGPA300", "CGPA400"]
t = np.array([1, 2, 3, 4], dtype=float)

def compute_slope(row):
    y = row[level_cols].values.astype(float)
    # simple OLS slope through 4 points
    t_mean = t.mean()
    y_mean = y.mean()
    slope = np.sum((t - t_mean) * (y - y_mean)) / np.sum((t - t_mean) ** 2)
    return slope

df["Trajectory_Slope"] = df.apply(compute_slope, axis=1)

THRESHOLD = 0.05
def classify(slope):
    if slope > THRESHOLD:
        return "Improving"
    elif slope < -THRESHOLD:
        return "Declining"
    else:
        return "Stable"

df["Trajectory_Class"] = df["Trajectory_Slope"].apply(classify)
```

Add `Trajectory_Slope` and `Trajectory_Class` to the enriched CSV output
and to `enrich.py` so the columns are always present after enrichment.

---

#### Component C: AR(1) Model on the Panel

The AR(1) (first-order autoregressive) model on the level-GPA sequence is:

```
GPA_t = φ₀ + φ₁ · GPA_(t-1) + ε_t
```

This answers the question: *how strongly does last year's GPA predict
this year's GPA, independent of any external factors?*

**Estimation approach:**

Stack the panel into long format — each student contributes three
(GPA_t, GPA_{t-1}) pairs (transitions 100→200, 200→300, 300→400):

```python
long_rows = []
for _, row in df.iterrows():
    gpas = [row["CGPA100"], row["CGPA200"], row["CGPA300"], row["CGPA400"]]
    for i in range(1, 4):
        long_rows.append({
            "student_id": row["ID No"],
            "level": i + 1,
            "GPA_t":   gpas[i],
            "GPA_lag": gpas[i - 1],
        })

panel = pd.DataFrame(long_rows)
```

Then fit OLS (or statsmodels OLS) on `GPA_t ~ GPA_lag`:

```python
import statsmodels.api as sm

X = sm.add_constant(panel["GPA_lag"])
y = panel["GPA_t"]
ar1_model = sm.OLS(y, X).fit()
print(ar1_model.summary())
```

Report:
- `φ₁` (autoregressive coefficient) — expected to be significant and
  positive, in the range 0.55–0.75
- R² of the AR(1) alone — this is the baseline "inertia" in GPA
- Residuals from AR(1) can be used as a "GPA surprise" variable

**Interpretation to write up:**
The AR(1) coefficient quantifies academic momentum. A φ₁ of 0.65, for
example, means that a student who scores 1.0 GPA point above the mean
in one year is expected to score 0.65 points above the mean the following
year, holding all else equal. This is the formal time series result the
project title requires.

---

#### Component D: Stationarity and Trend Test

On the cohort-mean GPA series (4 time points), perform:

1. **Visual trend check** — does the mean rise, fall, or stay flat
   from Level 100 to Level 400?
2. **Mann-Kendall trend test** (non-parametric, suitable for short
   series) using `scipy` or the `pymannkendall` package:

```python
# pip install pymannkendall
import pymannkendall as mk

mean_series = df[["CGPA100", "CGPA200", "CGPA300", "CGPA400"]].mean()
result = mk.original_test(mean_series)
print(result)  # trend, p-value, Tau, slope
```

Report the trend direction and p-value. Even if the cohort mean is
relatively flat, the test result is a legitimate time series finding.

---

#### Summary: What to Add to `model.ipynb`

Add a dedicated notebook section titled **"Time Series Analysis"**
with the following subsections:

1. Cohort mean GPA trajectory (plot)
2. Programme-level trajectories (faceted plot)
3. Per-student slope computation and trajectory classification
4. Distribution of Trajectory_Slope (histogram)
5. AR(1) panel model — table of coefficients and summary
6. Mann-Kendall trend test on cohort mean series
7. Trajectory class breakdown by programme and gender (crosstab)

---

## Fix 3 — Strengthen Genotype: Health-Pathway Interaction Model

### Problem

Genotype is currently synthesized as completely independent of CGPA
and of all other variables. This was done to avoid introducing spurious
direct correlation, but the consequence is that the regression coefficient
is effectively noise (β ≈ -0.0002, p = 0.977). The project promised
genotype as a meaningful health-related innovation in Chapter 1, but
the current implementation delivers a variable that is guaranteed to be
non-significant by construction.

The deeper issue is a misspecification of the causal pathway. The
literature on sickle cell disease (the basis for the SS genotype
inclusion) does not suggest that genotype directly suppresses GPA.
It suggests that genotype affects **health burden**, which affects
**attendance and consistency**, which then affects GPA. The correct
model encodes this indirect pathway.

### Solution — Two Parts

---

#### Part A: Update Genotype Synthesis in `enrich.py`

SS students should have a lower mean attendance rate and higher
attendance variance, reflecting the real-world burden of sickle cell
health episodes. AS students should have a marginal reduction relative
to AA.

**Update `synthesize_attendance` in `enrich.py`:**

```python
# Add genotype-conditional mean shift before blending
GENOTYPE_ATT_SHIFT = {"AA": 0.0, "AS": -2.5, "SS": -8.0}
GENOTYPE_ATT_NOISE_EXTRA = {"AA": 0.0, "AS": 0.5, "SS": 3.0}

def synthesize_attendance(
    cgpa: np.ndarray,
    genotype: np.ndarray,           # <-- new parameter
    rng: np.random.Generator
) -> np.ndarray:
    n           = len(cgpa)
    base        = rng.normal(ATT_BASE_MU, ATT_BASE_SIGMA, n)
    correlated  = ATT_CORR_BASE + ATT_CORR_BETA * cgpa
    noise       = rng.normal(0, ATT_BLEND_NOISE, n)
    blended     = BLEND_ALPHA * base + (1 - BLEND_ALPHA) * correlated + noise

    # Apply genotype-conditional attendance penalty
    shift = np.array([GENOTYPE_ATT_SHIFT.get(g, 0.0) for g in genotype])
    extra_noise_std = np.array([GENOTYPE_ATT_NOISE_EXTRA.get(g, 0.0) for g in genotype])
    extra_noise = rng.normal(0, 1, n) * extra_noise_std
    blended = blended + shift + extra_noise

    return np.clip(np.round(blended, 1), ATT_LO, ATT_HI)
```

Update the call in `enrich`:

```python
# Genotype must be synthesized BEFORE attendance so the shift can be applied
df["Genotype"] = synthesize_genotype(len(df), rng)
df["Attendance_Rate"] = synthesize_attendance(df["CGPA"].values, df["Genotype"].values, rng)
```

**Why these values:**
- AA (no sickle cell): no shift, baseline noise
- AS (sickle cell trait): mild shift (-2.5%), slightly higher noise
- SS (sickle cell disease): meaningful shift (-8%), substantially higher noise

These values are conservative and defensible. An SS student averaging
8 percentage points lower attendance than an AA student is well within
what the clinical literature on sickle cell disease would support.

---

#### Part B: Add Genotype × Attendance Interaction Term to the Regression

The formal interaction hypothesis is:

> For SS students, the slope of Attendance → CGPA400 is attenuated,
> because their absences are health-driven rather than effort-driven.
> A high-attendance SS student signals strong resilience; a low-attendance
> SS student may still be engaged but physically unable to attend.

The updated regression equation is:

```
CGPA400_i = β₀
           + β₁ · Previous_GPA_i
           + β₂ · Attendance_i
           + β₃ · StudyHours_i
           + β₄ · CourseLoad_i
           + β₅ · D_AS_i
           + β₆ · D_SS_i
           + β₇ · (D_SS_i × Attendance_i)    ← NEW INTERACTION TERM
           + ε_i
```

**Update `prediction_common.py` — `build_design_matrix`:**

```python
CONTINUOUS_PREDICTORS = [
    "Previous_GPA",
    "Attendance_Rate",
    "Study_Hours_Per_Week",
    "Course_Load",
]
GENOTYPE_DUMMY_COLS = ["Genotype_AS", "Genotype_SS"]
INTERACTION_COLS    = ["Genotype_SS_x_Attendance"]   # NEW
DESIGN_COLUMNS      = CONTINUOUS_PREDICTORS + GENOTYPE_DUMMY_COLS + INTERACTION_COLS

def build_design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    # ... existing validation code unchanged ...

    cont = df[CONTINUOUS_PREDICTORS].astype(float)
    gt   = df["Genotype"].astype(str)
    dums = pd.DataFrame(
        {
            "Genotype_AS": (gt == "AS").astype(float),
            "Genotype_SS": (gt == "SS").astype(float),
        },
        index=df.index,
    )
    # Interaction: SS dummy × Attendance_Rate
    interaction = pd.DataFrame(
        {
            "Genotype_SS_x_Attendance": (
                dums["Genotype_SS"] * df["Attendance_Rate"].astype(float)
            )
        },
        index=df.index,
    )
    return pd.concat([cont, dums, interaction], axis=1)
```

**What to expect from this interaction term:**

- If β₇ is **negative**: the attendance→GPA slope is weaker for SS
  students than for AA students. This supports the health-pathway
  hypothesis — attendance is a noisier signal of engagement for SS.
- If β₇ is **near zero or non-significant**: the interaction is not
  supported in this dataset under these synthesis assumptions. That is
  still a valid finding and should be reported as such.
- SS has only ~32 students in the sample (1.1% of 2,974). The
  interaction term will have low power. This must be noted as a
  limitation — the finding is indicative, not conclusive, and would
  require a larger sample of SS students to confirm.

---

#### Part C: Update `verify.py` for New Variables

Add checks for:

1. SS mean attendance is lower than AA mean attendance (directional check):

```python
aa_att = df[df["Genotype"] == "AA"]["Attendance_Rate"].mean()
ss_att = df[df["Genotype"] == "SS"]["Attendance_Rate"].mean()
results.append(check(
    "SS mean Attendance_Rate < AA mean (health-pathway synthesis)",
    ss_att < aa_att,
    f"AA mean = {aa_att:.2f}%, SS mean = {ss_att:.2f}%",
))
```

2. Trajectory columns present after enrichment:

```python
results.append(check(
    "Trajectory_Slope and Trajectory_Class columns present",
    "Trajectory_Slope" in df.columns and "Trajectory_Class" in df.columns,
))
```

---

## Summary Checklist for the Data Engineer

| # | File | Change | Priority |
|---|---|---|---|
| 1 | `prediction_common.py` | Change default DV to `CGPA400` everywhere | Critical |
| 2 | `prediction_common.py` | Add `Genotype_SS_x_Attendance` interaction to design matrix | Critical |
| 3 | `predict.py` | Update default target + output labels to `CGPA400` | Critical |
| 4 | `enrich.py` | Synthesize genotype before attendance | Critical |
| 5 | `enrich.py` | Add genotype-conditional attendance shift (SS: -8%, AS: -2.5%) | Critical |
| 6 | `enrich.py` | Add `Trajectory_Slope` and `Trajectory_Class` columns | High |
| 7 | `model.ipynb` | Add full time series section (A–D above) | High |
| 8 | `verify.py` | Update correlation checks for `CGPA400` target | High |
| 9 | `verify.py` | Add SS attendance check + trajectory column checks | High |
| 10 | `app.py` | Update all DV labels/columns from `CGPA` to `CGPA400` | Medium |
| 11 | `score.py` | Update output column name to `predicted_CGPA400` | Medium |

---

## Expected Outcomes After Fixes

| Metric | Before Fix | Expected After Fix |
|---|---|---|
| Dependent variable | CGPA (arithmetic overlap) | CGPA400 (genuine prediction target) |
| Adjusted R² | 0.953 (inflated) | ~0.70–0.85 (honest) |
| OOF R² (5-fold CV) | 0.953 (inflated) | ~0.65–0.80 |
| Previous_GPA overlap | Yes (3 of 4 components shared) | None |
| Genotype p-value | 0.977 (noise) | Interaction term estimable |
| Time series component | Cross-sectional means only | AR(1) + slopes + trend test |
| Trajectory columns | Absent | Trajectory_Slope, Trajectory_Class |

---

## Notes for Documentation Update (After Engineer Fixes)

Once the above changes are implemented and `verify.py` passes all checks,
the following documentation sections must be updated before supervisor submission:

- **Chapter 1 regression equation**: update from 3-predictor to 7-predictor
  specification (including interaction term)
- **Chapter 3 Section on DV**: rewrite to justify CGPA400 as the prediction
  target; explain why it eliminates arithmetic overlap
- **Chapter 3 Time Series section**: add AR(1) specification formally
- **Chapter 3 Genotype section**: update synthesis description to include
  health-pathway attendance shift and interaction term
- **Chapter 4 Results**: update all reported R², MAE, RMSE, and coefficient
  tables from new model runs
- **Chapter 4 New section**: Time Series Results (AR(1) coefficients,
  trajectory distribution, programme trajectories, Mann-Kendall result)
- **Chapter 5 Limitations**: revise L2 (overlap eliminated), update L7
  (genotype now carries indirect pathway), note low power for SS interaction

---

## Addendum — Trajectory_Slope_Prior in OLS / Scorer (May 2026)

**Problem:** Four-level `Trajectory_Slope` includes CGPA400 → cannot predict CGPA400 without leakage.

**Fix:** Add `Trajectory_Slope_Prior` (OLS slope through CGPA100–300 only) to `enrich.py`,
`prediction_common.py` design matrix, `app.py` scorer, and `score.py`.

**Keep:** Four-level `Trajectory_Slope` / `Trajectory_Class` for EDA and Trajectory tab only.

**Observed (seed 42, n=2974):** Adj R² ≈ 0.682; β(Trajectory_Slope_Prior) ≈ 0.448; OOF MAE ≈ 0.357.

**verify.py:** 52 checks (prior slope schema, VIF, significance).

---

*Brief prepared for internal use. All changes use fixed random seed 42
for full reproducibility. Re-run `python enrich.py && python verify.py`
after each change to confirm pipeline integrity before proceeding to
notebook and app updates.*
