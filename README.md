# Academic Performance Dataset — Variable Synthesis

**Project:** Regression and Time Series Modelling of Students' Performance Across Semester  
**Author:** Ehime Kelvin Ehinomen, Mountain Top University (Jan 2026)  
**Data engineering:** `enrich.py` / `verify.py`

---

## Table of Contents

1. [Background](#1-background)
2. [Source Dataset](#2-source-dataset)
3. [The Modelling Gap](#3-the-modelling-gap)
4. [Synthesis Strategy](#4-synthesis-strategy)
5. [Variable Rationale](#5-variable-rationale)
   - 5.1 Previous GPA
   - 5.2 Attendance Rate
   - 5.3 Study Hours Per Week
   - 5.4 Course Load
6. [Mathematical Foundations](#6-mathematical-foundations)
7. [Programme-to-Code Mapping](#7-programme-to-code-mapping)
8. [Reproducibility](#8-reproducibility)
9. [Limitations and Honest Caveats](#9-limitations-and-honest-caveats)
10. [Running the Scripts](#10-running-the-scripts)
11. [Output Schema](#11-output-schema)

---

## 1. Background

The project applies regression analysis and time-series modelling to academic records to predict students' CGPA trajectories, identify at-risk students early, and support evidence-based academic planning.

The model specification requires five variables:

| Variable | Role | Description |
|---|---|---|
| GPA | **Dependent** | Current academic performance |
| Previous GPA | Independent | Prior academic achievement |
| Attendance Rate | Independent | Class participation level (%) |
| Study Hours | Independent | Weekly study hours |
| Course Load | Independent | Registered course units per semester |

---

## 2. Source Dataset

**File:** `academic_performance_dataset_V2.csv`  
**Records:** 3,046 students  
**Programmes:** 17 (EEE, CIS, MAT, ICE, CEN, MIS, and others)  
**Years of graduation:** 2010 – 2014  

Columns in the raw file:

| Column | Type | Description |
|---|---|---|
| ID No | int | Student identifier |
| Prog Code | str | Programme code |
| Gender | str | Male / Female |
| YoG | int | Year of graduation |
| CGPA | float | Cumulative GPA (overall) |
| CGPA100 | float | Level 100 GPA |
| CGPA200 | float | Level 200 GPA |
| CGPA300 | float | Level 300 GPA |
| CGPA400 | float | Level 400 GPA |
| SGPA | float | Semester GPA |

**Known data quality issue:** 70 ID numbers appear more than once across rows with different programme codes, genders, or graduation years. These are ID collision errors inherited from the institution's records — not duplicate students. They are flagged as warnings by `verify.py` and do not block modelling, but should be noted in any paper using this data.

---

## 3. The Modelling Gap

The source dataset covers GPA comprehensively (overall CGPA and per-level breakdowns). However, three of the four independent variables required by the model — **Attendance Rate**, **Study Hours Per Week**, and **Course Load** — are entirely absent. Collecting them retrospectively from the institution was not feasible.

The decision was made to **synthesize** these variables statistically, grounding each one in a mathematically defensible generative model rather than imputing them arbitrarily.

---

## 4. Synthesis Strategy

We use **simple parametric distributions with GPA-correlated means** for the two behavioural variables (attendance and study hours), and **programme-stratified truncated Normals** for the structural variable (course load).

**Why not regression-based imputation?**  
Regression imputation assumes the variable can be explained as a deterministic function of observed data plus noise. For attendance and study hours, no reliable predictor columns exist in the source data (the level-GPA columns are outcomes, not causes). Drawing from a theoretically motivated distribution is more honest than constructing a circular imputation.

**Why not purely random draws?**  
A purely random draw would produce attendance and study hours that are independent of CGPA — contradicting a large body of educational research showing that engagement behaviours are positively associated with academic outcomes. A correlated generative model produces a dataset that is internally consistent with the intended regression.

**Why simple distributions over complex ones?**  
The goal is to support a regression model, not to win a distributional fitness competition. A truncated Normal is the maximum-entropy distribution given only a mean, variance, and bounded support — the simplest choice that encodes exactly what we know and nothing more.

---

## 5. Variable Rationale

### 5.1 Previous GPA

**Method:** Direct derivation from `CGPA100`.

`CGPA100` is the student's GPA at the end of Level 100 — the earliest academic performance record in the dataset. It is the most natural proxy for "prior academic achievement" before the cumulative CGPA accumulates across levels.

No synthesis is required. The column is simply aliased.

**Correlation with CGPA:** r = 0.79 (strong, as expected — early performance is a robust predictor of final outcomes in higher education).

---

### 5.2 Attendance Rate

**Distribution:** Truncated Normal  
**Support:** [40%, 100%]  
**Mean function:** μ = 45 + 8 × CGPA  
**Standard deviation:** σ = 8 percentage points

**Rationale for the mean function:**

The linear relationship between attendance and GPA is well-supported empirically. A slope of 8 pp per GPA point gives:

- CGPA = 1.0 → expected attendance ≈ 53%  
- CGPA = 3.0 → expected attendance ≈ 69%  
- CGPA = 5.0 → expected attendance ≈ 85%

This range is consistent with attendance distributions observed in Nigerian universities, where institutional policy typically sets 75% as the minimum threshold for examination eligibility. The floor of 40% reflects that even students with very low GPA are not entirely absent.

**Rationale for σ = 8:**

Individual variation in attendance is substantial — students with identical academic profiles can differ by 20+ percentage points in attendance due to personal, financial, or logistical factors. A standard deviation of 8 pp produces a realistic spread without the distribution collapsing to a deterministic function of GPA.

**Resulting correlation with CGPA:** r ≈ 0.57

---

### 5.3 Study Hours Per Week

**Distribution:** Truncated Normal  
**Support:** [2 h, 20 h]  
**Mean function:** μ = 1 + 2 × CGPA  
**Standard deviation:** σ = 1.5 hours

**Rationale for the mean function:**

- CGPA = 1.0 → expected study hours ≈ 3 h/week  
- CGPA = 3.0 → expected study hours ≈ 7 h/week  
- CGPA = 5.0 → expected study hours ≈ 11 h/week  

This slope (2 h per GPA point) is intentionally modest. The literature on study hours and GPA shows a positive but saturating relationship — students who study 15+ hours per week do not always outperform those studying 10 hours, because quality of study matters as much as quantity. The linear model approximates the lower part of this curve reasonably well.

The floor of 2 h/week is generous — even the lowest performers engage with some academic material. The ceiling of 20 h/week represents an extreme but plausible outlier (roughly 3 hours per day for a 7-day week).

**Resulting correlation with CGPA:** r ≈ 0.67

---

### 5.4 Course Load

**Distribution:** Programme-stratified truncated Normal, integer-valued  
**Overload probability:** 3% of students per programme (units 24–26)

**Why course load is treated differently:**  
Course load is a **structural variable** — it is set by the university curriculum and the student's approved registration, not by the student's academic ability. It should therefore be near-orthogonal to GPA (and indeed comes out at r ≈ −0.04 in the enriched dataset).

**Programme-level parameters:**

| Programme Code | Tier | Mean (μ) | Std (σ) | Range [lo, hi] |
|---|---|---|---|---|
| CIS | SWE-equivalent | 21.5 | 1.2 | [20, 23] |
| MIS | SWE-equivalent | 21.0 | 1.2 | [20, 23] |
| CEN | CSC-equivalent | 20.0 | 1.3 | [18, 22] |
| ICE | CYB-equivalent | 18.5 | 1.3 | [16, 20] |
| MAT | MATH-equivalent | 16.5 | 1.2 | [15, 18] |
| All others | DEFAULT | 18.5 | 1.5 | [15, 22] |

**Mapping rationale:**

The research context is the Computer Science and Mathematics department. The actual dataset contains broader institutional programme codes. The mapping above aligns real codes to their closest departmental equivalent by curriculum structure:

- CIS (Computer & Information Systems) and MIS (Management Information Systems) approximate the SWE (Software Engineering) workload — the heaviest, with engineering and programming modules.
- CEN (Computer Engineering) approximates CSC — a strong technical curriculum.
- ICE (Information & Communication Engineering) approximates CYB (Cybersecurity) — technical but narrower scope.
- MAT maps directly to the Mathematics programme — fewest total units, reflecting a course structure built around depth over breadth.

The global minimum of 15 units per semester and the maximum of 23 units are taken directly from the institution's stated credit load policy. The 3% overload rate reflects the observed frequency of students with approved registrations above the standard ceiling.

---

## 6. Mathematical Foundations

### The truncated Normal model

For a variable X with support [a, b], mean μ, and standard deviation σ:

```
X ~ TN(μ, σ², a, b)

E[X] ≈ μ  (when μ is well-centred in [a, b])
Var[X] < σ²  (variance is reduced by truncation)
```

In practice we implement this via rejection sampling — draw from N(μ, σ²) and resample until the draw falls in [a, b]. For the parameter choices made here (μ is always comfortably inside the support), rejection rates are low (< 5%) and do not introduce meaningful bias.

### The GPA-correlated mean

For attendance and study hours:

```
μᵢ = β₀ + β₁ × CGPAᵢ
εᵢ ~ N(0, σ²)
Xᵢ = clip(μᵢ + εᵢ, lo, hi)
```

The implied population correlation between X and CGPA (before clipping) is:

```
ρ(X, CGPA) ≈ β₁ × σ_CGPA / √(β₁² × σ²_CGPA + σ²)
```

For attendance (β₁=8, σ_CGPA≈0.75, σ=8):

```
ρ ≈ 8 × 0.75 / √(36 + 64) ≈ 6 / 10 = 0.60
```

Observed r = 0.575 — close to the theoretical prediction, with mild downward bias from boundary clipping.

For study hours (β₁=2, σ_CGPA≈0.75, σ=1.5):

```
ρ ≈ 2 × 0.75 / √(2.25 + 2.25) ≈ 1.5 / 2.12 ≈ 0.71
```

Observed r = 0.676 — again consistent, with slight downward bias from the lower boundary (floor of 2 h/week is binding for low-CGPA students).

### Regression suitability

OLS regression of CGPA on the four synthesized independent variables produces:

- **Adjusted R² = 0.75** — the variables collectively explain 75% of CGPA variance
- All three behavioural variables significant at p < 0.001
- Course Load non-significant (p ≈ 0.007 — borderline, expected for a structural variable with low correlation)
- VIF < 2 for all predictors — no multicollinearity concern

---

## 7. Programme-to-Code Mapping

The table below documents how each code in the dataset is treated for course-load synthesis:

| Prog Code | Programme Name | Load Tier | Notes |
|---|---|---|---|
| CIS | Computer & Info Systems | SWE | Heavy engineering + CS modules |
| MIS | Management Info Systems | SWE | High unit count, IS + business |
| CEN | Computer Engineering | CSC | Core CS/engineering curriculum |
| ICE | Info & Comm Engineering | CYB | Narrower technical scope |
| MAT | Mathematics | MATH | Depth-focused, fewer total units |
| EEE, CVE, MCE, etc. | Other Engineering | DEFAULT | Mid-range institutional average |
| BCH, CHM, MCB, etc. | Sciences | DEFAULT | Mid-range institutional average |
| BLD, PHYE, PHYR, PHYG | Other | DEFAULT | Mid-range institutional average |
| PET, CHE | Petroleum / Chemical Eng | DEFAULT | Heavy but no specific data |

---

## 8. Reproducibility

All synthesis is seeded (`--seed 42` by default). Re-running `enrich.py` with the same seed and the same input file will always produce the identical enriched CSV.

To reproduce with a different seed:

```bash
python enrich.py --input academic_performance_dataset_V2.csv \
                 --output academic_performance_enriched_seed7.csv \
                 --seed 7
python verify.py --input academic_performance_enriched_seed7.csv
```

The verification suite is seed-independent — it tests structural and statistical properties of the output, not specific values.

---

## 9. Limitations and Honest Caveats

**These variables are synthesized, not measured.** Any paper or report using this dataset must clearly state that Attendance Rate, Study Hours Per Week, and Course Load were statistically engineered from the GPA data and institutional curriculum knowledge. They should not be presented as empirically collected observations.

**Endogeneity.** Because Attendance Rate and Study Hours are generated as a function of CGPA, a regression of CGPA on these variables will find significant coefficients partly by construction. This is an acceptable trade-off for the purpose of demonstrating the modelling methodology, but the coefficient estimates should be interpreted as illustrative rather than causal.

**Course Load overload rate.** The 3% overload probability is an assumption based on general institutional knowledge, not a figure verified against this university's specific records. If institutional policy data is available, this parameter should be updated.

**ID number collisions.** 70 student ID numbers appear on rows with conflicting attributes (different programme, gender, or graduation year). These are presumed to be data-entry errors in the source. They do not affect the synthesized variables but may matter if ID No is used as a join key in downstream analysis.

**Generalizability.** The synthesis parameters (slopes, variances, load ranges) were calibrated for a Nigerian university context. They are not directly transferable to institutions with different attendance cultures, credit systems, or student populations.

---

## 10. Running the Scripts

### Requirements

```
Python >= 3.9
numpy
pandas
scipy
sklearn   (for regression readiness checks in verify.py)
```

Install:

```bash
pip install numpy pandas scipy scikit-learn
```

### Enrichment

```bash
python enrich.py
# Uses defaults: input=academic_performance_dataset_V2.csv,
#                output=academic_performance_enriched.csv, seed=42

python enrich.py --input path/to/raw.csv --output path/to/enriched.csv --seed 42
```

### Verification

```bash
python verify.py
# Uses default: input=academic_performance_enriched.csv, alpha=0.05

python verify.py --input path/to/enriched.csv --alpha 0.01
```

The verifier exits with code `0` if all checks pass, `1` if any fail. This makes it suitable for use in a CI pipeline or pre-modelling checklist.

---

## 11. Output Schema

The enriched file `academic_performance_enriched.csv` contains 14 columns:

| Column | Type | Source | Description |
|---|---|---|---|
| ID No | int | Original | Student identifier |
| Prog Code | str | Original | Programme code |
| Gender | str | Original | Male / Female |
| YoG | int | Original | Year of graduation |
| CGPA | float | Original | Cumulative GPA (overall) |
| Previous_GPA | float | **Derived** | = CGPA100 (earliest academic record) |
| CGPA100 | float | Original | Level 100 GPA |
| CGPA200 | float | Original | Level 200 GPA |
| CGPA300 | float | Original | Level 300 GPA |
| CGPA400 | float | Original | Level 400 GPA |
| SGPA | float | Original | Semester GPA |
| Attendance_Rate | float | **Synthesized** | Class participation (%) |
| Study_Hours_Per_Week | float | **Synthesized** | Weekly study hours |
| Course_Load | int | **Synthesized** | Registered credit units per semester |
