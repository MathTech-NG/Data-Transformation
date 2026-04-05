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
5. [Scientific Design Decisions](#5-scientific-design-decisions)
   - 5.1 Fix 1 — Endogeneity mitigation (blended synthesis)
   - 5.2 Fix 2 — Arithmetic overlap reduction (lagged cumulative GPA)
6. [Variable Rationale](#6-variable-rationale)
7. [Mathematical Foundations](#7-mathematical-foundations)
8. [Programme-to-Code Mapping](#8-programme-to-code-mapping)
9. [Honest Assessment of Remaining Limitations](#9-honest-assessment-of-remaining-limitations)
10. [Reproducibility](#10-reproducibility)
11. [Running the Scripts](#11-running-the-scripts)
12. [Output Schema](#12-output-schema)

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
| Course Load | Independent | Number of courses registered per semester |

---

## 2. Source Dataset

**File:** `academic_performance_dataset_V2.csv`  
**Records:** 3,046 students  
**Programmes:** 17 (EEE, CIS, MAT, ICE, CEN, MIS, and others)  
**Years of graduation:** 2010–2014  

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

**Known data quality issue:** 70 ID numbers appear on rows with different programme codes, genders, or graduation years. These are ID collision errors inherited from the institution's records — not duplicate students. They are flagged as warnings by `verify.py`. The paper should note this.

---

## 3. The Modelling Gap

The source dataset covers GPA comprehensively. However, three of the four independent variables required by the model — Attendance Rate, Study Hours Per Week, and Course Load — are entirely absent. Collecting them retrospectively was not feasible.

The decision was made to synthesize them statistically, grounding each in a mathematically defensible generative model.

---

## 4. Synthesis Strategy

We use **blended parametric distributions** for the two behavioural variables (attendance and study hours) and **programme-stratified truncated Normals** for the structural variable (course load).

This README documents two versions of the synthesis:

- **v1 (deprecated):** Attendance and Study Hours generated as direct functions of CGPA. Previous GPA set to CGPA100.
- **v2 (current):** Blended generative model to mitigate endogeneity. Previous GPA redefined as lagged three-year cumulative average.

The rationale for moving from v1 to v2 is documented in Section 5.

---

## 5. Scientific Design Decisions

### 5.1 Fix 1 — Endogeneity mitigation (blended synthesis)

**The problem in v1:**

Attendance and Study Hours were generated as:

```
X = f(CGPA) + noise
```

When you then regress CGPA on X, you are partially recovering the function you imposed. The R² of ~0.75 and the significant p-values in v1 proved the synthesis worked — not that the model discovered a real relationship. An examiner running OLS on this data would be right to reject the results as tautological.

**The fix in v2:**

A blended generative model:

```
X = α * base  +  (1 - α) * correlated_component  +  noise

where  base ~ N(μ_pop, σ_pop²)  is drawn independently of CGPA
       α = 0.60
```

`BLEND_ALPHA = 0.60` means 60% of each synthesized value comes from a CGPA-independent population distribution. The correlated component is retained at 40% weight to keep the distributions realistic — a completely CGPA-independent attendance variable would be implausible given the literature, but the correlation should be *weak*, not near-deterministic.

**Result:**
- v1: r(Attendance, CGPA) ≈ 0.57, r(StudyHours, CGPA) ≈ 0.67
- v2: r(Attendance, CGPA) ≈ 0.28, r(StudyHours, CGPA) ≈ 0.24

The v2 correlations reflect a genuine but modest behavioural relationship — consistent with the literature on engagement and academic outcomes, where study hours and attendance explain some but not most of GPA variance.

**What this does not fix:**

The behavioural variables are still synthesized from CGPA data. The endogeneity is reduced, not eliminated. Any paper using this dataset must state clearly that the regression results are illustrative of the modelling methodology, not evidence of empirical relationships.

---

### 5.2 Fix 2 — Arithmetic overlap reduction (lagged cumulative GPA)

**The problem in v1:**

`Previous_GPA = CGPA100`

CGPA100 is one of the four components of the final CGPA (which is approximately their mean across levels). Using it as a predictor of CGPA does not just introduce correlation — it introduces *arithmetic overlap*. Part of what the regression is explaining is a definitional relationship, not a behavioural one.

**The fix in v2:**

```
Previous_GPA = mean(CGPA100, CGPA200, CGPA300)
```

This is the student's cumulative academic record through their first three years — a genuinely *lagged* quantity. The Level 400 performance (CGPA400) is the new information not yet captured in `Previous_GPA`, making the final CGPA a meaningful outcome relative to prior trajectory.

**Result:**

r(Previous_GPA, CGPA) rises from 0.79 (v1) to 0.976 (v2) — because the three-year average is a much stronger predictor of the four-year average than CGPA100 alone. This is expected and reflects genuine predictive content.

**What this does not fully fix:**

`Previous_GPA` in v2 is still arithmetically related to CGPA — it shares three of the four components that make up the cumulative average. The overlap is reduced compared to v1 (one shared component → three shared out of four), but the underlying structure means `Previous_GPA` will always dominate R² in any regression. The adj R² of 0.954 observed in v2 is driven almost entirely by this variable, not by the behavioural predictors.

**The cleanest possible fix** (not implemented here, but recommended if data allows) would be to reframe the dependent variable as **CGPA400 alone** — the student's Level 400 performance — and define `Previous_GPA = mean(CGPA100, CGPA200, CGPA300)`. Under this framing, `Previous_GPA` is genuinely prior with zero arithmetic overlap. This reframing would require restructuring the model's research question, which is a decision for the paper's author.

---

## 6. Variable Rationale

### Previous GPA

**Definition:** `mean(CGPA100, CGPA200, CGPA300)`

The three-year cumulative average captures the student's full prior academic trajectory rather than only their earliest performance. It is the most defensible proxy for "prior academic achievement" given the available data structure.

### Attendance Rate

**Distribution:** Blended (see Section 5.1)  
**Base population:** N(70, 12²), clipped to [40, 100]  
**Blend weight:** 60% independent, 40% CGPA-correlated  
**Resulting r with CGPA:** ≈ 0.28

A floor of 40% reflects that even low-performing students are not entirely absent. A population mean of 70% is consistent with Nigerian university attendance norms, where 75% is a common minimum threshold for examination eligibility.

### Study Hours Per Week

**Distribution:** Blended (see Section 5.1)  
**Base population:** N(7, 2.5²), clipped to [2, 20]  
**Blend weight:** 60% independent, 40% CGPA-correlated  
**Resulting r with CGPA:** ≈ 0.24

A population mean of 7 h/week is realistic for undergraduate students in a Nigerian university context. The floor of 2 h/week is generous — even the weakest performers engage with some material. The ceiling of 20 h/week represents an extreme but plausible outlier.

### Course Load

**Distribution:** Programme-stratified truncated Normal, integer-valued  
**Unit of measurement:** Number of courses per semester (not credit units)  
**Overload rate:** ~5% per programme (16–20 courses)  
**Resulting r with CGPA:** ≈ −0.03 (structurally near-zero, as intended)

Course load is set by curriculum and approved registration — not by the student's academic ability. It should be near-orthogonal to GPA.

**Ground truth calibration:** A verified Mountain Top University transcript (BSc Mathematics, Years 1–3) shows 8–15 courses per semester across all levels, including zero-credit institutional courses (ESM, ICT, PIF, SDN series). Overload students with approved extra registrations are modelled up to 20 courses.

**Programme-level parameters:**

| Programme Code | Tier | Mean (μ) | Std (σ) | Normal Range | Overload |
|---|---|---|---|---|---|
| CIS | SWE-equivalent | 13.0 | 1.5 | [11, 15] | up to 20 |
| MIS | SWE-equivalent | 13.0 | 1.5 | [11, 15] | up to 20 |
| CEN | CSC-equivalent | 12.0 | 1.5 | [10, 14] | up to 20 |
| ICE | CYB-equivalent | 11.0 | 1.5 | [9, 13] | up to 20 |
| MAT | MATH-equivalent | 10.0 | 1.2 | [8, 12] | up to 20 |
| All others | DEFAULT | 11.0 | 1.5 | [8, 14] | up to 20 |

---

## 7. Mathematical Foundations

### Blended generative model

For a behavioural variable X with CGPA-independent base distribution N(μ_pop, σ_pop²):

```
base_i        ~ N(μ_pop, σ_pop²)
correlated_i  = β₀ + β₁ × CGPA_i
noise_i       ~ N(0, σ_noise²)
X_i           = clip(α × base_i + (1-α) × correlated_i + noise_i,  lo, hi)
```

The implied population correlation before clipping:

```
ρ(X, CGPA) ≈ (1-α) × β₁ × σ_CGPA  /  √[ α²σ²_pop + (1-α)²β₁²σ²_CGPA + σ²_noise ]
```

For attendance (α=0.60, β₁=8, σ_CGPA≈0.75, σ_pop=12, σ_noise=3):

```
numerator   = 0.40 × 8 × 0.75  = 2.4
denominator = √[ 0.36×144 + 0.16×36×0.5625 + 9 ]  ≈  √[ 51.84 + 3.24 + 9 ]  ≈  8.03
ρ ≈ 0.30
```

Observed r = 0.28 — consistent with theory, with slight downward bias from boundary clipping.

### Truncated Normal for course load

For a variable in [lo, hi] with mean μ and std σ, rejection sampling is used: draw from N(μ, σ²) and resample until the value falls in [lo, hi]. For all programme parameter choices, μ is well-centred in [lo, hi], making rejection rates low (< 5%).

### OLS regression summary (v2)

| Variable | r with CGPA | β (standardised) | Significance |
|---|---|---|---|
| Previous_GPA | 0.976 | dominates | p < 0.001 |
| Attendance_Rate | 0.280 | small positive | p < 0.001 (by construction) |
| Study_Hours_Per_Week | 0.238 | small positive | p < 0.001 (by construction) |
| Course_Load | −0.055 | near zero | p ≈ 0.0001 |
| **Adj R²** | | **0.954** | |

The high adj R² is driven by `Previous_GPA`, not by the behavioural variables. See Section 9.

---

## 8. Programme-to-Code Mapping

| Prog Code | Programme Name | Load Tier |
|---|---|---|
| CIS | Computer & Info Systems | SWE |
| MIS | Management Info Systems | SWE |
| CEN | Computer Engineering | CSC |
| ICE | Info & Comm Engineering | CYB |
| MAT | Mathematics | MATH |
| EEE, CVE, MCE, CHE, PET, etc. | Engineering/Sciences | DEFAULT |
| BCH, CHM, MCB, BLD, PHY* | Sciences/Other | DEFAULT |

---

## 9. Honest Assessment of Remaining Limitations

This section must be reproduced prominently in any paper or report using this dataset. Burying these points in a caveats section is insufficient — they concern the fundamental validity of regression results obtained from this data.

**L1 — Residual endogeneity (most critical)**

Attendance Rate and Study Hours are synthesized using CGPA as an input (at 40% weight). Running OLS regression of CGPA on these variables therefore recovers a relationship that is partly constructed. The significant p-values and positive coefficients do not constitute evidence that higher attendance or more study hours cause better academic outcomes. They demonstrate that the regression machinery functions correctly on a dataset where such a relationship was imposed. The paper must state this explicitly and prominently in the methodology section — not just in a limitations appendix.

**L2 — Arithmetic overlap in Previous_GPA**

`Previous_GPA = mean(CGPA100, CGPA200, CGPA300)` shares three of four components with the final CGPA. The adj R² of 0.954 is driven almost entirely by this definitional relationship. The behavioural variables contribute negligibly to explained variance once `Previous_GPA` is in the model. If the goal is to demonstrate the predictive contribution of attendance and study hours, the model should either exclude `Previous_GPA` and report the lower adj R² honestly, or reframe the dependent variable as CGPA400 (Level 400 performance) where `Previous_GPA` has no arithmetic overlap.

**L3 — Synthesized variables are not observed data**

Attendance Rate, Study Hours Per Week, and Course Load were never measured. They are statistically engineered. Any coefficient estimates for these variables carry no empirical weight and should not be used to make policy recommendations.

**L4 — Overload rate assumption**

The 5% overload probability and the 16–20 course overload ceiling are assumptions. The normal range (8–15 courses) is calibrated against a verified MTU transcript but a single transcript cannot speak for all programmes. If broader institutional data is available, `OVERLOAD_PROB`, `OVERLOAD_LO`, and `OVERLOAD_HI` in `enrich.py` should be updated accordingly.

**L5 — ID number collisions in source data**

70 student ID numbers appear on rows with conflicting attributes (different programme, gender, or graduation year). These are presumed to be data-entry errors. They do not affect the synthesized variables but should be noted if ID No is used as a join key.

**L6 — Generalizability**

The synthesis parameters were calibrated for a Nigerian university context. They are not directly transferable to other institutions.

---

## 10. Reproducibility

All synthesis is seeded (`--seed 42` by default). Re-running `enrich.py` with the same seed and input file will always produce an identical enriched CSV.

```bash
python enrich.py --input academic_performance_dataset_V2.csv \
                 --output academic_performance_enriched_seed7.csv \
                 --seed 7
python verify.py --input academic_performance_enriched_seed7.csv
```

---

## 11. Running the Scripts

### Requirements

```
Python >= 3.9
numpy
pandas
scipy
scikit-learn
```

```bash
pip install numpy pandas scipy scikit-learn
```

### Enrichment

```bash
python enrich.py
python enrich.py --input path/to/raw.csv --output path/to/enriched.csv --seed 42
```

### Verification

```bash
python verify.py
python verify.py --input path/to/enriched.csv --alpha 0.01
```

Exit codes: `0` = all checks passed, `1` = one or more failed.

---

## 12. Output Schema

| Column | Type | Source | Description |
|---|---|---|---|
| ID No | int | Original | Student identifier |
| Prog Code | str | Original | Programme code |
| Gender | str | Original | Male / Female |
| YoG | int | Original | Year of graduation |
| CGPA | float | Original | Cumulative GPA (overall) |
| Previous_GPA | float | **Derived** | mean(CGPA100, CGPA200, CGPA300) |
| CGPA100 | float | Original | Level 100 GPA |
| CGPA200 | float | Original | Level 200 GPA |
| CGPA300 | float | Original | Level 300 GPA |
| CGPA400 | float | Original | Level 400 GPA |
| SGPA | float | Original | Semester GPA |
| Attendance_Rate | float | **Synthesized** | Class participation (%) |
| Study_Hours_Per_Week | float | **Synthesized** | Weekly study hours |
| Course_Load | int | **Synthesized** | Registered credit units per semester |
