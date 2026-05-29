# memory.md
# Persistent Decisions — Academic Performance Modelling Project

This file records every significant decision made during the project,
the reasoning behind it, and who/what resolved it. It is append-only —
never delete or overwrite entries. Add new entries at the bottom of
each section.

---

## §1. Dataset and Source

**Decision:** Use `academic_performance_dataset_V2.csv` as the immutable source.  
**Reason:** 3,046 student records across 17 programme codes, years 2010–2014.
            Contains per-level CGPA (100–400) and overall CGPA. No nulls.  
**Resolved by:** Yaba (data engineer), session 1.

**Known issue:** 70 ID numbers appear on rows with conflicting attributes
(different Prog Code, Gender, or YoG). These are data-entry errors in the
institution's source records — not duplicate students. They are flagged as
WARN (not FAIL) in verify.py and do not block modelling.  
**Resolved by:** verify.py audit, session 1.

---

## §2. Endogeneity — Synthesis of Attendance and Study Hours

**Problem (v1):** Attendance_Rate and Study_Hours_Per_Week were generated
as `X = f(CGPA) + noise`. This made the regression partially tautological —
running OLS of CGPA on X partially recovers the generative function, not a
real relationship. v1 R² ≈ 0.75 was largely by construction.

**Fix (v2):** Blended generative model:
```
X = BLEND_ALPHA * base  +  (1 - BLEND_ALPHA) * correlated_component  +  noise
```
where `base ~ N(μ_pop, σ_pop²)` is drawn independently of CGPA.

**Parameter chosen:** `BLEND_ALPHA = 0.60`  
**Justification:** Tuned empirically to produce r(X, CGPA) ≈ 0.25–0.30,
consistent with published literature on engagement-GPA correlations.
At alpha=0.60: r(Attendance, CGPA) ≈ 0.28, r(StudyHours, CGPA) ≈ 0.24.

**What this does not fix:** Endogeneity is reduced, not eliminated. The
correlated component retains 40% weight. Regression results remain
illustrative, not empirical.  
**Resolved by:** Third-party review, session 2. Implemented in session 2.

---

## §3. Previous GPA — Arithmetic Overlap

**Problem (v1):** `Previous_GPA = CGPA100`.  
CGPA100 is one of four components of the final CGPA (≈ their mean).
This introduces arithmetic overlap, not just correlation. The regression
was partially explaining a definitional relationship.

**Fix (v2):** `Previous_GPA = mean(CGPA100, CGPA200, CGPA300)`  
This is the student's three-year lagged cumulative GPA — genuinely prior
to the Level 400 outcome period.

**Residual issue:** The three-year average still shares three of four
components with the final CGPA. r(Previous_GPA, CGPA) = 0.976. The adj R²
of 0.954 is dominated by this variable, not the behavioural predictors.

**Cleanest fix (implemented, senior review May 2026):** Reframe dependent variable as CGPA400.
Under this framing, Previous_GPA has zero arithmetic overlap with the DV.
**Status:** Implemented — default in `prediction_common.py`, `predict.py`, `score.py`, `app.py`, `verify.py`, `model.ipynb`.  
**Observed Adj R²:** ~0.65 (vs ~0.95 on overall CGPA).  
**Resolved by:** Senior review (`engineer_brief.md`), May 2026.

---

## §4. Course Load — Variable Definition Correction

**Problem (v1 and v2 initial):** Course_Load was measured in credit units
(range 15–26). The model specification requires number of courses, not units.

**Fix:** Course_Load redefined as number of courses per semester.

**Ground truth:** Verified against a Mountain Top University transcript
(BSc Mathematics, Yaba-Shiaka, S. W., Years 1–3):
- Y1 S1: 15 courses (11 credit-bearing)
- Y1 S2: 12 courses (9 credit-bearing)
- Y2 S1: 13 courses (9 credit-bearing)
- Y2 S2: 12 courses (9 credit-bearing)
- Y3 S1: 12 courses (8 credit-bearing)
- Y3 S2: 11 courses (8 credit-bearing)

**Decision:** Count all courses including zero-credit (ESM, ICT, PIF, SDN series).
Range: 8–15 normal, up to 20 for approved overload students.

**Programme parameters (current):**

| Code | Tier | μ | σ | [lo, hi] |
|---|---|---|---|---|
| CIS | SWE | 13.0 | 1.5 | [11, 15] |
| MIS | SWE | 13.0 | 1.5 | [11, 15] |
| CEN | CSC | 12.0 | 1.5 | [10, 14] |
| ICE | CYB | 11.0 | 1.5 | [9, 13] |
| MAT | MATH | 10.0 | 1.2 | [8, 12] |
| DEFAULT | — | 11.0 | 1.5 | [8, 14] |

Overload: 5% of students, 16–20 courses.

**Caveat:** Ground truth is one Mathematics transcript. Ranges for
CIS/MIS/CEN/ICE are extrapolated from programme-tier ranking logic.  
**Resolved by:** Yaba (transcript verification), session 3.

---

## §5. Verify.py Check Suite

**Current check count:** 52 (includes trajectory prior + four-level schema, SS attendance < AA, CGPA400 regression with Trajectory_Slope_Prior, VIF split for SS interaction block)  
**Exit code 0:** all checks passed  
**Exit code 1:** one or more failed  

Checks cover:
1. Schema & Integrity (3 checks)
2. Value Ranges (11 checks)
2b. Genotype (4 checks)
2c. Trajectory (5 checks — prior + four-level)
3. Distributional Plausibility (7 checks)
4. Correlation Structure (6 checks)
5. Regression Readiness (16 checks — CGPA400 adj R², Previous_GPA + Trajectory_Slope_Prior significance, VIF main effects + SS block)

**Rule:** Checks may only be added, never removed or weakened. VIF on SS×Attendance uses an expected-inflation pass, not a <10 threshold.  
**Last confirmed pass:** Trajectory scorer integration, all 52 checks pass.

---

## §8. Trajectory_Slope_Prior in OLS / Scorer

**Decision:** Add `Trajectory_Slope_Prior` (OLS slope through CGPA100–300 only) to the OLS design matrix and Streamlit/`score.py` scorer.  
**Reason:** Four-level `Trajectory_Slope` includes CGPA400 and must not predict CGPA400 (leakage). Prior-only slope captures momentum for forward prediction and aligns trajectory tab narrative with scoring.  
**Observed (seed 42, deduped n=2974):** Adj R² ≈ 0.682; β(Trajectory_Slope_Prior) ≈ 0.448; OOF MAE ≈ 0.357.  
**Scorer:** Derived from CGPA100–300 when provided; set to 0 with warning if only Previous_GPA supplied.  
**Resolved by:** Trajectory scorer integration plan, May 2026.

---

## §6. Limitations on Record

These six limitations (L1–L6) must appear in any paper using this dataset:

- **L1** — Residual endogeneity in synthesized behavioural variables
- **L2** — Arithmetic overlap between Previous_GPA and CGPA
- **L3** — Synthesized variables are not observed data
- **L4** — Overload rate (5%) and ceiling (20 courses) are unverified assumptions
- **L5** — 70 ID collisions in source data
- **L6** — Parameters calibrated for Nigerian university context only

---

## §7. Prediction layer (secondary goal)

**Decision:** Ship `prediction_common.py`, `predict.py` (k-fold OLS MAE / RMSE / out-of-sample R²
with optional `--also-cgpa400` exploratory block pending OQ-1), and `score.py` (batch CSV or JSON
scoring with the same four predictors as `app.py`). Streamlit tab **Predict & CV** reuses the
same helpers.

**Input contract for scoring:** Each row must include `Attendance_Rate`, `Study_Hours_Per_Week`,
`Course_Load`, and either `Previous_GPA` or all of `CGPA100`, `CGPA200`, `CGPA300` (previous GPA
derived as in `enrich.py`). Extra columns are ignored with a printed warning; `SGPA` is not used as
a predictor until OQ-4 is resolved.

**Reason:** Separates honest out-of-sample error from in-sample Adj R²; provides an operational
path for hypothetical or new rows without changing the synthesis or verify gates.

**Resolved by:** Implementation session (prediction layer), May 2026.

---

## §8. Genotype synthesis

**Decision:** Add `Genotype` column (AA, AS, SS) via independent multinomial draw in `enrich.py`.  
**Parameters:** `GENOTYPE_PROBS = (0.75, 0.24, 0.01)` — illustrative Nigerian-population priors; **not** from MTU medical records.  
**Modelling:** OLS uses reference category AA with dummies `Genotype_AS`, `Genotype_SS`.  
**Reason:** Align empirical pipeline with Chapters 1–2 health variable; draw is CGPA-independent (verify: $|r| < 0.15$).  
**Limitation:** L7 — synthesized genotype, same epistemic status as attendance/study hours.  
**Resolved by:** Genotype integration session, May 2026.
