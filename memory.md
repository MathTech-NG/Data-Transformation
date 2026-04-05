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

**Cleanest unimplemented fix:** Reframe dependent variable as CGPA400.
Under this framing, Previous_GPA has zero arithmetic overlap.
**Status:** Open — awaiting human decision (see context.md).  
**Resolved by:** Third-party review, session 2. Implemented in session 2.

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

**Current check count:** 34  
**Exit code 0:** all checks passed  
**Exit code 1:** one or more failed  

Checks cover:
1. Schema & Integrity (3 checks)
2. Value Ranges (11 checks)
3. Distributional Plausibility (7 checks)
4. Correlation Structure (4 checks)
5. Regression Readiness (9 checks — adj R², significance, VIF)

**Rule:** Checks may only be added, never removed or weakened.  
**Last confirmed pass:** Session 3, all 34 checks pass.

---

## §6. Limitations on Record

These six limitations (L1–L6) must appear in any paper using this dataset:

- **L1** — Residual endogeneity in synthesized behavioural variables
- **L2** — Arithmetic overlap between Previous_GPA and CGPA
- **L3** — Synthesized variables are not observed data
- **L4** — Overload rate (5%) and ceiling (20 courses) are unverified assumptions
- **L5** — 70 ID collisions in source data
- **L6** — Parameters calibrated for Nigerian university context only
