# context.md
# Current Session State — Academic Performance Modelling Project

Update this file at the start and end of every session.
It is the first thing an agent should read after AGENTS.md.

---

## Current State (as of Session 3)

**Pipeline status:** STABLE — all 34 verify.py checks pass  
**Active branch of work:** Data engineering complete. Ready for modelling.  
**Enriched dataset:** `academic_performance_enriched.csv` — 3,046 rows × 14 columns  

### Dataset summary

| Variable | Type | Source | Key stat |
|---|---|---|---|
| CGPA | float | Original | mean=3.56, std=0.62, range [1.46, 4.93] |
| Previous_GPA | float | Derived | mean(CGPA100,200,300), r(CGPA)=0.976 |
| Attendance_Rate | float | Synthesized | r(CGPA)=0.28, range [40, 100] |
| Study_Hours_Per_Week | float | Synthesized | r(CGPA)=0.24, range [2, 20] |
| Course_Load | int | Synthesized | r(CGPA)=-0.03, range [8, 20] |
| CGPA100–400 | float | Original | per-level GPA, all in [0, 5] |
| SGPA | float | Original | semester GPA |

### OLS snapshot (CGPA ~ all 4 predictors)

| Predictor | r | p-value | VIF |
|---|---|---|---|
| Previous_GPA | 0.976 | < 0.001 | 1.13 |
| Attendance_Rate | 0.280 | < 0.001 | 1.08 |
| Study_Hours_Per_Week | 0.238 | < 0.001 | 1.06 |
| Course_Load | -0.032 | 0.016 | 1.00 |
| **Adj R²** | | **0.953** | |

Note: The high adj R² is driven by Previous_GPA (arithmetic overlap), not
the behavioural variables. This is Limitation L2 — see README.md §9.

---

## Open Questions (must not be resolved autonomously)

### OQ-1 — Reframe dependent variable as CGPA400?
**Status:** Open  
**Why it matters:** If the DV is CGPA400 and Previous_GPA = mean(CGPA100,200,300),
there is zero arithmetic overlap between predictor and target. The regression
would be scientifically cleaner. Adj R² would likely drop to ~0.65–0.70,
which is more honest about what the behavioural variables actually explain.  
**Blocking:** Requires Kelvin and supervisor to agree on research question reframe.  
**Agent instruction:** Do not implement. Surface this in any methodology discussion.

### OQ-2 — Pooled model vs. per-programme-tier models?
**Status:** Open  
**Why it matters:** CIS/MIS students carry ~13 courses/semester vs. MAT students
at ~10. Course load distributions differ meaningfully. A stratified model
might reveal programme-specific GPA dynamics.  
**Blocking:** Research scope decision for Kelvin.  
**Agent instruction:** Do not implement. Note when discussing modelling strategy.

### OQ-3 — Verify 5% overload rate against MTU registrar data?
**Status:** Open  
**Why it matters:** Current 5% rate and 16–20 course ceiling are assumptions.
If wrong, Course_Load tail distribution is miscalibrated.  
**Blocking:** Requires institutional data access.  
**Agent instruction:** Flag as L4 limitation. Do not change the parameter
without verified data.

### OQ-4 — Include SGPA as a predictor?
**Status:** Open  
**Why it matters:** SGPA (semester GPA) is available in the raw data and
is a genuine observed variable — no synthesis required. However, it may
be collinear with CGPA or introduce its own overlap issues.  
**Blocking:** Modelling design decision.  
**Agent instruction:** Do not add to model. Surface as an option when
discussing predictor selection.

---

## Completed This Project

- [x] Source dataset loaded and profiled (3,046 rows, 10 columns, no nulls)
- [x] Missing variables identified: Attendance_Rate, Study_Hours_Per_Week, Course_Load
- [x] v1 synthesis implemented (direct f(CGPA) model)
- [x] v1 scientific issues identified by third-party review (endogeneity, arithmetic overlap)
- [x] v2 synthesis implemented (blended model, lagged Previous_GPA)
- [x] Course_Load corrected from credit units to course count
- [x] Course count range calibrated against verified MTU transcript
- [x] verify.py 34-check suite passing
- [x] README.md with full methodology and honest limitations (L1–L6)
- [x] AGENTS.md, memory.md, context.md, init.sh created

## Next Steps

- [ ] Resolve OQ-1 (DV reframe) with Kelvin and supervisor
- [ ] Build OLS regression script on enriched dataset
- [ ] Build time-series modelling script (per-student CGPA trajectory)
- [ ] Generate descriptive statistics table for paper methodology section
- [ ] Write methodology section language citing L1–L6 limitations explicitly
