# context.md
# Current Session State — Academic Performance Modelling Project

Update this file at the start and end of every session.
It is the first thing an agent should read after AGENTS.md.

---

## Current State (as of Trajectory scorer integration, May 2026)

**Pipeline status:** STABLE — all 52 verify.py checks pass  
**Active branch of work:** Trajectory_Slope_Prior in OLS/scorer; thesis figure integration; documentation sync.  
**Enriched dataset:** `academic_performance_enriched.csv` — 3,046 rows × 19 columns (includes `Trajectory_Slope_Prior`, `Trajectory_Class_Prior`, four-level `Trajectory_Slope` / `Trajectory_Class` for EDA)  

### Dataset summary

| Variable | Type | Source | Key stat |
|---|---|---|---|
| CGPA | float | Original | mean≈3.56, std≈0.62 |
| Previous_GPA | float | Derived | mean(CGPA100,200,300); no overlap with CGPA400 |
| Trajectory_Slope_Prior | float | Derived | OLS slope CGPA100–300; used in OLS/scorer |
| Trajectory_Slope | float | Derived | Four-level slope incl. CGPA400; EDA only |
| Attendance_Rate | float | Synthesized | r(CGPA)≈0.28, range [40, 100] |
| Study_Hours_Per_Week | float | Synthesized | r(CGPA)≈0.24, range [2, 20] |
| Course_Load | int | Synthesized | r(CGPA)≈−0.03, range [8, 20] |
| CGPA100–400 | float | Original | per-level GPA, all in [0, 5] |

### OLS snapshot (CGPA400; 8 predictors incl. genotype block)

| Predictor | Coef (approx.) | p-value (OLS) | VIF (main effects) |
|---|---|---|---|
| Previous_GPA | 0.851 | < 0.001 | 1.22 |
| Trajectory_Slope_Prior | 0.448 | < 0.001 | 1.10 |
| Attendance_Rate | 0.005 | < 0.001 | 1.10 |
| Study_Hours_Per_Week | −0.0003 | 0.939 | 1.07 |
| Course_Load | −0.006 | 0.166 | 1.01 |
| Genotype / SS×Attendance | — | NS | SS block inflated (expected) |
| **Adj R²** | **0.682** | | |
| **OOF MAE / RMSE / R²** (5-fold) | **0.357 / 0.453 / 0.680** | | |

Note: Four-level `Trajectory_Slope` must not be used to predict CGPA400 (leakage). Scorer derives `Trajectory_Slope_Prior` from CGPA100–300 when levels are supplied.

---

## Open Questions (must not be resolved autonomously)

### OQ-1 — Reframe dependent variable as CGPA400?
**Status:** Resolved (senior review, May 2026)  
**Decision:** Primary DV is **CGPA400** across `prediction_common.py`, `predict.py`, `score.py`, `app.py`, `verify.py`, and `model.ipynb`. Overall CGPA metrics remain optional for comparison.  
**Observed Adj R²:** ~0.682 (honest range vs ~0.95 on CGPA). OOF MAE ≈ 0.357, RMSE ≈ 0.453, R² ≈ 0.680.  
**Trajectory in scorer:** `Trajectory_Slope_Prior` (CGPA100–300 only) in OLS design matrix — see memory.md §8.

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
- [x] verify.py 52-check suite passing (includes Trajectory_Slope_Prior)
- [x] README.md with full methodology and honest limitations (L1–L7)
- [x] AGENTS.md, memory.md, context.md, init.sh created
- [x] Prediction layer: `prediction_common.py`, `predict.py`, `score.py`, Streamlit Predict & CV tab with breakdown
- [x] CGPA400 primary DV; genotype pathway; SS×Attendance interaction
- [x] Trajectory_Slope_Prior in enrich.py, OLS, scorer (May 2026)
- [x] Canonical thesis in `docs/PROJECT-MAIN/` with Ch4 figures (PNG refresh pending for OLS tab)

## Next Steps

- [ ] Re-capture Streamlit PNGs (`fig-4-3`–`fig-4-5`, optional `fig-A-8`) after `streamlit run app.py`
- [ ] Compile thesis PDF locally (`docs/PROJECT-MAIN/README.md`)
- [ ] Resolve OQ-2 / OQ-3 / OQ-4 with supervisor (pooled models, overload rate, SGPA)
- [x] Out-of-sample CV (`predict.py`) and batch scoring (`score.py`)
- [x] OQ-1 resolved: CGPA400 default DV
