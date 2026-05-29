# Talking Points — OLS Results & Trajectory Charts

**Project:** Regression and Time Series Modelling of Students' Performance Across Semester  
**Use:** Verbal presentation, viva defence, or supervisor discussion

---

## Opening — Setting Context

- "This analysis uses data from 3,046 students across 17 programmes at Mountain Top University,
  graduating between 2010 and 2014."
- "The **primary dependent variable is CGPA400** — final-year Level 400 GPA — not overall CGPA."
- "The regression uses **Previous_GPA** (mean of Years 1–3) and **Trajectory_Slope_Prior**
  (trend through CGPA100–300 only). The four-level trajectory slope in the charts **includes
  CGPA400** and is for description only — we do not use it to predict CGPA400."
- "Attendance, study hours, course load, and genotype were synthesized where absent from the
  export. I flag that wherever it affects interpretation (L1, L3, L7)."

---

## On the Headline R² (~0.68 on CGPA400)

- "The model explains about **68%** of variation in **CGPA400**. That is lower than legacy models
  on overall CGPA (~95%), and that is intentional."
- "Overall CGPA shares three of four level components with Previous_GPA — high R² there is partly
  algebra (L2). CGPA400 has **no overlap** with Previous_GPA."
- "Adding **Trajectory_Slope_Prior** connects the regression scorer to the time-series story:
  improving students get higher predicted CGPA400 than flat students at the same three-year mean."
- "Five-fold cross-validation: MAE ≈ 0.36, OOF R² ≈ 0.68 — illustrative under synthesis
  assumptions, not registrar-certified forecasting."

---

## On the Coefficient Table

- "**Previous_GPA ≈ 0.85** — level of performance over the first three years."
- "**Trajectory_Slope_Prior ≈ 0.45** — one GPA point per level steeper trend → about 0.45 higher
  CGPA400, holding other terms fixed."
- "**Attendance ≈ 0.005** — significant but small; partly synthetic (L1). I do not claim attendance
  *causes* Level 400 outcomes."
- "**Study hours** — not significant in the full model once trajectory is included."
- "Genotype terms — not significant; SS subsample is small (n ≈ 37)."

---

## On the Predict & CV Scorer

- "When you enter CGPA100, 200, and 300, the app derives both the mean and the **prior slope**."
- "The **prediction breakdown** shows coefficient × value per term — transparent, not a black box."
- "If you only enter Previous_GPA without levels, slope is zero — trend is not applied."
- "Example: a student improving 3.4 → 4.4 → 4.5 gets a **higher** prediction than the old
  mean-only model; that matches the narrative."

---

## On the Residual Diagnostics

- "Residuals vs fitted: roughly centred on zero — linear structure is acceptable for illustration."
- "Q-Q plot: approximate normality; synthesis and strong prior-GPA signal still shape residuals (L1)."
- "Do not over-claim 'perfect' diagnostics — they reflect the data-generating process as well as fit."

---

## On the Actual vs Predicted Chart

- "Moderate scatter around the identity line — honest for CGPA400 (Adj R² ≈ 0.68)."
- "Programme colours show heterogeneity, not separate programme models."

---

## On the Synthesized Variables (the honest caveat)

- "Forty percent of attendance and study hours were CGPA-linked in synthesis (L1)."
- "Significant attendance in OLS does not prove policy interventions would work."
- "Results are methodology demonstration under documented assumptions."

---

## On the Trajectory Charts

- "Based on **observed** CGPA100–400 — no synthesis in the level GPAs."
- "Lines are **cohort means at each level**, not the same students tracked calendar-year by calendar-year."
- "**Trajectory_Slope** (four levels) classifies Improving / Stable / Declining for EDA."
- "**Trajectory_Slope_Prior** (three levels) is what enters the scorer."

---

## On Specific Trajectory Patterns

- "Level 200 dip then recovery appears in several programmes — common 'middle-year' pattern."
- "Programme heatmap sorted by Level 400 performance — top rows finish strongest."
- "Year-of-graduation grouping can show cohort effects; interpret cautiously."

---

## Closing — What the Analysis Contributes

- "Regression + **prior trajectory in the scorer** demonstrates a unified story: level, trend, and
  illustrative behavioural covariates predicting CGPA400."
- "Time-series components (AR(1), trajectory classification, heatmaps) describe observed dynamics."
- "Limitations L1–L7 are documented — synthesized inputs, no causal claims, illustrative scope."

---

*Talking points for: Regression and Time Series Modelling of Students' Performance Across
Semester — Ehime Kelvin Ehinomen, Mountain Top University, Jan 2026.*
