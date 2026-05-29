# How to Interpret the OLS Results and Trajectory Charts

**Project:** Regression and Time Series Modelling of Students' Performance Across Semester  
**Dataset:** `academic_performance_enriched.csv` — Mountain Top University, Jan 2026

This guide walks through OLS Results, Predict & CV (scorer), and Trajectory Charts tabs, tied to
the **CGPA400 + Trajectory_Slope_Prior** model (Adj R² ≈ 0.68). Re-capture Streamlit screenshots
after refit if the UI still shows the old coefficient table.

---

## Part 1 — The OLS Regression Results (CGPA400)

**Primary dependent variable:** `CGPA400` (Level 400 GPA), not overall CGPA.  
**Primary predictors for academic history:** `Previous_GPA` (mean of CGPA100–300) and `Trajectory_Slope_Prior` (OLS slope through CGPA100–300 only). The four-level `Trajectory_Slope` (includes CGPA400) appears only in the Trajectory tab — never in the scorer.

### 1.1 What the three headline metrics mean

| Metric | This model's value | What it tells you |
|---|---|---|
| **Adj R²** | ~0.682 | About 68% of variation in **CGPA400** is explained by the predictors together (honest range; no CGPA400 overlap with Previous_GPA) |
| **F-statistic** | large | The model as a whole is statistically significant |
| **N observations** | 2,974 | Students after dropping duplicate ID rows |

Legacy models using overall **CGPA** as the DV can show Adj R² ≈ 0.95 — that is mostly arithmetic overlap (L2), not a better model.

---

### 1.2 How to read the Coefficient Table

Approximate values from the current refit (seed 42, deduped sample):

| Variable | Coefficient | p-value | Practical reading |
|---|---|---|---|
| Previous_GPA | **≈ 0.85** | < 0.001 | Higher three-year mean → higher CGPA400 |
| Trajectory_Slope_Prior | **≈ 0.45** | < 0.001 | Steeper improving trend (GPA/level) → higher CGPA400 |
| Attendance_Rate | **≈ 0.005** | < 0.001 | +1% attendance → +0.005 CGPA400 (small; partly synthetic — L1, L3) |
| Study_Hours_Per_Week | **≈ 0** | NS | Not significant once trajectory is included |
| Course_Load / Genotype block | near zero | mostly NS | Illustrative only (L3, L7) |

#### Coefficient (the slope)

- **Previous_GPA ≈ 0.85:** Level of performance over Years 1–3.
- **Trajectory_Slope_Prior ≈ 0.45:** *Direction* of performance (improving vs declining) before Level 400. An improving student (e.g. 3.4 → 4.4 → 4.5) gets a higher score than a flat student at the same mean.
- **Attendance_Rate ≈ 0.005:** Statistically significant but small; do not treat as proof that attendance *causes* outcomes (L1).

#### p-value

Green (p < 0.05) means distinguishable from zero in this sample — not necessarily large or causal.

---

### 1.3 How to read the Coefficient Bar Chart

`Previous_GPA` and `Trajectory_Slope_Prior` have the longest bars. Behavioural variables are much smaller. Genotype SS terms may show very wide error bars (small n_SS).

---

### 1.4 Why this R² is more honest than the old CGPA model

**Primary DV = CGPA400.** `Previous_GPA` uses only CGPA100–300 — **no shared components** with the outcome (L2 resolved for this spec).

**Trajectory_Slope_Prior** adds momentum without leaking CGPA400 into the predictor (unlike four-level `Trajectory_Slope`).

Adj R² ≈ 0.68 is lower than legacy CGPA models (~0.95) but reflects a genuine prediction problem: *given Years 1–3 and profile, what is Level 400 GPA?*

---

## Part 1b — Predict & CV tab (scorer)

When you enter **CGPA100 / 200 / 300**, the app computes:

- `Previous_GPA` = mean of the three levels  
- `Trajectory_Slope_Prior` = OLS slope through those three points  
- `Trajectory_Class_Prior` = Improving / Stable / Declining (±0.05 GPA per level)

The **prediction breakdown** table shows each term as coefficient × value. Sum ≈ predicted CGPA400.

**If you enter only Previous_GPA** (no levels), prior slope is set to **0** and a message warns that trend is not applied.

**Do not** crank attendance to 100% and interpret a high score as “effort replaces studying” — attendance is partly synthesized and secondary to prior GPA + trajectory (L1, L3).

---

## Part 2 — Residual Diagnostics (the 2×2 plot)

Residuals are the differences between what the model predicted and what actually happened:
`residual = actual CGPA400 − predicted CGPA400`. If the model is well-specified, residuals should
behave like random noise — no patterns, approximately normal, consistent variance.

---

### 2.1 Residuals vs Fitted (top-left)

**What you are looking at:** Each dot is one student. X-axis = predicted CGPA400, Y-axis =
prediction error.

**What good looks like:** A horizontal cloud of points scattered evenly around the y = 0 line,
with no obvious pattern.

**What to watch for:**

| Pattern | What it means |
|---|---|
| Fan shape (wider spread at high or low fitted values) | Heteroskedasticity — variance is not constant. OLS standard errors are unreliable |
| Curved band (residuals systematically positive then negative) | Non-linearity — a linear model is missing something |
| Horizontal cloud around zero | Good — no obvious problems |

In this model you will see a relatively clean cloud. This is partly because `Previous_GPA`
dominates and is itself nearly linear with `CGPA`. Slight compression at the tails is expected
due to the GPA scale being bounded at 0 and 5.

---

### 2.2 Q-Q Plot of Residuals (top-right)

**What you are looking at:** Each dot is one residual. X-axis = what the quantile *would be* if
residuals were perfectly Normal. Y-axis = the actual quantile from the data.

**What good looks like:** Points lying on or very close to the red diagonal line.

**What to watch for:**

| Pattern | What it means |
|---|---|
| Points on the diagonal | Residuals are approximately Normal — good |
| S-curve | Residuals are skewed |
| Heavy tails (points curve away from the line at both ends) | Outliers / fat tails in the residual distribution |

This model's Q-Q plot will look very well-behaved — nearly a straight line. However, **this is
partly an artefact of L2**, not a sign of excellent specification. Because `Previous_GPA` is a
computed mean of normally-distributed GPA scores, it pulls the residuals toward normality by
construction. Do not present the Q-Q plot as evidence of model quality without this caveat.

---

### 2.3 Residual Distribution (bottom-left)

**What you are looking at:** A histogram of all residuals with a normal curve overlaid in red.

**What good looks like:** The histogram bars closely follow the red curve — a bell shape
centred at zero.

This is the same information as the Q-Q plot, just easier to explain to a non-technical audience.
In your paper you can use this chart alongside the Q-Q plot to show normality of residuals.

---

### 2.4 Scale-Location Plot (bottom-right)

**What you are looking at:** X-axis = fitted values. Y-axis = square root of the absolute value
of standardised residuals. This is used to check **homoskedasticity** (equal variance of errors).

**What good looks like:** Points scattered randomly in a horizontal band — no upward or downward
trend.

**What to watch for:** A rising trend (funnel opening to the right) indicates that larger
predicted values have larger errors — heteroskedasticity.

---

### 2.5 Actual vs Predicted Scatter

**What you are looking at:** Each dot is one student. X = predicted CGPA400, Y = actual CGPA400.
The dashed line is the 45° identity line (perfect prediction would put every dot on it).

**What good looks like:** Points clustered tightly around the identity line.

You will see an extremely tight cluster here — almost no spread. This confirms the high R² but,
again, it is mainly because `Previous_GPA` arithmetically overlaps with `CGPA`. Colour-coding
by programme shows whether any programme systematically sits above or below the line — a
programme-level prediction bias.

---

## Part 3 — Trajectory Charts

The trajectory charts treat `CGPA100`, `CGPA200`, `CGPA300`, `CGPA400` as an ordered time series
of a student's academic progression across their four years of study.

> **Important framing:** Each student contributes one observation per level. The group lines you
> see are **means across all students at each level** — not the same students tracked over time.
> They are best read as "the typical GPA profile of a student in this programme/cohort."

---

### 3.1 Mean Trajectory Line Chart

**What you are looking at:** One line per programme (or year of graduation). X-axis = academic
level (100 → 400). Y-axis = mean GPA at that level.

**How to read it:**

| Line pattern | Interpretation |
|---|---|
| **Rising line** | Students in this programme tend to improve as they progress |
| **Falling line** | Students tend to decline — often indicates Level 300/400 courses are harder |
| **U-shape** | Recovery pattern — initial struggle, then improvement by final year |
| **Flat line** | Consistent performance across all years |
| **Large gap between programmes** | Structural curriculum differences, not just effort |

Look for:
- Which programmes show the steepest improvement into Level 400? These students may respond
  well to the higher-level courses or have strong final-year project performance.
- Which programmes show a dip at Level 300? This is a common pattern where third-year
  coursework peaks in difficulty.
- Does the gap between programmes widen or narrow over time? Widening suggests early differences
  compound; narrowing suggests convergence.

---

### 3.2 Trajectory by Year of Graduation (2010–2014)

When grouped by `YoG` instead of programme, you are comparing different graduating cohorts.

**What to look for:**

- Do some cohorts have systematically higher trajectories across all four levels? This could
  indicate year-on-year changes in admission standards, curriculum changes, or grading policy
  shifts.
- If one cohort is consistently 0.2–0.3 GPA points above the others at every level, that is
  likely an institutional effect (e.g. grade inflation or a different assessment year), not a
  student quality effect.
- If trajectories cross (one cohort leads at Level 100 but is overtaken by Level 400), that
  suggests different rates of development across cohorts.

---

### 3.3 GPA Distribution Boxplot Across Levels

**What you are looking at:** A boxplot for each academic level. The box = interquartile range
(middle 50% of students). The line inside = median. Whiskers = range excluding outliers.
Dots = outliers.

**How to read it:**

| Feature | Interpretation |
|---|---|
| **Box height** | Spread of performance. Taller box = more unequal outcomes |
| **Median position** | Typical student performance at this level |
| **Many outlier dots** | Some students performing very unusually (extremely high or low) |
| **Box shifting up from L100 to L400** | Performance tends to improve across the cohort |
| **Box shifting down** | Performance tends to decline |
| **Narrowing box at higher levels** | Convergence — weaker students either improve or leave |

In this dataset, look for whether the spread (box height) narrows from Level 100 to Level 400.
This would suggest that students who struggle early either catch up or are no longer represented
in the later levels.

---

### 3.4 Programme × Level Heatmap

**What you are looking at:** A grid where each row is a programme and each column is an academic
level. The colour represents mean GPA — darker/more saturated = higher GPA.

**How to read it:**

- **Read across a row** to see one programme's trajectory: does it get darker (improving) or
  lighter (declining) from Level 100 to 400?
- **Read down a column** to compare programmes at the same academic year: which programmes have
  the highest mean GPA at Level 200? At Level 400?
- **Look for the bottom-right corner** (Level 400, lowest-ranked programme): this is where
  the most academically struggling students finish their degree.
- **Look for the top-left corner** (Level 100, highest-ranked programme): this is the strongest
  entry performance.

The heatmap is sorted by Level 400 performance (final-year GPA), so the top rows are the
programmes whose students finish strongest, regardless of where they started.

---

## Part 4 — Putting It All Together

### The story this model tells

1. **Previous academic performance (Previous_GPA) is overwhelmingly the strongest predictor of
   final CGPA.** A student's first three years predict their overall outcome with r = 0.976.
   This is partly definitional (L2) but also reflects genuine academic momentum — past
   performance is the best predictor of future performance.

2. **Attendance and study hours have statistically significant but small effects.** They are
   significant largely because the sample is large (n ≈ 2,976) — with enough data, tiny effects
   become detectable. Their practical contribution to CGPA, once Previous_GPA is already in the
   model, is negligible.

3. **Course load is structurally determined, not performance-driven.** Its near-zero correlation
   with CGPA (r ≈ −0.03) and small coefficient confirm that it adds no meaningful predictive
   power — consistent with the design intent.

4. **The trajectory charts reveal the richest story.** Unlike the OLS results (which are partly
   artefact), the Level 100→400 trajectories show genuine patterns in how academic performance
   evolves across programmes and cohorts. These are directly observable data, not synthesized.

### What to highlight in the paper

- Lead with the trajectory charts as the empirical finding — they are based on observed data.
- Present the OLS results as a demonstration of the modelling methodology, not as empirical
  evidence of causal relationships.
- Quote L1 and L2 explicitly in the methodology section, not just in a limitations appendix.
- If R² is cited, always include the caveat that it is dominated by arithmetic overlap.

---

*Guide prepared for: Regression and Time Series Modelling of Students' Performance Across
Semester — Ehime Kelvin Ehinomen, Mountain Top University, Jan 2026.*
