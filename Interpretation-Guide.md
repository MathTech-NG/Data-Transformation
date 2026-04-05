# How to Interpret the OLS Results and Trajectory Charts

**Project:** Regression and Time Series Modelling of Students' Performance Across Semester  
**Dataset:** `academic_performance_enriched.csv` — Mountain Top University, Jan 2026

This guide walks through every output in the OLS Results and Trajectory Charts tabs, using the
actual numbers from this model. No generic theory — everything is tied directly to what you see
on screen.

---

## Part 1 — The OLS Regression Results

### 1.1 What the three headline metrics mean

| Metric | This model's value | What it tells you |
|---|---|---|
| **Adj R²** | 0.9530 | 95.3% of the variation in CGPA is explained by the four predictors *together* |
| **F-statistic** | ~15,000 | The model as a whole is statistically significant (all coefficients are not simultaneously zero) |
| **N observations** | ~2,976 | The number of students used after dropping duplicate ID rows |

**Adjusted R²** is the one that matters most for model evaluation. It penalises you for adding
extra predictors that do not help — unlike plain R², it does not automatically rise just because
you added more variables. A value of 0.953 sounds excellent. However, see Section 1.4 for why
you should be cautious about celebrating it here.

---

### 1.2 How to read the Coefficient Table

Here is what each column means using this model's output as the example:

| Variable | Coefficient | Std Error | t-statistic | p-value | CI Lower | CI Upper |
|---|---|---|---|---|---|---|
| const | small negative | small | large \|t\| | < 0.001 | — | — |
| Previous_GPA | **≈ 0.95–1.00** | very small | very large | < 0.001 | — | — |
| Attendance_Rate | **≈ 0.003–0.005** | small | moderate | < 0.001 | — | — |
| Study_Hours_Per_Week | **≈ 0.004–0.006** | small | moderate | < 0.001 | — | — |
| Course_Load | **≈ −0.002–0.002** | very small | near zero | ≈ 0.016 | — | — |

#### Coefficient (the slope)

The coefficient tells you: *holding all other predictors constant, by how much does CGPA change
for a one-unit increase in this variable?*

- **Previous_GPA ≈ 0.97:** A student with a one-point higher three-year average (e.g. 3.5 vs
  2.5) is predicted to have a CGPA about 0.97 points higher. This dominates the model.
- **Attendance_Rate ≈ 0.004:** A student who attends 10% more classes is predicted to have a
  CGPA only 0.04 higher (10 × 0.004). This is a tiny effect in practical terms.
- **Study_Hours_Per_Week ≈ 0.005:** One extra hour of study per week predicts a 0.005 CGPA
  increase. Again, a very small practical effect.
- **Course_Load ≈ −0.002:** Taking one more course predicts a negligible CGPA change.

#### Standard Error

How uncertain we are about each coefficient estimate. A small standard error means the estimate
is precise. The very small SE on `Previous_GPA` reflects that it is measured exactly (it is a
computed mean, not a survey response).

#### t-statistic

`t = Coefficient / Standard Error`. The larger the absolute value, the stronger the evidence
that the true coefficient is not zero. A rule of thumb: |t| > 2 is significant at the 5% level
for large samples.

#### p-value

The probability of observing a t-statistic this large *if the true coefficient were zero*.

- **p < 0.001** (green in the app): Very strong evidence the predictor matters.
- **p < 0.05** (yellow): Moderate evidence.
- **p ≥ 0.05** (red): Insufficient evidence at conventional thresholds.

All four predictors are significant here. That means the regression can distinguish their effects
from zero — but it does **not** mean the effects are large or meaningful. Significance is about
precision, not importance.

#### 95% Confidence Interval

The range within which the true coefficient falls with 95% probability (under frequentist
assumptions). If the interval does not cross zero, the predictor is significant at α = 0.05.
For `Previous_GPA` the interval is narrow and far from zero. For `Course_Load` the interval
is very narrow and straddles near-zero — significant only because the sample is large.

---

### 1.3 How to read the Coefficient Bar Chart

The bars show the magnitude of each coefficient. Error bars show ± 1.96 standard errors
(approximately the 95% CI).

**What to look for:**

- **Bar length** = how much a one-unit change in that predictor moves predicted CGPA.
- **Error bar width** = uncertainty. A very long error bar means imprecise estimation.
- **Bar crossing zero** = predictor is not distinguishable from no effect.

In this chart you will see `Previous_GPA` has a bar roughly 100–200× longer than the
behavioural variables. That single visual tells you the whole story of the model: one variable
is doing nearly all the work.

---

### 1.4 The critical caveat — why you should not over-interpret this R²

`Previous_GPA = mean(CGPA100, CGPA200, CGPA300)`

The final `CGPA` is approximately `mean(CGPA100, CGPA200, CGPA300, CGPA400)`.

So `Previous_GPA` contains **three of the four components** that make up `CGPA`. When you
regress something on part of itself, you will always get a high R² — not because your model is
insightful, but because of algebra. This is **Limitation L2**.

A more honest test of whether attendance and study hours *explain* anything would be to either:

1. Remove `Previous_GPA` and run the model on the behavioural variables alone — the R² will
   drop to around 0.10–0.15, which is more truthful.
2. Reframe the dependent variable as `CGPA400` only (Level 400 performance), keeping
   `Previous_GPA = mean(CGPA100, CGPA200, CGPA300)`. Under this framing the two variables no
   longer share any components, and the R² will reflect genuine predictive content.

---

## Part 2 — Residual Diagnostics (the 2×2 plot)

Residuals are the differences between what the model predicted and what actually happened:
`residual = actual CGPA − predicted CGPA`. If the model is well-specified, residuals should
behave like random noise — no patterns, approximately normal, consistent variance.

---

### 2.1 Residuals vs Fitted (top-left)

**What you are looking at:** Each dot is one student. X-axis = predicted CGPA, Y-axis =
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

**What you are looking at:** Each dot is one student. X = predicted CGPA, Y = actual CGPA.
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
