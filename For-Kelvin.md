# What This Project Does — A Plain English Walkthrough for Kelvin

**Written for:** Ehime Kelvin Ehinomen  
**Purpose:** Explain what every part of this project is doing, how the data is handled,
and how it all connects to the mathematics in your paper — no coding experience needed.

---

## The Big Picture First

Your paper studies whether things like attendance, study hours, and past academic performance
can predict a student's final CGPA. That is the research question.

To answer it, you need data — specifically, you need a dataset with all five variables:

| Variable | Role | Status in your dataset |
|---|---|---|
| CGPA | What you are predicting | ✅ Already in the data |
| Previous GPA | Predictor | ✅ Calculated from data |
| Attendance Rate | Predictor | ❌ Missing — had to be generated |
| Study Hours Per Week | Predictor | ❌ Missing — had to be generated |
| Course Load | Predictor | ❌ Missing — had to be generated |

The institution gave you real GPA data for 3,046 students. But three of your four
independent variables were never recorded. So the first job was to fill that gap
in a mathematically honest way. That is what the code does.

Think of the whole project as a production line with four stations:

```
Raw data  →  [enrich.py]  →  [verify.py]  →  [model.ipynb]  →  Results
```

Each station does one job. Here is what each one does in plain English.

---

## Station 1 — enrich.py (Filling the Gaps)

**What it does in one sentence:** Takes your real student data and adds the three missing
columns using statistical methods grounded in the literature.

### How it handles Previous GPA

This one is straightforward — no generation needed. The data already has each student's
GPA for Level 100, Level 200, and Level 300. Previous GPA is simply their average across
those three years:

```
Previous_GPA = (CGPA100 + CGPA200 + CGPA300) / 3
```

This represents everything the student has achieved *before* their final year. It is a
genuinely prior quantity — it was all in place before CGPA400 was recorded.

### How it generates Attendance Rate and Study Hours

This is the part that needs the most explanation, because it is the most important
methodological decision in your paper.

**The wrong way (what was tried first and rejected):**

The first attempt was to generate attendance like this:

```
Attendance ≈ some function of the student's CGPA + random noise
```

The problem: if you *build* attendance using CGPA, and then run a regression of CGPA
on attendance, the regression just recovers the function you put in. You would not be
discovering a relationship — you would be confirming your own construction. It is circular.
Your supervisor would rightly reject that.

**The right way (what is actually used):**

Instead, each value is generated using a *blend*:

```
Attendance = 60% × (random draw from the general population)
           + 40% × (a value anchored to that student's CGPA)
           + a little extra noise
```

The 60% part is drawn from a distribution that has *nothing to do* with any individual
student's CGPA — it reflects population-level variation. The 40% part introduces a weak
link to CGPA, because the literature genuinely shows that attendance and performance are
correlated (students who attend more tend to do better). But 40% keeps that link weak,
not dominant.

**The result:** The correlation between the generated Attendance Rate and CGPA comes out
at about 0.28. The correlation between Study Hours and CGPA comes out at about 0.24.
These are weak-to-moderate correlations — consistent with what real studies have found.

In your paper's methodology section, the correct language for this is:
> *"The behavioural variables were synthesized using a blended generative model calibrated
> to produce empirically plausible correlations with CGPA (r ≈ 0.25–0.28), consistent with
> the literature on engagement and academic outcomes."*

### How it generates Course Load

Course load (the number of courses a student takes per semester) is set by the curriculum —
not by how smart the student is. A brilliant student and a struggling student in the same
programme take the same courses. So Course Load should have *no relationship* with GPA.

The code generates it using a Normal distribution centred on the typical course count for
each programme, with no GPA input at all. The result is that the correlation between
Course Load and CGPA is approximately −0.03 — essentially zero, as expected.

Different programmes are given different averages based on a verified transcript from MTU:

| Programme | Typical courses per semester |
|---|---|
| CIS, MIS | 13 |
| CEN | 12 |
| ICE | 11 |
| MAT | 10 |
| All others | 11 |

About 5% of students in each programme are assumed to be on approved overload (up to 20
courses), reflecting students who registered extra courses with permission.

### What comes out of enrich.py

A CSV file called `academic_performance_enriched.csv` with 3,046 rows and 14 columns —
your original 10 columns plus the 4 new ones. This is the dataset your entire analysis
runs on.

---

## Station 2 — verify.py (Checking the Work)

**What it does in one sentence:** Runs 34 automated checks on the enriched dataset to
confirm it is mathematically sensible before any analysis starts.

You would not run a regression on data you have not checked. `verify.py` is that check.
It tests things like:

- Are all CGPA values between 0 and 5? (They should be — that is the scale.)
- Is Attendance Rate between 40% and 100% for every student?
- Is the correlation between Attendance and CGPA in the expected range (0.15 to 0.45)?
- Is the correlation between Course Load and CGPA near zero (|r| < 0.15)?
- Does the OLS regression produce a sensible Adjusted R²?
- Are the VIF values below 10? (This checks for multicollinearity — the problem where
  predictors are so correlated with each other that you cannot separate their effects.)

If all 34 checks pass, the terminal prints:
```
✓  ALL 34 CHECKS PASSED — dataset is fit for modelling
```

If any check fails, modelling stops until the issue is fixed. Think of it as a quality
control gate.

---

## Station 3 — model.ipynb (The Analysis — This Is Your Paper)

**What it does in one sentence:** Runs the full statistical analysis and produces all the
charts and tables that go in your paper.

The notebook has five sections. Here is what each one does.

### Section 1 — Setup and Data Loading

Loads the enriched CSV into memory and drops the 70 rows where a student ID number
appears twice with conflicting information (different programme, different gender — these
are data-entry errors in the original institutional records, not real duplicate students).

After dropping those rows, approximately 2,976 students remain. Every analysis runs on
this cleaned dataset.

### Section 2 — Descriptive Statistics and EDA

EDA stands for Exploratory Data Analysis — looking at the data before modelling it.
This section produces:

- A histogram of CGPA (what does the distribution of student performance look like?)
- Boxplots by programme (does CIS perform differently from MAT on average?)
- Boxplots by gender
- A correlation heatmap (a colour-coded table showing how strongly every variable
  relates to every other variable)

The correlation heatmap will show one very obvious thing: `Previous_GPA` has a correlation
of 0.976 with `CGPA`. This is very high. The reason is explained in the next section.

### Section 3 — OLS Regression

This is the mathematical core of your paper. Here is what the code is doing, mapped to
the mathematics you already know.

**The model in mathematical notation:**

```
CGPA_i = β₀ + β₁(Previous_GPA_i) + β₂(Attendance_i) + β₃(StudyHours_i) + β₄(CourseLoad_i) + ε_i
```

where i indexes each student, β₀ is the intercept, β₁–β₄ are the slope coefficients,
and ε_i is the error term.

**What the code does:**

```python
X = sm.add_constant(df[predictors])   # builds the matrix of predictors, adds a column of 1s for β₀
y = df["CGPA"]                         # the dependent variable vector
model = sm.OLS(y, X).fit()             # solves for β using ordinary least squares
```

The `sm.OLS(y, X).fit()` call is doing exactly what you would do by hand for OLS:
it finds the values of β₀, β₁, β₂, β₃, β₄ that minimise the sum of squared residuals:

```
minimise  Σ (CGPA_i − CGPA_hat_i)²
```

The `.fit()` call computes the closed-form OLS solution:

```
β_hat = (X'X)⁻¹ X'y
```

`model.summary()` then gives you everything: coefficients, standard errors, t-statistics,
p-values, confidence intervals, R², Adjusted R², F-statistic.

**Why the Adjusted R² is 0.953 — and why you should be careful about it:**

The model explains 95.3% of the variance in CGPA. That seems remarkable. But look at
what `Previous_GPA` is:

```
Previous_GPA = (CGPA100 + CGPA200 + CGPA300) / 3
```

And what is CGPA approximately?

```
CGPA ≈ (CGPA100 + CGPA200 + CGPA300 + CGPA400) / 4
```

You are regressing something on three-quarters of itself. The high R² is partly algebraic
inevitability, not a discovery. If you dropped `Previous_GPA` from the model and ran OLS
with only the three synthesized variables, the Adjusted R² would fall to roughly 0.10–0.15.
That is the honest answer to how much attendance and study hours explain.

Your paper should state this explicitly. The regression is a correct and valid demonstration
of the OLS methodology. The high R² does not mean attendance is a strong predictor — it
means previous GPA is arithmetically related to final GPA.

**The residual diagnostic plots:**

After fitting, the code produces four diagnostic plots. These exist to verify that the
assumptions of OLS are not badly violated:

| Plot | What it checks | OLS assumption being tested |
|---|---|---|
| Residuals vs Fitted | No pattern in errors as fitted values change | Linearity + homoskedasticity |
| Q-Q Plot | Are the errors normally distributed? | Normality of ε |
| Residual histogram | Same as Q-Q, easier to explain | Normality of ε |
| Scale-Location | Does variance of errors change with fitted values? | Homoskedasticity |

For your dataset, all four plots will look well-behaved. The Q-Q plot in particular will
show residuals tracking the normal line closely. Again, this is partly because `Previous_GPA`
pulls the residuals toward normality by construction — so mention it in the paper without
overstating it as a sign of excellent model fit.

### Section 4 — Time Series: CGPA Trajectory Across Levels 100→400

The "time series" in your paper's title refers to this section. Each student has four
GPA measurements — one per academic year (Level 100, 200, 300, 400). Treating these as
an ordered sequence lets you ask: how does the typical student's GPA change as they
progress through university?

The code **melts** the four level columns into a long-format table:

**Before melting (wide format):**
```
Student | CGPA100 | CGPA200 | CGPA300 | CGPA400
------- | ------- | ------- | ------- | -------
A       |   3.2   |   3.5   |   3.4   |   3.8
```

**After melting (long format):**
```
Student | Level    | GPA
------- | -------- | ---
A       | Level 100 | 3.2
A       | Level 200 | 3.5
A       | Level 300 | 3.4
A       | Level 400 | 3.8
```

This long format is what the plotting functions need to draw trajectory lines.

The code then computes the *mean* GPA across all students at each level, grouped by
programme or graduation year, and draws a line chart. A rising line means that programme's
students improve on average across their four years. A falling line means the reverse.

**Important framing for your paper:**

These are not the same students tracked over time like a longitudinal study. Each data
point is a mean across all students who reached that level. The correct way to describe
it is:

> *"CGPA100–CGPA400 represent per-level academic performance for each student.
> The group trajectories show the mean GPA of students at each academic level,
> forming a within-student progression sequence that is analysed as an ordered
> time series of academic development."*

### Section 5 — Limitations

The notebook ends with six limitations written out in full as required disclosures.
These are not weaknesses to apologise for — they are signs of methodological honesty.
Every quantitative study that uses synthetic data must disclose this. Examiners look for it.

---

## How the Streamlit App Fits In (app.py)

The Streamlit app (`app.py`) is a presentation tool — it shows the same results
as the notebook in an interactive web interface. It does not run any new analysis.
You would use it to present your work to your supervisor or in a seminar setting,
where clicking through charts is cleaner than scrolling through a notebook.

It has three tabs:
- **Data Overview** — the descriptive statistics and charts from Section 2
- **OLS Results** — the regression output from Section 3
- **Trajectory Charts** — the progression charts from Section 4

The sidebar on the left always shows the six limitations, so they are visible no matter
which tab you are on.

---

## The Complete Flow, One More Time

```
academic_performance_dataset_V2.csv
         │
         │  (3,046 students, 10 columns, 3 variables missing)
         ▼
    enrich.py
         │
         │  Generates Attendance_Rate, Study_Hours_Per_Week, Course_Load
         │  Derives Previous_GPA = mean(CGPA100, CGPA200, CGPA300)
         │  Writes 14-column CSV with 3,046 rows
         ▼
academic_performance_enriched.csv
         │
         ▼
    verify.py
         │
         │  34 checks — all must pass before modelling
         ▼
    model.ipynb
         │
         ├── Section 2: Descriptive stats and EDA
         ├── Section 3: OLS regression (your main model)
         ├── Section 4: CGPA trajectory analysis (your time series)
         └── Section 5: Limitations L1–L6
         │
         ▼
    Results → Paper
```

---

## What You Tell Your Supervisor

If your supervisor asks how the missing variables were handled:

> *"The three missing variables were synthesized statistically. Attendance Rate and Study
> Hours Per Week were generated using a blended model — 60% drawn from a CGPA-independent
> population distribution and 40% from a component correlated with CGPA — to produce
> realistic but weak correlations with the dependent variable (r ≈ 0.25–0.28). Course Load
> was drawn from a programme-stratified truncated Normal distribution with no GPA input,
> producing near-zero correlation (r ≈ −0.03), consistent with its structural nature.
> All synthesis is seeded for reproducibility. The dataset was validated through 34
> automated statistical checks before modelling. The regression results are presented as
> an illustration of the OLS methodology applied to a hybrid dataset, not as empirical
> evidence of causal relationships."*

That is the honest, academically defensible answer. It covers the method, the calibration,
the validation, and the correct interpretation — all in one paragraph.

---

*Written for Ehime Kelvin Ehinomen — Mountain Top University, April 2026.*
