# Academic Performance Data Pipeline
### Regression & Time Series Modelling of Students' Performance Across Semesters

---

## What This Is

This project is trying to answer one question: **can we look at how a student performed in their early years and predict — or at least explain — where they end up by the time they graduate?**

To do that, we need our data to be in the right shape. Raw data straight from the source is rarely model-ready. This pipeline takes the raw academic performance dataset and runs it through three stages: check it, reshape it, then verify that the reshaping didn't break anything or quietly introduce nonsense.

The dataset has **3,046 students** across multiple departments and year groups (2010–2014), with CGPA recorded per year level (100L through 400L) plus a final SGPA.

---

## The Three Stages

---

### Stage 1 — Look at What You Have (Pre-Transformation Diagnostics)

Before touching anything, you want to understand the shape and character of the data. This isn't busywork — if you skip it and something goes wrong in the model later, you won't know if it was the data's fault or yours.

**What we check:**

**Descriptive statistics** — the basics. What's the average CGPA at each level? What's the lowest and highest? How spread out are the scores? This gives you a feel for whether the data makes sense at all. For example, if someone's CGPA100 was 6.0 on a 5-point scale, that's a red flag you'd catch here.

**Skewness and Kurtosis** — these two tell you about the *shape* of the score distribution. Skewness tells you whether scores are bunching up on the high or low end. Kurtosis tells you whether the distribution is flat and spread out or sharply peaked in the middle. Why does this matter? Because the regression model we're building has an assumption baked into it — that errors in prediction follow a roughly bell-shaped pattern. If our data is wildly lopsided, we need to know upfront so we can account for it.

Our data came back with mild negative skew across all year levels, meaning slightly more students score above average than below. Not dramatic. No corrections needed, but good to document.

**Normality test** — we ran a formal test (Shapiro-Wilk) to check whether each year's GPA distribution follows a normal bell curve. None of them passed strictly, which is honest — real-world GPA data rarely does. But the departures from normality were mild, and at 3,000+ students, the model is robust enough to handle it.

**Correlation matrix** — this shows how strongly each year's CGPA is related to the others. The short answer: strongly. CGPA200 and CGPA300 are almost interchangeable in terms of what they predict about final CGPA. This is useful information — it tells us which years carry the most weight in predicting the final outcome.

---

### Stage 2 — Reshape the Data (Transformation)

The raw dataset has CGPA as a snapshot per year level. The model needs more than snapshots — it needs structure that captures *movement over time*. Here's what we built:

**Lag Features**

This is the most important transformation. A lag feature is just: *what was this student's GPA the year before?*

- Lag1 = CGPA100 (what they scored entering 200L)
- Lag2 = CGPA200 (what they scored entering 300L)
- Lag3 = CGPA300 (what they scored entering 400L)

Why does this matter? Because "previous performance predicts future performance" is the core claim of the entire project. The lag feature *is* that claim, made concrete. Without it, you're not doing time series modelling — you're just doing regular regression on snapshots.

**Delta Features**

These capture the *change* between levels:

- Delta_100_200 = CGPA200 minus CGPA100 (did they improve or decline from 100L to 200L?)
- Delta_200_300 = same idea
- Delta_300_400 = same idea

A student who goes from 2.5 to 3.8 between 100L and 200L is telling you something different from a student who goes from 4.2 to 3.1. The delta captures that story.

**Z-score Normalization**

All the GPA columns — original, lag, and delta — get rescaled so they're centered at zero with a consistent spread. The actual GPA values become: *how far is this student from the average, measured in standard deviations?*

A score of +1.5 means "1.5 standard deviations above average." A score of -0.8 means "below average but not drastically so."

This matters because regression models work better when all your variables are on the same scale. Without normalization, a variable that ranges from 1.5 to 5.0 will unfairly dominate a variable that ranges from 0 to 1, even if they're equally important. Normalization levels the playing field.

**Gender Encoding**

Gender (Male/Female) gets converted to (1/0) so the model can use it numerically. Simple, necessary, standard.

**Programme Code and Year of Graduation** are left as text — they're useful for slicing and analyzing results by group, but they're not fed directly into the model.

---

### Stage 3 — Verify That Nothing Broke (Post-Transformation Checks)

Transformation always carries risk. You might accidentally introduce patterns that weren't in the original data, or create features that are so similar to each other that the model gets confused. Stage 3 is how you catch that.

**VIF (Variance Inflation Factor)**

This checks whether any of your features are *too similar* to each other. If two features carry almost identical information, the model can't tell them apart and starts behaving unpredictably.

Scale: 1 is perfect, anything under 5 is fine, above 10 is a problem.

Our results:
- CGPA100: 2.03 ✓
- CGPA200: 3.57 ✓
- CGPA300: 3.86 ✓
- CGPA400: 3.18 ✓

All under 5. The features are related (as expected — they're all GPA) but not so related that they cancel each other out. We're good.

**Mutual Information (Feature Relevance)**

This measures how much each feature actually *knows* about the thing we're trying to predict (final CGPA). Think of it as asking: if I told you only this one number, how much would it tell you about where this student ended up?

Scale: 0 = tells you nothing, 1+ = highly informative.

Our results:
- CGPA100 / Lag1: ~0.53 (moderate — early performance has some signal but not everything)
- CGPA200 / Lag2: ~0.90 (high — second year performance is a strong predictor)
- CGPA300 / Lag3: ~0.90 (high — same)
- CGPA400: ~0.79 (high)

This is a meaningful finding: **200L and 300L performance are the strongest predictors of final CGPA**. 100L matters but less so — students often arrive unsettled and find their footing later. This is worth mentioning in the project discussion.

**Eigenvalue Analysis (Stability Check)**

This is a structural check on the relationship between all features taken together. Without getting into the linear algebra, we're checking: is there any direction in the data where the features completely collapse into each other?

The condition number came out at **18.62**. Think of it as a stability score — under 30 is stable, 30–100 is borderline, above 100 means the model will be fragile and easily broken by small changes in the data.

18.62 is comfortable. The feature set is structurally sound.

**Baseline R²**

Before building the full model, we ran a simple linear regression with all the transformed features and asked: *what fraction of the variation in final CGPA can these features explain?*

Result: **R² = 0.9817 (98.2%)**

This means the features we've built — lags, deltas, normalized GPAs — together explain 98% of the variation in a student's final CGPA. That's an unusually strong signal, and it makes intuitive sense: GPA at each year is directly cumulative.

This doesn't mean the final model will be this clean (the baseline uses all features together, including things we may trim), but it confirms the data has more than enough information to support the modelling exercise.

**GPA Variance Across Year Levels**

For time series work, you want to check whether the *spread* of scores is roughly stable across time or changing significantly. If variance doubles or triples from year to year, it suggests the series is not stationary — meaning the model needs extra steps to account for it.

Our variance:
- CGPA100: 0.46
- CGPA200: 0.61
- CGPA300: 0.74
- CGPA400: 0.64

The spread increases a bit from 100L to 300L (students diverge more as the years go on — some pull ahead, some fall behind) then tightens back at 400L. Nothing dramatic. The series is reasonably stable.

---

## Summary of Results

| Check | Result | What It Means |
|---|---|---|
| Normality | Not strictly normal | Mild departure, acceptable at this sample size |
| Skewness | Slight negative | More above-average than below — realistic |
| VIF | All under 4 | Features are independent enough — no confusion |
| Mutual Information | 0.53–0.90 | All features relevant; 200L and 300L strongest |
| Condition Number | 18.62 | Structurally stable |
| Baseline R² | 98.2% | Strong signal — data fully supports modelling |
| Variance stability | Gradual increase then plateau | Series is sufficiently stationary |

---

## Files

| File | Description |
|---|---|
| `academic_performance_dataset_V2.csv` | Original raw dataset |
| `pipeline.py` | Full transformation and verification script |
| `academic_performance_transformed.csv` | Output — model-ready dataset |
| `README.md` | This file |

---

## How to Run

```bash
# Make sure these are installed
pip install pandas numpy scipy scikit-learn

# Place the raw CSV in the same folder as pipeline.py, then:
python3 pipeline.py
```

The script prints all diagnostics to the terminal and saves the transformed dataset automatically.

---

## Project Context

This pipeline was built to support an undergraduate mathematics project on regression and time series modelling of student academic performance. The transformation decisions — particularly the lag feature construction and normalization — are mathematically grounded in standard time series preparation practices and standard regression assumptions. Every choice made in Stage 2 is verifiable through the checks in Stage 3.
