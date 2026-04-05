# Talking Points — OLS Results & Trajectory Charts

**Project:** Regression and Time Series Modelling of Students' Performance Across Semester  
**Use:** Verbal presentation, viva defence, or supervisor discussion

---

## Opening — Setting Context

- "This analysis uses data from 3,046 students across 17 programmes at Mountain Top University,
  graduating between 2010 and 2014."
- "Three of the four independent variables — Attendance Rate, Study Hours, and Course Load —
  were not collected in the original records, so they were synthesized statistically. I will flag
  where that affects interpretation."
- "The dependent variable throughout is CGPA — the student's overall cumulative GPA on a 5-point
  scale."

---

## On the Headline R² (0.953)

- "The model explains 95.3% of the variation in CGPA. That sounds very strong — but there is a
  structural reason for it that I need to be upfront about."
- "The strongest predictor, Previous GPA, is defined as the mean of the student's Level 100,
  200, and 300 GPAs. The final CGPA is itself approximately the mean of all four levels. So
  Previous GPA shares three of the four components that make up CGPA — they are arithmetically
  related by construction."
- "What this means in practice: the high R² is partly telling us that the average of three
  numbers predicts the average of four numbers well. That is algebra, not a finding."
- "The honest interpretation is that the model demonstrates the regression methodology works
  correctly. The R² should not be cited as evidence of strong predictive performance without
  this caveat."

---

## On the Coefficient Table

- "Previous GPA has a coefficient of approximately 0.97. That means a student with a one-point
  higher three-year average is predicted to finish 0.97 GPA points higher overall. It dominates
  every other variable in the model."
- "Attendance Rate and Study Hours both have positive coefficients and are statistically
  significant. However, the practical effect sizes are small. A student who attends 10% more
  classes is predicted to gain about 0.04 CGPA points — that is four hundredths of a point."
- "All four predictors are significant at p < 0.05. But I want to be careful here — statistical
  significance with nearly 3,000 observations is easy to achieve. It tells us the effect is
  distinguishable from zero, not that it is meaningful."
- "Course Load has a near-zero coefficient and is significant only marginally. This is by design
  — course load is set by the curriculum, not by student ability, so we expected it to have no
  real relationship with GPA."

---

## On the Residual Diagnostics

- "The residuals vs fitted plot shows a horizontal cloud around zero — no obvious fan shape or
  curve. This means the linear model structure is reasonable and variance is roughly constant
  across the range of predictions."
- "The Q-Q plot is very clean — the residuals track the theoretical normal line closely. This
  satisfies the normality assumption for OLS inference."
- "However — and this is important — the clean diagnostics are partly a consequence of the same
  arithmetic overlap. Because Previous GPA is a computed mean of normally distributed scores,
  it pulls the residuals toward normality by construction. I would not present the Q-Q plot as
  independent evidence of good model fit."
- "The scale-location plot shows no strong upward trend, which means we do not have a serious
  heteroskedasticity problem."

---

## On the Actual vs Predicted Chart

- "You can see the predictions sit very tightly around the 45-degree identity line — almost no
  scatter. Again, this is the arithmetic overlap at work, not a remarkable finding."
- "What is more interesting is whether any programme systematically sits above or below the
  line. A programme consistently above the line means the model is under-predicting its
  students — they perform better than their previous GPA would suggest."

---

## On the Synthesized Variables (the honest caveat)

- "I need to be transparent: Attendance Rate and Study Hours were synthesized using the students'
  own CGPA as one input. Forty percent of each value was derived from CGPA."
- "When you then regress CGPA back on those variables, you recover a relationship that was
  partially built in. The significant p-values and positive coefficients confirm that the
  regression machinery works — they do not constitute evidence that attendance or study hours
  actually cause better outcomes."
- "This is a methodological illustration. The results should not be used to recommend policy —
  for example, mandating higher attendance — because the causal direction is not established."

---

## On the Trajectory Charts

- "Unlike the regression results, the trajectory charts are based entirely on observed data —
  CGPA100 through CGPA400 were recorded directly. There is no synthesis here."
- "Each line represents the mean GPA of students in a programme across their four years of
  study. A rising line means students in that programme tend to improve as they progress. A
  falling line means the opposite."
- "An important caveat on how to read these: these are not the same students tracked over time.
  Each data point is the mean GPA of all students at that level. Think of it as the typical
  academic profile of a student in that programme — not a longitudinal panel."

---

## On Specific Trajectory Patterns

- "Programmes where the trajectory rises sharply from Level 300 to Level 400 are worth noting.
  This often reflects strong final-year project or dissertation performance pulling the average
  up."
- "Programmes with a dip at Level 300 are common in engineering and science curricula — the
  third year is typically the hardest, with the most theoretical content."
- "The gap between the highest and lowest performing programmes at Level 100 tells you about
  differential entry performance. If that gap narrows by Level 400, it suggests that initial
  differences diminish over time."
- "The Year of Graduation grouping is useful for spotting cohort effects — for example, if one
  graduating year performs consistently higher across all levels, that could indicate a change
  in grading policy or curriculum in that period."

---

## On the Heatmap

- "The heatmap gives you a compact summary across all programmes and all years at once. Darker
  cells mean higher mean GPA."
- "Reading across a row shows you one programme's trajectory. Reading down a column lets you
  rank programmes at the same academic level."
- "The rows are sorted by Level 400 performance — so the top row is the programme whose
  students finish with the highest mean GPA, regardless of where they started."
- "A programme that starts light-coloured at Level 100 but finishes dark at Level 400 is a
  late-bloomer programme — students who struggle early but finish strong. The reverse pattern
  is also possible and worth flagging."

---

## Closing — What the Analysis Contributes

- "The regression demonstrates that a five-variable OLS model can be specified and fitted on
  this dataset, and that the mechanics of regression — coefficient estimation, significance
  testing, residual diagnostics — all function as expected."
- "The trajectory analysis is the more empirically grounded contribution. It reveals genuine
  patterns in how academic performance evolves across programmes and cohorts, using only
  observed data."
- "Together, the two analyses provide a complete methodological demonstration of regression and
  time-series techniques applied to academic records — which is the stated aim of the project."
- "The limitations — particularly the synthesized variables and the arithmetic overlap — are
  not failures of the analysis. They are documented, understood, and disclosed. Any real-world
  application of this methodology would replace the synthesized variables with collected data."

---

*Talking points for: Regression and Time Series Modelling of Students' Performance Across
Semester — Ehime Kelvin Ehinomen, Mountain Top University, Jan 2026.*
