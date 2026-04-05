Mathematical Modelling and Statistical Analysis of 
Students’ Academic Performance Using 
Multivariate Factors 
EHIME KELVIN 
Department of Mathematics, MOUNTAIN TOP UNIVERSITY 
April 2026 
Abstract 
Contents 
1 
2 
3 
4 
5 
Introduction 
Literature Review and Theoretical Framework 
Methodology 
Results and Analysis 
Conclusion and Recommendations 
1 Introduction 
• Background and motivation 
• Problem statement 
• Aim and objectives 
• Research questions 
• Scope of the study 
2 Literature Review and Theoretical Framework 
• Review of related studies 
• Factors affecting academic performance 
• Mathematical modelling concepts 
• Regression analysis overview 
• Identified research gaps 
3 Methodology 
• Research design 
• Data sources (real + simulated) 
• •  Real-life dataset (secondary data)  
• •  Simulated dataset (to complete missing variables) 
• Data preprocessing 
• Model formulation (regression model) 
• Multiple Regression Model formulation:  
• Model implementation  
• Model training (hybrid dataset) 
• Model validation (real-life dataset) 
Tools and Software 
• Python (Pandas, NumPy, Scikit-learn) 
4 Results and Analysis 
• Data presentation 
• Descriptive statistics 
• Trend and correlation analysis 
• Regression results 
• Model performance (training vs testing) 
5 Conclusion and Recommendations 
• Summary of findings 
• Conclusion 
• Recommendations 
• Future work 
References 
Appendices 
• Dataset 
• Code 
• Additional results 

A few things worth noting:

**Good — he's already done the hard framing work**

"Real + simulated" and "hybrid dataset" in Section 3 is exactly the right language. It's honest without being self-defeating. "Simulated dataset to complete missing variables" is defensible and accurate. That framing should survive supervisor scrutiny.

**One mismatch to flag**

Section 4 says "Model training vs testing" — that's ML language, not OLS language. OLS doesn't have a train/test split in the traditional sense. If Kelvin is doing pure OLS, he should either reframe this as "Model fitting and validation" (e.g. using an 80/20 split for out-of-sample prediction error, which is valid for OLS), or remove the training/testing distinction entirely and just report in-sample fit metrics. As written it implies a ML workflow the project isn't doing. An examiner who knows stats will catch that.

**Tools listed don't match what was built**

Section 3 lists Scikit-learn but the model is `statsmodels` OLS. Scikit-learn is fine to include if it's used anywhere (e.g. VIF calculation used sklearn internally in verify.py), but `statsmodels` should be listed explicitly — it's the primary modelling library and the one that produces the `model.summary()` output the paper will reference.

**Section 3 should mention the seed**

Since the simulated variables are reproducible via `--seed 42`, Kelvin should note the random seed in the methodology. That's what makes the synthesis replicable, which matters for academic integrity.

**Everything else looks right.** The structure is clean, the scope matches what was built, and separating real vs simulated data sources explicitly in the methodology is the correct move.