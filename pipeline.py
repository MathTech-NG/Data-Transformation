"""
Student Academic Performance — Data Transformation & Verification Pipeline
Dataset: academic_performance_dataset_V2.csv
Project: Regression and Time Series Modelling of Students' Performance Across Semesters
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.feature_selection import mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 0. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv('academic_performance_dataset_V2.csv')
print("=" * 60)
print("DATASET LOADED")
print(f"  Rows: {len(df)}")
print(f"  Columns: {list(df.columns)}")
print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# 1. PRE-TRANSFORMATION DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[PHASE 1] PRE-TRANSFORMATION DIAGNOSTICS")
print("-" * 60)

gpa_cols = ['CGPA100', 'CGPA200', 'CGPA300', 'CGPA400', 'CGPA', 'SGPA']

# 1a. Descriptive statistics
print("\n-- Descriptive Statistics --")
print(df[gpa_cols].describe().round(3).to_string())

# 1b. Skewness and Kurtosis
print("\n-- Shape of Distributions --")
print(f"{'Column':<12} {'Skewness':>12} {'Kurtosis':>12}")
print("-" * 38)
for col in gpa_cols:
    skew = df[col].skew()
    kurt = df[col].kurt()
    print(f"{col:<12} {skew:>12.4f} {kurt:>12.4f}")

# 1c. Normality test (Shapiro-Wilk on sample of 500)
print("\n-- Normality Test (Shapiro-Wilk, n=500) --")
print(f"{'Column':<12} {'W-stat':>10} {'p-value':>12} {'Normal?':>10}")
print("-" * 46)
sample = df.sample(500, random_state=42)
for col in gpa_cols:
    stat, p = stats.shapiro(sample[col])
    normal = "Yes" if p >= 0.05 else "No"
    print(f"{col:<12} {stat:>10.4f} {p:>12.4f} {normal:>10}")

# 1d. Correlation matrix
print("\n-- Pearson Correlation Matrix --")
print(df[gpa_cols].corr().round(3).to_string())

print("\n[PHASE 1 COMPLETE]\n")


# ─────────────────────────────────────────────────────────────────────────────
# 2. TRANSFORMATION
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("[PHASE 2] DATA TRANSFORMATION")
print("-" * 60)

df_transformed = df.copy()

# 2a. Lag features — each level's CGPA becomes the "previous GPA" for the next
#     This is the core time-series structure: x(t-1) predicting x(t)
df_transformed['Lag1_GPA'] = df['CGPA100']   # previous GPA entering 200L
df_transformed['Lag2_GPA'] = df['CGPA200']   # previous GPA entering 300L
df_transformed['Lag3_GPA'] = df['CGPA300']   # previous GPA entering 400L
print("  [+] Lag features created (Lag1, Lag2, Lag3)")

# 2b. Delta features — rate of change between levels
#     Captures improvement or decline trajectory
df_transformed['Delta_100_200'] = df['CGPA200'] - df['CGPA100']
df_transformed['Delta_200_300'] = df['CGPA300'] - df['CGPA200']
df_transformed['Delta_300_400'] = df['CGPA400'] - df['CGPA300']
print("  [+] Delta (change) features created")

# 2c. Z-score normalization on all numeric GPA/score columns
#     Centers each variable around 0, scales by standard deviation
#     Keeps relative differences intact — just changes the unit of measurement
all_numeric = ['CGPA100', 'CGPA200', 'CGPA300', 'CGPA400', 'CGPA', 'SGPA',
               'Lag1_GPA', 'Lag2_GPA', 'Lag3_GPA',
               'Delta_100_200', 'Delta_200_300', 'Delta_300_400']

scaler = StandardScaler()
df_transformed[all_numeric] = scaler.fit_transform(df_transformed[all_numeric])
print("  [+] Z-score normalization applied to all numeric columns")

# 2d. Encode Gender as binary (Male=1, Female=0)
df_transformed['Gender_encoded'] = (df['Gender'] == 'Male').astype(int)
print("  [+] Gender encoded (Male=1, Female=0)")

# 2e. Keep Prog Code and YoG as-is for grouping/filtering purposes
#     (not fed into the model directly — used for stratified analysis)
print("  [~] Prog Code and YoG kept as-is for grouping")

print("\n  Sample of transformed data:")
print(df_transformed[['CGPA100', 'CGPA200', 'Lag1_GPA',
                        'Delta_100_200', 'Gender_encoded']].head(5).round(3).to_string())

# Save transformed dataset
df_transformed.to_csv('academic_performance_transformed.csv', index=False)
print("\n  [SAVED] academic_performance_transformed.csv")
print("\n[PHASE 2 COMPLETE]\n")


# ─────────────────────────────────────────────────────────────────────────────
# 3. POST-TRANSFORMATION VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("[PHASE 3] POST-TRANSFORMATION VERIFICATION")
print("-" * 60)

model_features = ['CGPA100', 'CGPA200', 'CGPA300', 'CGPA400']
X = df_transformed[model_features].values

# 3a. VIF — checks if features are too similar to each other
#     VIF > 10 = problem. VIF 1-5 = healthy.
print("\n-- VIF (Variance Inflation Factor) --")
print(f"{'Feature':<12} {'VIF':>8}  {'Status':>10}")
print("-" * 34)
for i, col in enumerate(model_features):
    y_vif = X[:, i]
    X_rest = np.delete(X, i, axis=1)
    r2 = r2_score(y_vif, LinearRegression().fit(X_rest, y_vif).predict(X_rest))
    vif = 1 / (1 - r2) if r2 < 1.0 else float('inf')
    status = "OK" if vif < 5 else ("Moderate" if vif < 10 else "HIGH")
    print(f"{col:<12} {vif:>8.2f}  {status:>10}")

# 3b. Mutual Information — how much each feature "knows" about CGPA
X_mi = df_transformed[['CGPA100', 'CGPA200', 'CGPA300', 'CGPA400',
                         'Lag1_GPA', 'Lag2_GPA', 'Lag3_GPA']].values
y_mi = df['CGPA'].values  # original (un-normalized) CGPA as target
mi_scores = mutual_info_regression(X_mi, y_mi, random_state=42)

print("\n-- Mutual Information (Feature Relevance to CGPA) --")
print(f"{'Feature':<12} {'MI Score':>10}  {'Relevance':>12}")
print("-" * 38)
mi_labels = ['CGPA100', 'CGPA200', 'CGPA300', 'CGPA400', 'Lag1', 'Lag2', 'Lag3']
for label, score in zip(mi_labels, mi_scores):
    relevance = "High" if score > 0.7 else ("Moderate" if score > 0.3 else "Low")
    print(f"{label:<12} {score:>10.4f}  {relevance:>12}")

# 3c. Eigenvalue analysis — structural health of the feature set
print("\n-- Eigenvalue Analysis (Correlation Matrix) --")
corr_matrix = df_transformed[model_features].corr().values
eigenvalues = np.sort(np.linalg.eigvalsh(corr_matrix))[::-1]
condition_number = eigenvalues[0] / eigenvalues[-1]
print(f"{'λ':<6} {'Value':>10}")
print("-" * 18)
for i, ev in enumerate(eigenvalues):
    print(f"  λ{i+1:<3} {ev:>10.4f}")
print(f"\n  Condition Number: {condition_number:.2f}")
cn_status = "Stable" if condition_number < 30 else ("Borderline" if condition_number < 100 else "Unstable")
print(f"  Assessment: {cn_status}")

# 3d. Baseline R² — early signal of how well features predict CGPA
X_base = df_transformed[['CGPA100', 'CGPA200', 'CGPA300', 'CGPA400',
                           'Lag1_GPA', 'Lag2_GPA', 'Lag3_GPA',
                           'Delta_100_200', 'Delta_200_300', 'Delta_300_400']].values
y_base = df['CGPA'].values
model = LinearRegression().fit(X_base, y_base)
y_pred = model.predict(X_base)
r2 = r2_score(y_base, y_pred)
adj_r2 = 1 - (1 - r2) * (len(y_base) - 1) / (len(y_base) - X_base.shape[1] - 1)

print(f"\n-- Baseline Linear Regression Fit --")
print(f"  R²          : {r2:.4f}  ({r2*100:.1f}% of variance explained)")
print(f"  Adjusted R² : {adj_r2:.4f}")
print(f"  Assessment  : {'Strong signal' if r2 > 0.8 else 'Moderate signal' if r2 > 0.5 else 'Weak signal'}")

# 3e. GPA variance across year levels (stationarity indicator)
print(f"\n-- GPA Variance Across Year Levels --")
print(f"  (Checks if the spread of scores stays reasonably consistent)")
print(f"{'Level':<12} {'Mean':>8} {'Variance':>10}")
print("-" * 32)
for col in ['CGPA100', 'CGPA200', 'CGPA300', 'CGPA400']:
    print(f"  {col:<10} {df[col].mean():>8.3f} {df[col].var():>10.4f}")

print("\n[PHASE 3 COMPLETE]\n")
print("=" * 60)
print("PIPELINE COMPLETE")
print("Output file: academic_performance_transformed.csv")
print("=" * 60)
