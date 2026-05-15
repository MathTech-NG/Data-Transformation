"""
app.py
──────────────────────────────────────────────────────────────────────────────
Streamlit presentation layer for the Academic Performance Modelling project.

  Regression and Time Series Modelling of Students' Performance Across Semester
  Ehime Kelvin Ehinomen — Mountain Top University, Jan 2026

Presentation only — Tabs 1–3 display the full-sample OLS fit and charts.
Tab 4 reports k-fold metrics and applies the same fitted equation to user inputs.
Tabs: Data Overview | OLS Results | Trajectory Charts | Predict & CV

Run:
    streamlit run app.py
"""

import os
import subprocess
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
import statsmodels.api as sm
import streamlit as st

from prediction_common import (
    DESIGN_COLUMNS,
    build_design_matrix,
    cross_val_ols_metrics,
    fit_reference_ols,
    prepare_scoring_features,
    score_dataframe,
    soft_validate_predictors,
)

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Academic Performance Model — MTU 2026",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── PATHS ────────────────────────────────────────────────────────────────────

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(BASE_DIR, "academic_performance_enriched.csv")
ENRICH_PY  = os.path.join(BASE_DIR, "enrich.py")
RAW_CSV    = os.path.join(BASE_DIR, "academic_performance_dataset_V2.csv")

# ─── DATA LOADING ─────────────────────────────────────────────────────────────

@st.cache_data
def load_data() -> pd.DataFrame:
    if not os.path.exists(CSV_PATH):
        st.warning("Enriched CSV not found — regenerating via enrich.py …")
        result = subprocess.run(
            [sys.executable, ENRICH_PY,
             "--input", RAW_CSV,
             "--output", CSV_PATH,
             "--seed", "42"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            st.error(f"enrich.py failed:\n{result.stderr}")
            st.stop()
    df = pd.read_csv(CSV_PATH)
    # Drop duplicate ID No rows (70 data-entry error collisions — see L5)
    df = df.drop_duplicates(subset="ID No", keep="first").reset_index(drop=True)
    return df


# ─── MODEL FITTING ────────────────────────────────────────────────────────────
# @st.cache_resource so Streamlit does not attempt to hash the DataFrame.
# The leading underscore on _df suppresses the hash-argument warning.

@st.cache_resource
def run_ols(_df: pd.DataFrame):
    return fit_reference_ols(_df, target="CGPA")


# ─── SIDEBAR — LIMITATIONS ────────────────────────────────────────────────────

LIMITATIONS = {
    "L1 — Residual Endogeneity (critical)": (
        "`Attendance_Rate` and `Study_Hours_Per_Week` are synthesized using `CGPA` as an input "
        "(at 40% weight). Running OLS on these variables therefore recovers a relationship that is "
        "**partly constructed**. The significant p-values do **not** constitute evidence that higher "
        "attendance or more study hours cause better outcomes. The paper must state this explicitly "
        "in the methodology section — not just in a limitations appendix."
    ),
    "L2 — Arithmetic Overlap in Previous_GPA": (
        "`Previous_GPA = mean(CGPA100, CGPA200, CGPA300)` shares three of four components with "
        "the final `CGPA`. The Adj R² ≈ 0.953 is driven almost entirely by this definitional "
        "relationship, not by the behavioural variables."
    ),
    "L3 — Synthesized Variables Are Not Observed Data": (
        "`Attendance_Rate`, `Study_Hours_Per_Week`, and `Course_Load` were never measured. "
        "They are statistically engineered. Coefficient estimates for these variables carry "
        "**no empirical weight** and should not drive policy recommendations."
    ),
    "L4 — Overload Rate Assumption": (
        "The 5% overload probability and 16–20 course ceiling are assumptions calibrated against "
        "a single verified MTU transcript. If broader institutional data is available, update "
        "`OVERLOAD_PROB` and `OVERLOAD_HI` in `enrich.py`."
    ),
    "L5 — ID Collisions in Source Data": (
        "70 student ID numbers appear on rows with conflicting programme/gender/YoG attributes. "
        "These are data-entry errors. This app drops duplicate ID rows (keep first) before "
        "analysis. Any use of `ID No` as a join key must account for this."
    ),
    "L6 — Generalizability": (
        "Synthesis parameters were calibrated for a Nigerian university context (Mountain Top "
        "University). They are not directly transferable to other institutions."
    ),
    "L7 — Synthesized genotype": (
        "`Genotype` (AA, AS, SS) was not collected from medical records. Values are drawn from "
        "fixed population proportions (75% / 24% / 1%) independent of CGPA. Coefficients for "
        "genotype dummies carry no clinical or empirical weight."
    ),
}

with st.sidebar:
    st.title("Scientific Limitations")
    st.error("Required disclosure — must appear in any paper or report using this data.")
    for i, (title, body) in enumerate(LIMITATIONS.items()):
        with st.expander(title, expanded=(i < 2)):  # L1 and L2 open by default
            st.markdown(body)
    st.divider()
    st.caption(
        "**Author:** Ehime Kelvin Ehinomen  \n"
        "**Data engineering:** Yaba-Shiaka, Shemaiah Wambebe  \n"
        "**Institution:** Mountain Top University, Jan 2026"
    )

# ─── LOAD DATA + FIT MODEL ────────────────────────────────────────────────────

df    = load_data()
model = run_ols(df)

# Derived quantities used across tabs
PREDICTORS   = DESIGN_COLUMNS
COEF_LABELS  = [
    "Previous_GPA", "Attendance_Rate", "Study_Hours_Per_Week", "Course_Load",
    "Genotype (AS vs AA)", "Genotype (SS vs AA)",
]
LEVEL_COLS   = ["CGPA100", "CGPA200", "CGPA300", "CGPA400"]
LEVEL_LABELS = ["Level 100", "Level 200", "Level 300", "Level 400"]
fitted       = model.fittedvalues
resid        = model.resid
std_resid    = resid / resid.std()

# ─── TABS ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Data Overview",
    "📈  OLS Results",
    "📉  Trajectory Charts",
    "🔮  Predict & CV",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATA OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.header("Data Overview")

    # ── Metric cards ──────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Students (post-dedup)", f"{len(df):,}")
    m2.metric("Programmes", df["Prog Code"].nunique())
    m3.metric("CGPA mean ± std",
              f"{df['CGPA'].mean():.2f} ± {df['CGPA'].std():.2f}")
    m4.metric("Graduation years",
              f"{df['YoG'].min()} – {df['YoG'].max()}")

    st.divider()

    # ── Descriptive statistics ────────────────────────────────────────────────
    with st.expander("Descriptive statistics", expanded=True):
        numeric_cols = [
            "CGPA", "Previous_GPA",
            "CGPA100", "CGPA200", "CGPA300", "CGPA400", "SGPA",
            "Attendance_Rate", "Study_Hours_Per_Week", "Course_Load",
        ]
        st.dataframe(df[numeric_cols].describe().T.round(3), width="stretch")
        st.caption("Genotype counts: " + df["Genotype"].value_counts().sort_index().to_string())

    st.divider()

    # ── Distributions ─────────────────────────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        fig_cgpa = px.histogram(
            df, x="CGPA", nbins=40,
            title="CGPA Distribution",
            labels={"CGPA": "CGPA"},
            color_discrete_sequence=["steelblue"],
        )
        fig_cgpa.add_vline(
            x=df["CGPA"].mean(), line_dash="dash", line_color="firebrick",
            annotation_text=f"Mean = {df['CGPA'].mean():.2f}",
        )
        st.plotly_chart(fig_cgpa, width="stretch")

    with col_r:
        fig_gen = px.violin(
            df, x="Gender", y="CGPA", box=True, points="outliers",
            title="CGPA by Gender",
            color="Gender", color_discrete_sequence=["#4CAF50", "#2196F3"],
        )
        st.plotly_chart(fig_gen, width="stretch")

    # ── Programme boxplot ─────────────────────────────────────────────────────
    prog_order = (
        df.groupby("Prog Code")["CGPA"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    fig_prog = px.box(
        df, x="Prog Code", y="CGPA",
        category_orders={"Prog Code": prog_order},
        title="CGPA by Programme (sorted by median)",
        color="Prog Code",
    )
    fig_prog.update_layout(showlegend=False)
    st.plotly_chart(fig_prog, width="stretch")

    # ── Synthesized variable distributions ───────────────────────────────────
    st.subheader("Synthesized Variable Distributions")
    c1, c2, c3 = st.columns(3)
    for col_widget, var, color in zip(
        [c1, c2, c3],
        ["Attendance_Rate", "Study_Hours_Per_Week", "Course_Load"],
        ["teal", "darkorange", "mediumpurple"],
    ):
        fig_v = px.histogram(
            df, x=var, nbins=30,
            title=var.replace("_", " "),
            color_discrete_sequence=[color],
        )
        fig_v.add_vline(
            x=df[var].mean(), line_dash="dash", line_color="black",
            annotation_text=f"Mean={df[var].mean():.1f}",
        )
        col_widget.plotly_chart(fig_v, width="stretch")

    # ── Correlation heatmap ───────────────────────────────────────────────────
    st.subheader("Pearson Correlation Matrix")
    corr_cols = [
        "CGPA", "Previous_GPA",
        "CGPA100", "CGPA200", "CGPA300", "CGPA400",
        "Attendance_Rate", "Study_Hours_Per_Week", "Course_Load",
    ]
    corr = df[corr_cols].corr().round(3)
    fig_heat = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Correlation Heatmap",
        aspect="auto",
    )
    fig_heat.update_layout(height=500)
    st.plotly_chart(fig_heat, width="stretch")
    st.caption(
        "Note: r(Previous_GPA, CGPA) = 0.976 reflects **arithmetic overlap** (L2), "
        "not a purely empirical relationship."
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — OLS RESULTS
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("OLS Regression Results")
    st.markdown(
        "**Model:** `CGPA ~ Previous_GPA + Attendance_Rate + Study_Hours_Per_Week + Course_Load`"
    )

    # ── Summary metrics ───────────────────────────────────────────────────────
    s1, s2, s3 = st.columns(3)
    s1.metric("Adj R²",     f"{model.rsquared_adj:.4f}")
    s2.metric("F-statistic", f"{model.fvalue:,.1f}")
    s3.metric("N observations", f"{int(model.nobs):,}")

    st.divider()

    # ── Coefficient table ─────────────────────────────────────────────────────
    st.subheader("Coefficient Table")
    ci = model.conf_int()
    coef_df = pd.DataFrame({
        "Variable":        model.params.index,
        "Coefficient":     model.params.values.round(4),
        "Std Error":       model.bse.values.round(4),
        "t-statistic":     model.tvalues.values.round(3),
        "p-value":         model.pvalues.values.round(4),
        "CI Lower (95%)":  ci[0].values.round(4),
        "CI Upper (95%)":  ci[1].values.round(4),
    }).set_index("Variable")

    def color_pval(val):
        if val < 0.001:
            return "background-color: #d4edda"
        elif val < 0.05:
            return "background-color: #fff3cd"
        return "background-color: #f8d7da"

    st.dataframe(
        coef_df.style.map(color_pval, subset=["p-value"]),
        width="stretch",
    )

    # ── Coefficient bar chart ─────────────────────────────────────────────────
    st.subheader("Coefficients (± 1.96 SE)")
    coefs  = model.params[DESIGN_COLUMNS]
    errors = model.bse[DESIGN_COLUMNS] * 1.96
    colors = ["#2196F3" if c > 0 else "#F44336" for c in coefs]

    fig_coef = go.Figure(go.Bar(
        y=COEF_LABELS,
        x=coefs.values,
        orientation="h",
        error_x=dict(type="data", array=errors.values, visible=True),
        marker_color=colors,
        opacity=0.85,
    ))
    fig_coef.add_vline(x=0, line_width=1, line_color="black")
    fig_coef.update_layout(
        title="OLS Coefficients with 95% Confidence Intervals",
        xaxis_title="Coefficient value",
        height=320,
    )
    st.plotly_chart(fig_coef, width="stretch")

    st.divider()

    # ── Residual diagnostics 2×2 ──────────────────────────────────────────────
    st.subheader("Residual Diagnostics")

    (osm, osr), (slope, intercept, _) = stats.probplot(resid.values, dist="norm")

    fig_diag = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Residuals vs Fitted",
            "Q-Q Plot of Residuals",
            "Residual Distribution",
            "Scale-Location",
        ),
    )

    # Residuals vs Fitted
    fig_diag.add_trace(
        go.Scatter(x=fitted.values, y=resid.values,
                   mode="markers", marker=dict(size=3, opacity=0.4, color="steelblue"),
                   name="Residuals"),
        row=1, col=1,
    )
    fig_diag.add_hline(y=0, line_dash="dash", line_color="firebrick", row=1, col=1)

    # Q-Q plot
    fig_diag.add_trace(
        go.Scatter(x=osm, y=osr,
                   mode="markers", marker=dict(size=3, opacity=0.4, color="steelblue"),
                   name="Q-Q"),
        row=1, col=2,
    )
    fig_diag.add_trace(
        go.Scatter(x=osm,
                   y=[slope * q + intercept for q in osm],
                   mode="lines", line=dict(color="firebrick", width=1.5),
                   name="Normal line"),
        row=1, col=2,
    )

    # Residual histogram + normal overlay
    fig_diag.add_trace(
        go.Histogram(x=resid.values, nbinsx=40, histnorm="probability density",
                     marker_color="steelblue", opacity=0.6, name="Residuals hist"),
        row=2, col=1,
    )
    x_norm = np.linspace(resid.min(), resid.max(), 200)
    fig_diag.add_trace(
        go.Scatter(x=x_norm,
                   y=stats.norm.pdf(x_norm, resid.mean(), resid.std()),
                   mode="lines", line=dict(color="firebrick", width=1.5),
                   name="Normal curve"),
        row=2, col=1,
    )

    # Scale-Location
    fig_diag.add_trace(
        go.Scatter(x=fitted.values, y=np.sqrt(np.abs(std_resid.values)),
                   mode="markers", marker=dict(size=3, opacity=0.4, color="steelblue"),
                   name="Scale-Loc"),
        row=2, col=2,
    )

    fig_diag.update_layout(height=650, showlegend=False,
                           title_text="OLS Residual Diagnostics")
    st.plotly_chart(fig_diag, width="stretch")

    st.info(
        "**Q-Q plot note:** Near-normal residuals here are partly an artifact of "
        "`Previous_GPA`'s arithmetic overlap with `CGPA` (Limitation L2). "
        "Three of the four components of `CGPA` are contained in `Previous_GPA`. "
        "The well-behaved diagnostics reflect a largely definitional relationship, "
        "not purely good model specification."
    )

    # ── Actual vs Predicted ───────────────────────────────────────────────────
    st.subheader("Actual vs Predicted CGPA")
    scatter_df = pd.DataFrame({
        "Predicted": fitted.values,
        "Actual":    df["CGPA"].values,
        "Programme": df["Prog Code"].values,
    })
    fig_avp = px.scatter(
        scatter_df, x="Predicted", y="Actual",
        color="Programme", opacity=0.4,
        title=f"Actual vs Predicted CGPA  (Adj R² = {model.rsquared_adj:.3f})",
        labels={"Predicted": "Predicted CGPA", "Actual": "Actual CGPA"},
    )
    lo = min(scatter_df["Predicted"].min(), scatter_df["Actual"].min())
    hi = max(scatter_df["Predicted"].max(), scatter_df["Actual"].max())
    fig_avp.add_trace(
        go.Scatter(x=[lo, hi], y=[lo, hi],
                   mode="lines",
                   line=dict(color="black", dash="dash", width=1.2),
                   name="Identity line")
    )
    fig_avp.update_layout(height=500)
    st.plotly_chart(fig_avp, width="stretch")
    st.caption(
        "The tight cluster around the identity line is driven by `Previous_GPA` "
        "dominating the regression (L2) — not by the behavioural predictors."
    )

    # ── Full statsmodels summary ──────────────────────────────────────────────
    with st.expander("Full statsmodels summary (text)"):
        st.code(str(model.summary()), language="text")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TRAJECTORY CHARTS
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.header("CGPA Trajectory Across Levels 100 → 400")
    st.caption(
        "Each student contributes one observation per level, so CGPA100→400 form a "
        "**within-student progression sequence**. The group trajectories shown here are "
        "**means across students at each level**, not repeated measurements of the same "
        "cohort tracked over calendar time."
    )

    # ── Build long-format ─────────────────────────────────────────────────────
    df_long = df[["ID No", "Prog Code", "YoG", "Gender"] + LEVEL_COLS].melt(
        id_vars=["ID No", "Prog Code", "YoG", "Gender"],
        value_vars=LEVEL_COLS,
        var_name="Level",
        value_name="GPA",
    )
    df_long["Level"] = df_long["Level"].map(dict(zip(LEVEL_COLS, LEVEL_LABELS)))
    df_long["Level"] = pd.Categorical(
        df_long["Level"], categories=LEVEL_LABELS, ordered=True
    )

    # Programmes with >= 20 students
    large_progs = (
        df["Prog Code"].value_counts()
        .loc[lambda s: s >= 20]
        .index.tolist()
    )

    st.divider()

    # ── Group selector ────────────────────────────────────────────────────────
    group_by = st.radio(
        "Group trajectories by:",
        ["Programme", "Year of Graduation"],
        horizontal=True,
    )

    if group_by == "Programme":
        group_col    = "Prog Code"
        all_options  = large_progs
        default_sel  = large_progs[:6] if len(large_progs) >= 6 else large_progs
        group_label  = "Programme"
    else:
        group_col    = "YoG"
        all_options  = sorted(df["YoG"].unique().tolist())
        default_sel  = all_options
        group_label  = "Year of Graduation"

    selected = st.multiselect(
        f"Select {group_label}(s):",
        options=all_options,
        default=default_sel,
    )

    if not selected:
        st.warning("Select at least one group to display.")
        st.stop()

    # ── Mean trajectory line chart ────────────────────────────────────────────
    subset = df_long[df_long[group_col].isin(selected)].copy()
    traj = (
        subset.groupby([group_col, "Level"], observed=True)["GPA"]
        .mean()
        .reset_index()
    )
    fig_traj = px.line(
        traj, x="Level", y="GPA",
        color=group_col,
        markers=True,
        title=f"Mean CGPA Trajectory by {group_label}",
        labels={"GPA": "Mean GPA", "Level": "Academic Level"},
        category_orders={"Level": LEVEL_LABELS},
    )
    fig_traj.update_layout(height=420)
    st.plotly_chart(fig_traj, width="stretch")

    # ── Distribution boxplot across levels ────────────────────────────────────
    st.subheader("GPA Distribution Across Levels")
    fig_box = px.box(
        subset, x="Level", y="GPA",
        color="Level",
        title="GPA Distribution per Academic Level",
        labels={"GPA": "GPA", "Level": "Academic Level"},
        category_orders={"Level": LEVEL_LABELS},
    )
    fig_box.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_box, width="stretch")

    # ── Programme heatmap ─────────────────────────────────────────────────────
    st.subheader("Programme × Level Heatmap (Mean GPA)")
    hm_data = (
        df_long[df_long["Prog Code"].isin(large_progs)]
        .groupby(["Prog Code", "Level"], observed=True)["GPA"]
        .mean()
        .unstack("Level")
        .round(3)
    )
    hm_data = hm_data[LEVEL_LABELS].sort_values("Level 400", ascending=False)

    fig_hm = px.imshow(
        hm_data,
        text_auto=".2f",
        color_continuous_scale="YlGnBu",
        zmin=2.0, zmax=4.5,
        title="Mean GPA per Programme × Level (sorted by Level 400 performance)",
        aspect="auto",
    )
    fig_hm.update_layout(height=420)
    st.plotly_chart(fig_hm, width="stretch")

    st.caption(
        "Data source: `academic_performance_enriched.csv` — "
        "Ehime Kelvin Ehinomen, Mountain Top University, Jan 2026."
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PREDICT & CV
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.header("Prediction layer (secondary goal)")
    st.caption(
        "Out-of-sample metrics use k-fold cross-validation (same logic as `predict.py`). "
        "Scores use the same OLS specification as Tab 2, fitted on this deduped dataset. "
        "L1–L3 apply — illustrative only, not registrar-official forecasts."
    )

    k_cv = st.slider("CV folds (k)", min_value=3, max_value=10, value=5, step=1)
    seed_cv = st.number_input("Random seed (CV splits)", value=42, step=1)

    m_cv = cross_val_ols_metrics(df, "CGPA", k=int(k_cv), seed=int(seed_cv))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE (OOF)", f"{m_cv['mae']:.4f}")
    c2.metric("RMSE (OOF)", f"{m_cv['rmse']:.4f}")
    c3.metric("R² (OOF)", f"{m_cv['r2_oof']:.4f}")
    c4.metric("Baseline MAE", f"{m_cv['baseline_mean_mae']:.4f}")
    st.caption(
        "Baseline = always predict each fold's training-set mean CGPA for held-out rows "
        "(same k and seed)."
    )

    if st.checkbox("Show exploratory CGPA400 metrics (OQ-1)", value=False):
        st.warning(
            "CGPA400 metrics are for discussion only until Kelvin and supervisor approve "
            "a dependent-variable reframe."
        )
        m4 = cross_val_ols_metrics(df, "CGPA400", k=int(k_cv), seed=int(seed_cv))
        d1, d2, d3 = st.columns(3)
        d1.metric("CGPA400 MAE", f"{m4['mae']:.4f}")
        d2.metric("CGPA400 RMSE", f"{m4['rmse']:.4f}")
        d3.metric("CGPA400 R² (OOF)", f"{m4['r2_oof']:.4f}")

    st.divider()
    st.subheader("Score new rows")
    mode = st.radio("Input mode", ["Manual form", "CSV upload"], horizontal=True)

    if mode == "Manual form":
        mode_prev = st.radio(
            "Prior GPA",
            ["Enter Previous_GPA directly", "Derive from CGPA100 / 200 / 300"],
            horizontal=True,
        )
        att = st.number_input(
            "Attendance_Rate (%)", min_value=0.0, max_value=100.0, value=75.0, step=0.1
        )
        sh = st.number_input(
            "Study_Hours_Per_Week", min_value=0.0, max_value=24.0, value=8.0, step=0.1
        )
        cl = st.number_input(
            "Course_Load (# courses)", min_value=0, max_value=25, value=11, step=1
        )
        genotype = st.selectbox("Genotype", ["AA", "AS", "SS"], index=0)
        if mode_prev == "Enter Previous_GPA directly":
            prev = st.number_input(
                "Previous_GPA", min_value=0.0, max_value=5.0, value=3.2, step=0.01
            )
            raw_in = pd.DataFrame([{
                "Previous_GPA": prev,
                "Attendance_Rate": att,
                "Study_Hours_Per_Week": sh,
                "Course_Load": int(cl),
                "Genotype": genotype,
            }])
        else:
            g1 = st.number_input(
                "CGPA100", min_value=0.0, max_value=5.0, value=3.0, step=0.01
            )
            g2 = st.number_input(
                "CGPA200", min_value=0.0, max_value=5.0, value=3.2, step=0.01
            )
            g3 = st.number_input(
                "CGPA300", min_value=0.0, max_value=5.0, value=3.4, step=0.01
            )
            raw_in = pd.DataFrame([{
                "CGPA100": g1,
                "CGPA200": g2,
                "CGPA300": g3,
                "Attendance_Rate": att,
                "Study_Hours_Per_Week": sh,
                "Course_Load": int(cl),
                "Genotype": genotype,
            }])
        if st.button("Compute predicted CGPA"):
            Xs, msgs = prepare_scoring_features(raw_in)
            for msg in msgs:
                st.info(msg)
            for w in soft_validate_predictors(Xs):
                st.warning(w)
            pred = float(score_dataframe(model, Xs).iloc[0])
            st.success(f"Predicted CGPA: **{pred:.4f}** (0–5 scale)")

    else:
        up = st.file_uploader(
            "Upload CSV (see README / `examples/example_score_rows.csv`)", type=["csv"]
        )
        if up is not None:
            raw_up = pd.read_csv(up)
            try:
                Xu, msgs = prepare_scoring_features(raw_up)
                for msg in msgs:
                    st.info(msg)
                for w in soft_validate_predictors(Xu):
                    st.warning(w)
                pred_s = score_dataframe(model, Xu)
                out_df = pd.concat([raw_up.reset_index(drop=True), pred_s], axis=1)
                st.dataframe(out_df)
                st.download_button(
                    "Download scored CSV",
                    data=out_df.to_csv(index=False).encode("utf-8"),
                    file_name="scored_upload.csv",
                    mime="text/csv",
                )
            except ValueError as e:
                st.error(str(e))
