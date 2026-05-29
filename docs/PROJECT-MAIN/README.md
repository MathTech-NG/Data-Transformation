# Thesis source (`PROJECT-MAIN`)

Canonical thesis: [`regression_time_series_students_performance.tex`](regression_time_series_students_performance.tex)

## Figures

Chapter 4 and Appendix A figures are PNG exports from the Streamlit dashboard:

```bash
# From repository root (after enrich + verify)
streamlit run app.py
```

Capture screenshots from each tab, then save under `img/` using the `fig-4-*` and `fig-A-*` naming scheme (no spaces in filenames). Current assets:

| File | Source tab / content |
|------|----------------------|
| `fig-4-1-data-overview.png` | Data Overview |
| `fig-4-2-correlation-heatmap.png` | Data Overview (correlation) |
| `fig-4-3-ols-results.png` | OLS Results (coefficient table) |
| `fig-4-4-residual-diagnostics.png` | OLS Results (diagnostics) |
| `fig-4-5-actual-vs-predicted.png` | OLS Results (scatter) |
| `fig-4-6-trajectory-heatmap.png` | Trajectory Charts (heatmap) |
| `fig-A-1` … `fig-A-6` | Supplementary EDA and trajectory views (see Appendix A in thesis) |

Optional: add `fig-A-8-predict-cv.png` from the **Predict & CV** tab (see Appendix A, Section on out-of-sample metrics).

### Screenshots to refresh (model includes Trajectory_Slope_Prior)

These PNGs may still show the pre-trajectory OLS table until you re-export from a running app:

| Priority | File | Why |
|----------|------|-----|
| High | `fig-4-3-ols-results.png` | Must show Trajectory_Slope_Prior row |
| High | `fig-4-4-residual-diagnostics.png` | Residuals from updated fit |
| High | `fig-4-5-actual-vs-predicted.png` | Adj R² ≈ 0.68 |
| Medium | `fig-A-5-ols-coefficients-ci.png` | CI plot includes new term |
| New | `fig-A-8-predict-cv.png` | OOF metrics + scorer breakdown (optional) |

Data Overview and correlation figures are still valid; trajectory heatmaps unchanged.

## Build PDF

From this directory (`docs/PROJECT-MAIN/`), with a LaTeX distribution installed:

```bash
pdflatex regression_time_series_students_performance.tex
bibtex regression_time_series_students_performance
pdflatex regression_time_series_students_performance.tex
pdflatex regression_time_series_students_performance.tex
```

Requires `natbib`, `graphicx`, and BibTeX entry file [`references.bib`](references.bib).
