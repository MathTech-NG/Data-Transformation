"""
score.py
──────────────────────────────────────────────────────────────────────────────
Operational CGPA400 predictions from the reference OLS fit on the full deduped
enriched dataset (same specification as app.py / model.ipynb).

Input rows must include Attendance_Rate, Study_Hours_Per_Week, Course_Load, Genotype, and
either Previous_GPA or (CGPA100, CGPA200, CGPA300). When level GPAs are supplied,
Trajectory_Slope_Prior is derived automatically (CGPA100–300 slope; excludes CGPA400).

Usage:
    python score.py --training academic_performance_enriched.csv \\
        --input new_rows.csv --output scored.csv

    python score.py --json rows.json --output scored.csv

Run verify.py on the training CSV before production use.
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from prediction_common import (
    fit_reference_ols,
    load_enriched_deduped,
    prediction_decomposition,
    prepare_scoring_features,
    print_limitation_banner,
    score_dataframe,
    soft_validate_predictors,
)


def main() -> int:
    p = argparse.ArgumentParser(description="Score new rows with the reference OLS model.")
    p.add_argument(
        "--training",
        default="academic_performance_enriched.csv",
        help="Enriched CSV used to refit the reference model.",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--input", help="CSV file with rows to score.")
    g.add_argument("--json", dest="json_path", help="JSON file: array of objects or newline JSON.")
    p.add_argument("--output", required=True, help="Output CSV path with predictions appended.")
    args = p.parse_args()

    print_limitation_banner()

    train = load_enriched_deduped(args.training)
    model = fit_reference_ols(train)

    if args.input:
        raw = pd.read_csv(args.input)
    else:
        with open(args.json_path, encoding="utf-8") as f:
            text = f.read().strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            try:
                payload = [json.loads(ln) for ln in lines]
            except json.JSONDecodeError as e:
                print(f"Could not parse JSON or newline-delimited JSON: {e}", file=sys.stderr)
                return 1
        if isinstance(payload, list):
            raw = pd.DataFrame(payload)
        elif isinstance(payload, dict):
            raw = pd.DataFrame([payload])
        else:
            print("JSON root must be an array of objects or a single object.", file=sys.stderr)
            return 1

    X, msgs = prepare_scoring_features(raw)
    for line in msgs:
        print(line)

    sv = soft_validate_predictors(X)
    for line in sv:
        print("Warning:", line)

    pred = score_dataframe(model, X)
    out = pd.concat([raw.reset_index(drop=True), pred.reset_index(drop=True)], axis=1)
    if "Trajectory_Slope_Prior" in X.columns:
        out["Trajectory_Slope_Prior"] = X["Trajectory_Slope_Prior"].values
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} row(s) with column 'predicted_CGPA400' → {args.output}")
    if len(X) == 1:
        print("\nPrediction breakdown (coefficient × value):")
        print(prediction_decomposition(model, X).to_string(index=False))
    print(
        "Reminder: predicted values depend on synthesized predictors for training rows "
        "(L1, L3) and are not registrar-official forecasts."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
