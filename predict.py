"""
predict.py
──────────────────────────────────────────────────────────────────────────────
K-fold out-of-sample evaluation for the OLS specification:
    CGPA ~ Previous_GPA + Attendance_Rate + Study_Hours_Per_Week + Course_Load

Optional exploratory metrics for CGPA400 (pending formal OQ-1 approval).

Usage:
    python predict.py [--training PATH] [--k 5] [--seed 42] [--also-cgpa400]

Run verify.py on the enriched CSV before relying on these numbers.
"""

from __future__ import annotations

import argparse
import sys

from prediction_common import (
    cross_val_ols_metrics,
    load_enriched_deduped,
    print_limitation_banner,
)


def main() -> int:
    p = argparse.ArgumentParser(description="K-fold OLS cross-validation on enriched data.")
    p.add_argument(
        "--training",
        default="academic_performance_enriched.csv",
        help="Path to enriched training CSV (default: academic_performance_enriched.csv)",
    )
    p.add_argument("--k", type=int, default=5, help="Number of CV folds (default: 5)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for KFold splits")
    p.add_argument(
        "--also-cgpa400",
        action="store_true",
        help="Also print exploratory CV metrics for CGPA400 (OQ-1 — not default DV until approved).",
    )
    args = p.parse_args()

    print_limitation_banner()

    df = load_enriched_deduped(args.training)
    m = cross_val_ols_metrics(df, "CGPA", k=args.k, seed=args.seed)
    print(
        f"Out-of-sample (stacked OOF predictions), target=CGPA, n={m['n']}, k={m['k']}, seed={args.seed}"
    )
    print(f"  MAE  (model):  {m['mae']:.4f}")
    print(f"  RMSE (model):  {m['rmse']:.4f}")
    print(f"  R²   (OOF):    {m['r2_oof']:.4f}")
    print(
        f"  Baseline (train-mean): MAE={m['baseline_mean_mae']:.4f}, "
        f"RMSE={m['baseline_mean_rmse']:.4f}"
    )

    if args.also_cgpa400:
        print(
            "\n*** EXPLORATORY — OQ-1 ***\n"
            "CGPA400 metrics below are for discussion only. The manuscript DV remains CGPA\n"
            "until Kelvin and supervisor formally approve a reframe.\n"
        )
        m4 = cross_val_ols_metrics(df, "CGPA400", k=args.k, seed=args.seed)
        print(
            f"Out-of-sample, target=CGPA400, n={m4['n']}, k={m4['k']}, seed={args.seed}"
        )
        print(f"  MAE  (model):  {m4['mae']:.4f}")
        print(f"  RMSE (model):  {m4['rmse']:.4f}")
        print(f"  R²   (OOF):    {m4['r2_oof']:.4f}")
        print(
            f"  Baseline (train-mean): MAE={m4['baseline_mean_mae']:.4f}, "
            f"RMSE={m4['baseline_mean_rmse']:.4f}"
        )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
