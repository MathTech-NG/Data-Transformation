"""
predict.py
──────────────────────────────────────────────────────────────────────────────
K-fold out-of-sample evaluation for the OLS specification predicting CGPA400:
    CGPA400 ~ Previous_GPA + Attendance + Study_Hours + Course_Load
              + Genotype dummies + Genotype_SS x Attendance

Optional legacy metrics for overall CGPA (--also-cgpa).

Usage:
    python predict.py [--training PATH] [--k 5] [--seed 42] [--also-cgpa]

Run verify.py on the enriched CSV before relying on these numbers.
"""

from __future__ import annotations

import argparse
import sys

from prediction_common import (
    DEFAULT_TARGET,
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
        "--also-cgpa",
        action="store_true",
        help="Also print CV metrics for legacy target CGPA (arithmetic overlap with Previous_GPA).",
    )
    args = p.parse_args()

    print_limitation_banner()

    df = load_enriched_deduped(args.training)
    m = cross_val_ols_metrics(df, DEFAULT_TARGET, k=args.k, seed=args.seed)
    print(
        f"Out-of-sample (stacked OOF predictions), target={DEFAULT_TARGET}, "
        f"n={m['n']}, k={m['k']}, seed={args.seed}"
    )
    print(f"  MAE  (model):  {m['mae']:.4f}")
    print(f"  RMSE (model):  {m['rmse']:.4f}")
    print(f"  R²   (OOF):    {m['r2_oof']:.4f}")
    print(
        f"  Baseline (train-mean): MAE={m['baseline_mean_mae']:.4f}, "
        f"RMSE={m['baseline_mean_rmse']:.4f}"
    )

    if args.also_cgpa:
        print(
            "\n*** LEGACY — arithmetic overlap ***\n"
            "CGPA shares components with Previous_GPA; metrics below are not the primary model.\n"
        )
        mc = cross_val_ols_metrics(df, "CGPA", k=args.k, seed=args.seed)
        print(f"Out-of-sample, target=CGPA, n={mc['n']}, k={mc['k']}, seed={args.seed}")
        print(f"  MAE  (model):  {mc['mae']:.4f}")
        print(f"  RMSE (model):  {mc['rmse']:.4f}")
        print(f"  R²   (OOF):    {mc['r2_oof']:.4f}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
