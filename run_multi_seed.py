"""Run DivideMix training multiple times with different seeds, then report
aggregated test-set results (mean, std, 95 % CI).

Usage example:
    python run_multi_seed.py \
        --train_dir data/train --val_dir data/val --test_dir data/test \
        --num_class 4 --n_runs 10 --start_seed 42 --num_epochs 200
"""

from __future__ import print_function

import copy
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from Train_imagefolder_optuna import build_parser, make_base_logger, train_one_run


# ------------------------------------------------------------------
# Extra CLI arguments
# ------------------------------------------------------------------
def _extend_parser():
    parser = build_parser()
    parser.add_argument("--n_runs", default=10, type=int, help="number of training runs")
    parser.add_argument("--start_seed", default=42, type=int, help="seed for the first run (incremented by 1 each run)")
    return parser


# ------------------------------------------------------------------
# Summary statistics (mirrors NoisyLabelDefectDetection pattern)
# ------------------------------------------------------------------
def calculate_summary_statistics(
    all_metrics: List[Dict[str, Any]],
) -> tuple:
    all_metrics_df = pd.DataFrame(all_metrics)
    summary_rows = []
    metrics_to_analyze = [m for m in all_metrics_df.columns if m not in ("run_idx", "seed", "run_dir")]

    for metric in metrics_to_analyze:
        values = pd.to_numeric(all_metrics_df[metric], errors="coerce").dropna()
        n = len(values)
        if n == 0:
            continue
        mean = values.mean()
        median = values.median()
        std = values.std(ddof=1) if n > 1 else 0.0
        se = std / np.sqrt(n) if n > 0 else 0.0

        if n > 1:
            t = stats.t.ppf(0.975, df=n - 1)
            margin = t * se
            ci_lower = mean - margin
            ci_upper = mean + margin
        else:
            ci_lower = ci_upper = mean

        summary_rows.append(
            {
                "metric": metric,
                "mean": mean,
                "median": median,
                "std": std,
                "se": se,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "n": n,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    return all_metrics_df, summary_df


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> None:
    parser = _extend_parser()
    args = parser.parse_args()

    if not args.test_dir:
        print("WARNING: --test_dir not set. Test metrics will not be collected.", file=sys.stderr)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    base_root = os.path.join("logs", f"{args.id}_multiseed_{timestamp}")
    os.makedirs(base_root, exist_ok=True)

    base_logger = make_base_logger(base_root, logger_name=f"{args.id}.multiseed.{timestamp}")
    base_logger.info(f"Multi-seed run: n_runs={args.n_runs}, start_seed={args.start_seed}")
    for arg, value in vars(args).items():
        base_logger.info(f"ARG {arg}: {value}")

    # Save hyperparameters
    hparams_path = os.path.join(base_root, "hyperparameters.json")
    with open(hparams_path, "w", encoding="utf-8") as f:
        json.dump(
            {k: v for k, v in vars(args).items()},
            f,
            indent=2,
            sort_keys=True,
            default=str,
        )

    all_metrics: List[Dict[str, Any]] = []

    for run_idx in range(1, args.n_runs + 1):
        seed = args.start_seed + (run_idx - 1)
        run_args = copy.deepcopy(args)
        run_args.seed = seed

        run_root = os.path.join(base_root, f"run_{run_idx}_seed_{seed}")
        os.makedirs(run_root, exist_ok=True)

        base_logger.info(f"=== Run {run_idx}/{args.n_runs} | seed={seed} ===")

        result = train_one_run({}, run_args, run_root, trial=None)

        row: Dict[str, Any] = {
            "run_idx": run_idx,
            "seed": seed,
            "val/best_score": result["best_score"],
            "best_epoch": result["best_epoch"],
            "run_dir": result["run_dir"],
        }

        if result["test_metrics"] is not None:
            for k, v in result["test_metrics"].items():
                row[k] = v

        all_metrics.append(row)

        # Log progress
        test_primary = row.get(f"test/{run_args.primary_metric}", None)
        msg = f"Run {run_idx} done | val_best={result['best_score']:.4f}"
        if test_primary is not None:
            msg += f" | test/{run_args.primary_metric}={test_primary:.4f}"
        base_logger.info(msg)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    summary_dir = os.path.join(base_root, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    all_metrics_df, summary_df = calculate_summary_statistics(all_metrics)
    all_metrics_df.to_csv(os.path.join(summary_dir, "all_runs_metrics.csv"), index=False)
    summary_df.to_csv(os.path.join(summary_dir, "summary_statistics.csv"), index=False)

    base_logger.info(f"\nAll runs complete. Results saved to: {summary_dir}")
    base_logger.info(f"\n{'=' * 70}")
    base_logger.info("SUMMARY STATISTICS")
    base_logger.info(f"{'=' * 70}")

    # Pretty-print key metrics
    for _, row in summary_df.iterrows():
        metric = row["metric"]
        base_logger.info(
            f"  {metric:40s}  {row['mean']:.4f} +/- {row['std']:.4f}  "
            f"(95% CI: [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}])  n={int(row['n'])}"
        )

    base_logger.info(f"{'=' * 70}")
    base_logger.info(f"Per-run metrics:     {os.path.join(summary_dir, 'all_runs_metrics.csv')}")
    base_logger.info(f"Summary statistics:  {os.path.join(summary_dir, 'summary_statistics.csv')}")


if __name__ == "__main__":
    main()
