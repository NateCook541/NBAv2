"""
shuffled_backtest.py

Null-model test for the COMBINED backtest.

Runs the exact combined over+under backtest pipeline, but the points model is
trained on RANDOMLY SHUFFLED targets each retrain period. The model therefore
learns no real feature->points mapping. Everything else is untouched: real
book lines, real calibrator fit (on the corrupted model's outputs), real
filters, real conflict resolution, and settlement against REAL actual points.

Logic
-----
The reported combined P&L is only "real" if it comes from the model predicting
outcomes the book's line does not already capture. If a model trained on noise
still prints a strong positive P&L, that profit is NOT coming from predictive
signal — it points to leakage, filter/threshold overfitting to the eval window,
or a settlement artifact.

    shuffled P&L ~ break-even or negative   -> PASS: real backtest edge is genuine
    shuffled P&L strongly positive          -> the "edge" is an artifact, not signal

This is a Monte Carlo null: one run is one draw. Run a few seeds to see the
spread; a single lucky/unlucky shuffle is not conclusive on its own.

Usage
-----
    source env/bin/activate
    python -m models.shuffled_backtest                         # seed 0, full range
    python -m models.shuffled_backtest --seed 1
    python -m models.shuffled_backtest --start-date 2024-10-01 --end-date 2025-06-30
"""

import argparse
import numpy as np

from pipeline.orchestrator import Pipeline
import models.points as points_mod


def _installShuffle(seed):
    """
    Wrap PointsBundle.train so the target vector y is permuted before the real
    training routine runs. Features, dates, split logic, calibrator, and the
    rest of the pipeline are all left exactly as-is.
    """
    rng = np.random.default_rng(seed)
    original = points_mod.PointsBundle.train.__func__  # unwrap classmethod

    def shuffledTrain(cls, X, y, dates, **kwargs):
        yShuf = y.copy()
        yShuf.iloc[:] = rng.permutation(y.to_numpy())
        print(
            f"[ShuffledBacktest] Target labels PERMUTED before training "
            f"({len(yShuf)} rows, seed-derived). Model is a null model."
        )
        return original(cls, X, yShuf, dates, **kwargs)

    points_mod.PointsBundle.train = classmethod(shuffledTrain)


def main():
    ap = argparse.ArgumentParser(description="Shuffled-label combined backtest (null test)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start-date", type=str, default=None)
    ap.add_argument("--end-date", type=str, default=None)
    ap.add_argument("--bankroll", type=float, default=1000.0)
    ap.add_argument("--over-edge-thresh", type=float, default=0.10)
    ap.add_argument("--under-edge-thresh", type=float, default=0.05)
    ap.add_argument("--retrain-every-months", type=int, default=1)
    args = ap.parse_args()

    print("=" * 68)
    print("SHUFFLED-LABEL COMBINED BACKTEST — data-leakage / overfit null test")
    print(f"seed={args.seed}")
    print("If this prints a strong positive P&L, the real edge is an artifact.")
    print("Note: ~15s calibrator fit x2 x ~13 periods -> expect ~10-15 min total.")
    print("=" * 68)

    _installShuffle(args.seed)

    pipeline = Pipeline()
    pipeline.backtestCombined(
        startDate=args.start_date,
        endDate=args.end_date,
        bankroll=args.bankroll,
        overEdgeThresh=args.over_edge_thresh,
        underEdgeThresh=args.under_edge_thresh,
        retrainEveryMonths=args.retrain_every_months,
    )


if __name__ == "__main__":
    main()
