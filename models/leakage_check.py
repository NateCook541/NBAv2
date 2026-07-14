"""
leakage_check.py

Randomized-labels (label-permutation) sanity check for data leakage.

Idea
----
If the feature matrix contains no leakage, a model trained on RANDOMLY
SHUFFLED targets has nothing real to learn: the mapping from features to the
(now meaningless) label is pure noise. Its out-of-sample performance should
collapse to that of a naive constant baseline, and its test R^2 should sit at
~0 (often slightly negative).

If instead the shuffled-label model still predicts the held-out target well,
some feature is carrying information about the target that it should not have
(target bleed, an index/id proxy, train/test row contamination, etc.).

What this script does
---------------------
1. Loads the SAME cached feature matrix the pipeline trains on.
2. Reproduces the pipeline's chronological train/holdout split, so leakage via
   train/test contamination is also exercised (shuffling is done WITHIN the
   train split only; the holdout keeps its true labels for evaluation).
3. Fits XGBoost on the residual target the model actually learns
   (points - avgPts10 in residual mode), exactly as models/points.py does.
4. Reports, on the held-out set, for BOTH real and shuffled labels:
       - MAE and R^2 on the residual target the model fits, and
       - a naive-baseline MAE (predict the train residual mean).
   The residual target is used deliberately: reconstructing points as
   (pred + avgPts10) would look predictive even under shuffling simply because
   avgPts10 is a legitimate baseline, not leakage. Evaluating the residual
   isolates what the FEATURES explain.

Interpretation
--------------
    real   R^2 clearly > 0 and MAE < baseline   -> model has genuine signal (expected)
    shuffled R^2 ~ 0 (|R^2| small) and MAE ~ baseline over all trials -> PASS, no leakage
    shuffled R^2 consistently > ~0.02 or MAE well below baseline       -> LEAKAGE SUSPECTED

Usage
-----
    source env/bin/activate
    python -m models.leakage_check                 # default 5 shuffles
    python -m models.leakage_check --trials 10 --holdout 0.20
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

from config import (
    POINTS_MODEL_PARAMS, HOLDOUT_RATIO, POINTS_TARGET_MODE,
    USE_RECENCY_WEIGHTS, RECENCY_WEIGHT_MIN, RECENCY_WEIGHT_MAX,
)
from features.cache import FeatureCache
from models.points import _splitChronologically


def _residualTarget(y, X, targetMode):
    """The quantity XGBoost actually fits."""
    if targetMode == "residual":
        return y.to_numpy(dtype=float) - X["avgPts10"].to_numpy(dtype=float)
    return y.to_numpy(dtype=float)


def _fitEval(XTrain, yTrainTarget, XTest, yTestTarget):
    """Fit XGBoost on the (possibly shuffled) residual target, evaluate on test."""
    model = XGBRegressor(**POINTS_MODEL_PARAMS)
    if USE_RECENCY_WEIGHTS and len(XTrain) > 1:
        w = np.linspace(RECENCY_WEIGHT_MIN, RECENCY_WEIGHT_MAX, len(XTrain))
        model.fit(XTrain, yTrainTarget, sample_weight=w)
    else:
        model.fit(XTrain, yTrainTarget)

    pred = model.predict(XTest)
    mae = mean_absolute_error(yTestTarget, pred)
    r2 = r2_score(yTestTarget, pred)
    return mae, r2


def runLeakageCheck(trials=5, holdoutRatio=HOLDOUT_RATIO, seed=0):
    targetMode = str(POINTS_TARGET_MODE).lower().strip()

    # 1. Load the exact feature matrix the pipeline trains on.
    X, y, dates = FeatureCache().loadOrBuild()

    # Mirror the pipeline: drop rows with no scoring history.
    mask = X["avgPts10"] > 0
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)
    dates = dates[mask].reset_index(drop=True)

    # 2. Same chronological split the model uses.
    XTrain, XTest, yTrain, yTest, _, _ = _splitChronologically(
        X, y, dates, holdoutRatio=holdoutRatio
    )

    yTrainTarget = _residualTarget(yTrain, XTrain, targetMode)
    yTestTarget = _residualTarget(yTest, XTest, targetMode)

    # Naive baseline: predict the mean train residual for every test row.
    baselinePred = np.full(len(yTestTarget), float(np.mean(yTrainTarget)))
    baselineMae = mean_absolute_error(yTestTarget, baselinePred)

    print("=" * 68)
    print("Randomized-labels leakage check")
    print("=" * 68)
    print(f"target_mode   : {targetMode} (evaluating the residual target the model fits)")
    print(f"rows          : train={len(XTrain)}  holdout={len(XTest)}")
    print(f"features      : {X.shape[1]}")
    print(f"baseline MAE  : {baselineMae:.4f}  (predict mean train residual)")
    print("-" * 68)

    # 3a. Real labels.
    realMae, realR2 = _fitEval(XTrain, yTrainTarget, XTest, yTestTarget)
    print(f"REAL labels     ->  MAE={realMae:.4f}   R2={realR2:+.4f}   "
          f"MAE vs baseline={baselineMae - realMae:+.4f}")

    # 3b. Shuffled labels: permute the TRAIN residual target only.
    print("-" * 68)
    rng = np.random.default_rng(seed)
    shufMaes, shufR2s = [], []
    for t in range(trials):
        perm = rng.permutation(len(yTrainTarget))
        yShuf = yTrainTarget[perm]
        mae, r2 = _fitEval(XTrain, yShuf, XTest, yTestTarget)
        shufMaes.append(mae)
        shufR2s.append(r2)
        print(f"SHUFFLED #{t + 1:<2}    ->  MAE={mae:.4f}   R2={r2:+.4f}   "
              f"MAE vs baseline={baselineMae - mae:+.4f}")

    shufMaes = np.array(shufMaes)
    shufR2s = np.array(shufR2s)

    print("-" * 68)
    print(f"shuffled MAE  : mean={shufMaes.mean():.4f}  std={shufMaes.std():.4f}")
    print(f"shuffled R2   : mean={shufR2s.mean():+.4f}  max={shufR2s.max():+.4f}")
    print("=" * 68)

    # 4. Verdict. Thresholds are deliberately loose — noise, not a hard test.
    #    A leaking model would show shuffled R2 well above 0 or a shuffled MAE
    #    meaningfully below the naive baseline.
    r2Suspicious = shufR2s.max() > 0.02
    maeSuspicious = (baselineMae - shufMaes.mean()) > 0.02 * baselineMae

    realHasSignal = (realR2 > 0.01) and (realMae < baselineMae)

    if r2Suspicious or maeSuspicious:
        print("VERDICT: LEAKAGE SUSPECTED")
        print("  Shuffled-label model beat the naive baseline. On randomized")
        print("  targets the features should explain nothing. Inspect features")
        print("  for target bleed, id/index proxies, or train/test contamination.")
    elif not realHasSignal:
        print("VERDICT: INCONCLUSIVE")
        print("  Shuffled labels look clean, but the REAL-label model also fails")
        print("  to beat the baseline on the residual target, so this run can't")
        print("  demonstrate the test would catch leakage. Check the feature set.")
    else:
        print("VERDICT: PASS — no leakage signal")
        print("  Real labels carry signal; shuffled labels collapse to the naive")
        print("  baseline (R2 ~ 0). Consistent with a leak-free feature matrix.")
    print("=" * 68)

    return {
        "baseline_mae": baselineMae,
        "real_mae": realMae,
        "real_r2": realR2,
        "shuffled_mae_mean": float(shufMaes.mean()),
        "shuffled_r2_mean": float(shufR2s.mean()),
        "shuffled_r2_max": float(shufR2s.max()),
    }


def main():
    ap = argparse.ArgumentParser(description="Randomized-labels leakage check")
    ap.add_argument("--trials", type=int, default=5,
                    help="number of label-shuffle trials (default 5)")
    ap.add_argument("--holdout", type=float, default=HOLDOUT_RATIO,
                    help=f"chronological holdout ratio (default {HOLDOUT_RATIO})")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    runLeakageCheck(trials=args.trials, holdoutRatio=args.holdout, seed=args.seed)


if __name__ == "__main__":
    main()
