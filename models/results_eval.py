"""
results_eval.py

Per-LAYER evaluation for the game-totals model. The headline MAE printed by
--train-results is misleading: the chronological split puts the line-less
2024-2026 games in the TEST fold, so that number can't see the market line's
value and mixes two populations. This harness reports each layer's own metric in
isolation, sliced by whether the game has a real market line:

  Layer 1 (point estimate) : MAE + residual std, on ALL / line-having / line-less
  Layer 2 (center bias)    : mean(pred) - mean(actual), the -1.2pt the model
                             currently never corrects
  Layer 3 (calibration)    : reliability table + Brier, on the held-out slice

Nothing here trains or saves. It rebuilds features, reproduces the exact
train/split/predict of ResultsBundle.train, and measures. Judge changes at the
layer they touch, not by final P&L.
"""
import sqlite3
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from config import DB_PATH, RESULTS_MODEL_PARAMS, RESULTS_TARGET_MODE
from features.resultsBuilder import RESULTS_FEATURES
from models.results import _buildResultsFeatures, _splitChronologically


def _resid_std(actual, pred):
    r = np.asarray(actual, dtype=float) - np.asarray(pred, dtype=float)
    return float(np.std(r))


def _slice_metrics(name, actual, pred):
    """MAE, residual std, center bias for one slice."""
    if len(actual) == 0:
        return {"slice": name, "n": 0, "mae": np.nan, "resid_std": np.nan, "bias": np.nan}
    mae = mean_absolute_error(actual, pred)
    bias = float(np.mean(pred) - np.mean(actual))  # + = model over-predicts
    return {
        "slice": name,
        "n": int(len(actual)),
        "mae": round(mae, 3),
        "resid_std": round(_resid_std(actual, pred), 3),
        "bias": round(bias, 3),
    }


def evaluateResultsLayers(caches, dbPath=DB_PATH, endDate=None, lineHavingSplit=False):
    """Train the totals model exactly as ResultsBundle.train does, then report
    each layer's metric in isolation. Returns the metrics dict and the held-out
    frame (pred/actual/has_line) for downstream calibrator eval.

    lineHavingSplit: the odds archive ends 2023-01-16 but the default 80/20 chrono
    split boundary is ~2024-04, so the default test fold has ZERO line-having games
    — the model trains on the market line but is never evaluated on it. With this
    flag we restrict to line-having games and split THOSE chronologically, so the
    test fold actually contains the market line. This is the split to tune any
    layer for the betting-relevant population.
    """
    # 1. Same game pull as ResultsBundle.train
    conn = sqlite3.connect(str(dbPath))
    games = pd.read_sql_query(
        f"""
        SELECT r.game_id, g.game_date, r.home_team_id, r.away_team_id,
               (r.home_score + r.away_score) AS total_pts
        FROM Results r JOIN Games g ON r.game_id = g.game_id
        {"WHERE g.game_date < '" + endDate + "'" if endDate else ""}
        ORDER BY g.game_date, r.game_id
        """,
        conn,
    )
    conn.close()
    if games.empty:
        raise ValueError("No completed games to evaluate on")

    # 2. Build features (carries market_total_close -> line-having flag)
    X, y, dates = _buildResultsFeatures(caches, games)

    if lineHavingSplit:
        keep = np.isfinite(X["market_total_close"].to_numpy(dtype=float))
        n_keep = int(keep.sum())
        if n_keep < 500:
            raise ValueError(f"Only {n_keep} line-having games — cannot split them")
        X = X[keep].reset_index(drop=True)
        y = y[keep].reset_index(drop=True)
        dates = dates[keep].reset_index(drop=True)
        print(f"[eval] line-having split: restricted to {n_keep} games with a market line")

    # 3. Chrono split (same helper; on the line-having subset when requested)
    XTrain, XTest, yTrain, yTest, trainDates, testDates = _splitChronologically(X, y, dates)

    targetMode = str(RESULTS_TARGET_MODE).lower().strip()
    baseTrain = XTrain["naive_total_projection"].to_numpy(dtype=float)
    baseTest = XTest["naive_total_projection"].to_numpy(dtype=float)
    yTrainTarget = (yTrain.to_numpy(dtype=float) - baseTrain) if targetMode == "residual" else yTrain

    model = XGBRegressor(**RESULTS_MODEL_PARAMS)
    model.fit(XTrain[RESULTS_FEATURES], yTrainTarget)

    testPred = model.predict(XTest[RESULTS_FEATURES])
    if targetMode == "residual":
        testPred = testPred + baseTest

    actual = yTest.to_numpy(dtype=float)
    hasLine = np.isfinite(XTest["market_total_close"].to_numpy(dtype=float))

    # ---- Layer 1 + 2: point estimate & bias, sliced by line availability ----
    rows = [
        _slice_metrics("ALL", actual, testPred),
        _slice_metrics("line-having", actual[hasLine], testPred[hasLine]),
        _slice_metrics("line-less", actual[~hasLine], testPred[~hasLine]),
    ]
    metrics = pd.DataFrame(rows)

    print("\n" + "=" * 64)
    print("RESULTS MODEL — PER-LAYER EVAL")
    print("=" * 64)
    print(f"target_mode={targetMode}  features={len(RESULTS_FEATURES)}  "
          f"train={len(XTrain)}  test={len(XTest)}")
    print(f"test dates: {testDates.iloc[0]} -> {testDates.iloc[-1]}")

    print("\n--- Layer 1/2: point estimate + center bias ---")
    print("(bias = mean_pred - mean_actual; + means model over-predicts)")
    print(metrics.to_string(index=False))
    print("\nRead: 'line-having' is the population that matters for betting. The")
    print("ALL/headline MAE is diluted by line-less games. 'bias' near 0 is the goal")
    print("— a persistent nonzero bias is a free MAE + calibration win to reclaim.")

    held = pd.DataFrame({
        "pred": testPred,
        "actual": actual,
        "has_line": hasLine,
        "date": testDates.to_numpy(),
    })
    return {"metrics": metrics, "held": held, "model": model}


def evaluateCalibrationLayer(held, calibrator):
    """Layer 3: reliability + Brier on the held-out slice, using an already-fit
    calibrator. Reports on ALL and line-having so calibration quality is judged
    on the betting-relevant population too."""
    from betting.resultsCalibrator import ResultsCalibrator  # noqa: F401 (type hint clarity)

    print("\n--- Layer 3: calibration (reliability + Brier) ---")
    for name, mask in (("ALL", np.ones(len(held), dtype=bool)),
                       ("line-having", held["has_line"].to_numpy())):
        sub = held[mask]
        if len(sub) < 50:
            print(f"[{name}] n={len(sub)} — too few to evaluate")
            continue
        brier = calibrator.brierScore(sub["pred"].to_numpy(), sub["actual"].to_numpy())
        print(f"\n[{name}] n={len(sub)}  Brier={brier:.4f}  (0.25 = coin flip)")
        table = calibrator.calibrationCheck(sub["pred"].to_numpy(), sub["actual"].to_numpy())
        print(table.to_string(index=False))
