import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from pathlib import Path

from config import (
    POINTS_MODEL_PARAMS, POINTS_TARGET_MODE, PREDICTION_CLIP_K,
    PREDICTION_CLIP_CANDIDATES, WALKFORWARD_SPLITS, WALKFORWARD_MIN_TRAIN_ROWS,
    CLIP_LOWER_K_CANDIDATES, CLIP_UPPER_K_CANDIDATES, CLIP_TAIL_PENALTY,
    WALKFORWARD_MAX_NEG_DELTA, WALKFORWARD_ALERT_DELTA,
    WF_TUNE_RATIO, USE_WF_RECENCY_WEIGHTS, USE_RECENCY_WEIGHTS,
    RECENCY_WEIGHT_MIN, RECENCY_WEIGHT_MAX,
    ENABLE_BASELINE_BLEND, ENABLE_REGIME_ALPHA, ALPHA_MIN, ALPHA_MAX,
    CHAOS_Q_MINSTD, CHAOS_Q_INJURY, TUNE_SCORE_LAMBDA_P95, TUNE_SCORE_LAMBDA_NEG_DELTA,
    TUNE_SCORE_LAMBDA_NEG_DELTA_SQ, TUNE_MIN_DELTA_FOR_MODEL, ALPHA_CHAOS_MAX_WHEN_WEAK,
    ALPHA_SHRINK_TO_BASELINE, ALPHA_MAX_WHEN_NO_CLIP,
)


def _printDirectionalDiagnostics(df):
    df = df.copy()
    df["line_error"] = df["pred"] - df["avg10"]
    df["actual_delta"] = df["actual"] - df["avg10"]
    df["pred_over"] = df["line_error"] > 0
    df["actual_over"] = df["actual_delta"] > 0
    direction_acc = float((df["pred_over"] == df["actual_over"]).mean())
    print(f"Directional accuracy vs avgPts10 line: {direction_acc:.3f}")

    df["abs_edge"] = df["line_error"].abs()
    df["edge_bucket"] = pd.cut(
        df["abs_edge"], bins=[0, 1, 2, 3, 99], labels=["0-1", "1-2", "2-3", "3+"]
    )
    grouped = df.groupby("edge_bucket", observed=True).agg(
        n=("actual", "count"),
        direction_acc=("pred_over", lambda s: float((s == df.loc[s.index, "actual_over"]).mean())),
        mae=("abs_error", "mean"),
    )
    print("\nDiagnostics by abs edge bucket:")
    print(grouped.to_string())


def _applyBiasCorrection(predictions, XTest, biasMeta):
    if not biasMeta:
        return np.asarray(predictions, dtype=float)
    corrected = np.asarray(predictions, dtype=float) + float(biasMeta.get("global_bias", 0.0))
    bucketBias = biasMeta.get("bucket_bias", {})
    if not bucketBias or "avgPts10" not in XTest.columns:
        return corrected
    avg = XTest["avgPts10"].to_numpy(dtype=float)
    corrected[avg < 12] += float(bucketBias.get("lt12", 0.0))
    corrected[(avg >= 12) & (avg < 20)] += float(bucketBias.get("12to20", 0.0))
    corrected[avg >= 20] += float(bucketBias.get("gte20", 0.0))
    return corrected


def _finalizePredictions(rawPred, XTest, targetMode, clipK, biasMeta=None, clipUpperK=None):
    pred = np.asarray(rawPred, dtype=float)
    if targetMode == "residual":
        baseline = XTest["avgPts10"].to_numpy(dtype=float)
        baseline = np.where(np.isfinite(baseline), baseline, 0.0)
        pred = pred + baseline
    pred = _applyBiasCorrection(pred, XTest, biasMeta)
    if clipK and clipK > 0 and "ptsStd10" in XTest.columns:
        center = XTest["avgPts10"].to_numpy(dtype=float)
        spread = np.maximum(2.0, XTest["ptsStd10"].to_numpy(dtype=float))
        center = np.where(np.isfinite(center), center, pred)
        spread = np.where(np.isfinite(spread), spread, 2.0)
        upperK = clipK if clipUpperK is None else clipUpperK
        lower = np.maximum(0.0, center - clipK * spread)
        upper = center + upperK * spread
        pred = np.clip(pred, lower, upper)
    pred = np.where(np.isfinite(pred), pred, 0.0)
    return pred


def evaluateModel(model, XTest, yTest, targetMode="absolute", clipK=0.0, biasMeta=None):
    predictions = _finalizePredictions(
        model.predict(XTest), XTest, targetMode, clipK, biasMeta=biasMeta
    )
    residuals = yTest.values - predictions

    mae = mean_absolute_error(yTest, predictions)
    rmse = np.sqrt(mean_squared_error(yTest, predictions))
    r2 = r2_score(yTest, predictions)
    bias = np.mean(predictions - yTest)

    print("\nEvaluation Metrics")
    print("------------------")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R2   : {r2:.3f}")
    print(f"Bias : {bias:.2f}")
    print(f"Mean residual: {residuals.mean():.3f}")
    print(f"Std residual : {residuals.std():.3f}")

    baseline = XTest["avgPts10"]
    baseline_mae = mean_absolute_error(yTest, baseline)
    print(f"\nBaseline (avgPts10) MAE: {baseline_mae:.2f}")

    debug = pd.DataFrame({"actual": yTest, "pred": predictions, "avg10": baseline})
    debug["error"] = debug["pred"] - debug["actual"]
    debug["abs_error"] = debug["error"].abs()

    print("\nWorst predictions:")
    print(debug.sort_values("abs_error", ascending=False).head(20).to_string(index=False))
    
    # TEMP
    print("\nMAE by avgPts10 bucket:")
    buckets = [(0,12,"<12"), (12,15,"12-15"), (15,18,"15-18"), 
           (18,22,"18-22"), (22,99,"22+")]
    for lo, hi, label in buckets:
        mask = (XTest["avgPts10"] >= lo) & (XTest["avgPts10"] < hi)
        if mask.sum() < 10:
            continue
        bucketPred = predictions[mask]
        bucketActual = yTest[mask].to_numpy()
        mae = float(np.mean(np.abs(bucketActual - bucketPred)))
        baseline_mae = float(np.mean(np.abs(
            bucketActual - XTest.loc[mask, "avgPts10"].to_numpy()
        )))
        print(f"  {label:<8} n={mask.sum():>5}  MAE={mae:.3f}  "
            f"baseline={baseline_mae:.3f}  delta={baseline_mae-mae:+.3f}")

        resid = bucketActual - bucketPred
        print(f"resid mean={resid.mean():+.3f}  "
            f"p10={np.percentile(resid,10):.1f}  "
            f"p90={np.percentile(resid,90):.1f}")

    _printDirectionalDiagnostics(debug)
    
    # Feature Importance
    if hasattr(model, "feature_importances_"):
        importance = pd.Series(
            model.feature_importances_,
            index=XTest.columns
        ).sort_values(ascending=False)

        print("\nTop 30 Feature Importances:")
        print(importance.to_string())
   
    return predictions


def _fitBiasMeta(rawPred, XRef, yRef):
    residual = yRef.to_numpy(dtype=float) - np.asarray(rawPred, dtype=float)
    avg = XRef["avgPts10"].to_numpy(dtype=float)
    return {
        "global_bias": float(residual.mean()),
        "bucket_bias": {
            "lt12": float(residual[avg < 12].mean()) if np.any(avg < 12) else 0.0,
            "12to20": float(residual[(avg >= 12) & (avg < 20)].mean()) if np.any((avg >= 12) & (avg < 20)) else 0.0,
            "gte20": float(residual[avg >= 20].mean()) if np.any(avg >= 20) else 0.0,
        },
    }

def _gridAlpha(modelPred, baseline, actual, alphaMin, alphaMax):
    alphas = np.linspace(alphaMin, alphaMax, 10)
    bestAlpha = float(alphaMax)
    bestMae = float("inf")
    for a in alphas:
        blended = a * modelPred + (1.0 - a) * baseline
        mae = mean_absolute_error(actual, blended)
        if mae < bestMae:
            bestMae = mae
            bestAlpha = float(a)
    return bestAlpha, bestMae

def _buildChaosMask(XRef, minStdThresh, injuryThresh):
    if "minStd10" not in XRef.columns or "injury_opportunity" not in XRef.columns:
        return np.zeros(len(XRef), dtype=bool)
    minStd = XRef["minStd10"].to_numpy(dtype=float)
    injury = XRef["injury_opportunity"].to_numpy(dtype=float)
    return (minStd >= minStdThresh) & (injury >= injuryThresh)


def walkForwardEvaluate(X, y, splits=WALKFORWARD_SPLITS, min_train_rows=WALKFORWARD_MIN_TRAIN_ROWS):
    if len(X) != len(y):
        raise ValueError("X and y length mismatch")
    if len(X) < (min_train_rows + 200):
        print("[WalkForward] Not enough rows for robust walk-forward evaluation.")
        return

    fold_size = max(200, (len(X) - min_train_rows) // splits)
    rows = []

    clipPairs = {(float(k), float(k)) for k in PREDICTION_CLIP_CANDIDATES}
    for lowerK in CLIP_LOWER_K_CANDIDATES:
        for upperK in CLIP_UPPER_K_CANDIDATES:
            clipPairs.add((float(lowerK), float(upperK)))
    clipPairs = sorted(clipPairs)

    prevFold = None
    for i in range(splits):
        train_end = min_train_rows + i * fold_size
        test_end = min(len(X), train_end + fold_size)
        if test_end - train_end < 100:
            continue

        X_train_full = X.iloc[:train_end].reset_index(drop=True)
        y_train_full = y.iloc[:train_end].reset_index(drop=True)
        X_test = X.iloc[train_end:test_end]
        y_test = y.iloc[train_end:test_end]

        tuneRows = max(500, int(len(X_train_full) * WF_TUNE_RATIO))
        tuneRows = min(tuneRows, len(X_train_full) - 500)
        if tuneRows <= 0:
            continue
        splitPoint = len(X_train_full) - tuneRows
        X_train = X_train_full.iloc[:splitPoint]
        y_train = y_train_full.iloc[:splitPoint]
        X_tune = X_train_full.iloc[splitPoint:]
        y_tune = y_train_full.iloc[splitPoint:]

        model = XGBRegressor(**POINTS_MODEL_PARAMS)
        targetMode = str(POINTS_TARGET_MODE).lower().strip()
        if targetMode == "residual":
            y_train_target = y_train - X_train["avgPts10"]
        else:
            y_train_target = y_train

        useWeights = USE_WF_RECENCY_WEIGHTS and USE_RECENCY_WEIGHTS and len(X_train) > 1
        if useWeights:
            w = np.linspace(RECENCY_WEIGHT_MIN, RECENCY_WEIGHT_MAX, len(X_train))
            model.fit(X_train, y_train_target, sample_weight=w)
        else:
            model.fit(X_train, y_train_target)

        trainRaw = model.predict(X_train)
        if targetMode == "residual":
            trainRaw = trainRaw + X_train["avgPts10"].to_numpy(dtype=float)
        biasMeta = _fitBiasMeta(trainRaw, X_train, y_train)

        rawTunePred = model.predict(X_tune)
        predTuneNoClip = _finalizePredictions(
            rawTunePred, X_tune, targetMode=targetMode, clipK=0.0, biasMeta=biasMeta
        )
        tuneBaseline = X_tune["avgPts10"].to_numpy(dtype=float)
        tuneBaselineMae = mean_absolute_error(y_tune, tuneBaseline)

        minStdThresh = float(X_train["minStd10"].quantile(CHAOS_Q_MINSTD)) if "minStd10" in X_train.columns else np.inf
        injuryThresh = float(X_train["injury_opportunity"].quantile(CHAOS_Q_INJURY)) if "injury_opportunity" in X_train.columns else np.inf
        tuneChaosMask = _buildChaosMask(X_tune, minStdThresh, injuryThresh)
        testChaosMask = _buildChaosMask(X_test, minStdThresh, injuryThresh)
        chaosRate = float(testChaosMask.mean()) if len(testChaosMask) else 0.0

        clipScores = []
        for lowerK, upperK in clipPairs:
            predClipTune = _finalizePredictions(
                rawTunePred,
                X_tune,
                targetMode=targetMode,
                clipK=float(lowerK),
                clipUpperK=float(upperK),
                biasMeta=biasMeta,
            )
            absErr = np.abs(y_tune.to_numpy(dtype=float) - predClipTune)
            modelMae = mean_absolute_error(y_tune, predClipTune)
            negDeltaPenalty = max(0.0, modelMae - tuneBaselineMae)
            p95 = float(np.quantile(absErr, 0.95))
            p99 = float(np.quantile(absErr, 0.99))
            score = (
                modelMae
                + TUNE_SCORE_LAMBDA_P95 * p95
                + CLIP_TAIL_PENALTY * p99
                + TUNE_SCORE_LAMBDA_NEG_DELTA * negDeltaPenalty
                + TUNE_SCORE_LAMBDA_NEG_DELTA_SQ * (negDeltaPenalty ** 2)
            )
            clipHitRate = float(np.mean(np.abs(predTuneNoClip - predClipTune) > 1e-9))
            chaosMae = mean_absolute_error(y_tune.iloc[np.where(tuneChaosMask)[0]], predClipTune[tuneChaosMask]) if np.any(tuneChaosMask) else modelMae
            normalMae = mean_absolute_error(y_tune.iloc[np.where(~tuneChaosMask)[0]], predClipTune[~tuneChaosMask]) if np.any(~tuneChaosMask) else modelMae
            clipScores.append((float(lowerK), float(upperK), modelMae, p99, score, clipHitRate, chaosMae, normalMae))

        bestLowerK, bestUpperK, _, _, _, _, _, _ = min(clipScores, key=lambda t: t[4])

        rawTestPred = model.predict(X_test)
        bestPred = _finalizePredictions(
            rawTestPred,
            X_test,
            targetMode=targetMode,
            clipK=float(bestLowerK),
            clipUpperK=float(bestUpperK),
            biasMeta=biasMeta,
        )

        alpha = 1.0
        alphaNormal = 1.0
        alphaChaos = 1.0
        baselineTest = X_test["avgPts10"].to_numpy(dtype=float)
        fallbackUsed = False
        tuneDeltaNormal = 0.0
        tuneDeltaChaos = 0.0
        if ENABLE_BASELINE_BLEND:
            predTuneBest = _finalizePredictions(
                rawTunePred,
                X_tune,
                targetMode=targetMode,
                clipK=float(bestLowerK),
                clipUpperK=float(bestUpperK),
                biasMeta=biasMeta,
            )
            tuneModelMae = mean_absolute_error(y_tune, predTuneBest)
            tuneDeltaOverall = tuneBaselineMae - tuneModelMae

            if np.any(~tuneChaosMask):
                tuneDeltaNormal = float(
                    mean_absolute_error(y_tune.iloc[np.where(~tuneChaosMask)[0]], tuneBaseline[~tuneChaosMask]) -
                    mean_absolute_error(y_tune.iloc[np.where(~tuneChaosMask)[0]], predTuneBest[~tuneChaosMask])
                )
            if np.any(tuneChaosMask):
                tuneDeltaChaos = float(
                    mean_absolute_error(y_tune.iloc[np.where(tuneChaosMask)[0]], tuneBaseline[tuneChaosMask]) -
                    mean_absolute_error(y_tune.iloc[np.where(tuneChaosMask)[0]], predTuneBest[tuneChaosMask])
                )

            alphaNormal, _ = _gridAlpha(
                predTuneBest, X_tune["avgPts10"].to_numpy(dtype=float), y_tune, ALPHA_MIN, ALPHA_MAX
            )
            # Conservative shrinkage to reduce tune-tail overfitting.
            alphaNormal = (1.0 - ALPHA_SHRINK_TO_BASELINE) * alphaNormal + ALPHA_SHRINK_TO_BASELINE * ALPHA_MIN
            alpha = alphaNormal
            if tuneDeltaOverall < TUNE_MIN_DELTA_FOR_MODEL:
                fallbackUsed = True
                alphaNormal = ALPHA_MIN

            if ENABLE_REGIME_ALPHA and "injury_opportunity" in X_test.columns and "minStd10" in X_test.columns:
                if np.any(tuneChaosMask):
                    alphaChaos, _ = _gridAlpha(
                        predTuneBest[tuneChaosMask],
                        X_tune["avgPts10"].to_numpy(dtype=float)[tuneChaosMask],
                        y_tune.iloc[np.where(tuneChaosMask)[0]],
                        ALPHA_MIN,
                        alphaNormal,
                    )
                else:
                    alphaChaos = alphaNormal
                alphaChaos = (1.0 - ALPHA_SHRINK_TO_BASELINE) * alphaChaos + ALPHA_SHRINK_TO_BASELINE * ALPHA_MIN
                if tuneDeltaChaos < TUNE_MIN_DELTA_FOR_MODEL:
                    fallbackUsed = True
                    alphaChaos = min(alphaChaos, ALPHA_CHAOS_MAX_WHEN_WEAK)

                # If no clipping was selected, enforce stronger conservatism.
                if bestLowerK == 0.0 and bestUpperK == 0.0:
                    fallbackUsed = True
                    alphaNormal = min(alphaNormal, ALPHA_MAX_WHEN_NO_CLIP)
                    alphaChaos = min(alphaChaos, min(ALPHA_CHAOS_MAX_WHEN_WEAK, ALPHA_MAX_WHEN_NO_CLIP))

                rowAlpha = np.where(testChaosMask, alphaChaos, alphaNormal).astype(float)
                bestPred = rowAlpha * bestPred + (1.0 - rowAlpha) * baselineTest
            else:
                if bestLowerK == 0.0 and bestUpperK == 0.0:
                    fallbackUsed = True
                    alphaNormal = min(alphaNormal, ALPHA_MAX_WHEN_NO_CLIP)
                bestPred = alphaNormal * bestPred + (1.0 - alphaNormal) * baselineTest

        bestModelMae = mean_absolute_error(y_test, bestPred)
        bestP99 = float(np.quantile(np.abs(y_test.to_numpy(dtype=float) - bestPred), 0.99))
        predTestNoClip = _finalizePredictions(
            rawTestPred, X_test, targetMode=targetMode, clipK=0.0, biasMeta=biasMeta
        )
        bestClipHitRate = float(np.mean(np.abs(predTestNoClip - bestPred) > 1e-9))
        base = X_test["avgPts10"].values
        base_mae = mean_absolute_error(y_test, base)
        delta = base_mae - bestModelMae
        foldAlert = "ALERT" if delta < WALKFORWARD_ALERT_DELTA else ""
        foldResidual = y_test.to_numpy(dtype=float) - bestPred

        driftScore = 0.0
        if prevFold is not None:
            driftScore = abs(prevFold["mean_avgPts10"] - float(X_test["avgPts10"].mean()))
            if "injury_opportunity" in X_test.columns:
                driftScore += abs(prevFold["injury_opp_rate"] - float((X_test["injury_opportunity"] > 0).mean()))

        rows.append({
            "fold": i + 1,
            "train_rows": len(X_train_full),
            "train_core_rows": len(X_train),
            "tuned_on_rows": len(X_tune),
            "test_rows": len(X_test),
            "mae_model": bestModelMae,
            "mae_avg10": base_mae,
            "delta_mae": delta,
            "alpha_selected": float(alpha),
            "alpha_normal": float(alphaNormal),
            "alpha_chaos": float(alphaChaos),
            "chaos_rate": float(chaosRate),
            "tune_delta_normal": float(tuneDeltaNormal),
            "tune_delta_chaos": float(tuneDeltaChaos),
            "fallback_used": bool(fallbackUsed),
            "best_clip_lower_k": bestLowerK,
            "best_clip_upper_k": bestUpperK,
            "tail_abs_err_p99": bestP99,
            "clip_hit_rate": bestClipHitRate,
            "mean_avgPts10": float(X_test["avgPts10"].mean()) if "avgPts10" in X_test.columns else np.nan,
            "mean_minStd10": float(X_test["minStd10"].mean()) if "minStd10" in X_test.columns else np.nan,
            "injury_opp_rate": float((X_test["injury_opportunity"] > 0).mean()) if "injury_opportunity" in X_test.columns else np.nan,
            "resid_p10": float(np.quantile(foldResidual, 0.10)),
            "resid_p50": float(np.quantile(foldResidual, 0.50)),
            "resid_p90": float(np.quantile(foldResidual, 0.90)),
            "drift_vs_prev_fold": float(driftScore),
            "fold_alert": foldAlert,
            "win_vs_baseline": bool(delta > 0),
        })
        prevFold = rows[-1]

    if not rows:
        print("[WalkForward] No valid folds produced.")
        return

    out = pd.DataFrame(rows)
    print("\nWalk-forward summary")
    print("--------------------")
    print(out.to_string(index=False))
    winRate = float(out["win_vs_baseline"].mean())
    worstDelta = float(out["delta_mae"].min())
    clipSummary = out.groupby(["best_clip_lower_k", "best_clip_upper_k"], observed=True).size().to_dict()
    maxNegGate = bool((out["delta_mae"] >= WALKFORWARD_MAX_NEG_DELTA).all())
    print(
        f"\nAvg MAE improvement vs avgPts10: {out['delta_mae'].mean():.3f} "
        f"(positive means model better)"
    )
    print(f"Folds beating baseline: {int(out['win_vs_baseline'].sum())}/{len(out)} ({winRate:.1%})")
    print(f"Worst-fold delta MAE: {worstDelta:.3f}")
    print(f"Best clip choices by fold: {clipSummary}")
    print(f"Pass max-negative-delta gate ({WALKFORWARD_MAX_NEG_DELTA:.2f}): {maxNegGate}")
    print("Leakage check: clip/alpha tuned on train tail only (no test rows used).")
    alertRows = out[out["fold_alert"] == "ALERT"]
    if not alertRows.empty:
        print("\nBad fold detector:")
        print(alertRows[["fold", "delta_mae", "mean_avgPts10", "mean_minStd10", "injury_opp_rate", "clip_hit_rate"]].to_string(index=False))
