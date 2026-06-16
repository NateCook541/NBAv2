import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor

from config import (
    MODEL_PATH, MODEL_META_PATH,
    POINTS_MODEL_PARAMS, HOLDOUT_RATIO,
    MIN_AVGPTS_CAL, MIN_AVGMIN_CAL,
    POINTS_TARGET_MODE, PREDICTION_CLIP_K,
    USE_RECENCY_WEIGHTS, RECENCY_WEIGHT_MIN, RECENCY_WEIGHT_MAX,
    BIAS_SHRINK_K
)

POINTS_BUNDLE_VERSION = 2


# Helpers

# Provides a test / train split that prevents data leakage
def _splitChronologically(X, y, dates, holdoutRatio=HOLDOUT_RATIO):
    if len(X) < 10:
        raise ValueError("Not enough rows for a chronological split")

    split = max(1, min(int(len(X) * (1 - holdoutRatio)), len(X) - 1))

    return (
            X.iloc[:split], X.iloc[split:],
            y.iloc[:split], y.iloc[split:],
            dates.iloc[:split], dates.iloc[split:],
    )

def _applyPropPlayerFilter(X, y, dates, minAvgPtsCal):
    mask = (X["avgPts10"] >= minAvgPtsCal) & (X["avgMin10"] >= MIN_AVGMIN_CAL)
    return (
            X[mask].reset_index(drop=True),
            y[mask].reset_index(drop=True),
            dates[mask].reset_index(drop=True)
    )


def _applyPredictionClip(predictions, X, clipK):
    pred = np.asarray(predictions, dtype=float)
    if clipK is None or clipK <= 0:
        return np.where(np.isfinite(pred), pred, 0.0)
    if "avgPts10" not in X.columns or "ptsStd10" not in X.columns:
        return np.where(np.isfinite(pred), pred, 0.0)

    center = X["avgPts10"].to_numpy(dtype=float)
    spread = np.maximum(2.0, X["ptsStd10"].to_numpy(dtype=float))
    center = np.where(np.isfinite(center), center, pred)
    spread = np.where(np.isfinite(spread), spread, 2.0)
    lower = np.maximum(0.0, center - (clipK * spread))
    upper = center + (clipK * spread)
    clipped = np.clip(pred, lower, upper)
    return np.where(np.isfinite(clipped), clipped, 0.0)


def _applyBiasCorrection(predictions, X, biasMeta):
    if not biasMeta:
        return np.asarray(predictions, dtype=float)

    corrected = np.asarray(predictions, dtype=float) + float(biasMeta.get("global_bias", 0.0))
    bucketBias = biasMeta.get("bucket_bias", {})
    if not bucketBias or "avgPts10" not in X.columns:
        return corrected

    avg = X["avgPts10"].to_numpy(dtype=float)
    lowMask = avg < 12
    midMask = (avg >= 12) & (avg < 20)
    highMask = avg >= 20
    corrected[lowMask] += float(bucketBias.get("lt12", 0.0))
    corrected[midMask] += float(bucketBias.get("12to20", 0.0))
    corrected[highMask] += float(bucketBias.get("gte20", 0.0))
    return corrected

def _computeBiasMeta(rawPredictions, actuals, X, shrinkK=BIAS_SHRINK_K):
    raw = np.asarray(rawPredictions, dtype=float)
    actual = np.asarray(actuals, dtype=float)
    residuals = actual - raw

    globalBias = float(residuals.mean())

    avg = X["avgPts10"].to_numpy(dtype=float)
    lowMask = avg < 12
    midMask = (avg >= 12) & (avg < 20)
    highMask = avg >= 20

    def _shrunk(mask):
        n = int(mask.sum())
        if n == 0:
            return 0.0, 0
        bucketMean = float(residuals[mask].mean())
        weight = n / (n + shrinkK)
        return weight * bucketMean + (1 - weight) * globalBias, n

    lt12Bias, lt12N = _shrunk(lowMask)
    midBias, midN = _shrunk(midMask)
    highBias, highN = _shrunk(highMask)

    print(
        f"[PointsBundle] Bias shrink (k={shrinkK}): "
        f"lt12 n={lt12N} -> {lt12Bias:+.3f}  "
        f"12to20 n={midN} -> {midBias:+.3f}  "
        f"gte20 n={highN} -> {highBias:+.3f}"
    )

    return {
        "global_bias": globalBias,
        "bucket_bias": {"lt12": lt12Bias, "12to20": midBias, "gte20": highBias},
    } 


# Public bundle class


class PointsBundle:
    """
    Wraps the trained XGBoost points model with its metadata

    After training the calibration split is exposed seperately so the betting layer can fit a calibrator without re splitting

    Attributes:
    model           : XGBRegressor
    meta            : dict
    calPredictions  : np array
    calActuals      : pd series
    calDates        : pd series
    """

    def __init__(self, model, meta, calPredictions=None,
                 calActuals=None, calDates=None):
        self.model = model
        self.meta = meta
        self.calPredictions = calPredictions
        self.calActuals = calActuals
        self.calDates = calDates


    # Prediction

    def predict(self, features):
        pred = self.predictBatch(features)
        return float(pred[0])
    

    def predictBatch(self, features):
        raw = self.model.predict(features)
        mode = self.meta.get("target_mode", "absolute")
        if mode == "residual":
            baseline = features["avgPts10"].to_numpy(dtype=float)
            baseline = np.where(np.isfinite(baseline), baseline, 0.0)
            raw = raw + baseline
        raw = _applyBiasCorrection(raw, features, self.meta.get("bias_correction", {}))
        clipK = float(self.meta.get("prediction_clip_k", 0.0))
        return _applyPredictionClip(raw, features, clipK)

    def featureImportance(self):
        # Prefer trained feature names when available.
        cols = getattr(self.model, "feature_names_in_", None)
        if cols is None:
            cols = list(range(len(self.model.feature_importances_)))
        return (
                pd.DataFrame({
                    "feature": cols,
                    "importance": self.model.feature_importances_,
                })
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
        )


# Persistence


    def save(self, modelPath=MODEL_PATH, metaPath=MODEL_META_PATH):
        joblib.dump(self.model, modelPath)
        joblib.dump(self.meta, metaPath)
        print(f"[PointsBundle] Saved model - {modelPath}")

    @classmethod
    def load(cls, modelPath=MODEL_PATH, metaPath=MODEL_META_PATH):
        if not Path(modelPath).exists():
            raise FileNotFoundError(f"No points model found at {modelPath}")

        model = joblib.load(modelPath)
        meta = joblib.load(metaPath) if Path(metaPath).exists() else {}

        return cls(model, meta)

    @classmethod
    def loadIfExists(cls):
        if not Path(MODEL_PATH).exists():
            return None
        return cls.load()


    # Training


    @classmethod
    def train(cls, X, y, dates, save=True, runMetrics=False, minAvgPtsCal=15):
        """
        Trains from a pre-built feature matrix (X, y, dates)

        Steps
        1. Drop rows with no scoring history (avgPts10 == 0).
        2. Chronological train / cal split (80/20 by default).
        3. Split cal further: first half → bias estimation, second half → calibrator fitting
        4. Apply prop player filter to both cal halves independently.
        5. Fit XGBoost on training split.
        6. Compute bias from the bias-estimation half (raw predictions only).
        7. Apply full inference pipeline to calibrator half → these are the
            predictions the Calibrator class will fit its t-dist + Platt on.
        8. Attach cal predictions + actuals to bundle for calibrator fitting
        """
        
        # 1. Remove rows with no scoring history
        
        mask = X["avgPts10"] > 0 
        X = X[mask].reset_index(drop=True)
        y = y[mask].reset_index(drop=True)
        dates = dates[mask].reset_index(drop=True)

        # 2. Chrono split
        
        XTrain, XHoldout, yTrain, yHoldout, trainDates, holdoutDates = (
                _splitChronologically(X, y, dates)
        )

        # Split holdout in half from chrono split data
        # First half is sent to bias estiamtion
        # Second half is sent for fitting

        splitIdx = max(1, len(XHoldout) // 2)
        XBias = XHoldout.iloc[:splitIdx].reset_index(drop=True)
        yBias = yHoldout.iloc[:splitIdx].reset_index(drop=True)
        XCal = XHoldout.iloc[splitIdx:].reset_index(drop=True)
        yCal = yHoldout.iloc[splitIdx:].reset_index(drop=True)
        calDates = holdoutDates.iloc[splitIdx:].reset_index(drop=True)

        # 4. Prop player filter on both holdout halves

        XBias, yBias, _ = _applyPropPlayerFilter(
                XBias, yBias,
                holdoutDates.iloc[:splitIdx].reset_index(drop=True),
                minAvgPtsCal=MIN_AVGPTS_CAL
        )
        XCal, yCal, calDates = _applyPropPlayerFilter(
                XCal, yCal, calDates,
                minAvgPtsCal=MIN_AVGPTS_CAL
        )

        if XBias.empty:
            raise ValueError(
                    "Bias estimation split is empty after filtering"
                    "Lower MIN_AVGPTS_CAL or use more data"
            )
        if XCal.empty:
            raise ValueError(
                    "Calibration estimation split is empty after filtering"
                    "Lower MIN_AVGPTS_CAL or use more data"
            )
        print(
                f"[PointsBundle] Cal set after prop filter: "
                f"{len(yCal)} rows mean actual: {yCal.mean():.1f}"
        )

        # 5. Fit the model

        model = XGBRegressor(**POINTS_MODEL_PARAMS)
        targetMode = str(POINTS_TARGET_MODE).lower().strip()
        if targetMode not in ("residual", "absolute"):
            raise ValueError(f"Unsupported POINTS_TARGET_MODE: {POINTS_TARGET_MODE}")

        if targetMode == "residual":
            yTrainTarget = yTrain - XTrain["avgPts10"]
        else:
            yTrainTarget = yTrain

        if USE_RECENCY_WEIGHTS and len(XTrain) > 1:
            sampleWeights = np.linspace(RECENCY_WEIGHT_MIN, RECENCY_WEIGHT_MAX, len(XTrain))
            model.fit(XTrain, yTrainTarget, sample_weight=sampleWeights)
        else:
            model.fit(XTrain, yTrainTarget)


        # 6. Compute bias from the bias half

        if targetMode == "residual":
            rawBiasPred = model.predict(XBias) + XBias["avgPts10"].to_numpy(dtype=float)
        else:
            rawBiasPred = model.predict(XBias)

        biasMeta = _computeBiasMeta(rawBiasPred, yBias, XBias) 

        print(
            f"[PointsBundle] Bias meta: "
            f"global={biasMeta['global_bias']:+.3f}  "
            f"buckets={biasMeta['bucket_bias']}"
        )
 
        # 7. Apply full inference pipeline to calibrator half
        
        if targetMode == "residual":
            rawCalPred = model.predict(XCal) + XCal["avgPts10"].to_numpy(dtype=float)
        else:
            rawCalPred = model.predict(XCal)
 
        calPredictions = _applyBiasCorrection(rawCalPred, XCal, biasMeta)
        calPredictions = _applyPredictionClip(calPredictions, XCal, PREDICTION_CLIP_K)
 
        print(
            f"[PointsBundle] Train rows: {len(XTrain)}  "
            f"Cal rows (filtered): {len(XCal)}"
        )
        print(
            f"[PointsBundle] Cal mean actual: {yCal.mean():.2f}  "
            f"mean predicted: {calPredictions.mean():.2f}"
        )

        if runMetrics:
            from models.evaluate import evaluateModel
            predictions = evaluateModel(
                model, XCal, yCal,
                targetMode=targetMode,
                clipK=PREDICTION_CLIP_K,
                biasMeta=biasMeta,
            )

        meta = {
                "train_start_date": trainDates.iloc[0],
                "train_last_date": trainDates.iloc[-1],
                "calibration_start_date": calDates.iloc[0],
                "calibration_end_date": calDates.iloc[-1],
                "train_rows": int(len(XTrain)),
                "calibration_rows": int(len(XCal)),
                "target_mode": targetMode,
                "points_bundle_version": POINTS_BUNDLE_VERSION,
                "prediction_clip_k": float(PREDICTION_CLIP_K),
                "bias_correction": {"global_bias": 0.0, "bucket_bias": {}},
                "recency_weights": bool(USE_RECENCY_WEIGHTS),
        }

        bundle = cls(
                model = model,
                meta = meta,
                calPredictions = calPredictions,
                calActuals = yCal,
                calDates = calDates
        )

        if save:
            bundle.save()

        return bundle


    # Backtest safety


    # Needed to check if safe for the backtest testing
    def isSafeFor(self, backtestStartDate):
        if int(self.meta.get("points_bundle_version", 0)) < POINTS_BUNDLE_VERSION:
            return False
        end = self.meta.get("train_end_date")
        if not end:
            last = self.meta.get("train_last_date", "")
            return str(last) < str(backtestStartDate)

        return end <= backtestStartDate
