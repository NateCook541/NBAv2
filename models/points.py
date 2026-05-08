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
)


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
        1. Filter out rows where avgPts10 == 0
        2. Chronological train / calibration split
        3. Apply prop player filter to calibration split only 
        4. Fit XGBoost on training split
        5. Return bundle with cak split attached for calibrator fitting
        """
        
        # 1. Remove rows with no scoring history
        
        mask = X["avgPts10"] > 0 
        X = X[mask].reset_index(drop=True)
        y = y[mask].reset_index(drop=True)
        dates = dates[mask].reset_index(drop=True)

        # 2. Chrono split
        
        XTrain, XCal, yTrain, yCal, trainDates, calDates = (
                _splitChronologically(X, y, dates)
        )
        # 3. Filter cal split to prop-relevant players only

        XCalFiltered, yCalFiltered, calDatesFiltered = (
            _applyPropPlayerFilter(XCal, yCal, calDates, minAvgPtsCal)
        )
        if XCalFiltered.empty:
            raise ValueError("Calibration split is empty after filtering; lower thresholds or use more data.")
        print(
            f"[PointsBundle] Cal set after prop filter: "
            f"{len(yCalFiltered)} rows"
            f"mean actual: {yCalFiltered.mean():.1f}"
        )

        # 4. Fit the model

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


        # 5. Evaluate and gather prediction on filtered cal set

        if targetMode == "residual":
            rawCalPred = model.predict(XCalFiltered) + XCalFiltered["avgPts10"].to_numpy(dtype=float)
        else:
            rawCalPred = model.predict(XCalFiltered)

        globalBias = float((yCalFiltered.to_numpy(dtype=float) - rawCalPred).mean())
        avgCal = XCalFiltered["avgPts10"].to_numpy(dtype=float)
        residualCal = yCalFiltered.to_numpy(dtype=float) - rawCalPred
        bucketBias = {
            "lt12": float(residualCal[avgCal < 12].mean()) if np.any(avgCal < 12) else 0.0,
            "12to20": float(residualCal[(avgCal >= 12) & (avgCal < 20)].mean()) if np.any((avgCal >= 12) & (avgCal < 20)) else 0.0,
            "gte20": float(residualCal[avgCal >= 20].mean()) if np.any(avgCal >= 20) else 0.0,
        }
        biasMeta = {"global_bias": globalBias, "bucket_bias": bucketBias}

        if runMetrics:
            from models.evaluate import evaluateModel
            predictions = evaluateModel(
                model,
                XCalFiltered,
                yCalFiltered,
                targetMode=targetMode,
                clipK=PREDICTION_CLIP_K,
                biasMeta=biasMeta,
            )
        else:
            predictions = rawCalPred
            predictions = _applyBiasCorrection(predictions, XCalFiltered, biasMeta)
            predictions = _applyPredictionClip(predictions, XCalFiltered, PREDICTION_CLIP_K)

        print(
            f"[PointsBundle] Train rows: {len(XTrain)}"
            f"Cal rows (filtered): {len(XCalFiltered)}"
        )
        print(
            f"[PointsBundle] Cal mean actual: {yCalFiltered.mean():.2f}"
            f"mean predicted: {predictions.mean():.2f}"
        )

        meta = {
                "train_start_date": trainDates.iloc[0],
                "train_last_date": trainDates.iloc[-1],
                "calibration_start_date": calDatesFiltered.iloc[0],
                "calibration_end_date": calDatesFiltered.iloc[-1],
                "train_rows": int(len(XTrain)),
                "calibration_rows": int(len(XCalFiltered)),
                "target_mode": targetMode,
                "prediction_clip_k": float(PREDICTION_CLIP_K),
                "bias_correction": biasMeta,
                "recency_weights": bool(USE_RECENCY_WEIGHTS),
        }

        bundle = cls(
                model = model,
                meta = meta,
                calPredictions = predictions,
                calActuals = yCalFiltered,
                calDates = calDatesFiltered
        )

        if save:
            bundle.save()

        return bundle


    # Backtest safety


    # Needed to check if safe for the backtest testing
    def isSafeFor(self, backtestStartDate):
        end = self.meta.get("train_end_date")
        if not end:
            last = self.meta.get("train_last_date", "")
            return str(last) < str(backtestStartDate)

        return end <= backtestStartDate
