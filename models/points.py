import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor

from config import (
    MODEL_PATH, MODEL_META_PATH,
    POINTS_MODEL_PARAMS, HOLDOUT_RATIO,
    MIN_AVGPTS_CAL, MIN_AVGMIN_CAL,
)


# Helpers

# Provides a test / train split that prevents data leakage
def _split_chronologically(X, y, dates, holdoutRatio=HOLDOUT_RATIO):
    if len(X) < 10:
        raise ValueError("Not enough rows for a chronological split")

    split = max(1, min(int(len(X) * (1 - holdoutRatio)), len(X) - 1))

    return (
            X.iloc[:split], X.iloc[split:],
            y.iloc[:split], y.iloc[split:],
            dates.iloc[:split], dates.iloc[split:],
    )

def _applyPropPlayerFilter(X, y, dates):
    mask = (X["avgPts10"] >= MIN_AVGPTS_CAL) & (X["avgMin10"] >= MIN_AVGMIN_CAL)
    return (
            X[mask].reset_index(drop=True),
            y[mask].reset_index(drop=True),
            dates[mask].reset_index(drop=True)
    )


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

    def __init__(self, model, meta, calPredictions,
                 calActuals, calDates):
        self.model = model
        self.meta = meta
        self.calPredictions = calPredictions
        self.calActuals = calActuals
        self.calDates = calDates


    # Prediction

    def predict(self, features):
        return float(self.model.predict(features)[0])
    

    def predictBatch(self, features):
        return self.model.predict(features)

    def featureImportance(self):
        # Cal actuals doubles as a feature column reference after training
        cols = (
                self.cal_actuals.index.tolist()
                if isinstance(self.calActuals, pd.DataFrame)
                else list(range(len(self.model.feature_importances_)))
        )
        return (
                pd.DataFrame({
                    "features": cols,
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
    def train(cls, X, y, dates, save, runMetrics):
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
        dates = date[mask].reset_index(drop=True)

        # 2. Chrono split
        
        XTrain, XCal, yTrain, trainDates, calDate = (
                _applyPropPlayerFilter(XCal, yCal, calDates)
        )

        # 3. Filter cal split to prop-relevant players only

        XCalFiltered, yCalFiltered, calDatesFiltered = (
            _applyPropPlayerFilter(XCal, yCal, calDates)
        )
        print(
            f"[PointsBundle] Cal set after prop filter: "
            f"{len(y_cal_filtered)} rows"
            f"mean actual: {y_cal_filtered.mean():.1f}"
        )

        # 4. Fit the model

        model = XGBRegressor(**POINTS_MODEL_PARAMS)
        model.fit(XTrain, yTrain)


        # 5. Evaluate and gather prediction on filtered cal set

        if runMetrics:
            from models.evaluate import evaluateModel
            predictions = evaluateModel(model, XCalFiltered, yCalFiltered)
        else:
            predictions = model.predict(XCalFiltered)

        print(
            f"[PointsBundle] Train rows: {len(X_train)}"
            f"Cal rows (filtered): {len(X_cal_filtered)}"
        )
        print(
            f"[PointsBundle] Cal mean actual: {y_cal_filtered.mean():.2f}"
            f"mean predicted: {predictions.mean():.2f}"
        )

        meta = {
                "train_start_date": trainDates.iloc[0],
                "train_last_date": trainDates.iloc[-1],
                "calibration_start_date": calDatesFiltered.iloc[0],
                "calibration_end_date": calDatesFiltered.iloc[-1],
                "train_rows": int(len(XTrain)),
                "calibration_row": int(len(XCalFilter)),
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
            return last < backtestStartDate

        return end <= backtestStartDate

