import numpy as np
import pandas as pd
from models.points import PointsBundle

class TieredPointsBundle:
    def __init__(self, low, mid, high):
        self.low = low # avgPts10 < 15
        self.mid = mid # avgPts10 15-22
        self.high = high # avgPts10 > 22

    def predict(self, features):
        avg = features["avgPts10"].iloc[0]
        if avg < 15:
            return self.low.predict(features)
        elif avg < 22:
            return self.mid.predict(features)
        else:
            return self.mid.predict(features)

    @classmethod
    def train(cls, X, y, dates, save=True):
        lowMask = X["avgPts10"] < 15
        midMask = (X["avgPts10"]  >= 15) & (X["avgPts10"] < 22)
        highMask = X["avgPts10"] >= 22
        
        print(f"Tier sizes — low: {lowMask.sum()}, mid: {midMask.sum()}, high: {highMask.sum()}")
        
        low = PointsBundle.train(X[lowMask], y[lowMask], dates[lowMask], save=False, minAvgPtsCal=3)
        mid = PointsBundle.train(X[midMask], y[midMask], dates[midMask], save=False, minAvgPtsCal=12)
        high = PointsBundle.train(X[highMask], y[highMask], dates[highMask], save=False, minAvgPtsCal=18)

        combinedPredictions = np.concatenate([
            low.calPredictions,
            mid.calPredictions,
            high.calPredictions,
        ])
        combinedActuals = pd.concat([
            low.calActuals,
            mid.calActuals,
            high.calActuals,
        ]).reset_index(drop=True)
        combinedCalDates = pd.concat([
            low.calDates,
            mid.calDates,
            high.calDates,
        ]).reset_index(drop=True)

        bundle = cls(low, mid, high)
        bundle.calPredictions = combinedPredictions
        bundle.calActuals = combinedActuals
        bundle.calDates = combinedCalDates
        bundle.meta = {
            "calibration_start_date": combinedCalDates.min(),
            "calibration_end_date": combinedCalDates.max(),
            "calibration_rows": int(len(combinedActuals)),
        }

        if save:
            bundle.save()
        return bundle


    # Persistance


    def save(self, path=None):
        import joblib
        from config import MODELS_DIR

        savePath = path or (MODELS_DIR / "nba_tiered_model.joblib")
        joblib.dump({
            "low":  self.low.model,
            "mid":  self.mid.model,
            "high": self.high.model,
            "meta": self.meta,
        }, savePath)

        print(f"[TieredPointsBundle] Saved -> {savePath}")

    @classmethod
    def load(cls, path=None):
        import joblib
        from config import MODELS_DIR
        from models.points import PointsBundle
        
        loadPath = path or (MODELS_DIR / "nba_tiered_model.joblib")
        if not loadPath.exists():
            raise FileNotFoundError(f"No tiered model at {loadPath}")
        
        data = joblib.load(loadPath)
        low = PointsBundle(model=data["low"],  meta={})
        mid = PointsBundle(model=data["mid"],  meta={})
        high = PointsBundle(model=data["high"], meta={})
        bundle = cls(low, mid, high)
        bundle.meta = data.get("meta", {})
        return bundle

    @classmethod
    def loadIfExists(cls, path=None):
        from config import MODELS_DIR
        
        checkPath = path or (MODELS_DIR / "nba_tiered_model.joblib")
        if not checkPath.exists():
            return None
        return cls.load(checkPath)

    def isSafeFor(self, backtestStartDate):
        end = self.meta.get("train_end_date")
        if not end:
            last = self.meta.get("train_last_date", "")
            return last < backtestStartDate
        return end <= backtestStartDate

