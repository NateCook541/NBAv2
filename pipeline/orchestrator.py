# Main controller for full project

import sqlite3
from pathlib import Path

from config import (
    DB_PATH, FEATURE_CACHE_PATH,
    MODEL_PATH, CALIBRATOR_PATH, MINUTES_PATH,
    DEFAULT_EDGE_THRESH, DEFAULT_BANKROLL, DEFAULT_KELLY_FRAC,
)
from features.cache import FeatureCache, preloadCaches
from models.minutes import MinutesBundle
from models.points import PointsBundle
from betting.calibrator import Calibrator

class Pipeline:
    """
    Orchestrates the full prediction pipeline

    Is instanitated once in main and then is used to call the needed methods

    """

    def __init__(self, dbPath=DB_PATH, featureCachePath=FEATURE_CACHE_PATH):
        self.dbPath = Path(dbPath)
        self.featureCachePath = FeatureCache(featureCachePath)


    # Helpers


    def _loadOrTrainBundle(self, backtestStartDate):
        from models.tieredPoints import TieredPointsBundle
        points = TieredPointsBundle.loadIfExists()
        minutes = MinutesBundle.loadIfExists()
        calibrator = Calibrator.loadIfExists()
               
        allSafe = (
                points is not None and points.isSafeFor(backtestStartDate) and
                minutes is not None and minutes.isSafeFor(backtestStartDate) and
                calibrator is not None and calibrator.isSafeFor(backtestStartDate)
        )

        if allSafe:
            print(
                    f"Saved bundle is leakage safe for {backtestStartDate}"
                    f"\nUsing saved model"
            )
            return points, minutes, calibrator

        print(
                f"Saved bundle is not leakage safe for {backtestStartDate}"
                f"\nNot using saved model"
        )
        return self.train(endDate=backtestStartDate, save=False) 


    def _earliestPropDate(self):
        conn = sqlite3.connect(str(self.dbPath))
        res = conn.execute("SELECT MIN(game_date) FROM PROPS").fetchone()
        conn.close()

        if res and res[0]:
            return str(res[0])
        raise ValueError("No props found in database")


    # Train


    def train(self, endDate=None, save=True, runMetrics=False):
        """
        Full model train
        1. Train minutes model
        2. Build feature matrix
        3. Train points model
        4. Fit calibrator
        5. Save if option enabled
        """
        print("\n" + "=" * 55)
        print("TRAIN: Start")
        print("=" * 55)

        # 1. Minutes model

        print("\n--- Step 1. Minutes model ---")
        conn = sqlite3.connect(str(self.dbPath))
        caches = preloadCaches(conn)
        conn.close()
        
        savedMinutes = MinutesBundle.loadIfExists()
        minutesChanged = (
                savedMinutes is None or
                not savedMinutes.isSafeFor(endDate or "9999-99-99")
        )
        if minutesChanged:
            print(f"[Pipeline] Training fresh minutes bundle")
            minutes = MinutesBundle.train(
                playerLogCache = caches.playerLogCache,
                statusDF = caches.statusDF,
                posCache = caches.posCache,
                dbPath = self.dbPath,
                endDate = endDate,
                save = save
            )
        else:
            print(f"[Pipeline] Saved minutes model is current, resuing")
            minutes = savedMinutes

        # 2. Feature matrix
        print("\n--- Step 2. Feature matrix ---")
        X, y, dates = self.featureCachePath.loadOrBuild(
                endDate = endDate,
                dbPath = self.dbPath,
                minutesBundle = minutes if minutesChanged else None
        )

        # 3. Points model
        print("\n--- Step 3. Points model ---")
        
        # Using a tiered model system to train seperate models
        # based on their avg pts per 10 to adjust for diffirent
        # calibers of players in backtest preds
        from models.tieredPoints import TieredPointsBundle
        points = TieredPointsBundle.train(
                X,
                y,
                dates,
                save=save
        )

        # 4. Calibrator
        print("\n--- Step 4. Calibrator ---")
        calibrator = Calibrator.fit(
                predictions = points.calPredictions,
                actuals = points.calActuals,
                savePath = CALIBRATOR_PATH if save else None,
                metadata = {
                    "train_end_date": endDate,
                    "calibration_start_date": points.meta.get("calibration_start_date"),
                    "calibration_end_date": points.meta.get("calibration_end_date"),
                    "calibration_rows": points.meta.get("calibration_rows")
                }
        )
        
        print("\n" + "=" * 55)
        print("TRAIN: Complete")
        print("=" * 55)
        
        return points, minutes, calibrator


# Backtest
        

    def backtest(self, startDate=None, endDate=None, 
                 edgeThresh=DEFAULT_EDGE_THRESH, bankroll=DEFAULT_BANKROLL):
        """
        Run a full backtest on props data

        Will use saved model if its leakage safe (trained before start
        of prop data) and if not save will build a new model
        """
        from betting.backtest import BacktestEngine

        print("\n" + "=" * 55)
        print("BACKTEST: Start")
        print("=" * 55)

        # Find the backtest start date if not supplied
        backtestStart = startDate or self._earliestPropDate()

        points, minutes, calibrator = self._loadOrTrainBundle(
                backtestStartDate = backtestStart
        )

        engine = BacktestEngine(
                pointsBundle = points,
                minutesBundle = minutes,
                calibrator = calibrator,
                dbPath = self.dbPath
        )

        results = engine.run(
                startDate = startDate,
                endDate = endDate,
                edgeThresh = edgeThresh,
                bankroll = bankroll
        )

        print("\n" + "=" * 55)
        print("BACKTEST: Complete")
        print("=" * 55)


# Cache features


    def cacheFeatures(self):
        print("\n--- Building and saving feature cache ---")
        X, y, dates = self.featureCachePath.buildAndSave(dbPath=self.dbPath)
        print(f"Feature cache ready: {len(X)} rows")
        return X, y, dates
    

# Refit calibrator

    
    # Refit the calibrator on the saved model for testing calibration changes
    def refitCalibrator(self, save=True):
        from models.tieredPoints import TieredPointsBundle
        from models.points import _splitChronologically, _applyPropPlayerFilter
        from config import HOLDOUT_RATIO

        print("\n--- Refit calibrator ---")

        pointsModel = TieredPointsBundle.load()
        X, y, dates = self.featureCachePath.loadOrBuild(dbPath=self.dbPath)

        mask = X["avgPts10"] > 0
        X, y, dates = (
                X[mask].reset_index(drop=True),
                y[mask].reset_index(drop=True),
                dates[mask].reset_index(drop=True),
        )

        _, XCal, _, yCal, calDates = _splitChronologically(
                X, y, dates, holdoutRatio=HOLDOUT_RATIO
        )
        XCal, yCal, calDates = _applyPropPlayerFilter(Xcal, yCal, calDates)
        
        predictions = pointsModel.predictBatch(XCal)

        calibrator = Calibrator.fit(
                predictions = predictions,
                actuals = yCal,
                savePath = CALIBRATOR_PATH if save else None,
                metadata = {
                    "calibration_start_date": calDates.iloc[0],
                    "calibration_end_date": calDates.iloc[-1],
                    "calibration_rows": int(len(yCal)),
                    "refitted_only": True,
                }
        )
        print(
                f"Calibrator refitted"
                f"sigma={calibrator.sigma:.3f}, df={calibrator.df:.2f}"
        )
        return calibrator


# Evaluate model


    def evaluateModel(self):
        from models.evaluate import evaluateModel
        from models.tieredPoints import TieredPointsBundle

        print("\n--- Evaluating saved calibrator ---")
        points = TieredPointsBundle.load()

        X, y, _ = self.featureCachePath.loadOrBuild(dbPath=self.dbPath)

        split = int(len(X) * 0.8)
        XTest = X.iloc[split:]
        yTest = y.iloc[split:]

        lowMask  = XTest["avgPts10"] < 15
        midMask  = (XTest["avgPts10"] >= 15) & (XTest["avgPts10"] < 22)
        highMask = XTest["avgPts10"] >= 22

        print("\n-- Low tier (<15 avgPts10) ---")
        if lowMask.sum() > 0:
            evaluateModel(points.low.model, XTest[lowMask], yTest[lowMask])
        print("\n-- Mid tier (15-22 avgPts10) ---")
        if midMask.sum() > 0:
            evaluateModel(points.mid.model, XTest[midMask], yTest[midMask])
        print("\n-- High tier (22+ avgPts10) ---")
        if highMask.sum() > 0:
            evaluateModel(points.high.model, XTest[highMask], yTest[highMask])


# Evaluate calibrator


    def evaluateCalibrator(self):
        from models.points import _splitChronologically, _applyPropPlayerFilter
        from config import HOLDOUT_RATIO

        print("\n--- Evaluating calibrator ---")
        pointsModel = TieredPointsBundle.load()
        calibrator = Calibrator.load()

        X, y, dates = self.featureCachePath.loadOrBuild(dbPath=self.dbPath)

        mask = X["avgPts10"] > 0
        X, y, dates = (
                X[mask].reset_index(drop=True),
                y[mask].reset_index(drop=True),
                dates[mask].reset_index(drop=True),
        )

        _, XCal, _, yCal, _, calDates = _splitChronologically(
                X, y, dates, holdoutRatio=HOLDOUT_RATIO
        )
        XCal, yCal, _ = _applyPropPlayerFilter(XCal, yCal, calDates)
        
        predictions = pointsModel.predictBatch(XCal)

        print(f"Mean predicted: {predictions.mean():.2f}")
        print(f"Mean actual: {float(yCal.mean()):.2f}")
        print(f"Pred std: {predictions.std():.2f}")
        print(f"Actual std: {float(yCal.std()):.2f}")

        calibrator.printExamples(predMean=float(predictions.mean()))

