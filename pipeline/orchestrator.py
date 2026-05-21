# Main controller for full project

import sqlite3
from pathlib import Path

from config import (
    DB_PATH, FEATURE_CACHE_PATH,
    MODEL_PATH, CALIBRATOR_PATH, MINUTES_PATH,
    DEFAULT_EDGE_THRESH,
    OVER_EDGE_THRESH_CANDIDATES,
    OVER_TARGET_BETS_MIN, OVER_TARGET_BETS_MAX,
    DEFAULT_BANKROLL, DEFAULT_KELLY_FRAC,
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
        points = PointsBundle.loadIfExists()
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
        
        points = PointsBundle.train(
                X,
                y,
                dates,
                save=save,
                runMetrics=runMetrics,
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
        

    def backtest(self, startDate=None, endDate=None, edgeThresh=DEFAULT_EDGE_THRESH,
                 bankroll=DEFAULT_BANKROLL):
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

        points, minutes, calibrator = self._loadOrTrainBundle(backtestStartDate = backtestStart)

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


    # Backtest multi edge threshold testing


    def sweepOverThresholds(self, startDate=None, endDate=None, thresholds=None, bankroll=DEFAULT_BANKROLL):
        """
        Runs multiple over-only backtests at different edge thresholds and prints
        a compact comparison table for selecting a 900-1000 bet operating point.
        """
        from betting.backtest import BacktestEngine

        testThresholds = list(thresholds or OVER_EDGE_THRESH_CANDIDATES)
        backtestStart = startDate or self._earliestPropDate()
        points, minutes, calibrator = self._loadOrTrainBundle(backtestStartDate=backtestStart)
        engine = BacktestEngine(pointsBundle=points, minutesBundle=minutes, calibrator=calibrator, dbPath=self.dbPath)

        rows = []
        for thresh in testThresholds:
            results = engine.run(
                startDate=startDate,
                endDate=endDate,
                edgeThresh=float(thresh),
                bankroll=bankroll,
            )
            finalBank = float(results["bankroll"].iloc[-1]) if not results.empty else bankroll
            summary = BacktestEngine.summarizeResults(results, startingBank=bankroll, finalBank=finalBank)
            rows.append(
                {
                    "edge_thresh": float(thresh),
                    "bets": summary["bets"],
                    "win_rate": summary["win_rate"],
                    "roi": summary["roi"],
                    "total_pnl": summary["total_pnl"],
                    "final_bank": summary["final_bank"],
                }
            )

        import pandas as pd
        table = pd.DataFrame(rows).sort_values(["win_rate", "roi"], ascending=[False, False]).reset_index(drop=True)
        print("\n--- Over Threshold Sweep ---")
        print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        return table

    
    # Backtest fold testing


    def walkForwardOverThresholds(self, startDate=None, endDate=None, thresholds=None,
                                  bankroll=DEFAULT_BANKROLL, folds=4):
        """
        Walk-forward threshold selection:
        choose threshold on prior window, evaluate on subsequent window.
        """
        import pandas as pd
        from betting.backtest import BacktestEngine, _loadProps

        testThresholds = list(thresholds or OVER_EDGE_THRESH_CANDIDATES)
        backtestStart = startDate or self._earliestPropDate()
        points, minutes, calibrator = self._loadOrTrainBundle(backtestStartDate=backtestStart)
        engine = BacktestEngine(pointsBundle=points, minutesBundle=minutes, calibrator=calibrator, dbPath=self.dbPath)

        conn = sqlite3.connect(str(self.dbPath))
        props = _loadProps(conn, startDate, endDate)
        conn.close()
        dates = sorted(pd.to_datetime(props["game_date"]).dt.date.unique())
        if len(dates) < (folds * 4):
            print("[WalkForwardOver] Not enough dates for robust walk-forward.")
            return pd.DataFrame()

        step = max(1, len(dates) // (folds + 1))
        rows = []

        for i in range(1, folds + 1):
            train_end_idx = min(len(dates) - 2, i * step)
            test_end_idx = min(len(dates) - 1, (i + 1) * step)
            train_start = str(dates[0])
            train_end = str(dates[train_end_idx])
            test_start = str(dates[train_end_idx + 1])
            test_end = str(dates[test_end_idx])

            best = None
            for thresh in testThresholds:
                trainRes = engine.run(startDate=train_start, endDate=train_end, edgeThresh=float(thresh), bankroll=bankroll)
                trainFinal = float(trainRes["bankroll"].iloc[-1]) if not trainRes.empty else bankroll
                s = BacktestEngine.summarizeResults(trainRes, bankroll, trainFinal)
                bets = s["bets"]
                inBand = (OVER_TARGET_BETS_MIN <= bets <= OVER_TARGET_BETS_MAX)
                score = s["win_rate"] + (0.25 * s["roi"])
                if (best is None) or (inBand and not best["in_band"]) or (
                    inBand == best["in_band"] and score > best["score"]
                ):
                    best = {"thresh": float(thresh), "score": float(score), "in_band": inBand}

            chosen = best["thresh"]
            testRes = engine.run(startDate=test_start, endDate=test_end, edgeThresh=chosen, bankroll=bankroll)
            testFinal = float(testRes["bankroll"].iloc[-1]) if not testRes.empty else bankroll
            ts = BacktestEngine.summarizeResults(testRes, bankroll, testFinal)
            rows.append(
                {
                    "fold": i,
                    "train_range": f"{train_start}..{train_end}",
                    "test_range": f"{test_start}..{test_end}",
                    "chosen_thresh": chosen,
                    "bets": ts["bets"],
                    "win_rate": ts["win_rate"],
                    "roi": ts["roi"],
                    "total_pnl": ts["total_pnl"],
                }
            )

        out = pd.DataFrame(rows)
        print("\n--- Walk-Forward Over Thresholds ---")
        print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))
        if not out.empty:
            print(
                f"\n[WalkForwardOver] mean bets={out['bets'].mean():.1f} "
                f"mean win_rate={out['win_rate'].mean():.4f} mean roi={out['roi'].mean():.4f}"
            )
        return out


# Cache features


    def cacheFeatures(self):
        print("\n--- Building and saving feature cache ---")
        X, y, dates = self.featureCachePath.buildAndSave(dbPath=self.dbPath)
        print(f"Feature cache ready: {len(X)} rows")
        return X, y, dates
    

# Refit calibrator

    
    # Refit the calibrator on the saved model for testing calibration changes
    def refitCalibrator(self, save=True):
        from models.points import _splitChronologically, _applyPropPlayerFilter
        from config import HOLDOUT_RATIO

        print("\n--- Refit calibrator ---")

        pointsModel = PointsBundle.load()
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
        XCal, yCal, calDates = _applyPropPlayerFilter(
            XCal, yCal, calDates, minAvgPtsCal=5
        )
        
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
        from models.evaluate import evaluateModel, walkForwardEvaluate

        print("\n--- Evaluating saved points model ---")
        points = PointsBundle.load()

        X, y, _ = self.featureCachePath.loadOrBuild(dbPath=self.dbPath)

        split = int(len(X) * 0.8)
        XTest = X.iloc[split:].reset_index(drop=True)
        yTest = y.iloc[split:].reset_index(drop=True)
        evaluateModel(
            points.model,
            XTest,
            yTest,
            targetMode=points.meta.get("target_mode", "absolute"),
            clipK=float(points.meta.get("prediction_clip_k", 0.0)),
            biasMeta=points.meta.get("bias_correction", {}),
        )
        walkForwardEvaluate(X, y)


# Evaluate calibrator


    def evaluateCalibrator(self):
        from models.points import _splitChronologically, _applyPropPlayerFilter
        from config import HOLDOUT_RATIO

        print("\n--- Evaluating calibrator ---")
        pointsModel = PointsBundle.load()
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
        XCal, yCal, _ = _applyPropPlayerFilter(XCal, yCal, calDates, minAvgPtsCal=5)
        
        predictions = pointsModel.predictBatch(XCal)

        print(f"Mean predicted: {predictions.mean():.2f}")
        print(f"Mean actual: {float(yCal.mean()):.2f}")
        print(f"Pred std: {predictions.std():.2f}")
        print(f"Actual std: {float(yCal.std()):.2f}")

        calibrator.printExamples(predMean=float(predictions.mean()))
