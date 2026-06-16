# Main controller for full project

import sqlite3
from pathlib import Path

from config import (
    DB_PATH, FEATURE_CACHE_PATH,
    MODEL_PATH, CALIBRATOR_PATH, MINUTES_PATH,
    DEFAULT_EDGE_THRESH,
    DEFAULT_BANKROLL, DEFAULT_KELLY_FRAC,
)
from features.cache import FeatureCache, preloadCaches
from models.minutes import MinutesBundle
from models.points import PointsBundle
from betting.calibrator import Calibrator
from betting.filters import FilterSet


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


    def _propDateRange(self):
        conn = sqlite3.connect(str(self.dbPath))
        res = conn.execute(
            "SELECT MIN(game_date), MAX(game_date) FROM Props"
        ).fetchone()

        conn.close()
        if res and res[0] and res[1]:
            return str(res[0]), str(res[1])
        raise ValueError("No props found in database")

    def _propDates(self, startDate=None, endDate=None):
        conn = sqlite3.connect(str(self.dbPath))
        query = """
            SELECT DISTINCT game_date
            FROM Props
            WHERE over_odds IS NOT NULL AND under_odds IS NOT NULL
        """
        params = []
        if startDate:
            query += " AND game_date >= ?"
            params.append(startDate)
        if endDate:
            query += " AND game_date <= ?"
            params.append(endDate)
        query += " ORDER BY game_date"
        rows = conn.execute(query, params).fetchall()
        conn.close()

        dates = [str(row[0]) for row in rows]
        if not dates:
            raise ValueError("No props found for the requested timeframe")
        return dates

    def _splitPropDatesByMonths(self, startDate=None, endDate=None, months=1):
        if months < 1:
            raise ValueError("months must be >= 1")

        import pandas as pd

        propDates = self._propDates(startDate=startDate, endDate=endDate)
        parsed = pd.to_datetime(propDates)
        first = parsed.min()
        firstMonthIndex = first.year * 12 + first.month

        groups = {}
        for raw, date in zip(propDates, parsed):
            monthIndex = date.year * 12 + date.month
            periodIdx = (monthIndex - firstMonthIndex) // months
            groups.setdefault(periodIdx, []).append(raw)

        return [(dates[0], dates[-1]) for _, dates in sorted(groups.items())]



    # Train


    def train(self, endDate=None, save=True, runMetrics=False,
              forceRetrain=False, useCachedFeatures=False):
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
        if useCachedFeatures and savedMinutes is None:
            raise ValueError(
                "useCachedFeatures=True requires a saved minutes model for "
                "backtest-time feature generation."
            )

        minutesChanged = (
                (not useCachedFeatures) and (
                    forceRetrain or
                    savedMinutes is None or
                    not savedMinutes.isSafeFor(endDate or "9999-99-99")
                )
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
                 bankroll=DEFAULT_BANKROLL, retrainEveryMonths=None,
                 retrainMinutes=False):
        """
        Run a full backtest on props data

        Will use saved model if its leakage safe (trained before start
        of prop data) and if not save will build a new model
        """
        from betting.backtest import BacktestEngine

        print("\n" + "=" * 55)
        print("BACKTEST: Start")
        print("=" * 55)

        if retrainEveryMonths is not None and retrainEveryMonths > 0:
            results = self._backtestWithPeriodicRetraining(
                startDate=startDate,
                endDate=endDate,
                bankroll=bankroll,
                retrainEveryMonths=retrainEveryMonths,
                retrainMinutes=retrainMinutes,
            )
            print("\n" + "=" * 55)
            print("BACKTEST: Complete")
            print("=" * 55)
            return results

        # Find the backtest start date if not supplied
        backtestStart = startDate or self._earliestPropDate()

        points, minutes, calibrator = self._loadOrTrainBundle(backtestStartDate = backtestStart)

        fs = FilterSet.edgeDonutHole() 
        
        engine = BacktestEngine(
                pointsBundle = points,
                minutesBundle = minutes,
                calibrator = calibrator,
                dbPath = self.dbPath,
                filterSet = fs
        )

        results = engine.run(
                startDate = startDate,
                endDate = endDate,
                bankroll = bankroll
        )

        print("\n" + "=" * 55)
        print("BACKTEST: Complete")
        print("=" * 55)

        return results

    def _backtestWithPeriodicRetraining(self, startDate, endDate,
                                        bankroll, retrainEveryMonths,
                                        retrainMinutes):
        import pandas as pd
        from betting.backtest import BacktestEngine
        from metrics.reporter import Reporter

        periods = self._splitPropDatesByMonths(
            startDate=startDate,
            endDate=endDate,
            months=retrainEveryMonths,
        )

        print(
            f"[Backtest] Periodic retraining every {retrainEveryMonths} "
            f"month(s) across {len(periods)} test periods"
        )
        print(f"[Backtest] Retrain minutes each period: {retrainMinutes}")
        for idx, (periodStart, periodEnd) in enumerate(periods, start=1):
            print(f"  period {idx}: {periodStart} → {periodEnd}")

        fs =FilterSet(
            name="edgeDonut_under22",
            #donutLow=0.08,
            #donutHigh=0.11,
            maxPredicted=22.0,
        )

        currentBank = bankroll
        allResults = []

        for idx, (periodStart, periodEnd) in enumerate(periods, start=1):
            print(f"\n{'='*60}")
            print(f"BACKTEST PERIOD {idx}/{len(periods)}: {periodStart} → {periodEnd}")
            print(f"{'='*60}")
            print(
                f"[Backtest] Training bundle through {periodStart} "
                f"(exclusive)"
            )

            points, minutes, calibrator = self.train(
                endDate=periodStart,
                save=False,
                forceRetrain=retrainMinutes,
                useCachedFeatures=not retrainMinutes,
            )

            engine = BacktestEngine(
                pointsBundle=points,
                minutesBundle=minutes,
                calibrator=calibrator,
                dbPath=self.dbPath,
                filterSet=fs,
            )

            results = engine.run(
                startDate=periodStart,
                endDate=periodEnd,
                bankroll=currentBank,
            )

            if not results.empty:
                results = results.copy()
                results["retrain_period"] = idx
                results["model_train_end"] = periodStart
                currentBank = float(results["bankroll"].iloc[-1])
                allResults.append(results)

        combined = pd.concat(allResults, ignore_index=True) if allResults else pd.DataFrame()
        combined.to_csv("backtest_results.csv", index=False)

        print(f"\n{'='*52}")
        print("PERIODIC RETRAIN BACKTEST SUMMARY")
        print(f"{'='*52}")
        Reporter.backtestSummary(combined, bankroll, currentBank)

        return combined


    # Walk forward fold test
    

    def walkForwardOverThresholds(self, startDate = None, endDate = None, nFolds = 5, 
                                  edgeThresh = DEFAULT_EDGE_THRESH, 
                                  bankroll = DEFAULT_BANKROLL, filterSets = None,
                                  retrainEachFold = True,
                                  retrainMinutesEachFold = False):
        """
        Walk-forward backtest over held-out time folds.
 
        For each fold the points model and calibrator are trained only on data
        before that fold's start date, then reused across filter comparisons.
 
        A baseline FilterSet (no optional filters) is always run first so
        every other FilterSet result can be compared against it directly.
        This makes it immediately visible whether a filter is adding real
        value or just cherry picking a lucky in sample period.
        """

        import pandas as pd
        from betting.backtest import BacktestEngine
        from betting.filters import FilterSet
        from metrics.reporter import Reporter
 
        conn = sqlite3.connect(str(self.dbPath))
        propDates = pd.read_sql_query(
            """
            SELECT DISTINCT game_date
            FROM Props
            WHERE over_odds IS NOT NULL AND under_odds IS NOT NULL
            ORDER BY game_date
            """,
            conn,
        )["game_date"].tolist()
        conn.close()
 
        if startDate:
            propDates = [d for d in propDates if d >= startDate]
        if endDate:
            propDates = [d for d in propDates if d <= endDate]
 
        if len(propDates) < nFolds:
            raise ValueError(
                f"Only {len(propDates)} prop dates available — "
                f"cannot split into {nFolds} folds."
            )
 
        # Split the sorted date list into nFolds equal-ish chunks
        foldSize = len(propDates) // nFolds
        folds = []
        for i in range(nFolds):
            chunkStart = i * foldSize
            chunkEnd   = (i + 1) * foldSize if i < nFolds - 1 else len(propDates)
            folds.append((propDates[chunkStart], propDates[chunkEnd - 1]))
 
        print(f"\n[WalkForward] {nFolds} folds across {len(propDates)} prop dates")
        print(f"[WalkForward] Full range: {propDates[0]} → {propDates[-1]}")
        print(f"[WalkForward] Retrain each fold: {retrainEachFold}")
        print(f"[WalkForward] Retrain minutes each fold: {retrainMinutesEachFold}")
        for i, (fs, fe) in enumerate(folds):
            print(f"  fold {i + 1}: {fs} → {fe}")

        # FilterSet list always include baseline
        baseline = FilterSet.baseline()
        if filterSets is None:
            filterSets = [FilterSet.edgeDonutHole(), FilterSet.baseline()]  
        else:
            names = [f.name for f in filterSets]
            if baseline.name not in names:
                filterSets = [baseline] + list(filterSets) 

        # Run every FilterSet across every fold 
        allResults = {}
        foldBundles = {}
 
        for fs in filterSets:
            print(f"\n{'='*60}")
            print(f"WALK-FORWARD  [filter={fs.name}]")
            print(f"{'='*60}")
 
            foldResults = []
 
            for foldIdx, (foldStart, foldEnd) in enumerate(folds):
                print(f"\nFold {foldIdx + 1}/{nFolds}: {foldStart} → {foldEnd}")
 
                # Leakage-safe model for this fold's start date.
                if foldStart not in foldBundles:
                    if retrainEachFold:
                        print(
                            f"[WalkForward] Training fold bundle through "
                            f"{foldStart} (exclusive)"
                        )
                        foldBundles[foldStart] = self.train(
                            endDate=foldStart,
                            save=False,
                            forceRetrain=retrainMinutesEachFold,
                            useCachedFeatures=not retrainMinutesEachFold,
                        )
                    else:
                        foldBundles[foldStart] = self._loadOrTrainBundle(
                            backtestStartDate=foldStart
                        )

                points, minutes, calibrator = foldBundles[foldStart]
 
                engine = BacktestEngine(
                    pointsBundle = points,
                    minutesBundle = minutes,
                    calibrator = calibrator,
                    dbPath = self.dbPath,
                    filterSet = fs,
                )
 
                results = engine.run(
                    startDate = foldStart,
                    endDate = foldEnd,
                    edgeThresh = edgeThresh,
                    bankroll = bankroll,
                )
 
                bets = results[results["bet"]] if not results.empty else results
                if bets.empty:
                    foldPnl = 0.0
                    foldBets = 0
                    foldWr = 0.0
                    foldRoi = 0.0
                else:
                    foldPnl = float(bets["pnl"].sum())
                    foldBets = len(bets)
                    foldWr = float((bets["pnl"] > 0).mean())
                    stakeSum = float(bets["stake"].sum())
                    foldRoi = foldPnl / stakeSum if stakeSum > 0 else 0.0
 
                foldResult = {
                    "fold": foldIdx + 1,
                    "start": foldStart,
                    "end": foldEnd,
                    "bets": foldBets,
                    "win_rate": round(foldWr, 4),
                    "total_pnl": round(foldPnl, 2),
                    "roi": round(foldRoi, 4),
                }
                foldResults.append(foldResult)
 
                # Per fold edge bucket report
                if not bets.empty:
                    print(f"\nEdge bucket performance (fold {foldIdx + 1}):")
                    Reporter.edgeBucketReport(bets)
 
                    print(f"\nMonthly P&L (fold {foldIdx + 1}):")
                    Reporter.monthlyPnl(bets)
 
            allResults[fs.name] = foldResults
 
            # Aggregate fold summary
            # Compare against baseline if available
            baselinePnl = None
            if fs.name != baseline.name and baseline.name in allResults:
                baselinePnl = sum(
                    r["total_pnl"] for r in allResults[baseline.name]
                )
 
            Reporter.walkForwardSummary(
                foldResults  = foldResults,
                filterName   = fs.name,
                baselinePnl  = baselinePnl,
            )
 
        # Cross filter comparison table
        self._printCrossFilterSummary(allResults, filterSets)
 
        return allResults

    def _printCrossFilterSummary(self, allResults, filterSets):
        import pandas as pd
        from metrics.reporter import Reporter

        rows = []
        for fs in filterSets:
            foldResults = allResults.get(fs.name, [])
            if not foldResults:
                continue
            
            totalPnl = sum(r["total_pnl"] for r in foldResults)
            totalBets = sum(r["bets"] for r in foldResults)
            profFolds = sum(1 for r in foldResults if r["total_pnl"] > 0)
            avgWr = (
                sum(r["win_rate"] for r in foldResults) / len(foldResults)
            )
            row = {
                "filter": fs.name,
                "total_pnl": round(totalPnl, 2),
                "total_bets": totalBets,
                "avg_win_rate": round(avgWr, 4),
                "prof_folds": f"{profFolds}/{len(foldResults)}",
                **fs.asDict(),
            }
            rows.append(row)
 
        Reporter.filterSweepTable(rows)

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
                f"sigma_left={calibrator.sigmaLeft:.3f}, "
                f"sigma_right={calibrator.sigmaRight:.3f}"
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
