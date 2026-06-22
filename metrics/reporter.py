import pandas as pd
import numpy as np


class Reporter:

    # Set this to false to slience everything
    verbose = True

    @classmethod
    def _out(cls, *args, **kwargs):
        if cls.verbose:
            print(*args, **kwargs)


    # Training


    @classmethod
    def minutesTrained(cls, mae, trainRows, valRows):
        cls._out(f"[MinutesBundle] MAE: {mae:.2f} | "
                 f"train={trainRows} val={valRows}"
        )

    @classmethod
    def trainingComplete(cls, importanceDF, trainRows, calRows,
                         calMeanActuals, calMeanPred):
        cls._out(
            f"\n[PointsBundle] Train rows: {trainRows} | "
            f"Cal rows (filtered): {calRows}"
        )
        cls._out(
            f"[PointsBundle] Cal mean actual: {calMeanActuals:.2f} | "
            f"mean predicted: {calMeanPred:.2f}"
        )
        cls._out("\nFeature importances:")
        cls._out(importanceDF.to_string(index=False))
    

    # Calibration

    
    @classmethod
    def calibratorFit(cls, residualStd, df, sigma, predMean, calibrator):
        cls._out(f"\n[Calibrator] residual std={residualStd:.3f}  "
                 f"df={df:.2f}  sigma={sigma:.3f}")
        
        if cls.verbose:
            calibrator.printExamples(predMean=predMean)


    @classmethod
    def calibrationDiagnostics(cls, predictions, actuals, predMean,
                               predStd, actualMean, actualStd):

        cls._out(f"\nMean predicted : {predMean:.2f}")
        cls._out(f"Mean actual : {actualMean:.2f}")
        cls._out(f"Pred std : {predStd:.2f}")
        cls._out(f"Actual std : {actualStd:.2f}")
 
        cls._out(f"\n  {'Line':>6}  {'Actual hit rate':>16}")
        for line in [10, 15, 20, 25, 30]:
            rate = float(np.mean(np.asarray(actuals) > line))
            cls._out(f"  {line:>6}  {rate:>16.3f}")


    # Backtest


    @classmethod
    def skipBreakdown(cls, skips):
        cls._out(f"\n--- Skip breakdown ---")
        cls._out(f"No player match : {skips.noPlayerMatch}")

        # No opp
        cls._out(f"No opp match : {skips.noOppMatch}")        
        if getattr(skips, "noOppMatchByMonth", None):
            cls._out("No opp match by month:")
            for month, count in sorted(skips.noOppMatchByMonth.items()):
                cls._out(f"  {month}: {count}")
                cls._out(f"No actuals : {skips.noActuals}")
        
        cls._out(f"No features : {skips.noFeatures}")
        cls._out(f"Line filtered : {skips.noLine}")
        cls._out(f"No rolling history: {skips.noRollingHistory}")


    @classmethod
    def backtestSummary(cls, df, startingBank, finalBank):
        if df.empty:
            cls._out("No results to summarise.")
            return
 
        bets = df[df["bet"]].copy()
 
        cls._out(f"\n{'='*52}")
        cls._out("BACKTEST SUMMARY")
        cls._out(f"{'='*52}")
        cls._out(f"Props evaluated : {len(df)}")
        cls._out(f"Bets placed : {len(bets)}")
 
        if bets.empty:
            cls._out("No bets placed.")
            return
 
        wins = (bets["pnl"] > 0).sum()
        losses = (bets["pnl"] < 0).sum()
        winRate = wins / len(bets)
        totalPnl = bets["pnl"].sum()
        roi = totalPnl / bets["stake"].sum()
 
        cls._out(f"Win / Loss : {wins}W / {losses}L ({winRate:.2%})")
        cls._out(f"Total P&L : ${totalPnl:.2f}")
        cls._out(f"ROI : {roi:.1%}")
        cls._out(f"Starting bank : ${startingBank:.2f}")
        cls._out(f"Final bank : ${finalBank:.2f}")
        cls._out(
            f"Return : "
            f"{((finalBank - startingBank) / startingBank):.1%}"
        )

        # Win rate by predicted score bucket
        bets["predBucket"] = pd.cut(
            bets["predicted"],
            bins=[0, 12, 15, 18, 22, 99],
            labels=["<12", "12-15", "15-18", "18-22", "22+"],
        )
        cls._out("\nWin rate by predicted score:")
        predSummary = bets.groupby("predBucket", observed=True).agg(
            bets=("pnl", "count"),
            winRate=("pnl", lambda x: (x > 0).mean()),
            avgEdge=("edge", "mean"),
            totalPnl=("pnl", "sum"),
        )
        cls._out(predSummary.to_string(float_format=lambda x: f"{x:.4f}"))
 
        # Top / Worst bets
        cols = ["date", "player", "line", "predicted", "actual", "predDiff", "edge", "pnl"]
        cls._out("\nTop 5 bets by edge:")
        cls._out(
            bets.sort_values("edge", ascending=False)[cols]
            .head(5).to_string(index=False)
        )
        cls._out("\nWorst 5 bets by P&L:")
        cls._out(
            bets.sort_values("pnl")[cols]
            .head(5).to_string(index=False)
        )

        # Check if higher edge bets are actually winning more
        bets["edge_bucket"] = pd.cut(bets["edge"], 
            bins=[0, 0.05, 0.10, 0.15, 0.20, 1.0],
            labels=["0-5%", "5-10%", "10-15%", "15-20%", "20%+"]
        )
        
        print("\nWin rate by edge bucket:")
        print(bets.groupby("edge_bucket", observed=True).agg(
            bets=("pnl", "count"),
            win_rate=("pnl", lambda x: (x > 0).mean()),
            total_pnl=("pnl", "sum")
        ).to_string())


    # Edge distribution

    
    @classmethod
    def edgeDistribution(cls, df):
        bets = df[df["bet"]]
        if bets.empty:
            return

        cls._out("\nEdge distribution of bets placed:")
        cls._out(bets["edge"].describe().to_string())


    @classmethod
    def edgeBucketReport(cls, bets):
        if bets.empty:
            cls._out("no bets placed")
            return

        bets = bets.copy()
        bets["edgeBucket"] = pd.cut(
            bets["edge"],
            bins=[0, 0.05, 0.08, 0.11, 0.15, 0.20, 1.0],
            labels=["0-5%", "5-8%", "8-11%", "11-15%", "15-20%", "20%+"],
        )
        summary = bets.groupby("edgeBucket", observed=True).agg(
            bets    = ("pnl", "count"),
            winRate = ("pnl", lambda x: (x > 0).mean()),
            avgEdge = ("edge", "mean"),
            totalPnl= ("pnl", "sum"),
            roi     = ("pnl", lambda x: x.sum() / (x.count() * bets["stake"].mean())
                        if x.count() > 0 else 0.0),
        )
        cls._out(summary.to_string(float_format=lambda x: f"{x:.4f}"))
 
        winRates = summary["winRate"].dropna().values
        if len(winRates) >= 3:
            drops = [(i, winRates[i], winRates[i + 1])
                     for i in range(len(winRates) - 1)
                     if winRates[i + 1] < winRates[i] - 0.05]
            if drops:
                cls._out(
                    "\n  ⚠  Non-monotonic edge→win-rate detected "
                    "(possible edge signal over-fit):"
                )
                for idx, hi, lo in drops:
                    cls._out(
                        f"     bucket[{idx}]={hi:.2%} → bucket[{idx+1}]={lo:.2%}"
                    )
            
            cls.calibrationAccuracy(bets)
            cls.marginBucketReport(bets)
    
    @classmethod
    def walkForwardSummary(cls, foldResults, filterName = "default", baselinePnl = None):
        if not foldResults:
            cls._out(" (no fold results)")
            return
 
        cls._out(f"\n{'='*60}")
        cls._out(f"WALK-FORWARD SUMMARY  [filter={filterName}]")
        cls._out(f"{'='*60}")
 
        foldDF = pd.DataFrame(foldResults)
        cls._out(foldDF.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        ))
 
        totalPnl   = foldDF["total_pnl"].sum()
        totalBets  = foldDF["bets"].sum()
        profFolds  = (foldDF["total_pnl"] > 0).sum()
        totalFolds = len(foldDF)
 
        cls._out(f"\nTotal P&L across folds : ${totalPnl:.2f}")
        cls._out(f"Total bets            : {int(totalBets)}")
        cls._out(
            f"Profitable folds       : {profFolds}/{totalFolds} "
            f"({profFolds/totalFolds:.0%})"
        )
 
        if baselinePnl is not None:
            delta = totalPnl - baselinePnl
            cls._out(
                f"vs baseline (no filter): ${delta:+.2f}  "
                f"({'better' if delta >= 0 else 'WORSE — filter may be over-fit'})"
            )
    
    @classmethod
    def monthlyPnl(cls, bets: pd.DataFrame):
        if bets.empty:
            cls._out("  (no bets)")
            return
 
        bets = bets.copy()
        bets["month"] = pd.to_datetime(bets["date"]).dt.to_period("M")
        monthly = bets.groupby("month").agg(
            bets    = ("pnl", "count"),
            wins    = ("pnl", lambda x: (x > 0).sum()),
            winRate = ("pnl", lambda x: (x > 0).mean()),
            pnl     = ("pnl", "sum"),
        )
        monthly["cumPnl"] = monthly["pnl"].cumsum()
 
        cls._out(monthly.to_string(float_format=lambda x: f"{x:.2f}"))
 
        # Summary flag: what fraction of months were profitable?
        profMonths = (monthly["pnl"] > 0).sum()
        totalMonths = len(monthly)
        cls._out(
            f"\n  Profitable months: {profMonths}/{totalMonths} "
            f"({profMonths/totalMonths:.0%})"
        )

    @staticmethod
    def filterSweepTable(rows: list[dict]) -> None:
        """
        Prints a compact table comparing all FilterSets side by side.
        The baseline row is always shown first.
        """
        if not rows:
            print("  [FilterSweep] No data.")
            return
 
        print(f"\n{'='*60}")
        print("FILTER COMPARISON TABLE")
        print(f"{'='*60}")
        print(
            f"  {'Filter':<20} {'P&L':>8}  {'Bets':>6}  "
            f"{'Avg WR':>7}  {'Prof folds':>11}"
        )
        print(
            f"  {'-'*20} {'-'*8}  {'-'*6}  {'-'*7}  {'-'*11}"
        )
 
        baselinePnl = None
        for row in rows:
            if row.get("filter") == "baseline":
                baselinePnl = row["total_pnl"]
                break
 
        for row in rows:
            name  = row.get("filter", "?")
            pnl   = row.get("total_pnl", 0.0)
            bets  = row.get("total_bets", 0)
            wr    = row.get("avg_win_rate", 0.0)
            pf    = row.get("prof_folds", "?")
 
            # Mark filters worse than baseline
            flag = ""
            if baselinePnl is not None and name != "baseline":
                flag = " ✓" if pnl > baselinePnl else " ✗"
 
            print(
                f"  {name:<20} {pnl:>+8.2f}  {bets:>6}  "
                f"{wr:>7.1%}  {pf:>11}{flag}"
            )
 
        print(f"{'='*60}")
        if baselinePnl is not None:
            print(
                "  ✓ = better than baseline  "
                "✗ = worse than baseline  "
                "(baseline = no optional filters)"
            )

    @classmethod
    def calibrationAccuracy(cls, bets: pd.DataFrame) -> None:
        """
        For each edge bucket shows mean predicted probability vs actual win rate.
        The gap between them is the calibration error — if myProb is consistently
        higher than actual WR, the calibrator is inflating edges.
        """
        if bets.empty or "myProb" not in bets.columns:
            return

        bets = bets.copy()
        bets["edgeBucket"] = pd.cut(
            bets["edge"],
            bins=[0, 0.05, 0.08, 0.11, 0.15, 0.20, 1.0],
            labels=["0-5%", "5-8%", "8-11%", "11-15%", "15-20%", "20%+"],
        )

        cls._out("\nCalibration accuracy by edge bucket:")
        cls._out(
            f"  {'Bucket':<10} {'Bets':>5}  {'myProb':>8}  "
            f"{'bookProb':>9}  {'ActualWR':>9}  {'CalGap':>8}"
        )
        cls._out(
            f"  {'-'*10} {'-'*5}  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*8}"
        )

        for bucket, group in bets.groupby("edgeBucket", observed=True):
            if group.empty:
                continue
            myProb   = float(group["myProb"].mean())
            bookProb = float(group["bookProb"].mean())
            actualWR = float((group["pnl"] > 0).mean())
            calGap   = actualWR - myProb   # negative = calibrator over-confident
            flag     = "  ⚠" if calGap < -0.05 else ""

            cls._out(
                f"  {str(bucket):<10} {len(group):>5}  {myProb:>8.3f}  "
                f"{bookProb:>9.3f}  {actualWR:>9.3f}  {calGap:>+8.3f}{flag}"
            )

        # Overall
        overall_myProb = float(bets["myProb"].mean())
        overall_wr     = float((bets["pnl"] > 0).mean())
        cls._out(
            f"\n  Overall: myProb={overall_myProb:.3f}  "
            f"actualWR={overall_wr:.3f}  "
            f"gap={overall_wr - overall_myProb:+.3f}"
        )

    @classmethod
    def marginBucketReport(cls, bets: pd.DataFrame) -> None:
        """
        Shows whether the model's point margin over the book line is behaving
        monotonically before odds/calibration are considered.
        """
        if bets.empty or "predDiff" not in bets.columns:
            return

        bets = bets.copy()
        bets["marginBucket"] = pd.cut(
            bets["predDiff"],
            bins=[-99, 1, 2, 3, 4, 5, 99],
            labels=["<=1", "1-2", "2-3", "3-4", "4-5", "5+"],
        )
        summary = bets.groupby("marginBucket", observed=True).agg(
            bets=("pnl", "count"),
            winRate=("pnl", lambda x: (x > 0).mean()),
            avgPredDiff=("predDiff", "mean"),
            avgRawProb=("rawProb", "mean"),
            avgMyProb=("myProb", "mean"),
            totalPnl=("pnl", "sum"),
        )

        cls._out("\nWin rate by predicted-line margin:")
        cls._out(summary.to_string(float_format=lambda x: f"{x:.4f}"))

