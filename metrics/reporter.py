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
        cols = ["date", "player", "line", "predicted", "actual", "edge", "pnl"]
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
