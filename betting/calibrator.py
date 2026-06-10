import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import t as t_dist
from sklearn.linear_model import LogisticRegression

from config import (
    CALIBRATOR_PATH, CAL_FIT_LINES, PLATT_FIT_LINES, SIGMA_BOUNDS, DF_BOUNDS,
)


def _probOverT(predicted, line, sigma, df):
    return float(1 - t_dist.cdf(line, df=df, loc=predicted, scale=sigma))


def _estimateResidualStd(predictions, actuals):
    residuals = np.asarray(actuals) - np.asarray(predictions)
    q75, q25 = np.percentile(residuals, [75, 25])
    iqrStd = (q75 - q25) / 1.349
    plainStd = float(np.std(residuals))
    return max(iqrStd, plainStd)


def _fitDfAndSigma(predictions, actuals, lines=None):
    if lines is None:
        lines = CAL_FIT_LINES
    actuals = np.asarray(actuals)

    def error(params):
        df, sigma = params
        if df < DF_BOUNDS[0] or sigma < SIGMA_BOUNDS[0]:
            return 1e9
        errs = []
        for line in lines:
            predRate = float(np.mean(1 - t_dist.cdf(line, df=df, loc=predictions, scale=sigma)))
            actualRate = float(np.mean(actuals > line))
            errs.append((predRate - actualRate) ** 2)
        return float(np.mean(errs))

    bestLoss = np.inf
    bestParams = (5, 8.0)
    for df0 in [3, 5, 10, 20]:
        for sigma0 in [6, 8, 10, 12]:
            result = minimize(
                error,
                x0=[df0, sigma0],
                method="Nelder-Mead",
                options={"xatol": 0.1, "fatol": 1e-5, "maxiter": 500},
            )
            if result.fun < bestLoss:
                bestLoss = result.fun
                bestParams = result.x

    bestDF = float(np.clip(bestParams[0], *DF_BOUNDS))
    bestSigma = float(np.clip(bestParams[1], *SIGMA_BOUNDS))
    print(f"[Calibrator] joint fit: df={bestDF:.2f}\n[Calibrator] sigma={bestSigma:.3f}, loss={bestLoss:.6f}")
    return bestDF, bestSigma
    

# SPLIT NORMAL ATTEMPT


# Fits a split normal distribution, left for below and right for above   
def _fitSplitSigma(predictions, actuals):
    residuals = np.asarray(actuals) - np.asarray(predictions)

    left = residuals[residuals <= 0]        
    right = residuals[residuals > 0]

    sigmaLeft = float(np.sqrt(np.mean(left ** 2))) if len(left) > 10 else 7.0
    sigmaRight = float(np.sqrt(np.mean(right ** 2))) if len(right) > 10 else 7.0

    sigmaLeft = float(np.clip(sigmaLeft, *SIGMA_BOUNDS))
    sigmaRight = float(np.clip(sigmaRight, *SIGMA_BOUNDS))

    print(f"[Calibrator] split-normal: sigma_left={sigmaLeft:.3f}, sigma_right={sigmaRight:.3f}")
    print(f"[Calibrator] left tail n={len(left)}, right tail n={len(right)}")
    return sigmaLeft, sigmaRight
    
def _probOverSplitNormal(predicted, line, sigmaLeft, sigmaRight):
    from scipy.stats import norm
    delta = line - predicted

    if delta <= 0:
        return float(norm.sf(delta / sigmaRight))
    else:
        return float(norm.sf(delta / sigmaLeft))


def _buildPlattData(predictions, actuals, lines, sigmaLeft, sigmaRight):
    actuals = np.asarray(actuals)
    rawProbsAll, hitsAll = [], []
    for line in lines:
        raw = np.array([
            _probOverSplitNormal(p, line, sigmaLeft, sigmaRight)
            for p in predictions
        ])
        hit = (actuals > line).astype(float)
        rawProbsAll.append(raw)
        hitsAll.append(hit)
    return np.concatenate(rawProbsAll), np.concatenate(hitsAll)


class Calibrator:
    def __init__(self, platt, sigmaLeft, sigmaRight, residualStd, meta):
        self.platt = platt
        self.sigmaLeft = sigmaLeft
        self.sigmaRight = sigmaRight
        self.residualStd = residualStd
        self.meta = meta

    
    # Core probablity methods


    # Returns calibrated P(acutal > line) given a raw point pred
    # NOTE: Only method backtest and prediction layers needs to call
    def probOver(self, predicted, line):
        raw = _probOverSplitNormal(predicted, line, self.sigmaLeft, self.sigmaRight)
        return float(self.platt.predict_proba([[raw]])[0, 1])

    # Uncalibrated prob (needed for debuging)
    def rawProbOver(self, predicted, line):
        return _probOverSplitNormal(predicted, line, self.sigmaLeft, self.sigmaRight)


    # Convience


    @property
    def profitableEdgeCap(self):
        """
        Max edge that calibrator considers as reliable
        """
        return float(self.meta.get("profitable_edge_cap", self.meta.get("profitable_edge_Cap", 0.15)))


    # Diagnostics


    def printExamples(self, predMean=18.0, lines=None):
        if lines is None:
            lines = [10, 15, 20, 25, 30, 35, 40]
        print(f"\n[Calibrator] sigma_left={self.sigmaLeft:.3f}  sigma_right={self.sigmaRight:.3f}")
        print(f"\n[Calibrator] {'Line':>6}  {'Raw':>8}  {'Calibrated':>12}")
        for line in lines:
            raw = self.rawProbOver(predMean, line)
            cal = self.probOver(predMean, line)
            print(f"{line:>6}  {raw:>8.3f}  {cal:>12.3f}")

    # Returns a DF of line, predicted prob bucket, acutal hit rate, n
    # Reading this lets us diagnose over/under confidense on prop range
    def calibrationCheck(self, predictions, actuals, lines=None):
        if lines is None:
            lines = np.arange(5, 46, 2.5)
        actuals = np.asarray(actuals.values if hasattr(actuals, "values") else actuals, dtype=float)
        results = []
        for line in lines:
            probs = np.array([self.probOver(p, line) for p in predictions])
            bins = np.linspace(0, 1, 11)
            for i in range(len(bins) - 1):
                mask = (probs >= bins[i]) & (probs < bins[i + 1])
                if mask.sum() < 20:
                    continue
                results.append(
                    {
                        "line": line,
                        "predicted prob": float(probs[mask].mean()),
                        "actual rate": float((actuals[mask] > line).mean()),
                        "n": int(mask.sum()),
                    }
                )
        return pd.DataFrame(results)

    def plotCalibration(self, calDF, savePath=None):
        plt.figure(figsize=(7, 7))
        plt.scatter(calDF["predicted prob"], calDF["actual rate"], alpha=0.4, c=calDF["line"], cmap="viridis")
        plt.colorbar(label="Line")
        plt.plot([0, 1], [0, 1], "r--", label="Perfect calibration")
        plt.xlabel("Predicted probability")
        plt.ylabel("Actual hit rate")
        plt.title("Calibration plot (coloured by line)")
        plt.legend()
        if savePath:
            Path(savePath).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(savePath)
            print(f"[Calibrator] Plot saved -> {savePath}")
        else:
            plt.show()
        plt.close()


    # Persistance


    def save(self, path=CALIBRATOR_PATH):
        bundle = {
            "platt": self.platt,
            "sigma left": self.sigmaLeft,
            "sigma right": self.sigmaRight,
            "residual std": self.residualStd,
            **self.meta,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, path)
        print(f"[Calibrator] Saved -> {path}")

    @classmethod
    def load(cls, path=CALIBRATOR_PATH):
        if not Path(path).exists():
            raise FileNotFoundError(f"No calibrator at {path}")
        bundle = joblib.load(path)
        return cls(
            platt=bundle["platt"],
            sigmaLeft=bundle["sigma left"],
            sigmaRight=bundle["sigma right"],
            residualStd=bundle.get("residual std", bundle.get("sigma")),
            meta={k: v for k, v in bundle.items() if k not in ("platt", "plattUnder", "plattUnderHigh", "df", "sigma", "residual std")},
        )

    @classmethod
    def loadIfExists(cls, path=CALIBRATOR_PATH):
        if not Path(path).exists():
            return None
        return cls.load(path)


    # Fitting
    

    @classmethod
    def fit(cls, predictions, actuals, savePath=None, metadata=None, targetMode="absolute"):
        """
        Fits a new calibrator from holdout preds and actuals
    
        Steps
        1. Esimate residual std
        2. Print diagnostics for hit rates
        3. Jointly optimizes df and sigma across full line range
        4. Build (rawProb, hit) pairs across all prop relevant lines
        5. Fit platt scale
        6. Check high end compression and set edge cap
        7. Optional save
        """
        
        predictions = np.asarray(predictions, dtype=float)
        actuals = np.asarray(actuals.values if hasattr(actuals, "values") else actuals, dtype=float)

        # 1. Estimate residual std on holdout

        residualStd = _estimateResidualStd(predictions, actuals)
        print(f"[Calibrator] residual std = {residualStd:.3f}")

        # 2. Hit rate diagnoistics

        meanPred = float(predictions.mean())
        meanAct = float(actuals.mean())
        print(f"\n[Calibrator] Hit rates (mean pred={meanPred:.1f}, mean actual={meanAct:.1f}):")
        print(f"  {'Line':>6}  {'Actual hit rate':>16}  {'Model > line':>14}")
        for line in [10, 15, 20, 25, 30]:
            actRate = float(np.mean(actuals > line))
            predRate = float(np.mean(predictions > line))
            print(f"  {line:>6}  {actRate:>16.3f}  {predRate:>14.3f}")

        # 3. Fit split nromal sigmas
        
        sigmaLeft, sigmaRight = _fitSplitSigma(predictions, actuals)

        # 4. Build platt data with a line restriction

        plattLines = [line for line in PLATT_FIT_LINES if 5 <= line <= 35]
        rawProbs, hits = _buildPlattData(predictions, actuals, plattLines, sigmaLeft, sigmaRight)

        # 5. Platt scaling (logistic regression on raw probs)

        nBins = 20
        binEdges = np.linspace(0.0, 1.0, nBins + 1)
        binMidpoints = []
        binHitRates = []

        for i in range(nBins):
            lo, hi = binEdges[i], binEdges[i + 1]
            mask = (rawProbs >= lo) & (rawProbs < hi)
            if mask.sum() < 5:   # skip bins with too few observations
                continue
            binMidpoints.append(float(rawProbs[mask].mean()))
            binHitRates.append(float(hits[mask].mean()))

        binMidpoints = np.array(binMidpoints)
        binHitRates = np.array(binHitRates)

        platt = LogisticRegression(solver="lbfgs", max_iter=2000)
        platt.fit(rawProbs.reshape(-1, 1), hits)

        # 6. Compression check

        calAt60 = float(platt.predict_proba([[0.60]])[0, 1])
        calAt80 = float(platt.predict_proba([[0.80]])[0, 1])
        calAt90 = float(platt.predict_proba([[0.90]])[0, 1])
        highCompression = calAt90 - calAt60
        midCompression = calAt80 - calAt60

        print(f"[Calibrator] Platt output: @60%={calAt60:.3f}, @80%={calAt80:.3f}, @90%={calAt90:.3f}")
        print(f"[Calibrator] Compression: 60→90 span={highCompression:.3f}, 60→80 span={midCompression:.3f}")

        if highCompression < 0.10:
            profitableEdgeCap = 0.10
            print(f"[Calibrator] Severe compression — capping profitable edge at {profitableEdgeCap:.0%}")
        elif midCompression < 0.08:
            profitableEdgeCap = 0.12
            print(f"[Calibrator] Moderate compression — capping profitable edge at {profitableEdgeCap:.0%}")
        else:
            profitableEdgeCap = 0.18
            print(f"[Calibrator] Compression acceptable — profitable edge cap={profitableEdgeCap:.0%}")

        profitableEdgeCap = min(profitableEdgeCap, 0.15)
        print(f"[Calibrator] Final enforced edge cap={profitableEdgeCap:.0%}")

        instance = cls(
            platt=platt,
            sigmaLeft=sigmaLeft,
            sigmaRight=sigmaRight,
            residualStd=residualStd,
            meta={
                **(metadata or {}),
                "target_mode": targetMode,
                "profitable_edge_cap": profitableEdgeCap,
                "mean_pred": meanPred,
                "platt_60": calAt60,
                "platt_80": calAt80,
                "platt_90": calAt90,
            },
        )
        instance.printExamples(predMean=meanPred)

        # 7. Optional save

        if savePath:
            instance.save(savePath)
        return instance


    # Backtest safety
    def isSafeFor(self, backtestStartDate):
        end = self.meta.get("calibration_end_date", "")
        return bool(end) and end < backtestStartDate

