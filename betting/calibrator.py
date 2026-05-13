import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import t as t_dist
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

from config import (
    CALIBRATOR_PATH,
    CAL_FIT_LINES, PLATT_FIT_LINES,
    SIGMA_BOUNDS, DF_BOUNDS,
    DEFAULT_UNDER_CALIBRATION_MODE,
    DEFAULT_UNDER_HIGH_CUTOFF,
    DEFAULT_UNDER_RELIABILITY_SHRINK,
)


# File helpers


def _probOverT(predicted, line, sigma, df):
    return float(1 - t_dist.cdf(line, df=df, loc=predicted, scale=sigma))

def _rawPushProb(predicted, line, sigma, df):
    """
    Rough push probability for integer lines assuming integer outcomes.
    For half-point lines this should be near zero.
    """
    if not float(line).is_integer():
        return 0.0
    upper = float(t_dist.cdf(line + 0.5, df=df, loc=predicted, scale=sigma))
    lower = float(t_dist.cdf(line - 0.5, df=df, loc=predicted, scale=sigma))
    return max(0.0, upper - lower)

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
            result = minimize(error, x0=[df0, sigma0],
                             method="Nelder-Mead",
                             options={"xatol": 0.1, "fatol": 1e-5, "maxiter": 500})
            if result.fun < bestLoss:
                bestLoss = result.fun
                bestParams = result.x

    bestDF = float(np.clip(bestParams[0], *DF_BOUNDS))
    bestSigma = float(np.clip(bestParams[1], *SIGMA_BOUNDS))
    print(
        f"[Calibrator] joint fit: df={bestDF:.2f}\n" 
        f"[Calibrator] sigma={bestSigma:.3f}, loss={bestLoss:.6f}"
    )

    return bestDF, bestSigma


def _buildPlattData(predictions, actuals, lines, sigma, df):
    actuals = np.asarray(actuals)
    rawProbsAll, hitsAll = [], []

    for line in lines:
        raw = np.array([_probOverT(p, line, sigma, df) for p in predictions])
        hit = (actuals > line).astype(float)
        rawProbsAll.append(raw)
        hitsAll.append(hit)

    return np.concatenate(rawProbsAll), np.concatenate(hitsAll)

def _buildPlattDataUnder(predictions, actuals, lines, sigma, df):
    actuals = np.asarray(actuals)
    rawProbsAll, hitsAll = [], []

    for line in lines:
        raw = np.array([
            float(t_dist.cdf(line, df=df, loc=p, scale=sigma))
            for p in predictions
        ])

        hit = (actuals < line).astype(float)
        rawProbsAll.append(raw)
        hitsAll.append(hit)

    return np.concatenate(rawProbsAll), np.concatenate(hitsAll)

# Calibrator class


class Calibrator:
    def __init__(self, platt, plattUnder, plattUnderHigh, df, sigma, residualStd, meta):
        self.platt = platt
        self.plattUnder = plattUnder
        self.plattUnderHigh = plattUnderHigh
        self.df = df
        self.sigma = sigma
        self.residualStd = residualStd
        self.meta = meta


    # Core probability methods


    # Returns calibrated P(acutal > line) given a raw point pred
    # NOTE: Only method backtest and prediction layers needs to call
    def probOver(self, predicted, line):
        raw =  float(1 - t_dist.cdf(
            line, df=self.df, loc=predicted, scale=self.sigma
        ))
        return float(self.platt.predict_proba([[raw]])[0, 1])

    # Returns calibrated P(acutal < line) given a raw point pred
    # NOTE: Only (other) method backtest and prediction layers needs to call
    def probUnder(self, predicted, line):
        mode = str(
            self.meta.get("under_calibration_mode", DEFAULT_UNDER_CALIBRATION_MODE)
        ).lower().strip()
        highCut = float(self.meta.get("under_high_cutoff", DEFAULT_UNDER_HIGH_CUTOFF))
        useHigh = (predicted >= highCut) or (line >= highCut)

        # Complement baseline for coherence.
        over = self.probOver(predicted, line)
        push = _rawPushProb(predicted, line, self.sigma, self.df)
        complement = float(np.clip(1.0 - over - push, 0.0, 1.0))

        # Dedicated under calibration path.
        rawUnder = float(t_dist.cdf(line, df=self.df, loc=predicted, scale=self.sigma))
        underGlobal = float(self.plattUnder.predict_proba([[rawUnder]])[0, 1])
        underHigh = (
            float(self.plattUnderHigh.predict_proba([[rawUnder]])[0, 1])
            if self.plattUnderHigh is not None
            else underGlobal
        )
        dedicated = underHigh if useHigh else underGlobal

        if mode == "complement":
            return complement
        if mode == "dedicated":
            return float(np.clip(dedicated, 0.0, 1.0))
        # hybrid: use dedicated on high regime, complement elsewhere
        value = dedicated if useHigh else complement
        return float(np.clip(value, 0.0, 1.0))

    def underExtraShrink(self, predicted, line):
        highCut = float(self.meta.get("under_high_cutoff", DEFAULT_UNDER_HIGH_CUTOFF))
        if not ((predicted >= highCut) or (line >= highCut)):
            return 0.0
        base = float(
            self.meta.get("under_reliability_shrink", DEFAULT_UNDER_RELIABILITY_SHRINK)
        )
        highBrier = float(self.meta.get("under_high_brier", 0.25))
        # Increase shrink when reliability is poor in high under regime.
        uncertainty = float(np.clip((highBrier - 0.22) / 0.10, 0.0, 1.0))
        return float(np.clip(base * (1.0 + uncertainty), 0.0, 0.35))


    # Uncalibrated prob (Needed for debugging)
    def rawProbOver(self, predicted, line):
        return _probOverT(predicted, line, self.sigma, self.df)


    # Convenience

    
    @property
    def profitableEdgeCap(self):
        """
        Max edge that calibrator considers reliable, as if a edge is too far
        then we want to just drop it as its most likely a error and we shouldn't
        bet on it.

        Computed in fit and used in backtest to filter out most likely bad bets
        """
        return float(
            self.meta.get(
                "profitable_edge_cap",
                self.meta.get("profitable_edge_Cap", 0.15)
            )
        )


    # Diagnostics

    
    def printExamples(self, predMean=18.0, lines=None):
        if lines is None:
            lines = [10, 15, 20, 25, 30, 35, 40]
        print(f"\n[Calibrator] sigma={self.sigma:.3f}  df={self.df:.2f}")
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
        
        actuals = np.asarray(
                actuals.values if hasattr(actuals, "values") else actuals,
                dtype=float,
        )
        results = []

        for line in lines:
            probs = np.array([self.probOver(p, line) for p in predictions])
            bins = np.linspace(0, 1, 11)
            for i in range(len(bins) - 1):
                mask = (probs >= bins[i]) & (probs < bins[i+1])
                if mask.sum() < 20:
                    continue
                results.append({
                    "line": line,
                    "predicted prob": float(probs[mask].mean()),
                    "actual rate": float((actuals[mask] > line).mean()),
                    "n": int(mask.sum())
                })

        return pd.DataFrame(results)

    def plotCalibration(self, calDF, savePath= None):
        plt.figure(figsize=(7, 7))
        plt.scatter(
            calDF["predicted prob"], calDF["actual rate"],
            alpha=0.4, c=calDF["line"], cmap="viridis",
        )
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
            "plattUnder": self.plattUnder,
            "plattUnderHigh": self.plattUnderHigh,
            "df": self.df,
            "sigma": self.sigma,
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
                platt = bundle["platt"],
                plattUnder = bundle.get("plattUnder", bundle["platt"]),
                plattUnderHigh = bundle.get("plattUnderHigh", bundle.get("plattUnder", bundle["platt"])),
                df = bundle["df"],
                sigma = bundle["sigma"],
                residualStd = bundle.get("residual std", bundle.get("sigma")),
                meta = {
                    k: v for k, v in bundle.items()
                    if k not in ("platt", "plattUnder", "plattUnderHigh", "df", "sigma", "residual std")
                },
        )


    @classmethod
    def loadIfExists(cls, path=CALIBRATOR_PATH):
        if not Path(path).exists():
            return None
        return cls.load(path)

    
    # Fitting

   
    @classmethod
    def fit(cls, predictions, actuals, savePath=None, metadata=None, 
            targetMode="absolute", underCalibrationMode=DEFAULT_UNDER_CALIBRATION_MODE,
            underHighCutoff=DEFAULT_UNDER_HIGH_CUTOFF,
            underReliabilityShrink=DEFAULT_UNDER_RELIABILITY_SHRINK):
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
        actuals = np.asarray(
            actuals.values if hasattr(actuals, "values") else actuals,
            dtype=float
        )

        # 1. Estimate residual std on holdout

        residualStd = _estimateResidualStd(predictions, actuals)
        print(f"[Calibrator] residual std = {residualStd:.3f}")

        # 2. Hit rate diagnoistics

        meanPred = float(predictions.mean())
        meanAct = float(actuals.mean())

        print(
            f"\n[Calibrator] Hit rates "
            f"(mean pred={meanPred:.1f}, mean actual={meanAct:.1f}):"
        )
        print(f"  {'Line':>6}  {'Actual hit rate':>16}  {'Model > line':>14}")
        for line in [10, 15, 20, 25, 30]:
            actRate  = float(np.mean(actuals > line))
            predRate = float(np.mean(predictions > line))
            print(f"  {line:>6}  {actRate:>16.3f}  {predRate:>14.3f}")


        # 3. Fit df and sigma

        df, optimalSigma = _fitDfAndSigma(predictions, actuals)
        print(f"[Calibrator] best df = {df:.2f}, sigma={optimalSigma:.3f}")


        # 4. Build platt data with a line restriction

        plattLines = [line for line in PLATT_FIT_LINES if 5 <= line <= 35]
        
        rawProbs, hits = _buildPlattData(predictions, actuals, plattLines, optimalSigma, df)
        rawProbsUnder, hitsUnder = _buildPlattDataUnder(predictions, actuals, plattLines, optimalSigma, df)
        highLines = [line for line in plattLines if line >= float(underHighCutoff)]
        rawProbsUnderHigh, hitsUnderHigh = _buildPlattDataUnder(
            predictions, actuals, highLines, optimalSigma, df
        )

        # 5. Platt scaling (logistic regression on raw probs)
        
        platt = LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
        )
        platt.fit(rawProbs.reshape(-1,1), hits)
        
        
        plattUnder = LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
        )
        
        plattUnder.fit(rawProbsUnder.reshape(-1,1), hitsUnder)

        plattUnderHigh = LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
        )
        if len(hitsUnderHigh) >= 200 and len(np.unique(hitsUnderHigh)) > 1:
            plattUnderHigh.fit(rawProbsUnderHigh.reshape(-1, 1), hitsUnderHigh)
        else:
            plattUnderHigh = plattUnder
        

        # 6. Compression check
        
        calAt60 = float(platt.predict_proba([[0.60]])[0, 1])
        calAt80 = float(platt.predict_proba([[0.80]])[0, 1])
        calAt90 = float(platt.predict_proba([[0.90]])[0, 1])
        highCompression = calAt90 - calAt60
        midCompression  = calAt80 - calAt60
 
        print(
            f"[Calibrator] Platt output: "
            f"@60%={calAt60:.3f}, @80%={calAt80:.3f}, @90%={calAt90:.3f}"
        )
        print(
            f"[Calibrator] Compression: "
            f"60→90 span={highCompression:.3f}, "
            f"60→80 span={midCompression:.3f}"
        )
 
        if highCompression < 0.10:
            profitableEdgeCap = 0.10
            print(
                f"[Calibrator] Severe compression — "
                f"capping profitable edge at {profitableEdgeCap:.0%}"
            )
        elif midCompression < 0.08:
            profitableEdgeCap = 0.12
            print(
                f"[Calibrator] Moderate compression — "
                f"capping profitable edge at {profitableEdgeCap:.0%}"
            )
        else:
            profitableEdgeCap = 0.18
            print(
                f"[Calibrator] Compression acceptable — "
                f"profitable edge cap={profitableEdgeCap:.0%}"
            )

        # Historical backtests showed edges above 15% are typically not reliable.
        profitableEdgeCap = min(profitableEdgeCap, 0.15)
        print(f"[Calibrator] Final enforced edge cap={profitableEdgeCap:.0%}")

        # Unders

        underAt40 = float(plattUnder.predict_proba([[0.40]])[0, 1])
        underAt60 = float(plattUnder.predict_proba([[0.60]])[0, 1])
        underAt80 = float(plattUnder.predict_proba([[0.80]])[0, 1])
        underHighAt60 = float(plattUnderHigh.predict_proba([[0.60]])[0, 1])
        print(
            f"[Calibrator] Under Platt: "
            f"@40%={underAt40:.3f}, @60%={underAt60:.3f}, @80%={underAt80:.3f}"
        )
        print(
            f"[Calibrator] Under High ({underHighCutoff:.1f}+): "
            f"@60%={underHighAt60:.3f}"
        )
        underGlobalBrier = float(np.mean((rawProbsUnder - hitsUnder) ** 2))
        underHighBrier = float(np.mean((rawProbsUnderHigh - hitsUnderHigh) ** 2)) if len(hitsUnderHigh) else underGlobalBrier

        instance = cls(
            platt = platt,
            plattUnder = plattUnder,
            plattUnderHigh = plattUnderHigh,
            df = df, 
            sigma = optimalSigma,
            residualStd = residualStd,
            meta = {
                **(metadata or {}),
                "target_mode": targetMode,
                "under_calibration_mode": str(underCalibrationMode).lower().strip(),
                "under_high_cutoff": float(underHighCutoff),
                "under_reliability_shrink": float(underReliabilityShrink),
                "profitable_edge_cap": profitableEdgeCap,
                "mean_pred": meanPred,
                "platt_60": calAt60,
                "platt_80": calAt80,
                "platt_90": calAt90,
                "under_platt_60": underAt60,
                "under_high_platt_60": underHighAt60,
                "under_global_brier": underGlobalBrier,
                "under_high_brier": underHighBrier,
            }
        )
        instance.printExamples(predMean=meanPred)


        # 5. Optional save


        if savePath:
            instance.save(savePath)

        return instance


    # Backtest safety
    def isSafeFor(self, backtestStartDate):
        end = self.meta.get("calibration_end_date", "")
        return bool(end) and end < backtestStartDate
