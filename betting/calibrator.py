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
    SIGMA_BOUNDS, DF_BOUNDS
)


# File helpers


def _probOverT(predicted, line, residualStd, df):
    return float(1 - t_dist.cdf(line, df=df, loc=predicted, scale=residualStd))

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


# Calibrator class


class Calibrator:
    def __init__(self, platt, df, sigma, residualStd, meta):
        self.platt = platt
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

    # Uncalibrated prob (Needed for debugging)
    def rawProbOver(self, predicted, line):
        return _probOverT(predicted, line, self.sigma, self.df)


    # Diagnostics

    
    def printExamples(self, predMean=18.0, lines=None):
        if lines is None:
            lines = [10, 15, 20, 25, 30, 35, 40]
        print(f"\n[Calibrator] sigma={self.sigma:.3f}  df={self.df:.2f}")
        print(f"\n[Calibrator] {'Line':>6}  {'Raw':>8}  {'Calibrated':>12}")
        
        for line in lines:
            raw = self.rawProbOver(predMean, line)
            cal = self.probOver(predMean, line)
            print(f"{Line:>6}  {Raw:>8.3f}  {cal:>12.3f}")
        
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
            bins = np.linespace(0, 1, 11)
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
            print(f"[Calibrator] Plot saved -> {save_path}")
        else:
            plt.show()
        plt.close()


# Persistance


    def save(self, path=CALIBRATOR_PATH):
        bundle = {
            "platt": self.platt,
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
                paltt = bundle["platt"],
                df = bundle["df"],
                sigma = bundle["sigma"],
                residualStd = bundle.get("residual std", bundle.get("sigma")),
                meta = {
                    k: v for k, v in bundle.items()
                    if k not in ("platt", "df", "sigma", "residual std")
                },
        )


    @classmethod
    def loadIfExists(cls, path=CALIBRATOR_PATH):
        if not Path(path).exists():
            return None
        return cls.load(path)

    
    # Fitting

    
    def fit(cls, predictions, actuals, savePath=None, metadata=None):
        """
        Fits a new calibrator from holdout preds and actuals
    
        Steps
        1. Esimate residual std
        2. Jointly optimizes df and sigma across full line range
        3. Build (rawProb, hit) pairs across all prop relevant lines
        4. Fit platt scale
        5. Optional save
        """
        predictions = np.asarray(predictions, dtype=float)
        actuals = np.asarray(
            actuals.values if hasattr(actuals, "values") else actuals,
            dtype=float
        )

        # 1. Estimate residual std on holdout

        if residualStd is None:
            residualStd = _estimateResidualStd(predictions, actuals)
        print(f"[Calibrator] residual std = {residualStd:.3f}")

        # 2. Fit df empirically and find best sigma

        print("\n[Calibrator] Actual hit rates vs naive prediction at mean:")
        meanPred = float(predictions.mean())
        for line in [10, 15, 20, 25, 30]:
            actualRate = float(np.mean(actuals > line))
            print(f"  line={line}  actual_hit_rate={actualRate:.3f}  mean_pred={meanPred:.1f}")
    
        df, optimalSigma = _fitDfAndSigma(predictions, actuals)
        print(f"[Calibrator] best df = {df:.2f}, sigma={optimalSigma:.3f}")

        # 3. Build (raw prob and hit) pairs across mutiple lines

        rawProbsAll, hitsAll = [], []
        for line in PLATT_FIT_LINES:
            raw = np.array([_probOverT(p, line, optimalSigma, df)
                for p in predictions
            ])
            hit = (actuals > line).astype(float)
            rawProbsAll.append(raw)
            hitsAll.append(hit)

        rawProbsAll = np.concatenate(rawProbsAll)
        hitsAll = np.concatenate(hitsAll)

        # 4. Platt scaling (logistic regression on raw probs)

        platt = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
        )
        platt.fit(rawProbsAll.reshape(-1,1), hitsAll)

        instance = cls(
            platt = platt,
            df = df, 
            sigma = optimalSigma,
            residualStd = residualStd,
            meta = metadata or {},
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

