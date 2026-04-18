import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import t as t_dist
from scipy.optimize import minimize_scalar, minimize
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression

# HELPERS

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
        lines = np.arange(10, 40, 2.5)
    
    actuals = np.asarray(actuals)

    def error(params):
        df, sigma = params
        if df < 2.1 or sigma < 2:
            return 1e9
        errs = []
        for line in lines:
            predRate = float(np.mean(1 - t_dist.cdf(line, df=df, loc=predictions, scale=sigma)))
            actualRate = float(np.mean(actuals > line))
            errs.append((predRate - actualRate) ** 2)
        return np.mean(errs)

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

    bestDF = float(np.clip(bestParams[0], 2.1, 30))
    bestSigma = float(np.clip(bestParams[1], 3, 25))
    print(f"[calibrator] joint fit: df={bestDF:.2f}, sigma={bestSigma:.3f}, loss={bestLoss:.6f}")
    return bestDF, bestSigma


# Not helpers!

def probOverTDist(predicted, line, residualStd, df=5):
    scale = residualStd
    return 1 - t_dist.cdf(line, df=df, loc=predicted, scale=scale)

def cailbratedProbOver(predicted, line, residualStd, calibrator, df=5):
    sigma = calibrator.get("sigma", residualStd)
    raw = float(probOverTDist(predicted, line, sigma, df=df))
    platt = calibrator["platt"]
    cal = float(platt.predict_proba([[raw]])[0, 1])
    
    return cal

def fitCailbrator(predictions, yTest, residualStd=None, df=None, savePath=None, metadata=None):
    predictions = np.asarray(predictions, dtype=float)
    actuals = np.asarray(
            yTest.values if hasattr(yTest, "values") else yTest,
            dtype=float
    )

    # 1. Estimate residual std on holdout

    if residualStd is None:
        residualStd = _estimateResidualStd(predictions, actuals)
    print(f"[calibrator] residual std = {residualStd:.3f}")

    # 2. Fit df empirically and find best sigma

    print("\n[DEBUG] Actual hit rates vs naive prediction at mean:")
    meanPred = float(predictions.mean())
    for line in [10, 15, 20, 25, 30]:
        actualRate = float(np.mean(actuals > line))
        print(f"  line={line}  actual_hit_rate={actualRate:.3f}  mean_pred={meanPred:.1f}")
    
    df, optimalSigma = _fitDfAndSigma(predictions, actuals)
    print(f"[calibrator] best df = {df:.2f}, sigma={optimalSigma:.3f}")

    # 3. Build (raw prob and hit) pairs across mutiple lines

    # Sample lines uniformly across the realistic prop range
    lines = np.arange(10, 46, 2.5)
    rawProbsAll = []
    hitsAll = []

    for line in lines:
        raw = np.array([_probOverT(p, line, optimalSigma, df)
                        for p in predictions])
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

    # 5. Diagonstic print
    
    print("\n[calibrator] Platt correction examples:")
    print(f"  {'Raw':>8} {'Calibrated':>12}")
    for raw in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        cal = float(platt.predict_proba([[raw]])[0, 1])
        print(f"  {raw:>8.2f} {cal:>12.3f}")
    
    # 6. Bundle and save

    bundle = {
            "platt": platt,
            "df": df,
            "residualStd": residualStd,
            "sigma": optimalSigma,
    }
    if metadata:
        bundle.update(metadata)

    if savePath:
        joblib.dump(bundle, savePath)
        print(f"[calibrator] Saved to {savePath}")
    
    return bundle


def calibrationCheck(predictions, y_test, residual_std, calibrator=None, df=5):
    actuals = np.asarray(
        y_test.values if hasattr(y_test, "values") else y_test,
        dtype=float
    )
    lines = np.arange(5, 46, 2.5)
    results = []
 
    for line in lines:
        if calibrator is not None:
            probs = np.array([
                cailbratedProbOver(p, line, residual_std, calibrator, df=df)
                for p in predictions
            ])
        else:
            probs = np.array([
                _probOverT(p, line, residual_std, df)
                for p in predictions
            ])
 
        bins = np.linspace(0, 1, 11)
        for i in range(len(bins) - 1):
            mask = (probs >= bins[i]) & (probs < bins[i + 1])
            if mask.sum() < 20:
                continue
            results.append({
                "line":           line,
                "predicted_prob": float(probs[mask].mean()),
                "actual_rate":    float((actuals[mask] > line).mean()),
                "n":              int(mask.sum()),
            })
 
    return pd.DataFrame(results)

# Print out metrics on the calibrator
def printCalMetrics(model, XTest, yTest):
    predictions = model.predict(XTest)
    residuals = yTest.values - predictions
    residualStd = residuals.std()
    
    calibratorPath = Path("models/nba_calibrator.joblib")
    calibrator = fitCailbrator(predictions, yTest, savePath=calibratorPath)

    calDF = calibrationCheck(predictions, yTest, residualStd, calibrator=calibrator, df=3)
    displayCalibration(calDF)


def displayCalibration(calDF, savePath=None):
    plt.figure(figsize=(7, 7))
    plt.scatter(calDF["predicted_prob"], calDF["actual_rate"],
                alpha=0.4, c=calDF["line"], cmap="viridis")
    plt.colorbar(label="Line")
    plt.plot([0, 1], [0, 1], "r--", label="Perfect calibration")
    plt.xlabel("Predicted probability")
    plt.ylabel("Actual hit rate")
    plt.title("Calibration Plot (coloured by line)")
    plt.legend()
 
    if savePath:
        Path(savePath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savePath)
        print(f"[calibrator] Plot saved to {savePath}")
    else:
        plt.show()
 
    plt.close()

