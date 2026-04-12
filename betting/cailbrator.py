import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from scipy.stats import t as t_dist
from pathlib import Path
from sklearn.isotonic import IsotonicRegression

def printCalMetrics(model, XTest, yTest):
    predictions = model.predict(XTest)
    residuals = yTest.values - predictions
    residualStd = residuals.std()
    
    calibratorPath = Path("models/nba_calibrator.joblib")
    calibrator = fitCalibrator(predictions, yTest, residualStd, df=3, savePath=calibratorPath)

    calDF = calibrationCheck(predictions, yTest, residualStd, calibrator=calibrator, df=3)
    displayCalibration(calDF)
    
    print("\nCalibration correction examples:")
    print(f"{'Raw prob':>10} {'Calibrated':>12}")
    for raw in [0.30, 0.45, 0.60, 0.75, 0.90]:
        cal = float(calibrator.predict([raw])[0])
        print(f"{raw:>10.2f} {cal:>12.3f}")

# Calibration check and fit cailbration lines

def calibrationCheck(predictions, yTest, residualStd, calibrator=None, df=3): 
    # Test lines
    lines = np.arange(5, 45, 2.5)
    actuals = yTest.values if hasattr(yTest, "values") else yTest
    results = []

    for line in lines:
        if calibrator is not None:
            predictedProbs = np.array([
                calibratedProbOver(p, line, residualStd, calibrator, df=df)
                for p in predictions
            ])
        else:
            predictedProbs = probOverTDist(predictions, line, residualStd, df=df)


        # Bucket predictions into confidence bins
        bins = np.linspace(0, 1, 11)

        for i in range(len(bins) - 1):
            mask = (predictedProbs >= bins[i]) & (predictedProbs < bins[i+1])
            if mask.sum() < 20:
                continue
            
            avgPredicted = predictedProbs[mask].mean()
            actualRate = (actuals[mask] > line).mean()

            results.append({
                "line": line,
                "predicted_prob": avgPredicted,
                "actual_rate": actualRate,
                "n": mask.sum()
            })

    return pd.DataFrame(results)

# Fits a isotonic regression calibrator for a playerAvg line and saves it
# This lets us know the true prob -> actual hit rate mapping
def fitCalibrator(predictions, yTest, residualsStd, df=3, savePath=None, metadata=None):

    actuals = yTest.values if hasattr(yTest, "values") else yTest

    # Uses the median line as the reference point for fitting (middle of distribution has the most data)
    referenceLine = np.median(actuals)

    rawProbs = probOverTDist(predictions, referenceLine, residualsStd, df=df)
    yBinary = (actuals > referenceLine).astype(int)

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(rawProbs, yBinary)

    if savePath:
        payload = {"calibrator": calibrator, "df": df, "residualStd": residualsStd}
        if metadata:
            payload.update(metadata)
        joblib.dump(payload, savePath)
        print(f"Calibator saved to {savePath}")

    return calibrator

# Use this instead of raw norm.cdf when generating bet signals
def calibratedProbOver(predicted, line, residualStd, calibrator, df=3):
    rawProb = float(probOverTDist(predicted, line, residualStd, df=df))
    return float(calibrator.predict([rawProb])[0])

def probOverTDist(predicted, line, residualStd, df=3):
    scale = residualStd * np.sqrt((df - 2) / df)
    return 1 - t_dist.cdf(line, df=df, loc=predicted, scale=scale)

# Saves a graph displaying the calibration graph
def displayCalibration(calDF):
    # Perfect calibration = points on the diagonal
    plt.figure(figsize=(7, 7))
    plt.scatter(calDF["predicted_prob"], calDF["actual_rate"], alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--', label="Perfect calibration")
    plt.xlabel("Predicted probability")
    plt.ylabel("Actual hit rate")
    plt.title("Calibration Plot")
    plt.legend()

    savePath = Path("/mnt/c/Users/natec/Pictures/calibration_plot.png")
    savePath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savePath)
    plt.close()
    print(f"\nCalibration plot saved to {savePath}")
