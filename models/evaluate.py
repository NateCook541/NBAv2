import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

OUTPUT_DIR = Path("/mnt/c/Users/natec/Pictures/NBA")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def evaluateModel(model, XTest, yTest):

    predictions = model.predict(XTest)

    mae = mean_absolute_error(yTest, predictions)
    rmse = np.sqrt(mean_squared_error(yTest, predictions))
    r2 = r2_score(yTest, predictions)
    bias = np.mean(predictions - yTest)

    print("\nEvaluation Metrics")
    print("------------------")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R2   : {r2:.3f}")
    print(f"Bias : {bias:.2f}")

    baseline = XTest["avgPts10"]
    baseline_mae = mean_absolute_error(yTest, baseline)

    print(f"\nBaseline (avgPts10) MAE: {baseline_mae:.2f}")

    debug = pd.DataFrame({
        "actual": yTest,
        "pred": predictions,
        "avg10": XTest["avgPts10"]
    })

    debug["error"] = debug["pred"] - debug["actual"]
    debug["abs_error"] = debug["error"].abs()

    print("\nWorst predictions:")
    print(debug.sort_values("abs_error", ascending=False).head(20))

    return predictions

def plotPredictions(yTest, predictions):

    plt.figure(figsize=(6,6))
    plt.scatter(yTest, predictions, alpha=0.3)
    plt.plot([0,50],[0,50], color="red")
    plt.xlabel("Actual Points")
    plt.ylabel("Predicted Points")
    plt.title("Prediction vs Actual")
    path = OUTPUT_DIR / "prediction_vs_actual.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved plot → {path}")

def plotResiduals(yTest, predictions):

    residuals = predictions - yTest

    plt.figure(figsize=(6,4))
    plt.hist(residuals, bins=40)
    plt.title("Residual Distribution")
    plt.xlabel("Prediction Error")
    path = OUTPUT_DIR / "residual_histogram.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved plot → {path}")

def plotResidualVsPred(predictions, yTest):

    residuals = predictions - yTest

    plt.figure(figsize=(6,4))
    plt.scatter(predictions, residuals, alpha=0.3)
    plt.axhline(0, color="red")
    plt.xlabel("Predicted Points")
    plt.ylabel("Residual")
    plt.title("Residual vs Prediction")
    path = OUTPUT_DIR / "residual_vs_pred.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved plot → {path}")

