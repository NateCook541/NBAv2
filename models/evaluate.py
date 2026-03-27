import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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


#FIXME: Add plots here mabye?

