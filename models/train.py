import sqlite3
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from models.evaluate import evaluateModel, plotPredictions, plotResiduals, plotResidualVsPred

from features.featureCollector import buildFeatures, featureOrder

def generateTrainingData():
    conn = sqlite3.connect('NBA.db')

    logsQuery = """
        SELECT
            pgl.player_id,
            pgl.game_id,
            pgl.points          AS actual_points,
            g.game_date,
            g.home_team_id,
            g.away_team_id,
            p.team_id,
            CASE WHEN g.home_team_id = p.team_id
                 THEN g.away_team_id
                 ELSE g.home_team_id
            END                 AS opp_team_id
        FROM Player_game_logs pgl
        JOIN Games   g ON pgl.game_id   = g.game_id
        JOIN Players p ON pgl.player_id = p.player_id
        WHERE pgl.minutes > 0
        ORDER BY g.game_date
        """
    logs = pd.read_sql_query(logsQuery, conn)
    logs = logs.dropna(subset=["team_id", "opp_team_id"])
    print(f"Building features for {len(logs)} log rows")
        
    featureRows = []
    targets = []
    skipped = 0

    for _, row in logs.iterrows():
        features = buildFeatures(
                playerID = int(row["player_id"]),
                date = row["game_date"],
                teamID = int(row["team_id"]),
                oppTeamID = int(row["opp_team_id"]),
                conn = conn,
        )
        if features is None or features.isnull().any(axis=1).iloc[0]:
            skipped += 1
            continue

        featureRows.append(features)
        targets.append(float(row["actual_points"]))

    conn.close()
        
    print(f"Built {len(featureRows)} rows  |  skipped {skipped} (insufficient history)")
    X = pd.concat(featureRows, ignore_index=True)
    y = pd.Series(targets, name="points")
    return X, y


def trainModel(save=True, plot=False):
    X, y = generateTrainingData()

    mask = X["avgPts10"] > 0
    X, y = X[mask], y[mask]

    # Uses basic hyperparameters (Note - Test other models/parameters (Mabye TPOT might help here?))
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(
        n_estimators  = 100,
        max_depth     = 5,
        learning_rate = 0.1,
        objective     = "reg:squarederror",
        random_state  = 42,
    ) 

    model.fit(XTrain, yTrain)

    predictions = evaluateModel(model, XTest, yTest)
    if plot:
        plotPredictions(yTest, predictions)
        plotResiduals(yTest, predictions)
        plotResidualVsPred(predictions, yTest)
    
    # Get the feature importance data as well
    importance = pd.DataFrame({
        "feature": XTest.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    print("\nFeature importances:")
    print(importance.to_string(index=False))
    
    if save:
        modelPath = Path(__file__).parent / "nba_model.joblib"
        joblib.dump(model, modelPath)
        print(f"\nModel saved to {modelPath}")

    return model

