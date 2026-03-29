import sqlite3
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from models.evaluate import evaluateModel

from features.featureCollector import buildFeatures, featureOrder

def preloadCaches(conn):
    print(f"Loading caches")

    # Player logs
    allLogs = pd.read_sql_query("""
        SELECT pgl.player_id, g.game_date, pgl.points, pgl.minutes, 
               pgl.fg_pct, pgl.is_home, pgl.rest_days
        FROM Player_game_logs pgl
        JOIN Games g ON pgl.game_id = g.game_id
        ORDER BY pgl.player_id, g.game_date
    """, conn)

    playerLogCache = {
        pid: grp.reset_index(drop=True)
        for pid, grp in allLogs.groupby("player_id")
    }

    # Player positions
    players = pd.read_sql_query("SELECT player_id, position FROM Players", conn)
    posCache = dict(zip(players.player_id, players.position))

    # Team stats
    teams = pd.read_sql_query("SELECT team_id, date, def_rtg, pace FROM Teams", conn)
    teams = teams.sort_values(["team_id", "date"])
    teamCache = {
        tid: grp.reset_index(drop=True)
        for tid, grp in teams.groupby("team_id")
    }

    # Status table
    status = pd.read_sql_query("SELECT * FROM Status", conn)

    # Player avg points precompute
    playerAvg = pd.read_sql_query("""
        SELECT player_id, AVG(points) as avg_pts
        FROM Player_game_logs
        GROUP BY player_id
    """, conn)
    playerAvgCache = dict(zip(playerAvg.player_id, playerAvg.avg_pts))

    # Player usage rate
    teamGameTotals = (
        allLogs.groupby(["game_date", "is_home"])["points"]
        .sum()
        .to_dict()
    )
        
    print("Cache loading done")
    return playerLogCache, posCache, teamCache, status, playerAvgCache, teamGameTotals

def generateTrainingData():
    conn = sqlite3.connect('NBA.db')
    
    playerLogCache, posCache, teamCache, statusDF, playerAvgCache, teamGameTotals = preloadCaches(conn) 

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
            END AS opp_team_id
        FROM Player_game_logs pgl
        JOIN Games   g ON pgl.game_id   = g.game_id
        JOIN Players p ON pgl.player_id = p.player_id
        WHERE pgl.minutes >= 3 -- train only when greater than 10 minutes played
        ORDER BY g.game_date
        """
    logs = pd.read_sql_query(logsQuery, conn)
    logs = logs.dropna(subset=["team_id", "opp_team_id"])
    print(f"Building features for {len(logs)} log rows")
        
    featureRows = []
    targets = []
    skipped = 0

    for row in logs.itertuples(index=False):
        features = buildFeatures(
            playerID=row.player_id,
            date=row.game_date,
            teamID=row.team_id,
            oppTeamID=row.opp_team_id,
            cache=playerLogCache,
            posCache=posCache,
            teamCache=teamCache,
            statusDF=statusDF,
            playerAvgCache=playerAvgCache,
            usageRateCache=teamGameTotals,
        )
        if features is None:
            skipped += 1
            continue

        
        featureRows.append(features)
        targets.append(row.actual_points)

    conn.close()
        
    print(f"Built {len(featureRows)} rows  |  skipped {skipped} (insufficient history)")
    X = pd.concat(featureRows, ignore_index=True)
    y = pd.Series(targets, name="points")
    return X, y

def trainModel(save=True, metrics=False):
    X, y = generateTrainingData()

    mask = X["avgPts10"] > 0
    X, y = X[mask], y[mask]

    # Have to use this instead of test, train, split as to prevent r2 score being inflated from games already being seen
    splitIdx = int(len(X) * 0.8)
    XTrain, XTest = X.iloc[:splitIdx], X.iloc[splitIdx:]
    yTrain, yTest = y.iloc[:splitIdx], y.iloc[splitIdx:]

    # Uses basic hyperparameters (Note - Test other models/parameters (Mabye TPOT might help here?))
    model = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        n_jobs=-1,
        objective="reg:squarederror",
        random_state=42,
    )

    model.fit(XTrain, yTrain)
    
    if metrics:
        predictions = evaluateModel(model, XTest, yTest)
    
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

