import sqlite3
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from datetime import datetime, timedelta

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

    playerUsageCache = {}
    for pid, grp in allLogs.groupby("player_id"):
        shares = []
        for _, row in grp.iterrows():
            teamTotal = teamGameTotals.get((row["game_date"], row["is_home"]), 0)
            if teamTotal > 0:
                shares.append(row["points"] / teamTotal)
        playerUsageCache[pid] = float(sum(shares) / len(shares)) if shares else 0.0

    # Pos opp query
    oppPosLogs = pd.read_sql_query("""
        SELECT 
            g.game_date,
            g.home_team_id,
            g.away_team_id,
            pgl.is_home,
            pgl.points,
            p.position
        FROM Player_game_logs pgl
        JOIN Games   g ON pgl.game_id   = g.game_id
        JOIN Players p ON pgl.player_id = p.player_id
        WHERE p.position IS NOT NULL
        ORDER BY g.game_date
    """, conn)     

    oppPosCache = {}
    for _, row in oppPosLogs.iterrows():
        defendingTeam = row["home_team_id"] if not row["is_home"] else row["away_team_id"]
        key = (int(defendingTeam), row["position"])
        if key not in oppPosCache:
            oppPosCache[key] = []
        oppPosCache[key].append({
            "game_date": row["game_date"],
            "points":    row["points"]
        })
    
    oppPosCache = {
        key: pd.DataFrame(row).sort_values("game_date").reset_index(drop=True)
        for key, row in oppPosCache.items()
    }

    print("Cache loading done")

    return playerLogCache, posCache, teamCache, status, playerAvgCache, playerUsageCache, oppPosCache, teamGameTotals

def generateTrainingData():
    conn = sqlite3.connect('NBA.db')
    
    playerLogCache, posCache, teamCache, statusDF, playerAvgCache, playerUsageCache, oppPosCache, teamGameTotals = preloadCaches(conn) 

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

    minutesModelPath = Path("models/nba_minutes_model.joblib")
    minutesModel = joblib.load(minutesModelPath) if minutesModelPath.exists() else None

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
            usageRateCache=playerUsageCache,
            oppPosCache=oppPosCache,
            teamGameTotals=teamGameTotals,
            minutesModel=minutesModel
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


# Minutes model training


def trainMinutes(save=True):
    conn = sqlite3.connect("NBA.db")
    playerLogCache, posCache, teamCache, statusDF, playerAvgCache, playerUsageCache, oppPosCache, teamGameTotals = preloadCaches(conn)

    logsQuery = """
        SELECT pgl.player_id, pgl.minutes AS actual_minutes,
               g.game_date, p.team_id,
               CASE WHEN g.home_team_id = p.team_id
                    THEN g.away_team_id ELSE g.home_team_id
               END AS opp_team_id
        FROM Player_game_logs pgl
        JOIN Games g   ON pgl.game_id   = g.game_id
        JOIN Players p ON pgl.player_id = p.player_id
        WHERE pgl.minutes >= 3
        ORDER BY g.game_date
    """
    logs = pd.read_sql_query(logsQuery, conn)
    conn.close()

    featureRows = []
    targets = []

    for row in logs.itertuples(index=False):
        rolling = playerLogCache.get(row.player_id)
        if rolling is None:
            continue

        past = rolling[rolling["game_date"] < row.game_date].tail(10)
        if len(past) < 5:
            continue

        # Basic stats
        avgMin10 = float(past["minutes"].mean())
        minStd10 = float(past["minutes"].std() or 0)
        last3Mins = float(past.head(3)["minutes"].mean())
        last1Mins = float(past.iloc[0]["minutes"])
        minTrend = float(past.head(5)["minutes"].mean()) - avgMin10

        # Injury context
        playerStatus = statusDF[
                (statusDF.player_id == row.player_id) &
                (statusDF.scrape_date == (
                    datetime.strptime(row.game_date, "%Y-%m-%d") - timedelta(days=1)
                ).strftime("%Y-%m-%d"))
        ]
        isQuestionable = 1 if not playerStatus.empty and playerStatus.iloc[0]["status"] == "Questionable" else 0

        pos = posCache.get(row.player_id, None)
        posVal = {"PG": 1, "SG": 2, "SF": 3, "PF": 4, "C": 5}.get(pos, 3)

        featureRows.append({
            "avgMin10": avgMin10,
            "minStd10": minStd10,
            "last3Mins": last3Mins,
            "last1Mins": last1Mins,
            "minTrend": minTrend,
            "isQuestionable": isQuestionable,
            "pos": posVal,
        })
        targets.append(row.actual_minutes)


    X = pd.DataFrame(featureRows)
    y = pd.Series(targets)

    splitIdx = int(len(X) * 0.8)
    XTrain, XTest = X.iloc[:splitIdx], X.iloc[splitIdx:]
    yTrain, yTest = y.iloc[:splitIdx], y.iloc[splitIdx:]

    model = XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            min_child_weight = 5,
            n_jobs=-1,
            random_state=42,
    )
    model.fit(XTrain, yTrain)

    mae = mean_absolute_error(yTest, model.predict(XTest))
    print(f"Minutes model MAE: {mae:.2f}")

    if save:
        path = Path("models/nba_minutes_model.joblib")
        joblib.dump(model, path)
        print(f"Minutes model saved to {path}")
    
    return model

