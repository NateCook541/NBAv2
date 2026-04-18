import os
import sqlite3
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from datetime import datetime, timedelta

from models.evaluate import evaluateModel
from features.featureCollector import buildFeatures, featureOrder
from betting.cailbrator import fitCailbrator


def _modelMetaPath():
    return Path(__file__).parent / "nba_model_meta.joblib"


def _minutesMetaPath():
    return Path(__file__).parent / "nba_minutes_model_meta.joblib"


def _splitChronologically(X, y, dates, holdout_ratio=0.2):
    if len(X) < 10:
        raise ValueError("Not enough rows to create a chronological split")

    splitIdx = max(1, int(len(X) * (1 - holdout_ratio)))
    splitIdx = min(splitIdx, len(X) - 1)

    return (
        X.iloc[:splitIdx],
        X.iloc[splitIdx:],
        y.iloc[:splitIdx],
        y.iloc[splitIdx:],
        dates.iloc[:splitIdx],
        dates.iloc[splitIdx:],
    )

def preloadCaches(conn):
    print(f"Loading caches")

    # Player logs
    allLogs = pd.read_sql_query("""
        SELECT pgl.player_id, pgl.game_id, g.game_date, pgl.points, pgl.minutes, 
               pgl.fg_pct, pgl.is_home, pgl.rest_days,
               CASE WHEN pgl.is_home = 1 THEN g.home_team_id ELSE g.away_team_id END AS team_id
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

    teamGameTotals = allLogs.groupby(["game_id", "team_id"])["points"].sum().to_dict()

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

    return playerLogCache, posCache, teamCache, status, oppPosCache, teamGameTotals

def generateTrainingData(dbPath="NBA.db", startDate=None, endDate=None, minutesModel=None, cachePath=None):
    if (cachePath and os.path.exists(cachePath) and minutesModel is None):
        print("Loading saved training data")
        df = pd.read_parquet(cachePath)
        X = df.drop(columns=["__target__", "__date__"])
        y = df["__target__"].rename("points")
        dates = df["__date__"].rename("game_date")
        if endDate:
            mask = dates < endDate
            X, y, dates = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True), dates[mask].reset_index(drop=True)
        
        return X, y, dates

    conn = sqlite3.connect(dbPath)
    playerLogCache, posCache, teamCache, statusDF, oppPosCache, teamGameTotals = preloadCaches(conn) 

    logsQuery = """
        SELECT
            pgl.player_id,
            pgl.game_id,
            pgl.points          AS actual_points,
            pgl.is_home,
            pgl.rest_days,
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
    """
    params = []
    if startDate:
        logsQuery += " AND g.game_date >= ?"
        params.append(startDate)
    if endDate:
        logsQuery += " AND g.game_date < ?"
        params.append(endDate)
    logsQuery += " ORDER BY g.game_date, pgl.game_id, pgl.player_id"
    logs = pd.read_sql_query(logsQuery, conn, params=params)
    logs = logs.dropna(subset=["team_id", "opp_team_id"])
    print(f"Building features for {len(logs)} log rows")
        
    featureRows = []
    targets = []
    validDates = []
    skipped = 0

    if minutesModel is None:
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
            oppPosCache=oppPosCache,
            teamGameTotals=teamGameTotals,
            minutesModel=minutesModel,
            currentIsHome=row.is_home,
            currentRestDays=row.rest_days,
        )
        if features is None:
            skipped += 1
            continue
        
        featureRows.append(features)
        targets.append(row.actual_points)
        validDates.append(row.game_date)

    conn.close()
        
    print(f"Built {len(featureRows)} rows  |  skipped {skipped} (insufficient history)")
    X = pd.concat(featureRows, ignore_index=True)
    y = pd.Series(targets, name="points")
    dates = pd.Series(validDates, name="game_date")
    return X, y, dates

def trainModel(save=True, metrics=False, dbPath="NBA.db", train_end_date=None, cachePath=None):
    minutesModel = trainMinutes(save=save, dbPath=dbPath, endDate=train_end_date)
    X, y, dates = generateTrainingData(
            dbPath=dbPath, 
            endDate=train_end_date, 
            minutesModel=minutesModel,
            cachePath=cachePath 
    )

    mask = X["avgPts10"] > 0
    X, y, dates = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True), dates[mask].reset_index(drop=True)

    XTrain, XCal, yTrain, yCal, trainDates, calDates = _splitChronologically(X, y, dates)

    propMask = yCal >= 12  # only players scoring 12+ on average
    XCal = XCal[propMask].reset_index(drop=True)
    yCal = yCal[propMask].reset_index(drop=True)
    calDates = calDates[propMask].reset_index(drop=True)
    propPlayerMask = (XCal["avgPts10"] >= 12) & (XCal["avgMin10"] >= 15)
    print(f"[calibrator] Cal set after prop filter: {len(yCal)} rows, mean actual: {yCal.mean():.1f}")

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
        predictions = evaluateModel(model, XCal, yCal)
    else:
        predictions = model.predict(XCal)

    print(f"[DEBUG] Train rows: {len(XTrain)}, Cal rows: {len(XCal)}")
    print(f"[DEBUG] Cal date range: {calDates.iloc[0]} to {calDates.iloc[-1]}")
    print(f"[DEBUG] Cal mean actual: {yCal.mean():.2f}, mean predicted: {predictions.mean():.2f}")

    residualStd = float(np.std(yCal.values - predictions))
    calibratorPath = Path(__file__).parent / "nba_calibrator.joblib"
    calibrator = fitCailbrator(
        predictions,
        yCal,
        savePath=calibratorPath if save else None,
        metadata={
            "train_end_date": train_end_date,
            "calibration_start_date": calDates.iloc[0],
            "calibration_end_date": calDates.iloc[-1],
            "rows": int(len(yCal)),
        },
    )
    
    # Get the feature importance data as well
    importance = pd.DataFrame({
        "feature": XCal.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    print("\nFeature importances:")
    print(importance.to_string(index=False))
    
    if save:
        modelPath = Path(__file__).parent / "nba_model.joblib"
        joblib.dump(model, modelPath)
        joblib.dump({
            "train_end_date": train_end_date,
            "train_start_date": trainDates.iloc[0],
            "train_last_date": trainDates.iloc[-1],
            "calibration_start_date": calDates.iloc[0],
            "calibration_end_date": calDates.iloc[-1],
            "train_rows": int(len(XTrain)),
            "calibration_rows": int(len(XCal)),
        }, _modelMetaPath())
        print(f"\nModel saved to {modelPath}")

    return model, calibrator


# Minutes model training


def trainMinutes(save=True, dbPath="NBA.db", endDate=None):
    conn = sqlite3.connect(dbPath)
    playerLogCache, posCache, teamCache, statusDF, oppPosCache, teamGameTotals = preloadCaches(conn)

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
    """
    params = []
    if endDate:
        logsQuery += " AND g.game_date < ?"
        params.append(endDate)
    logsQuery += " ORDER BY g.game_date, pgl.game_id, pgl.player_id"
    logs = pd.read_sql_query(logsQuery, conn, params=params)
    conn.close()

    featureRows = []
    targets = []
    validDates = []

    for row in logs.itertuples(index=False):
        rolling = playerLogCache.get(row.player_id)
        if rolling is None:
            continue

        past = rolling[rolling["game_date"] < row.game_date].tail(10).sort_values("game_date", ascending=False)
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
        validDates.append(row.game_date)


    X = pd.DataFrame(featureRows)
    y = pd.Series(targets)

    dates = pd.Series(validDates, name="game_date")
    XTrain, XTest, yTrain, yTest, trainDates, testDates = _splitChronologically(X, y, dates)
    
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
        joblib.dump({
            "train_end_date": endDate,
            "train_start_date": trainDates.iloc[0],
            "train_last_date": trainDates.iloc[-1],
            "validation_start_date": testDates.iloc[0],
            "validation_end_date": testDates.iloc[-1],
            "train_rows": int(len(XTrain)),
            "validation_rows": int(len(XTest)),
        }, _minutesMetaPath())
        print(f"Minutes model saved to {path}")
    
    return model
