import sqlite3
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from config import (
    DB_PATH, MINUTES_PATH, MINUTES_META_PATH,
    MINUTES_MODEL_PARAMS, HOLDOUT_RATIO, MIN_MINUTES_TRAIN,
)

MINUTES_FEATURES = [
    "avgMin10",
    "minStd10",
    "last3Mins",
    "last1Mins",
    "minTrend",
    "isQuestionable",
    "pos",
]

POS_MAP = {"PG": 1, "SG": 2, "SF": 3, "PF": 4, "C": 5}

# Provides a test / train split that prevents data leakage
def _split_chronologically(X, y, dates, holdoutRatio=HOLDOUT_RATIO):
    if len(X) < 10:
        raise ValueError("Not enough rows for a chronological split")

    split = max(1, min(int(len(X) * (1 - holdoutRatio)), len(X) - 1))

    return (
            X.iloc[:split], X.iloc[split:],
            y.iloc[:split], y.iloc[split:],
            dates.iloc[:split], dates.iloc[split:],
    )

def _buildMinsFeatures(playerLogCache, statusDF, posCache, logsDF):
    rows, targets, validDates = [], [], []

    for row in logsDF.itertuples(index=False):
        rolling = playerLogCache.get(row.player_id)
        if rolling is None:
            continue

        past = (
                rolling[rolling["game_date"] < row.game_date]
                .tail(10)
                .sort_values("game_date", ascending=False)
        )
        if len(past) < 5:
            continue

        avgMin10 = float(past["minutes"].mean())
        minStd10 = float(past["minutes"].std() or 0)
        last3Mins = float(past.head(3)["minutes"].mean())
        last1Mins = float(past.iloc[0]["minutes"])
        minTrend = float(past.head(5)["minutes"].mean()) - avgMin10

        dayBefore = (
            datetime.strptime(row.game_date, "%Y-%m-%d") - timedelta(days=1)
        ).strftime("%Y-%m-%d")

        playerStatus = statusDF[
                (statusDF.player_id == row.player_id) &
                (statusDF.scrape_date == dayBefore)
        ]
        
        isQuestionable = (
                1 if not playerStatus.empty
                and playerStatus.iloc[0]["status"] == "Questionable"
                else 0
        )

        posStr = posCache.get(row.playerID, None)
        posVal = POS_MAP.get(posStr, 3)

        rows.append({
            "avgMin10": avgMin10,
            "minStd10": minStd10,
            "last3Mins": last3Mins,
            "last1Mins": last1Mins,
            "minTrend": minTrend,
            "isQuestionable": isQuestionable,
            "pos": posVal,
        })
        targets.append(row.actualMinutes)
        validDates.append(row.game_date)

    X = pd.DataFrame(rows, columns=MINUTES_FEATURES)
    y = pd.Series(targets, name="minutes")
    dates = pd.Series(validDates, name="game_date")
    return X, y, dates


# Public bundle class


class MinutesBundle:
    """
    Wraps the trained XGBoost minutes model with its metadata

    Attributes:
    model : XGBRegressor
    meta  : (dict) Training dates, row counts, MAE
    """

    def __init__(self, model):
        self.model = model
        self.meta  = meta


    # Prediction


    def predict(self, features):
        return float(self.model.predict(features[MINUTES_FEATURES])[0])

    def predictBatch(self, features):
        return self.model.predict(features[MINUTES_FEATURES])


    # Persistence


    def save(self, modelPath=MINUTES_PATH, metaPath=MINUTES_META_PATH):
        joblib.dump(self.model, modelPath)
        joblib.dump(self.meta, metaPath)
        print(f"[MinutesBundle] Saved model - {model_path}")

    @classmethod
    def load(cls, modelPath=MINUTES_PATH, metaPath=MINUTES_META_PATH):
        if not Path(modelPath).exists():
            raise FileNotFoundError(f"No minutes model at {modelPath}")

        model = joblib.load(modelPath)
        meta = joblib.load(metaPath) if Path(metaPath).exists() else {}

        return cls(model, meta)

    @classmethod
    def loadIfExists(cls, modelPath=MINUTES_PATH):
        if not Path(modelPath).exists():
            return None
        return cls.load(modelPath)
    

    # Training


    @classmethod
    def train(cls, playerLogCache, statusDF, posCache,
              dbPath=DB_PATH, endDate=None, save=True):
        
        conn = sqlite3.connect(str(dbPath))
        query = f"""
            SELECT pgl.player_id, pgl.minutes AS actual_minutes,
                   g.game_date, p.team_id,
                   CASE WHEN g.home_team_id = p.team_id
                        THEN g.away_team_id ELSE g.home_team_id
                   END AS opp_team_id
            FROM Player_game_logs pgl
            JOIN Games   g ON pgl.game_id   = g.game_id
            JOIN Players p ON pgl.player_id = p.player_id
            WHERE pgl.minutes >= {MIN_MINUTES_TRAIN}
            {"AND g.game_date < '" + end_date + "'" if end_date else ""}
            ORDER BY g.game_date, pgl.game_id, pgl.player_id
        """
        logs = pd.read_sql_query(query, conn)
        conn.close()

        X, y, dates = _buildMinutesFeatures(
                playerLogCache, statusDF, posCache, logs
        )
        
        XTrain, XTest, yTrain, yTest, trainDates, testDates = (
                _splitChronologically(X, y, dates)
        )

        model = XGBRegressor(**MINUTES_MODEL_PARAMS)
        model.fit(XTrain, yTrain)

        mae = mean_absolute_error(yTest, model.predict(XTest))
        print(f"[MinutesBundle] MAE: {mae:.2f}")

        meta = {
                "train_end_date": endDate,
                "train_start_date": trainDates.iloc[0],
                "train_last_date": trainDates.iloc[-1],
                "validation_start": testDates.iloc[0],
                "validation_end": testDates.iloc[-1],
                "trainRows": int(len(XTrain)),
                "validationRows": int(len(XTest)),
                "mae": round(mae, 3),
        }

        bundle = cls(model, meta)
        if save:
            bundle.save()
        return bundle


    # Backtest safety check


    # Needed to check if safe for the backtest testing
    def isSafeFor(self, backtestStartDate):
        end = self.meta.get("train_end_date")
        if not end:
            return false

        return end <= backtestStartDate

