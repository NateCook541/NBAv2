import sqlite3
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
sklearn.metrics import mean_absolute_error
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











