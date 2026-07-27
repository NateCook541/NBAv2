import sqlite3
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from config import (
    DB_PATH, RESULTS_MODEL_PATH, RESULTS_META_PATH,
    RESULTS_MODEL_PARAMS, RESULTS_TARGET_MODE, HOLDOUT_RATIO,
)

from features.resultsBuilder import buildTotalsFeatures, featureOrder, RESULTS_FEATURES

RESULTS_BUNDLE_VERSION = 3   # v3: added market-line features (market_total_close, line_minus_naive, open_close_move)


# Helpers


# Provides a test / train split that prevents data leakage
def _splitChronologically(X, y, dates, holdoutRatio=HOLDOUT_RATIO):
    if len(X) < 10:
        raise ValueError("Not enough rows for a chronological split")

    split = max(1, min(int(len(X) * (1 - holdoutRatio)), len(X) - 1))

    return (
            X.iloc[:split], X.iloc[split:],
            y.iloc[:split], y.iloc[split:],
            dates.iloc[:split], dates.iloc[split:],
    )


# Builds the full totals feature matrix from a games dataframe.
# One row per game (home-team perspective) so each game appears once.
# Target is the final combined total points of the game.
def _buildResultsFeatures(caches, gamesDF):
    rows, targets, validDates = [], [], []
    skipped = 0

    for game in gamesDF.itertuples(index=False):
        features = buildTotalsFeatures(
                gameID = game.game_id,
                date = game.game_date,
                teamID = int(game.home_team_id),
                oppTeamID = int(game.away_team_id),
                teamGameCache = caches.teamGameCache,
                statusDF = caches.statusDF,
                playerLogCache = caches.playerLogCache,
                teamGameTotals = caches.teamGameTotals,
                h2hCache = caches.h2hCache,
                oddsCache = caches.oddsCache,
        )

        if features is None:
            skipped += 1
            continue

        rows.append(features)
        targets.append(game.total_pts)
        validDates.append(game.game_date)

    print(
            f"[ResultsBundle] Built {len(rows)} rows "
            f"(skipped {skipped} for insufficient history)"
    )

    if not rows:
        raise ValueError("No results feature rows built — is the DB populated?")

    X = pd.concat(rows, ignore_index=True)
    y = pd.Series(targets, name="total_pts")
    dates = pd.Series(validDates, name="game_date")
    return X, y, dates


# Public bundle class


class ResultsBundle:
    """
    Wraps the trained XGBoost game-totals model with its metadata.

    Predicts the combined total points of a game, for over/under work.

    Attributes:
    model : XGBRegressor
    meta  : (dict) Training dates, row counts, MAE
    """

    def __init__(self, model, meta):
        self.model = model
        self.meta = meta


    # Prediction


    def predict(self, features):
        return float(self.predictBatch(features)[0])

    def predictBatch(self, features):
        raw = self.model.predict(features[RESULTS_FEATURES])
        # In residual mode the model learned (total - naive projection); add it back.
        if self.meta.get("target_mode", "absolute") == "residual":
            base = features["naive_total_projection"].to_numpy(dtype=float)
            base = np.where(np.isfinite(base), base, 0.0)
            raw = raw + base
        return raw

    def featureImportance(self):
        # Prefer trained feature names when available.
        cols = getattr(self.model, "feature_names_in_", None)
        if cols is None:
            cols = list(range(len(self.model.feature_importances_)))
        return (
                pd.DataFrame({
                    "feature": cols,
                    "importance": self.model.feature_importances_,
                })
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
        )


    # Persistence


    def save(self, modelPath=RESULTS_MODEL_PATH, metaPath=RESULTS_META_PATH):
        joblib.dump(self.model, modelPath)
        joblib.dump(self.meta, metaPath)
        print(f"[ResultsBundle] Saved model - {modelPath}")

    @classmethod
    def load(cls, modelPath=RESULTS_MODEL_PATH, metaPath=RESULTS_META_PATH):
        if not Path(modelPath).exists():
            raise FileNotFoundError(f"No results model found at {modelPath}")

        model = joblib.load(modelPath)
        meta = joblib.load(metaPath) if Path(metaPath).exists() else {}

        return cls(model, meta)

    @classmethod
    def loadIfExists(cls, modelPath=RESULTS_MODEL_PATH):
        if not Path(modelPath).exists():
            return None
        return cls.load(modelPath)


    # Training


    @classmethod
    def train(cls, caches, dbPath=DB_PATH, endDate=None, save=True):
        """
        Trains the game-totals model from the DB.

        Steps
        1. Pull every completed game (Results joined to Games) up to endDate.
        2. Build one totals feature row per game (home perspective).
        3. Chronological train / test split (80/20 by default).
        4. Fit XGBoost on the pruned RESULTS_FEATURES with the residual target
           (total - naive projection), report MAE on the held-out tail.
        """

        # 1. Every completed game with its final total, in date order
        conn = sqlite3.connect(str(dbPath))
        query = f"""
            SELECT
                r.game_id,
                g.game_date,
                r.home_team_id,
                r.away_team_id,
                (r.home_score + r.away_score) AS total_pts
            FROM Results r
            JOIN Games g ON r.game_id = g.game_id
            {"WHERE g.game_date < '" + endDate + "'" if endDate else ""}
            ORDER BY g.game_date, r.game_id
        """
        games = pd.read_sql_query(query, conn)
        conn.close()

        if games.empty:
            raise ValueError("No completed games found to train on")

        print(f"[ResultsBundle] Training on {len(games)} completed games")

        # 2. Build features
        X, y, dates = _buildResultsFeatures(caches, games)

        # 3. Chrono split
        XTrain, XTest, yTrain, yTest, trainDates, testDates = (
                _splitChronologically(X, y, dates)
        )

        targetMode = str(RESULTS_TARGET_MODE).lower().strip()
        if targetMode not in ("residual", "absolute"):
            raise ValueError(f"Unsupported RESULTS_TARGET_MODE: {RESULTS_TARGET_MODE}")

        # 4. Fit on the pruned feature set. In residual mode the tree learns only the
        #    correction on top of naive_total_projection (which alone is a strong predictor).
        baseTrain = XTrain["naive_total_projection"].to_numpy(dtype=float)
        baseTest = XTest["naive_total_projection"].to_numpy(dtype=float)

        if targetMode == "residual":
            yTrainTarget = yTrain.to_numpy(dtype=float) - baseTrain
        else:
            yTrainTarget = yTrain

        model = XGBRegressor(**RESULTS_MODEL_PARAMS)
        model.fit(XTrain[RESULTS_FEATURES], yTrainTarget)

        testPred = model.predict(XTest[RESULTS_FEATURES])
        if targetMode == "residual":
            testPred = testPred + baseTest

        mae = mean_absolute_error(yTest, testPred)
        print(
                f"[ResultsBundle] Train rows: {len(XTrain)}  Test rows: {len(XTest)}\n"
                f"[ResultsBundle] Target mode: {targetMode}  Features: {len(RESULTS_FEATURES)}\n"
                f"[ResultsBundle] Test MAE: {mae:.2f}  "
                f"(mean actual total: {yTest.mean():.1f})"
        )

        meta = {
                "train_end_date": endDate,
                "train_start_date": trainDates.iloc[0],
                "train_last_date": trainDates.iloc[-1],
                "validation_start": testDates.iloc[0],
                "validation_end": testDates.iloc[-1],
                "trainRows": int(len(XTrain)),
                "validationRows": int(len(XTest)),
                "mae": round(float(mae), 3),
                "target_mode": targetMode,
                "results_bundle_version": RESULTS_BUNDLE_VERSION,
        }

        bundle = cls(model, meta)

        # Expose the held-out slice so a totals calibrator can be fit on it without
        # re-splitting or re-predicting (mirrors PointsBundle.calPredictions/calActuals).
        # Transient — not persisted with the model.
        bundle.calPredictions = np.asarray(testPred, dtype=float)
        bundle.calActuals = yTest.to_numpy(dtype=float)
        bundle.calDates = testDates.reset_index(drop=True)

        if save:
            bundle.save()
        return bundle


    # Backtest safety


    # Needed to check if safe for the backtest testing
    def isSafeFor(self, backtestStartDate):
        if int(self.meta.get("results_bundle_version", 0)) < RESULTS_BUNDLE_VERSION:
            return False

        end = self.meta.get("train_end_date")
        if not end:
            last = self.meta.get("train_last_date", "")
            return str(last) < str(backtestStartDate)

        return end <= backtestStartDate

    # Check for if results model is new or can reuse its training
    def isCurrent(self, endDate=None):
        modelEnd = self.meta.get("train_end_date")
        if endDate is None:
            return modelEnd is None
        if modelEnd is None:
            return True
        return modelEnd >= endDate
