"""
cache.py

Does two things
preloadCaches - Loads all DB tables into memory dicts and DFs
to speed up build features as it only needs to be called
once per run

FeatureCache - Save and loads the full training/testing data
to a parquet file so feature building doesn't need to be rebuilt
for backtestin/calibrating/etc

"""

import os
import sqlite3
import joblib
import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timedelta


# Using a dataclass here so callers can unpack caches by name

@dataclass
class Caches:
    playerLogCache # player_id -> DataFrame of game logs
    posCache # player_id -> position string
    teamCache # team_id -> DataFrame of team stats
    statusDF # Just get the injury status LOL
    oppPosCache # team_id, position -> DataFrame
    teamGameTotals: # game_id, team_id -> total points 

def preloadCaches():
    print(f"[cache] Loading caches")

    # Player game logs
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

    # Team game totals (Needed for usage rate)
    teamGameTotals = allLogs.groupby(["game_id", "team_id"])["points"].sum().to_dict()

    # Opp points allowed per position
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

    print("[cache] Cache loading done")

    return Cache(
            playerLogCache = playerLogCache, 
            posCache = posCache, 
            teamCache = teamCache, 
            statusDF = statusDF, 
            oppPosCache = oppPosCache, 
            teamGameTotals = teamGameTotals
    )


# Feature cache

# Here to manage the parquet file that stores the feature matrix
class FeatureCache:

    def __init__(self, cachePath=FEATURE_CACHE_PATH):
        self.cachePath = Path(cachePath)


    # Internal helpers


    def _load(self, endDate=None):
        print(f"[FeatureCache] Loading from {self.cachePath}")
        df = pd.read_parquet(self.cachePath)

        X = df.drop(columns=["__target__", "__date__"])
        y = df["__target__"].rename("points")
        dates = df["__date__"].rename("game_date")

        if endDate:
            mask = dates < endDate
            X = X[mask].reset_index(drop=True)
            y = y[mask].reset_index(drop=True)
            dates = dates[mask].reset_index(drop=True)
            print(f"[Feature Cache] Filtered to {len(X)} rows before {endDate}")

        return X, y, dates


    def _save(self, X, y, dates):
        self.cachePath.parent.mkdir(parents=True, exist_ok=True)
        df = X.copy()
        df["__target__"] = y.values
        df["__date__"] = dates.values
        df.to_parquet(self.cachePath)
        print(f"[Feature Cache] Saved {len(df)} rows to {self.cachePath}")


    # Full feature matrix building
    def _build(self, dbPath, endDate, minutesBundle):
        from features.featureCollector import buildFeatures, featureOrder
        
        conn = sqlite3.connect(str(dbPath))
        caches = preloadCaches(conn)

        query = """
            SELECT
                pgl.player_id,
                pgl.game_id,
                pgl.points      AS actual_points,
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
            WHERE pgl.minutes >= {MIN_MINUTES_TRAIN}
            {"AND g.game_date < '" + end_date + "'" if end_date else ""}
            ORDER BY g.game_date, pgl.game_id, pgl.player_id
        """
        logs = pd.read_sql_query(query, conn)
        logs = logs.dropna(subset=["team_id", "opp_team_id"])
        conn.close()

        print(f"[FeatureCache] Building features for {len(logs)} log rows")

        featureRows, targets, validDates = [], [], []
        skipped = 0

        for row in logs.itertuples(index=False):
            features = buildFeatures(
                    playerID = row.player_id,
                    date = row.game_date,
                    teamID = row.team_id,
                    oppTeamID, row.opp_team_id,
                    cache = caches.playerLogCache,
                    posCache = caches.posCache,
                    teamCache = caches.teamCache,
                    statusDF = caches.statusDF,
                    oppPosCache = caches.oppPosCache,
                    teamGameTotals = cache.teamGameTotals,
                    minutesModel = minuteBundle,
                    currentIsHome = row.is_home,
                    currentRestDay = row.rest_day
            )
            if features is None:
                skipped += 1
                continue

            featureRows.append(features)
            targets.append(row.actualPoints)
            validDates.append(row.game_date)

        print(
                f"[Feature Cache] Built {len(featureRows)} rows \n"
                f"[Feature Cache] Skipped {skipped} (insufficient history)"
        )

        X = pd.concat(featureRows, ignore_index=True)
        y = pd.Series(targets, name="points")
        dates = pd.Series(validDates, name="game_date")
        return X, y, dates


    # Public inteface


    def exists(self):
        return self.cachePath.exists()

    def loadOrBuild(
        self,
        endDate=None,
        dbPath=DB_PATH,
        minutesBundle=None):

        # If a mins bunlde is supplied then cache is skipped and features rebuilt as that would be using stale data

        if minutesBundle is None and self.exists():
            return self._load(endDate=endDate)
        
        return self._build(
                dbPath=dbPath,
                endDate=endDate,
                minutesBundle=minutesBundle
        )

    # Full rebuild and save it to a parquet file
    def buildAndSave(self, dbPath=dbPath, minutesBundle=None):
        X, y, dates = self._build(
                dbPath=dbPath,
                endDate=None,
                minutesBundle=minutesBundle
        )
        
        self._save(X, y, dates)
        return X, y, dates

    # Just deletes the parquet file
    def invalidate(self):
        if self.cachePath.exists():
            self.cachePath.unlink()
            print(f"[FeatureCache] Cache deleted at {self.cachePath}")

