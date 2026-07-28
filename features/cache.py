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

from config import (
    DB_PATH, FEATURE_CACHE_PATH, MODELS_DIR,
    MIN_MINUTES_TRAIN,
)

# Using a dataclass here so callers can unpack caches by name

@dataclass
class Caches:
    playerLogCache: dict # player_id -> DataFrame of game logs
    posCache: dict # player_id -> position string
    teamCache: dict # team_id -> DataFrame of team stats
    statusDF: pd.DataFrame # Just get the injury status LOL
    oppPosCache: dict # team_id, position -> DataFrame
    teamGameTotals: dict # game_id, team_id -> total points
    teamGameCache: dict # team_id -> DataFrame of per-game team results (for totals model)
    h2hCache: dict # (min_team_id, max_team_id) -> DataFrame of head-to-head totals
    oddsCache: dict # (game_date, home_team_id, away_team_id) -> {total_open, total_close}

def preloadCaches(conn):
    print(f"[cache] Loading caches")

    # Player game logs
    allLogs = pd.read_sql_query("""
               SELECT pgl.player_id, pgl.game_id, g.game_date,
               pgl.points, pgl.minutes, pgl.fg_pct,
               pgl.is_home, pgl.rest_days,
               CASE WHEN pgl.is_home = 1
                    THEN g.home_team_id
                    ELSE g.away_team_id
               END AS team_id,
               CASE WHEN pgl.is_home = 1
                    THEN g.away_team_id
                    ELSE g.home_team_id
               END AS opp_team_id
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

    # Per-game team results (for the totals / results model)
    # One row per team per game, from each team's own perspective.
    teamGameCache, h2hCache = _buildTeamGameCaches(conn)

    # Historical market totals (opening/closing lines), keyed for a direct join
    # to each game. Present only for 2016-2023 (the odds archive); games without
    # a line simply aren't in the dict and features fall back to NaN.
    oddsCache = _buildOddsCache(conn)

    print("[cache] Cache loading done")

    return Caches(
            playerLogCache = playerLogCache,
            posCache = posCache,
            teamCache = teamCache,
            statusDF = status,
            oppPosCache = oppPosCache,
            teamGameTotals = teamGameTotals,
            teamGameCache = teamGameCache,
            h2hCache = h2hCache,
            oddsCache = oddsCache
    )


def _buildOddsCache(conn):
    """(game_date, home_team_id, away_team_id) -> {total_open, total_close}
    from the Odds_archive table. Only ~2016-2023 games have rows."""
    try:
        odds = pd.read_sql_query(
            """SELECT game_date, home_team_id, away_team_id, total_open, total_close
               FROM Odds_archive""", conn)
    except Exception:
        # Table may not exist in older DBs — degrade to no lines.
        return {}
    cache = {}
    for r in odds.itertuples(index=False):
        cache[(r.game_date, int(r.home_team_id), int(r.away_team_id))] = {
            "total_open": r.total_open,
            "total_close": r.total_close,
        }
    print(f"[cache] Loaded {len(cache)} odds-archive lines")
    return cache


def _buildTeamGameCaches(conn):
    """
    Builds the two caches the totals model needs from Results + Games + Teams.

    teamGameCache: team_id -> DataFrame (one row per game that team played), newest
        rows appended in date order, with the columns resultsBuilder reads:
            date, game_id, opp_team_id, pts_scored, pts_allowed, total_pts,
            pace, off_rtg, def_rtg, rest_days
    h2hCache: frozen (min_id, max_id) team pair -> DataFrame of shared games
        with columns date, total_pts.

    off_rtg / def_rtg / pace come from the per-game Team_game_ratings table (NBA
    stats API) when available, so those are true per-game values. For any game with
    no rating row (e.g. not yet backfilled) it falls back to the season-start
    snapshot in the Teams table, so the cache degrades gracefully rather than zeroing.
    """
    # Scores joined to schedule for dates, opponent and season.
    games = pd.read_sql_query("""
        SELECT
            r.game_id,
            g.game_date  AS date,
            g.season,
            r.home_team_id,
            r.away_team_id,
            r.home_score,
            r.away_score
        FROM Results r
        JOIN Games g ON r.game_id = g.game_id
        ORDER BY g.game_date, r.game_id
    """, conn)

    if games.empty:
        return {}, {}

    # Season-start rating / pace snapshots. One row per (team, season). Used as a
    # fallback when a game has no per-game rating row.
    teamStats = pd.read_sql_query("SELECT team_id, off_rtg, def_rtg, pace, date FROM Teams", conn)
    # date like "2023-10-01" -> season 2024 (season is the calendar year it ends in)
    teamStats["season"] = teamStats["date"].str[:4].astype(int) + 1
    seasonLookup = {
        (int(r.team_id), int(r.season)): (r.off_rtg, r.def_rtg, r.pace)
        for r in teamStats.itertuples(index=False)
    }

    # Real per-game ratings, keyed (game_id, team_id).
    gameRatings = pd.read_sql_query(
        "SELECT game_id, team_id, off_rtg, def_rtg, pace FROM Team_game_ratings", conn
    )
    gameRatingLookup = {
        (str(r.game_id), int(r.team_id)): (r.off_rtg, r.def_rtg, r.pace)
        for r in gameRatings.itertuples(index=False)
    }

    def _ratings(gameID, teamID, season):
        # Prefer the real per-game row; fall back to the season snapshot; then zeros.
        perGame = gameRatingLookup.get((str(gameID), int(teamID)))
        if perGame is not None and all(v is not None for v in perGame):
            return perGame
        return seasonLookup.get((int(teamID), int(season)), (0.0, 0.0, 0.0))

    # Explode each game into two team-perspective rows.
    rows = []
    for g in games.itertuples(index=False):
        for teamID, oppID, scored, allowed in (
            (g.home_team_id, g.away_team_id, g.home_score, g.away_score),
            (g.away_team_id, g.home_team_id, g.away_score, g.home_score),
        ):
            offRtg, defRtg, pace = _ratings(g.game_id, teamID, g.season)
            rows.append({
                "team_id":     int(teamID),
                "opp_team_id": int(oppID),
                # Venue of this game == the home team's arena. Recorded so the
                # totals model can trace where a team last played (travel features).
                "home_team_id": int(g.home_team_id),
                "game_id":     g.game_id,
                "date":        g.date,
                "season":      int(g.season),
                "pts_scored":  int(scored),
                "pts_allowed": int(allowed),
                "total_pts":   int(scored) + int(allowed),
                "off_rtg":     float(offRtg) if offRtg is not None else 0.0,
                "def_rtg":     float(defRtg) if defRtg is not None else 0.0,
                "pace":        float(pace) if pace is not None else 0.0,
            })

    teamGames = pd.DataFrame(rows).sort_values(["team_id", "date"]).reset_index(drop=True)

    # Rest days: gap since that team's previous game. First game of a season -> 20 (same
    # convention scrapeLogs uses so early-season rows aren't treated as back-to-backs).
    teamGames["_prev_date"] = teamGames.groupby("team_id")["date"].shift(1)
    prevDT = pd.to_datetime(teamGames["_prev_date"], errors="coerce")
    curDT = pd.to_datetime(teamGames["date"], errors="coerce")
    restDays = (curDT - prevDT).dt.days
    teamGames["rest_days"] = restDays.fillna(20).clip(upper=20).astype(int)
    teamGames = teamGames.drop(columns=["_prev_date"])

    teamGameCache = {
        tid: grp.reset_index(drop=True)
        for tid, grp in teamGames.groupby("team_id")
    }

    # Head-to-head totals, keyed by the unordered team pair (dedupe the two perspectives).
    h2h = games.copy()
    h2h["total_pts"] = h2h["home_score"] + h2h["away_score"]
    h2h["pair"] = h2h.apply(
        lambda r: tuple(sorted((int(r.home_team_id), int(r.away_team_id)))), axis=1
    )
    h2hCache = {
        pair: grp[["date", "total_pts"]].sort_values("date").reset_index(drop=True)
        for pair, grp in h2h.groupby("pair")
    }

    return teamGameCache, h2hCache


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
        from features.builder import buildFeatures, featureOrder
        
        conn = sqlite3.connect(str(dbPath))
        caches = preloadCaches(conn)

        query = f"""
            SELECT
                pgl.player_id,
                pgl.game_id,
                pgl.points      AS actualPoints,
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
            {"AND g.game_date < '" + endDate + "'" if endDate else ""}
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
                    oppTeamID = row.opp_team_id,
                    cache = caches.playerLogCache,
                    posCache = caches.posCache,
                    teamCache = caches.teamCache,
                    statusDF = caches.statusDF,
                    oppPosCache = caches.oppPosCache,
                    teamGameTotals = caches.teamGameTotals,
                    minutesModel = minutesBundle,
                    currentIsHome = row.is_home,
                    currentRestDays = row.rest_days
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
        if minutesBundle is None and self.exists():
            X, y, dates = self._load(endDate=endDate)
            try:
                from features.builder import featureOrder
                missing = [c for c in featureOrder if c not in X.columns]
            except Exception:
                missing = []
            if not missing:
                return X, y, dates
            print(f"[FeatureCache] Missing columns in cached features ({len(missing)}). Rebuilding features.")

        X, y, dates = self._build(
                dbPath=dbPath,
                endDate=endDate,
                minutesBundle=minutesBundle
        )
        # Persist rebuilt features so follow-up commands in the same run
        # (for example --evaluate after --train) can reuse them instantly.
        if endDate is None:
            self._save(X, y, dates)
        return X, y, dates

    # Full rebuild and save it to a parquet file
    def buildAndSave(self, dbPath=DB_PATH, minutesBundle=None):
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
