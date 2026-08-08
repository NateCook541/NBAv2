"""
Single day live scorer + CLV computation

scoreLiveDay: for a given date, score todays OPEN prop snapshot with the same
math the backtest uses, sourcing opp/home/rest PREGAME from the
Games schedule (never from played logs). Records bettable candidates to CLVLedger
with the open line/odds

computeCLV: after a CLOSE snapshot is captured, fills close line/odds and CLV
metrics (clv_prob primary, clv_points, beat_close) per candidate

settleCLV (deferred): after games finish, fills actual_points/won
"""

import sqlite3
from datetime import datetime

import pandas as pd

from config import DEFAULT_EDGE_THRESH, MAX_BET_EDGE
from features.cache import preloadCaches
from models.points import PointsBundle
from models.minutes import MinutesBundle
from betting.calibrator import Calibrator
from betting.under_calibrator import UnderCalibrator
from betting.filters import FilterSet, UnderFilterSet
from betting.backtest import (
    scoreOnce, _loadPlayerMap, _loadScheduleMap, _normalize,
    _removeVig, _deduplicateProps,
)
from data.dbManager import DBManager


def _loadPlayerTeams(conn):
    """player_id -> team_id from the Players table (current roster)."""
    df = pd.read_sql_query("SELECT player_id, team_id FROM Players", conn)
    return dict(zip(df.player_id, df.team_id))


def _deriveRestDays(playerLogCache, playerID, date):
    """Days since the player's last game strictly before date. None if no
    prior game (rookies / first game) — caller decides a default."""
    logs = playerLogCache.get(playerID)
    if logs is None or logs.empty:
        return None
    prior = logs[logs["game_date"] < date]
    if prior.empty:
        return None
    lastDate = prior["game_date"].max()
    return (datetime.strptime(date, "%Y-%m-%d")
            - datetime.strptime(lastDate, "%Y-%m-%d")).days


def _loadSnapshot(conn, date, snapshotType):
    """Load a live snapshot for date, both odds present, dedup alternate lines."""
    df = pd.read_sql_query(
        """
        SELECT game_date, player_name, line, over_odds, under_odds
        FROM PropSnapshots
        WHERE game_date = ? AND snapshot_type = ?
          AND over_odds IS NOT NULL AND under_odds IS NOT NULL
        """,
        conn,
        params=[date, snapshotType],
    )
    if df.empty:
        return df
    return _deduplicateProps(df)


def scoreLiveDay(dbPath="NBA.db", date=None, overThresh=DEFAULT_EDGE_THRESH,
                 underThresh=DEFAULT_EDGE_THRESH, record=True):
    """
    Score todays OPEN snapshot and record bettable candidates to CLVLedger
    Returns the list of candidate dicts and logs unmatched players
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    points = PointsBundle.loadIfExists()
    minutes = MinutesBundle.loadIfExists()
    overCal = Calibrator.loadIfExists()
    underCal = UnderCalibrator.loadIfExists()

    if points is None or overCal is None:
        raise RuntimeError("Points model / over calibrator not found")
    if underCal is None:
        raise RuntimeError(
            "Under calibrator not found"
        )

    overFilter = FilterSet.production()
    underFilter = UnderFilterSet.production()

    conn = sqlite3.connect(str(dbPath))
    caches = preloadCaches(conn)
    playerMap = _loadPlayerMap(conn)
    scheduleMap = _loadScheduleMap(conn)
    playerTeams = _loadPlayerTeams(conn)
    props = _loadSnapshot(conn, date, "open")
    conn.close()

    if props.empty:
        print(f"[live] No open snapshot props for {date}.")
        return []

    candidates = []
    misses = {"noPlayer": 0, "noTeam": 0, "noGame": 0, "noFeatures": 0}
    missNames = []

    for _, prop in props.iterrows():
        pid = playerMap.get(_normalize(prop.player_name))
        if pid is None:
            misses["noPlayer"] += 1
            missNames.append(prop.player_name)
            continue
        teamID = playerTeams.get(pid)
        if teamID is None:
            misses["noTeam"] += 1
            continue
        sched = scheduleMap.get((date, int(teamID)))
        if sched is None:
            misses["noGame"] += 1
            continue

        rest = _deriveRestDays(caches.playerLogCache, pid, date)
        ctx = {
            "player_id": pid,
            "team_id": sched["team_id"],
            "opp_team_id": sched["opp_team_id"],
            "is_home": sched["is_home"],
            "rest_days": rest if rest is not None else 2,  # neutral default
        }

        over = scoreOnce(prop, ctx, caches, points, minutes, overCal,
                         overFilter, "over", overThresh, MAX_BET_EDGE)
        under = scoreOnce(prop, ctx, caches, points, minutes, underCal,
                          underFilter, "under", underThresh, MAX_BET_EDGE)
        if over is None or under is None:
            misses["noFeatures"] += 1
            continue

        # keep bettable sides if both, keep higher edge (combined backtest rule)
        picks = [s for s in (over, under) if s["bettable"]]
        if not picks:
            continue
        pick = max(picks, key=lambda s: s["edge"])
        pick["player_id"] = pid
        candidates.append(pick)

    print(f"[live] {date}: {len(candidates)} candidate(s) from {len(props)} props "
          f"| skips: {misses}")
    if missNames:
        print(f"[live] unmatched players ({len(missNames)}): "
              f"{', '.join(sorted(set(missNames))[:15])}"
              f"{' ...' if len(set(missNames)) > 15 else ''}")

    if record and candidates:
        recordedAt = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        rows = [{
            "game_date":      c["date"],
            "player_name":    c["player"],
            "player_id":      c["player_id"],
            "side":           c["side"],
            "open_line":      c["line"],
            "open_side_odds": c["sideOdds"],
            "predicted":      c["predicted"],
            "pred_diff":      round(c["predicted"] - c["line"], 2),
            "my_prob":        c["myProb"],
            "fair_open":      c["bookProb"],
            "edge":           c["edge"],
            "recorded_at":    recordedAt,
        } for c in candidates]
        DBManager(dbPath).insertCLVCandidates(rows)

    return candidates

