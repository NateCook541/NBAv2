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
            "my_prob":        c["myProb"],
            "fair_open":      c["bookProb"],
            "edge":           c["edge"],
            "recorded_at":    recordedAt,
        } for c in candidates]
        DBManager(dbPath).insertCLVCandidates(rows)

    return candidates


def _fairForSide(overOdds, underOdds, side):
    fairOver, fairUnder = _removeVig(overOdds, underOdds)
    return fairOver if side == "over" else fairUnder


def computeCLV(dbPath="NBA.db", date=None):
    """
    Match each open candidate to the CLOSE snapshot and compute CLV

    clv_prob = fair_close - fair_open (>0 means market moved toward us)
    clv_points = (close-open) for over (open-close) for under
    beat_close = 1 if clv_prob > 0
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    conn = sqlite3.connect(str(dbPath))
    ledger = pd.read_sql_query(
        "SELECT * FROM CLVLedger WHERE game_date = ?", conn, params=[date]
    )
    close = pd.read_sql_query(
        """
        SELECT player_name, line, over_odds, under_odds
        FROM PropSnapshots
        WHERE game_date = ? AND snapshot_type = 'close'
          AND over_odds IS NOT NULL AND under_odds IS NOT NULL
        """,
        conn, params=[date],
    )
    conn.close()

    if ledger.empty:
        print(f"[clv] No ledger rows for {date}.")
        return []
    if close.empty:
        print(f"[clv] No close snapshot for {date} — run --snapshot-close first.")
        return []

    close["norm"] = close["player_name"].apply(_normalize)
    updates = []
    for _, row in ledger.iterrows():
        cand = close[close["norm"] == _normalize(row.player_name)]
        if cand.empty:
            continue
        # nearest close line to the open line (books may re-line)
        cand = cand.assign(_d=(cand["line"] - row.open_line).abs())
        best = cand.sort_values("_d").iloc[0]

        fairOpen = row.fair_open
        fairClose = _fairForSide(best.over_odds, best.under_odds, row.side)
        closeSideOdds = best.over_odds if row.side == "over" else best.under_odds
        clvProb = fairClose - fairOpen
        clvPoints = (best.line - row.open_line) if row.side == "over" \
            else (row.open_line - best.line)

        updates.append({
            "game_date": row.game_date,
            "player_name": row.player_name,
            "side": row.side,
            "close_line": float(best.line),
            "close_side_odds": int(closeSideOdds),
            "fair_close": round(float(fairClose), 4),
            "clv_prob": round(float(clvProb), 4),
            "clv_points": round(float(clvPoints), 2),
            "beat_close": 1 if clvProb > 0 else 0,
        })

    if updates:
        DBManager(dbPath).updateCLVClose(updates)

    beat = sum(u["beat_close"] for u in updates)
    n = len(updates)
    if n:
        meanProb = sum(u["clv_prob"] for u in updates) / n
        meanPts = sum(u["clv_points"] for u in updates) / n
        print(f"[clv] {date}: {n} matched | beat-close {beat}/{n} "
              f"({beat/n:.0%}) | mean clv_prob {meanProb:+.4f} | "
              f"mean clv_points {meanPts:+.2f}")
    return updates


def clvReport(dbPath="NBA.db", startDate=None, endDate=None):
    """Aggregate CLV across the ledger
    beat close_rate, mean clv_prob/points, broken down by side"""
    conn = sqlite3.connect(str(dbPath))
    q = "SELECT * FROM CLVLedger WHERE clv_prob IS NOT NULL"
    params = []
    if startDate:
        q += " AND game_date >= ?"; params.append(startDate)
    if endDate:
        q += " AND game_date <= ?"; params.append(endDate)
    df = pd.read_sql_query(q, conn, params=params)
    conn.close()

    if df.empty:
        print("[clv-report] No graded ledger rows yet.")
        return

    def summarize(sub, label):
        n = len(sub)
        beat = int(sub["beat_close"].sum())
        print(f"  {label:<8} n={n:<4} beat-close {beat}/{n} ({beat/n:.0%}) | "
              f"mean clv_prob {sub['clv_prob'].mean():+.4f} | "
              f"mean clv_points {sub['clv_points'].mean():+.2f}")

    print("=" * 60)
    print(f"CLV report ({df['game_date'].min()} → {df['game_date'].max()})")
    print("=" * 60)
    summarize(df, "ALL")
    for side in ("over", "under"):
        sub = df[df["side"] == side]
        if not sub.empty:
            summarize(sub, side)
    if "won" in df and df["won"].notna().any():
        settled = df[df["won"].notna()]
        wr = settled["won"].mean()
        print(f"  settled n={len(settled)} win-rate {wr:.1%} "
              f"(secondary confirmation)")
    print("=" * 60)

