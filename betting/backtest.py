import sqlite3
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from config import (
    DB_PATH,
    DEFAULT_EDGE_THRESH,
    MAX_BET_EDGE,
    DEFAULT_BANKROLL,
    FLAT_STAKE,
    MIN_LINE,
    MAX_LINE_DIFF,
    OVER_MIN_DISAGREEMENT,
    DEFAULT_KELLY_FRAC,
    MIN_AVGPTS_CAL,
    MIN_AVGMIN_CAL
)
from features.builder import buildFeatures
from features.cache import preloadCaches
from metrics.reporter import Reporter
from betting.filters import FilterSet

@dataclass
class BetRecord:
    date: str
    player: str
    line: float
    predicted: float
    actual: float
    predDiff: float
    rawProb: float
    myProb: float
    bookProb: float
    edge: float
    bet: bool
    stake: float
    pnl: float
    bankroll: float
    betSide: str = "over"
    betOdds: float = 0.0
    # NOTE: These are for debugging the calibrator/model layer. Could be removed in a future build
    avgPts10: float = 0.0
    last1Pts: float = 0.0

@dataclass
class SkipCounters:
    noPlayerMatch: int = 0
    noOppMatch: int = 0
    noActuals: int = 0
    noFeatures: int = 0
    noRollingHistory: int = 0
    noLine: int = 0
    noOppMatchByMonth: Counter = field(default_factory=Counter, repr=False)
    filteredOut: int = 0
    filterReasons: Counter = field(default_factory=Counter, repr=False)
    noOppMatchByMonth: Counter = field(default_factory=Counter, repr=False)


def _impliedProb(usOdds):
    if usOdds < 0:
        return abs(usOdds) / (abs(usOdds) + 100)
    return 100 / (usOdds + 100)


def _removeVig(overOdds, underOdds):
    over = _impliedProb(overOdds)
    under = _impliedProb(underOdds)
    total = over + under
    return over / total, under / total


def _payoutMultiplier(usOdds):
    if usOdds > 0:
        return usOdds / 100
    return 100 / abs(usOdds)

def _kellyFractional(edge, usOdds, fraction=DEFAULT_KELLY_FRAC):
    b = _payoutMultiplier(usOdds)
    p = edge + _impliedProb(usOdds)
    q = 1 - p
    kelly = (b * p - q) / b
    return max(0.0, kelly * fraction)

def _normalize(name):
    base = "".join(
            c for c in unicodedata.normalize("NFD", name)
            if unicodedata.category(c) != "Mn"
            ).lower().strip()
    cleaned = base.replace(".", " ").replace("'", "").replace("-", " ")
    tokens = [t for t in cleaned.split() if t]
    merged = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and len(tokens[i]) == 1 and len(tokens[i + 1]) == 1:
            merged.append(tokens[i] + tokens[i + 1])
            i += 2
            continue
        merged.append(tokens[i])
        i += 1
    return " ".join(merged)


def scoreOnce(prop, ctx, caches, points, minutes, calibrator, filterSet,
              side, edgeThresh, edgeCap):
    """
    Settlement-free, side-agnostic scoring for a single prop.

    Shared source of truth for backtest and live scoring. Given an already
    resolved context (team/opp/home/rest) and loaded bundles, builds features,
    predicts, calibrates the chosen side, computes edge vs de-vigged fair prob,
    applies the filter set, and flags `bettable`. NO actuals, NO bankroll.

    ctx keys: player_id, team_id, opp_team_id, is_home, rest_days.
    side: 'over' or 'under'. `calibrator` and `filterSet` must match the side.
    Returns a scored dict, or None if features can't be built.
    """
    features = buildFeatures(
        playerID=ctx["player_id"],
        date=prop.game_date,
        teamID=ctx["team_id"],
        oppTeamID=ctx["opp_team_id"],
        cache=caches.playerLogCache,
        posCache=caches.posCache,
        teamCache=caches.teamCache,
        statusDF=caches.statusDF,
        oppPosCache=caches.oppPosCache,
        teamGameTotals=caches.teamGameTotals,
        minutesModel=minutes,
        currentIsHome=ctx["is_home"],
        currentRestDays=ctx["rest_days"],
    )
    if features is None:
        return None

    predicted = points.predict(features)
    pos = float(features["pos"].iloc[0]) if "pos" in features.columns else None

    if side == "over":
        rawProb     = calibrator.rawProbOver(predicted, prop.line)
        myProb      = calibrator.probOver(predicted, prop.line)
        fair, _     = _removeVig(prop.over_odds, prop.under_odds)
        edge        = myProb - fair
        predDiff    = round(predicted - prop.line, 2)
        sideOdds    = prop.over_odds
        passed, reason = filterSet.passes(
            predicted=predicted, propLine=prop.line, edge=edge, pos=pos,
        )
    else:
        rawProb     = calibrator.rawProbUnder(predicted, prop.line)
        myProb      = calibrator.probUnder(predicted, prop.line)
        _, fair     = _removeVig(prop.over_odds, prop.under_odds)
        edge        = myProb - fair
        predDiff    = round(prop.line - predicted, 2)
        sideOdds    = prop.under_odds
        passed, reason = filterSet.passes(
            predicted=predicted, propLine=prop.line, edge=edge, pos=pos,
            betOdds=prop.under_odds, predDiff=predDiff,
        )

    bettable = passed and (edgeThresh < edge <= edgeCap)

    return {
        "date":       prop.game_date,
        "player":     prop.player_name,
        "line":       prop.line,
        "side":       side,
        "predicted":  round(predicted, 1),
        "predDiff":   predDiff,
        "rawProb":    round(rawProb, 3),
        "myProb":     round(myProb, 3),
        "bookProb":   round(fair, 3),
        "edge":       round(edge, 3),
        "sideOdds":   sideOdds,
        "avgPts10":   float(features["avgPts10"].iloc[0]),
        "last1Pts":   float(features["last1Pts"].iloc[0]),
        "passed":     passed,
        "filterReason": reason,
        "bettable":   bettable,
        # exact prediction retained for parity checks / stake sizing
        "_predictedExact": predicted,
    }


def _loadProps(conn, startDate=None, endDate=None):
    query = """
        SELECT prop_id, game_date, player_name, line, over_odds, under_odds
        FROM Props
        WHERE over_odds IS NOT NULL AND under_odds IS NOT NULL
    """
    params = []
    if startDate:
        query += " AND game_date >= ?"
        params.append(startDate)
    if endDate:
        query += " AND game_date <= ?"
        params.append(endDate)
    query += " ORDER BY game_date"
    df = pd.read_sql_query(query, conn, params=params)
    return _deduplicateProps(df)


def _deduplicateProps(df):
    """
    When multiple props exist for the same player on the same date
    (DraftKings alternate lines), keep the single row whose line is
    closest to the group median.  For even-sized groups where the true
    median falls between two rows, the lower line is kept — a
    conservative choice that is harder for the model to beat.

    Rows are selected by choosing the prop_id with the minimum absolute
    distance to the group median line, breaking ties by taking the
    lower line (min prop_id among tied rows).
    """
    dup_mask = df.duplicated(subset=["game_date", "player_name"], keep=False)
    n_dups = dup_mask.sum()
    if n_dups == 0:
        return df

    clean = df[~dup_mask].copy()
    dupes = df[dup_mask].copy()

    def _pick_median_row(group):
        median_line = group["line"].median()
        group = group.copy()
        group["_dist"] = (group["line"] - median_line).abs()
        min_dist = group["_dist"].min()
        candidates = group[group["_dist"] == min_dist]
        return candidates.loc[[candidates["line"].idxmin()]]

    deduped = (
        dupes
        .groupby(["game_date", "player_name"], group_keys=False)[df.columns.tolist()]
        .apply(_pick_median_row)
        .drop(columns=["_dist"], errors="ignore")
        .reset_index(drop=True)
    )

    n_removed = n_dups - len(deduped)
    print(f"[Props] Deduplicated {n_dups} rows → kept {len(deduped)} "
          f"(removed {n_removed} alternate lines)")

    return pd.concat([clean, deduped], ignore_index=True).sort_values("game_date").reset_index(drop=True)


def _loadActuals(conn):
    df = pd.read_sql_query(
            """
        SELECT pgl.player_id, g.game_date, pgl.points
        FROM Player_game_logs pgl
        JOIN Games g ON pgl.game_id = g.game_id
        """,
        conn,
        )
    return df.set_index(["player_id", "game_date"])["points"].to_dict()

def _loadPlayerMap(conn):
    df = pd.read_sql_query(
            "SELECT player_id, name FROM Players", conn
            )

    df["name_norm"] = df["name"].apply(_normalize)
    df = df.sort_values("player_id").drop_duplicates(
            subset="name_norm", keep="last"
            )

    return df.set_index("name_norm")["player_id"].to_dict()


def _loadOppMap(conn):
    df = pd.read_sql_query(
            """
        SELECT pgl.player_id, g.game_date,
               g.home_team_id, g.away_team_id,
               pgl.is_home, pgl.rest_days
        FROM Player_game_logs pgl
        JOIN Games g ON pgl.game_id = g.game_id
        """,
        conn,
        )
    result = {}

    for _, row in df.iterrows():
        if row.is_home:
            teamID    = row.home_team_id
            oppTeamID = row.away_team_id
        else:
            teamID    = row.away_team_id
            oppTeamID = row.home_team_id

        result[(int(row.player_id), row.game_date)] = {
                "team_id":     teamID,
                "opp_team_id": oppTeamID,
                "is_home":     int(row.is_home),
                "rest_days":   int(row.rest_days) if pd.notna(row.rest_days) else 0,
                }

    return result


def _loadScheduleMap(conn):
    df = pd.read_sql_query(
            """
        SELECT game_id, game_date, home_team_id, away_team_id
        FROM Games
        """,
        conn,
        )
    result = {}

    for _, row in df.iterrows():
        result[(row.game_date, int(row.home_team_id))] = {
                "team_id": int(row.home_team_id),
                "opp_team_id": int(row.away_team_id),
                "is_home": 1,
                "game_id": row.game_id,
                }
        result[(row.game_date, int(row.away_team_id))] = {
                "team_id": int(row.away_team_id),
                "opp_team_id": int(row.home_team_id),
                "is_home": 0,
                "game_id": row.game_id,
                }

    return result


# Main backtest class


class BacktestEngine:
    def __init__(self, pointsBundle, minutesBundle, calibrator, dbPath=DB_PATH, filterSet=None):
        self.points     = pointsBundle
        self.minutes    = minutesBundle
        self.calibrator = calibrator
        self.dbPath     = Path(dbPath)
        self.edgeCap    = MAX_BET_EDGE
        self.filterSet  = filterSet if filterSet is not None else FilterSet.baseline()

    def run(self, startDate=None, endDate=None, edgeThresh=DEFAULT_EDGE_THRESH, bankroll=DEFAULT_BANKROLL):
        conn = sqlite3.connect(str(self.dbPath))
        props = _loadProps(conn, startDate, endDate)

        if props.empty:
            raise ValueError("No props found for the requested timeframe")

        actuals     = _loadActuals(conn)
        playerMap   = _loadPlayerMap(conn)
        oppMap      = _loadOppMap(conn)
        scheduleMap = _loadScheduleMap(conn)
        caches      = preloadCaches(conn)
        conn.close()

        print(
            f"[BacktestEngine] {len(props)} props loaded \n"
            f"edge thresh={edgeThresh} | max edge={self.edgeCap:.0%} | "
            f"bankroll=${bankroll:.0f}"
        )
        print(
            f"[BacktestEngine] prop date range: "
            f"{props['game_date'].min()} -> {props['game_date'].max()}"
        )

        records     = []
        currentBank = bankroll
        skips       = SkipCounters()

        for _, prop in props.iterrows():
            record, currentBank, skips = self._evaluateProp(
                prop=prop,
                actuals=actuals,
                playerMap=playerMap,
                oppMap=oppMap,
                scheduleMap=scheduleMap,
                caches=caches,
                edgeThresh=edgeThresh,
                currentBank=currentBank,
                skips=skips,
            )
            if record is not None:
                records.append(record)

        resultsDF = pd.DataFrame([vars(r) for r in records])

        Reporter.skipBreakdown(skips)
        Reporter.backtestSummary(resultsDF, bankroll, currentBank)

        return resultsDF

    @staticmethod
    def summarizeResults(resultsDF, startingBank, finalBank):
        if resultsDF.empty:
            return {
                "props": 0, "bets": 0, "wins": 0, "losses": 0,
                "win_rate": 0.0, "total_pnl": 0.0, "roi": 0.0,
                "final_bank": float(finalBank),
            }
        bets = resultsDF[resultsDF["bet"]].copy()
        if bets.empty:
            return {
                "props": int(len(resultsDF)), "bets": 0, "wins": 0, "losses": 0,
                "win_rate": 0.0, "total_pnl": 0.0, "roi": 0.0,
                "final_bank": float(finalBank),
            }
        wins       = int((bets["pnl"] > 0).sum())
        losses     = int((bets["pnl"] < 0).sum())
        totalPnl   = float(bets["pnl"].sum())
        totalStake = float(bets["stake"].sum())
        return {
            "props":      int(len(resultsDF)),
            "bets":       int(len(bets)),
            "wins":       wins,
            "losses":     losses,
            "win_rate":   float(wins / len(bets)),
            "total_pnl":  totalPnl,
            "roi":        float(totalPnl / totalStake) if totalStake > 0 else 0.0,
            "final_bank": float(finalBank),
        }

    def scoreAllProps(self, startDate, endDate, edgeThresh):
        """
        Score all props for the period without settling any bets.
        Returns (candidatesByDate, skips) where candidatesByDate is
        {date: [scored_dict, ...]} containing only bettable candidates.
        Used by the combined backtest.
        """
        conn      = sqlite3.connect(str(self.dbPath))
        props     = _loadProps(conn, startDate, endDate)
        actuals   = _loadActuals(conn)
        playerMap = _loadPlayerMap(conn)
        oppMap    = _loadOppMap(conn)
        caches    = preloadCaches(conn)
        conn.close()

        candidatesByDate = {}
        skips            = SkipCounters()

        for date, dayProps in props.groupby("game_date", sort=True):
            for _, prop in dayProps.iterrows():
                result = self._scoreProp(
                    prop=prop, actuals=actuals, playerMap=playerMap,
                    oppMap=oppMap, caches=caches,
                    edgeThresh=edgeThresh, skips=skips,
                )
                if result is None:
                    continue
                scored, skips = result
                if scored["bettable"]:
                    candidatesByDate.setdefault(date, []).append(scored)

        return candidatesByDate, skips

    def _scoreProp(self, prop, actuals, playerMap, oppMap, caches, edgeThresh, skips):
        """
        Pure scoring — no bankroll mutation. Returns (scored_dict, skips) or None.
        scored_dict has bettable=True when edge is within [edgeThresh, edgeCap].
        Used by scoreAllProps() for the combined backtest.
        """
        nameNorm = _normalize(prop.player_name)
        date     = prop.game_date

        playerID = playerMap.get(nameNorm)
        if playerID is None:
            skips.noPlayerMatch += 1
            return None

        ctx = oppMap.get((playerID, date))
        if ctx is None:
            skips.noOppMatch += 1
            skips.noOppMatchByMonth[str(date)[:7]] += 1
            return None

        actualPts = actuals.get((playerID, date))
        if actualPts is None:
            skips.noActuals += 1
            return None

        features = buildFeatures(
            playerID=playerID,
            date=date,
            teamID=ctx["team_id"],
            oppTeamID=ctx["opp_team_id"],
            cache=caches.playerLogCache,
            posCache=caches.posCache,
            teamCache=caches.teamCache,
            statusDF=caches.statusDF,
            oppPosCache=caches.oppPosCache,
            teamGameTotals=caches.teamGameTotals,
            minutesModel=self.minutes,
            currentIsHome=ctx["is_home"],
            currentRestDays=ctx["rest_days"],
        )
        if features is None:
            skips.noFeatures += 1
            return None

        predicted = self.points.predict(features)

        rawProb     = self.calibrator.rawProbOver(predicted, prop.line)
        myProb      = self.calibrator.probOver(predicted, prop.line)
        fairOver, _ = _removeVig(prop.over_odds, prop.under_odds)
        edge        = myProb - fairOver

        pos = float(features["pos"].iloc[0]) if "pos" in features.columns else None
        passed, filterReason = self.filterSet.passes(
            predicted=predicted,
            propLine=prop.line,
            edge=edge,
            pos=pos,
        )
        if not passed:
            skips.filteredOut += 1
            skips.filterReasons[filterReason] += 1
            return None

        bettable = edgeThresh < edge <= self.edgeCap

        scored = {
            "date":      date,
            "player":    prop.player_name,
            "line":      prop.line,
            "predicted": round(predicted, 1),
            "actual":    actualPts,
            "predDiff":  round(predicted - prop.line, 2),
            "rawProb":   round(rawProb, 3),
            "myProb":    round(myProb, 3),
            "bookProb":  round(fairOver, 3),
            "edge":      round(edge, 3),
            "over_odds": prop.over_odds,
            "avgPts10":  float(features["avgPts10"].iloc[0]),
            "last1Pts":  float(features["last1Pts"].iloc[0]),
            "bettable":  bettable,
        }
        return scored, skips

    def _evaluateProp(self, prop, actuals, playerMap, oppMap, scheduleMap, caches,
                      edgeThresh, currentBank, skips):
        nameNorm = _normalize(prop.player_name)
        date     = prop.game_date

        playerID = playerMap.get(nameNorm)
        if playerID is None:
            skips.noPlayerMatch += 1
            return None, currentBank, skips

        ctx = oppMap.get((playerID, date))
        if ctx is None:
            skips.noOppMatch += 1
            skips.noOppMatchByMonth[str(date)[:7]] += 1
            return None, currentBank, skips

        actualPts = actuals.get((playerID, date))
        if actualPts is None:
            skips.noActuals += 1
            return None, currentBank, skips

        features = buildFeatures(
            playerID=playerID,
            date=date,
            teamID=ctx["team_id"],
            oppTeamID=ctx["opp_team_id"],
            cache=caches.playerLogCache,
            posCache=caches.posCache,
            teamCache=caches.teamCache,
            statusDF=caches.statusDF,
            oppPosCache=caches.oppPosCache,
            teamGameTotals=caches.teamGameTotals,
            minutesModel=self.minutes,
            currentIsHome=ctx["is_home"],
            currentRestDays=ctx["rest_days"],
        )
        if features is None:
            skips.noFeatures += 1
            return None, currentBank, skips

        predicted = self.points.predict(features)

        rawProb      = self.calibrator.rawProbOver(predicted, prop.line)
        myProb       = self.calibrator.probOver(predicted, prop.line)
        fairOver, _  = _removeVig(prop.over_odds, prop.under_odds)
        edge         = myProb - fairOver

        pos = float(features["pos"].iloc[0]) if "pos" in features.columns else None
        passed, filterReason = self.filterSet.passes(
            predicted=predicted,
            propLine=prop.line,
            edge=edge,
            pos=pos,
        )
        if not passed:
            skips.filteredOut += 1
            skips.filterReasons[filterReason] += 1
            return None, currentBank, skips

        if edge <= edgeThresh or edge > self.edgeCap:
            return BetRecord(
                date=date,
                player=prop.player_name,
                line=prop.line,
                predicted=round(predicted, 1),
                actual=actualPts,
                predDiff=round(predicted - prop.line, 2),
                rawProb=round(rawProb, 3),
                myProb=round(myProb, 3),
                bookProb=round(fairOver, 3),
                edge=round(edge, 3),
                bet=False,
                stake=0.0,
                pnl=0.0,
                bankroll=round(currentBank, 2),
                betOdds=prop.over_odds,
                avgPts10=float(features["avgPts10"].iloc[0]),
                last1Pts=float(features["last1Pts"].iloc[0]),
            ), currentBank, skips

        stake = _kellyFractional(edge, prop.over_odds) * currentBank
        stake = round(min(stake, currentBank * 0.10), 2)

        won = actualPts > prop.line
        pnl = stake * _payoutMultiplier(prop.over_odds) if won else -stake
        currentBank += pnl

        return BetRecord(
            date=date,
            player=prop.player_name,
            line=prop.line,
            predicted=round(predicted, 1),
            actual=actualPts,
            predDiff=round(predicted - prop.line, 2),
            rawProb=round(rawProb, 3),
            myProb=round(myProb, 3),
            bookProb=round(fairOver, 3),
            edge=round(edge, 3),
            bet=True,
            stake=round(stake, 2),
            pnl=round(pnl, 2),
            bankroll=round(currentBank, 2),
            betOdds=prop.over_odds,
            avgPts10=float(features["avgPts10"].iloc[0]),
            last1Pts=float(features["last1Pts"].iloc[0]),
        ), currentBank, skips

