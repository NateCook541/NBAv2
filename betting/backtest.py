import sqlite3
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
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
from data.dbManager import connect
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
    Optional ctx keys (present on the live path, absent in the backtest):
    game_id, team_abbr, opp_abbr. When absent the matching output fields are
    None, so backtest callers are unaffected.
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
        # identity fields, consumed by the hub emitter. None on the backtest
        # path, where ctx comes from oppMap and no schedule row is resolved.
        "playerID":   ctx.get("player_id"),
        "gameID":     ctx.get("game_id"),
        "teamID":     ctx.get("team_id"),
        "oppTeamID":  ctx.get("opp_team_id"),
        "teamAbbr":   ctx.get("team_abbr"),
        "oppAbbr":    ctx.get("opp_abbr"),
        "isHome":     ctx.get("is_home"),
        "bookmaker":  getattr(prop, "bookmaker", None),
        "propType":   "points",
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
        SELECT pgl.player_id, pgl.game_id, g.game_date,
               g.home_team_id, g.away_team_id,
               pgl.is_home, pgl.rest_days
        FROM Player_game_logs pgl
        JOIN Games g ON pgl.game_id = g.game_id
        """,
        conn,
        )
    abbrs = _loadTeamAbbrMap(conn)
    result = {}

    for _, row in df.iterrows():
        if row.is_home:
            teamID    = row.home_team_id
            oppTeamID = row.away_team_id
        else:
            teamID    = row.away_team_id
            oppTeamID = row.home_team_id

        result[(int(row.player_id), row.game_date)] = {
                "player_id":   int(row.player_id),
                "team_id":     teamID,
                "opp_team_id": oppTeamID,
                "is_home":     int(row.is_home),
                "rest_days":   int(row.rest_days) if pd.notna(row.rest_days) else 0,
                "game_id":     row.game_id,
                "team_abbr":   abbrs.get(int(teamID)),
                "opp_abbr":    abbrs.get(int(oppTeamID)),
                }

    return result


def _loadTeamAbbrMap(conn):
    """team_id -> abbreviation (Teams.name is already the abbr, e.g. 'BOS').

    Teams carries one row per team per ratings snapshot date, so collapse to
    one abbr per team_id.
    """
    df = pd.read_sql_query(
        "SELECT DISTINCT team_id, name AS abbr FROM Teams", conn
    )
    # NOTE: aliased to `abbr` because `row.name` on an itertuples/iterrows row
    # resolves to the pandas index label, not the `name` column.
    return {int(r.team_id): r.abbr for _, r in df.iterrows()}


def _loadScheduleMap(conn):
    df = pd.read_sql_query(
            """
        SELECT game_id, game_date, home_team_id, away_team_id
        FROM Games
        """,
        conn,
        )
    abbrs = _loadTeamAbbrMap(conn)
    result = {}

    for _, row in df.iterrows():
        homeID = int(row.home_team_id)
        awayID = int(row.away_team_id)
        result[(row.game_date, homeID)] = {
                "team_id": homeID,
                "opp_team_id": awayID,
                "is_home": 1,
                "game_id": row.game_id,
                "team_abbr": abbrs.get(homeID),
                "opp_abbr": abbrs.get(awayID),
                }
        result[(row.game_date, awayID)] = {
                "team_id": awayID,
                "opp_team_id": homeID,
                "is_home": 0,
                "game_id": row.game_id,
                "team_abbr": abbrs.get(awayID),
                "opp_abbr": abbrs.get(homeID),
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
            "player_id": playerID,
            "line":      prop.line,
            "predicted": round(predicted, 1),
            "actual":    actualPts,
            "predDiff":  round(predicted - prop.line, 2),
            "rawProb":   round(rawProb, 3),
            "myProb":    round(myProb, 3),
            "bookProb":  round(fairOver, 3),
            "edge":      round(edge, 3),
            "over_odds": prop.over_odds,
            "under_odds": prop.under_odds,
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



# KALSHI LADDER BACKTEST


# KXNBAPTS fee_type is "quadratic" with multiplier 1 (series endpoint, 2026-10-22):
# taker fee per contract = 0.07 * P * (1 - P). Kalshi rounds the order total up to the cent, ignored here
KALSHI_FEE_RATE = 0.07
# Maker (resting order) fee. Kalshi only charges makers on some markets and the series endpoint doesn't
# say which, so assume the published maker rate (1/4 of taker) to stay conservative until confirmed
KALSHI_MAKER_FEE_RATE = 0.0175

# Kalshi names that don't normalize onto Players.name
KALSHI_NAME_ALIASES = {
    "Jimmy Butler III": "Jimmy Butler",
    "Egor Dëmin": "Egor Dёmin",  # Players stores a Cyrillic ё
}


def _kalshiFee(price, orderType="taker"):
    rate = KALSHI_FEE_RATE if orderType == "taker" else KALSHI_MAKER_FEE_RATE
    return rate * price * (1 - price)


def _loadKalshiQuotes(conn, startDate, endDate, decisionLeadHours, maxQuoteAgeHours=2):
    """
    One row per points-ladder market with the quote we could have traded at decision time
    (latest hourly candle at least decisionLeadHours before tip, no older than maxQuoteAgeHours
    beyond that) and the last pre-tip quote for CLV.
    """
    query = """
        SELECT m.market_ticker, m.game_date, m.player_name, m.strike, m.est_tip_utc,
               m.result, m.settlement_value, c.end_ts, c.yes_bid, c.yes_ask
        FROM KalshiMarkets m
        JOIN KalshiCandles c ON c.market_ticker = m.market_ticker
        WHERE m.series_ticker = 'KXNBAPTS' AND m.result IN ('yes', 'no', 'scalar')
          AND c.end_ts <= m.est_tip_utc
    """
    params = []
    if startDate:
        query += " AND m.game_date >= ?"
        params.append(startDate)
    if endDate:
        query += " AND m.game_date <= ?"
        params.append(endDate)
    df = pd.read_sql_query(query, conn, params=params)
    if df.empty:
        return df

    df["end_ts"] = pd.to_datetime(df["end_ts"])
    tip = pd.to_datetime(df["est_tip_utc"])
    decisionTime = tip - pd.Timedelta(hours=decisionLeadHours)

    keys = ["market_ticker", "game_date", "player_name", "strike", "est_tip_utc", "result", "settlement_value"]

    # Decision quote: latest candle at or before decision time, and not stale
    atDecision = df[(df["end_ts"] <= decisionTime) &
                    (df["end_ts"] >= decisionTime - pd.Timedelta(hours=maxQuoteAgeHours))]
    decision = (atDecision.sort_values("end_ts").groupby("market_ticker").tail(1)
                [keys + ["end_ts", "yes_bid", "yes_ask"]]
                .rename(columns={"end_ts": "quote_ts"}))

    # Close quote: last candle before tip
    close = (df.sort_values("end_ts").groupby("market_ticker").tail(1)
             [["market_ticker", "yes_bid", "yes_ask"]]
             .rename(columns={"yes_bid": "close_bid", "yes_ask": "close_ask"}))

    return decision.merge(close, on="market_ticker", how="left").reset_index(drop=True)


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fitMarketShrink(rungs, minRungs=300):
    """
    Weight w on the model in logit space, the rest on the decision-time mid:
    logit(p) = w * logit(model) + (1 - w) * logit(mid). Grid-searched for log loss on settled
    two-sided rungs. Returns None when there isn't enough history to fit
    """
    graded = rungs[rungs["result"].isin(["yes", "no"]) & rungs["mid"].notna()]
    if len(graded) < minRungs:
        return None
    y = (graded["result"] == "yes").astype(float).values
    lm, lk = _logit(graded["p_yes"]), _logit(graded["mid"])
    weights = np.linspace(0, 1, 101)
    losses = []
    for w in weights:
        p = np.clip(1 / (1 + np.exp(-(w * lm + (1 - w) * lk))), 1e-4, 1 - 1e-4)
        losses.append(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return float(weights[int(np.argmin(losses))])


def applyMarketShrink(rungs, weight):
    """
    Adds p_bet, the probability bets are picked on. weight=None (raw model) bets every rung on p_yes;
    otherwise p_bet is the shrunk probability and only exists on two-sided rungs
    """
    rungs = rungs.copy()
    if weight is None:
        rungs["p_bet"] = rungs["p_yes"]
        return rungs
    p = 1 / (1 + np.exp(-(weight * _logit(rungs["p_yes"]) + (1 - weight) * _logit(rungs["mid"].fillna(0.5)))))
    rungs["p_bet"] = np.where(rungs["mid"].notna(), p, np.nan)
    return rungs


def _sidePayout(side, result, settlementValue):
    """What one contract of `side` pays at settlement. scalar = DNP, settles at a fair price"""
    if result == "scalar":
        yesValue = float(settlementValue or 0.0)
        return yesValue if side == "yes" else 1.0 - yesValue
    return 1.0 if result == side else 0.0


class KalshiBacktestEngine:
    """
    Prices every rung of a player's points ladder with the model's P(points > strike)
    and buys at the real ask (+ taker fee). One bet max per player-game: the best-EV rung/side.
    """

    def __init__(self, pointsBundle, minutesBundle, calibrator, dbPath=DB_PATH):
        self.points     = pointsBundle
        self.minutes    = minutesBundle
        self.calibrator = calibrator
        self.dbPath     = Path(dbPath)

    def run(self, startDate=None, endDate=None, decisionLeadHours=2):
        """Scores every rung of every ladder. Bets are picked separately by selectBets"""
        conn = connect(self.dbPath)
        quotes = _loadKalshiQuotes(conn, startDate, endDate, decisionLeadHours)
        if quotes.empty:
            raise ValueError("No Kalshi quotes found for the requested timeframe")

        actuals   = _loadActuals(conn)
        playerMap = _loadPlayerMap(conn)
        for kalshiName, dbName in KALSHI_NAME_ALIASES.items():
            if _normalize(dbName) in playerMap:
                playerMap[_normalize(kalshiName)] = playerMap[_normalize(dbName)]
        oppMap    = _loadOppMap(conn)
        caches    = preloadCaches(conn)
        conn.close()

        print(f"[KalshiBacktest] {len(quotes)} markets with a decision quote "
              f"({quotes['game_date'].min()} -> {quotes['game_date'].max()}), lead={decisionLeadHours}h")

        rungRows = []
        skips = SkipCounters()

        for (date, name), ladder in quotes.groupby(["game_date", "player_name"], sort=True):
            playerID = playerMap.get(_normalize(name))
            if playerID is None:
                skips.noPlayerMatch += 1
                continue
            # No log row = player didn't play (these settle "scalar" at a fair price, ~0 EV)
            ctx = oppMap.get((playerID, date))
            if ctx is None:
                skips.noOppMatch += 1
                continue

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
                continue

            predicted = self.points.predict(features)
            actualPts = actuals.get((playerID, date))

            for q in ladder.itertuples(index=False):
                twoSided = (pd.notna(q.yes_bid) and pd.notna(q.yes_ask) and 0 < q.yes_bid and q.yes_ask < 1)
                closeMid = ((q.close_bid + q.close_ask) / 2
                            if pd.notna(q.close_bid) and pd.notna(q.close_ask) and 0 < q.close_bid < q.close_ask < 1
                            else None)
                rungRows.append({
                    "date": date, "player": name, "player_id": playerID, "strike": q.strike,
                    "predicted": round(predicted, 2), "p_yes": self.calibrator.probOver(predicted, q.strike),
                    "yes_bid": q.yes_bid, "yes_ask": q.yes_ask,
                    "mid": (q.yes_bid + q.yes_ask) / 2 if twoSided else None, "close_mid": closeMid,
                    "result": q.result, "settlement_value": q.settlement_value, "actual": actualPts,
                })

        Reporter.skipBreakdown(skips)
        return pd.DataFrame(rungRows)

    @staticmethod
    def selectBets(rungs, evThresh=0.03, minPrice=0.20, maxPrice=0.80, maxSpread=None, stake=10.0,
                   orderType="taker"):
        """
        Picks bets on rungs priced by p_bet (see applyMarketShrink). One bet max per player-game: the best-EV rung/side.
        taker: buys at the real ask (+ taker fee).
        maker: rests a limit on our side's bid, 1c better when the spread leaves room (+ maker fee), whether it
        would fill is decided later (paper trader only, hourly candles can't tell).
        Rungs without a result column (live) come back as picks with no settlement fields
        """
        opts = []
        priced = rungs[rungs["p_bet"].notna()]
        improve = (priced["yes_ask"] - priced["yes_bid"] >= 0.02 - 1e-9) * 0.01
        for side in ("yes", "no"):
            # Taker buys YES at the ask, NO at 1 - yes_bid. Maker bids YES at yes_bid, NO at 1 - yes_ask
            if side == "yes":
                quote = "yes_ask" if orderType == "taker" else "yes_bid"
                ok = priced[quote].between(0, 1, inclusive="neither")
                o = priced[ok].copy()
                o["price"] = o[quote] if orderType == "taker" else o[quote] + improve[ok]
                o["my_prob"] = o["p_bet"]
            else:
                quote = "yes_bid" if orderType == "taker" else "yes_ask"
                ok = priced[quote].between(0, 1, inclusive="neither")
                o = priced[ok].copy()
                o["price"] = 1 - o[quote] if orderType == "taker" else 1 - o[quote] + improve[ok]
                o["my_prob"] = 1 - o["p_bet"]
            o["side"] = side
            opts.append(o)
        opts = pd.concat(opts, ignore_index=True)
        opts["price"] = opts["price"].round(2)
        opts["spread"] = opts["yes_ask"] - opts["yes_bid"]
        opts["fee"] = _kalshiFee(opts["price"], orderType)
        opts["ev"] = opts["my_prob"] - opts["price"] - opts["fee"]

        keep = opts["price"].between(minPrice, maxPrice) & (opts["ev"] > evThresh)
        if maxSpread is not None:
            keep &= opts["spread"] <= maxSpread + 1e-9
        bets = (opts[keep].sort_values("ev").groupby(["date", "player"]).tail(1)
                .sort_values(["date", "player"]).reset_index(drop=True))
        if bets.empty:
            return bets

        # Flat stake: spend `stake` dollars including fees
        bets["contracts"] = stake / (bets["price"] + bets["fee"])
        bets["staked"] = stake
        if "result" not in bets.columns:
            return bets
        payout = np.array([_sidePayout(b.side, b.result, b.settlement_value) for b in bets.itertuples()])
        bets["won"] = np.where(bets["result"] == "scalar", np.nan, (payout > bets["price"]).astype(float))
        bets["pnl"] = bets["contracts"] * (payout - bets["price"] - bets["fee"])
        # CLV in probability points: where the close mid ended up vs what we paid for our side
        closeMid = bets["close_mid"].astype(float)
        closeSide = np.where(bets["side"] == "yes", closeMid, 1 - closeMid)
        bets["clv"] = closeSide - bets["price"]
        return bets



    @staticmethod
    def summarize(rungs, bets):
        def _logloss(p, y):
            p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
            return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

        print(f"\n{'=' * 60}\nKALSHI LADDER BACKTEST\n{'=' * 60}")

        # 1. Model vs market on every two-sided rung (does the model know anything the price doesn't?)
        graded = rungs[rungs["result"].isin(["yes", "no"]) & rungs["mid"].notna()].copy()
        if not graded.empty:
            y = (graded["result"] == "yes").astype(float).values
            print(f"\nProbability quality on {len(graded)} two-sided rungs (lower is better)")
            print(f"  {'':<16}{'log loss':>10}{'brier':>10}")
            candidates = [("model", graded["p_yes"]), ("market mid", graded["mid"])]
            # p_bet differs from p_yes when bets were picked on a market-shrunk probability
            shrunk = graded["p_bet"].notna() & (graded["p_bet"] != graded["p_yes"])
            if shrunk.any():
                print(f"  shrunk on {shrunk.sum()} rungs (periods without enough history to fit stay raw)")
                candidates.append(("shrunk", graded["p_bet"].where(shrunk, graded["p_yes"])))
            for label, p in candidates:
                p = np.asarray(p, dtype=float)
                print(f"  {label:<16}{_logloss(p, y):>10.4f}{float(np.mean((p - y) ** 2)):>10.4f}")

            # Sanity check: Kalshi result vs our box score
            hasActual = graded["actual"].notna()
            agree = ((graded.loc[hasActual, "actual"] > graded.loc[hasActual, "strike"])
                     == (graded.loc[hasActual, "result"] == "yes")).mean()
            print(f"  Kalshi result agrees with box score on {agree:.2%} of {hasActual.sum()} rungs")

        if bets.empty:
            print("\nNo bets placed")
            return

        # 2. Bets
        settled = bets[bets["result"] != "scalar"]
        staked = bets["staked"].sum()
        print(f"\nBets: {len(bets)} | win rate {settled['won'].mean():.1%} "
              f"(predicted {settled['my_prob'].mean():.1%}) | "
              f"avg price {bets['price'].mean():.3f} | avg model EV {bets['ev'].mean():+.3f}")
        print(f"PnL ${bets['pnl'].sum():+.2f} on ${staked:.0f} staked | ROI {bets['pnl'].sum() / staked:+.2%}")
        clv = bets["clv"].dropna()
        if not clv.empty:
            print(f"CLV: mean {clv.mean():+.4f} | beat close {(clv > 0).mean():.1%} | n={len(clv)}")

        def _table(groupCol, title):
            g = bets.groupby(groupCol, observed=True)
            out = pd.DataFrame({
                "bets": g.size(),
                "win%": g["won"].mean() * 100,
                "avg_price": g["price"].mean(),
                "avg_ev": g["ev"].mean(),
                "roi%": g["pnl"].sum() / g["staked"].sum() * 100,
                "clv": g["clv"].mean(),
            }).round(3)
            print(f"\nBy {title}\n{out.to_string()}")

        bets = bets.copy()
        bets["ev_bucket"] = pd.cut(bets["ev"], [0, 0.03, 0.05, 0.08, 0.12, 1.0])
        bets["price_bucket"] = pd.cut(bets["price"], [0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0])
        bets["month"] = bets["date"].str[:7]
        _table("side", "side")
        _table("ev_bucket", "model EV")
        _table("price_bucket", "price paid")
        bets["spread_bucket"] = pd.cut(bets["spread"], [0, 0.015, 0.035, 0.065, 0.155, 1.0],
                                       labels=["1c", "2-3c", "4-6c", "7-15c", ">15c"])
        _table("spread_bucket", "spread at decision")
        _table("month", "month")
