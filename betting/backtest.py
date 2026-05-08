import sqlite3
import unicodedata
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
 
from config import (
    DB_PATH,
    DEFAULT_EDGE_THRESH, DEFAULT_BANKROLL, FLAT_STAKE,
    MIN_LINE, MAX_LINE_DIFF, DEFAULT_KELLY_FRAC,
)

from features.cache import preloadCaches
from features.builder import buildFeatures
from metrics.reporter import Reporter


# Mutiple small dataclasses
# Desgin to keep the loops in backtesting more readable


@dataclass
class BetRecord:
    date: str
    player: str
    line: float
    predicted: float
    actual: float
    myProb: float
    bookProb: float
    edge: float
    bet: bool
    stake: float
    pnl: float
    bankroll: float


@dataclass
class SkipCounters:
    noPlayerMatch: int=0
    noOppMatch: int=0
    noActuals: int=0
    noFeatures: int=0
    noLine: int=0

    def total(self):
        return (
                self.noPlayerMatch + self.noOppMatch +
                self.noActuals + self.noFeatures + self.noLine
        )


# Odds helpers (Just functions, no state)


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



# DB Loaders


def _normalize(name):
    return "".join(
        c for c in unicodedata.normalize("NFD", name)
        if unicodedata.category(c) != "Mn"
    ).lower().strip()
 

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
    return pd.read_sql_query(query, conn, params=params)


def _loadActuals(conn):
    df = pd.read_sql_query(
        """
        SELECT p.name, g.game_date, pgl.points
        FROM Player_game_logs pgl
        JOIN Games   g ON pgl.game_id   = g.game_id
        JOIN Players p ON pgl.player_id = p.player_id
        """,
        conn
    )
    df["name_norm"] = df["name"].apply(_normalize)
    return df.set_index(["name_norm", "game_date"])["points"].to_dict()


def _loadPlayerMap(conn):
    df = pd.read_sql_query(
        "SELECT player_id, name, team_id FROM Players", conn
    )
    df["name_norm"] = df["name"].apply(_normalize)
    df  = (
            df.sort_values("player_id").drop_duplicates(
                subset="name_norm", keep="last")
    )
    return df.set_index("name_norm")[["player_id", "team_id"]].to_dict("index")

def _loadOppMap(conn):
    df = pd.read_sql_query(
    """
        SELECT pgl.player_id, g.game_date,
               g.home_team_id, g.away_team_id,
               pgl.is_home, pgl.rest_days
        FROM Player_game_logs pgl
        JOIN Games g ON pgl.game_id = g.game_id
    """,
    conn)
    
    result = {}
    for _, row in df.iterrows():
        if row.is_home:
            teamID = row.home_team_id
            oppTeamID = row.away_team_id
        else:
            teamID = row.away_team_id
            oppTeamID = row.home_team_id

        result[(int(row.player_id), row.game_date)] = {
            "team_id": teamID,
            "opp_team_id": oppTeamID,
            "is_home": int(row.is_home),
            "rest_days": int(row.rest_days) if pd.notna(row.rest_days) else 0,
        }

    return result


# Backtest engine


class BacktestEngine:
    """
    Evalutes a trained model and calibrator against historical NBA props
    
    Goes through almost all other classes so flow is desgined to be simple
    """
    
    def __init__(self, pointsBundle, minutesBundle, calibrator, dbPath=DB_PATH):
        self.points = pointsBundle
        self.minutes = minutesBundle
        self.calibrator = calibrator
        self.dbPath = Path(dbPath)


    # Public entry point


    def run(self, startDate=None, endDate=None, 
            edgeThresh=DEFAULT_EDGE_THRESH, bankroll=DEFAULT_BANKROLL):
        
        # Load data from db

        conn = sqlite3.connect(str(self.dbPath))
        props = _loadProps(conn, startDate, endDate)

        if props.empty:
            raise ValueError("No props found for the requested timeframe")

        actuals = _loadActuals(conn)
        playerMap = _loadPlayerMap(conn)
        oppMap = _loadOppMap(conn)
        caches = preloadCaches(conn)

        conn.close()

        print(
            f"[BacktestEngine] {len(props)} props loaded \n"
            f"edge thresh={edgeThresh:.0%} | bankroll=${bankroll:.0f}"
        )

        # Main loop

        records = []
        currentBank = bankroll
        skips = SkipCounters()

        for _, prop in props.iterrows():
            record, currentBank, skips = self._evaluateProp(
                prop = prop,
                actuals = actuals,
                playerMap = playerMap,
                oppMap = oppMap,
                caches = caches,
                edgeThresh = edgeThresh,
                currentBank = currentBank,
                skips = skips
            )
            if record is not None:
                records.append(record)

        resultsDF = pd.DataFrame([vars(r) for r in records])

        # Report

        Reporter.skipBreakdown(skips)
        Reporter.backtestSummary(resultsDF, bankroll, currentBank)

        return resultsDF


    # Single prop evaulation


    def _evaluateProp(self, prop, actuals, playerMap, oppMap, caches,
                      edgeThresh, currentBank, skips):
        nameNorm = _normalize(prop.player_name)
        date = prop.game_date

        # Player lookup
        if nameNorm not in playerMap:
            skips.noPlayerMatch += 1
            return None, currentBank, skips
    
        playerID = playerMap[nameNorm]["player_id"]

        # Game context
        ctx = oppMap.get((playerID, date))
        if ctx is None:
            skips.noOppMatch += 1
            return None, currentBank, skips

        # Actuals
        actualPts = actuals.get((nameNorm, date))
        if actualPts is None:
            skips.noActuals += 1
            return None, currentBank, skips


        # Feature building
        features = buildFeatures(
            playerID = playerID,
            date = date,
            teamID = ctx["team_id"],
            oppTeamID = ctx["opp_team_id"],
            cache = caches.playerLogCache,
            posCache = caches.posCache,
            teamCache = caches.teamCache,
            statusDF = caches.statusDF,
            oppPosCache = caches.oppPosCache,
            teamGameTotals = caches.teamGameTotals,
            minutesModel = self.minutes,
            currentIsHome = ctx["is_home"],
            currentRestDays = ctx["rest_days"],
        )
        if features is None:
            skips.noFeatures += 1
            return None, currentBank, skips


        # Line sanity filter
        avgPts = float(features["avgPts10"].iloc[0])
        if prop.line < MIN_LINE or abs(prop.line - avgPts) > MAX_LINE_DIFF:
            skips.noLine += 1
            return None, currentBank, skips

        
        # Prediction
        predicted = self.points.predict(features)


        # Probailites
        myProb = self.calibrator.probOver(predicted, prop.line)
        fairOver, _ = _removeVig(prop.over_odds, prop.under_odds)
        edge = myProb - fairOver

        # No bet record
        if edge <= edgeThresh:
            return BetRecord(
                date = date,
                player = prop.player_name,
                line = prop.line,
                predicted = round(predicted, 1),
                actual = actualPts,
                myProb = round(myProb, 3),
                bookProb = round(fairOver, 3),
                edge = round(edge, 3),
                bet = False,
                stake = 0.0,
                pnl = 0.0,
                bankroll = round(currentBank, 2),
            ), currentBank, skips

        # Bet sizing
        # Kelly is unimplmented for now until preformance improves
        # with current setup
        # This is to reduce noise for working on improvments
        stake = FLAT_STAKE

        won = actualPts > prop.line
        pnl = stake * _payoutMultiplier(prop.over_odds) if won else -stake
        currentBank += pnl

        return BetRecord(
                date = date,
                player = prop.player_name,
                line = prop.line,
                predicted = round(predicted, 1),
                actual = actualPts,
                myProb = round(myProb, 3),
                bookProb = round(fairOver, 3),
                edge = round(edge, 3),
                bet = True,
                stake = round(stake, 2),
                pnl = round(pnl, 2),
                bankroll = round(currentBank, 2),
            ), currentBank, skips
