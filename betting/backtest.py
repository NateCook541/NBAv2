import sqlite3
import unicodedata
from collections import Counter
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
 
from config import (
    DB_PATH,
    DEFAULT_EDGE_THRESH, DEFAULT_BANKROLL, FLAT_STAKE,
    DEFAULT_UNDER_EDGE_THRESH,
    DEFAULT_SELECTION_MODE, DEFAULT_BET_BUDGET,
    DEFAULT_BET_BUDGET_TOLERANCE, DEFAULT_MARKET_PROB_SHRINK,
    UNDER22_SCORE_MODE, UNDER22_USE_EV_MARGIN, UNDER22_SCORE_W_EV_MARGIN,
    UNDER22_SCORE_W_CONFIDENCE, UNDER22_SCORE_W_ODDS_COST,
    UNDER22_SCORE_W_RELIABILITY, UNDER22_SCORE_W_EDGE,
    UNDER22_BREAKEVEN_BASE, UNDER22_BREAKEVEN_SOFT_ADJUST,
    UNDER22_MARKET_ANCHOR_ENABLED,
    MIN_LINE, MAX_LINE_DIFF, DEFAULT_KELLY_FRAC,
    UNDER_MIN_DISAGREEMENT, OVER_MIN_DISAGREEMENT,
    UNDER_MIN_PREDICTED_POINTS, OVER_BLOCK_PREDICTED_RANGE,
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
    betSide: str
    betOdds: float
    score: float = 0.0
    evPerDollar: float = 0.0
    evMargin: float = 0.0
    breakevenProb: float = 0.0
    oddsCost: float = 0.0
    under22Score: float = 0.0
    kellyFrac: float = 0.0

@dataclass
class SkipCounters:
    noPlayerMatch: int=0
    noOppMatch: int=0
    noActuals: int=0
    noFeatures: int=0
    noLine: int=0
    noOppMatchByMonth: Counter = field(default_factory=Counter, repr=False)

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

def _kellyFractional(edge, fairProb, usOdds, fraction=DEFAULT_KELLY_FRAC):
    b = _payoutMultiplier(usOdds)
    p = fairProb + edge
    q = 1 - p
    kelly = (b * p - q) / b
    return max(0.0, kelly * fraction)


def _under22BreakevenBucketAdjust(breakevenProb):
    p = float(breakevenProb)
    if p <= 0.52:
        return float(UNDER22_BREAKEVEN_SOFT_ADJUST.get("le_52", 0.0))
    if p <= 0.54:
        return float(UNDER22_BREAKEVEN_SOFT_ADJUST.get("52_54", 0.0))
    if p <= 0.56:
        return float(UNDER22_BREAKEVEN_SOFT_ADJUST.get("54_56", 0.0))
    return float(UNDER22_BREAKEVEN_SOFT_ADJUST.get("gt_56", 0.0))



# DB Loaders


def _normalize(name):
    base = "".join(
        c for c in unicodedata.normalize("NFD", name)
        if unicodedata.category(c) != "Mn"
    ).lower().strip()
    # Normalize punctuation and collapse initial-based variants:
    # "C.J. McCollum" and "CJ McCollum" -> "cj mccollum".
    cleaned = (
        base.replace(".", " ")
        .replace("'", "")
        .replace("-", " ")
    )
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
        SELECT pgl.player_id, g.game_date, pgl.points
        FROM Player_game_logs pgl
        JOIN Games   g ON pgl.game_id   = g.game_id
        """,
        conn
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
    """
    Returns {(player_id, game_date): {team_id, opp_team_id, is_home, rest_days}}
    Keyed on player + date so mid-season trades are handled correctly.
    """

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
    """
    Returns {(game_date, team_id): {team_id, opp_team_id, is_home, game_id}}
    """
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
        # We are reading the profitable edge cap directly from the calibrator
        self.edgeCap = calibrator.profitableEdgeCap
       
        
    # Helpers

    
    # Returns true for high variance games where model doesn't preform well
    # which is then used to filter out lines.
    def _isChaosGame(self, features):
        ptsStd = float(features["ptsStd10"].iloc[0])
        minStd = float(features["minStd10"].iloc[0])
        predicted = float(features["avgPts10"].iloc[0])

        # Player has high game to game scoring variance
        if ptsStd > 8.0:
            return True
 
        # Player has high minutes variance (role unclear)
        if minStd > 7.0:
            return True
 
        # Large injury opportunity making lineup disrupted
        # injury_opportunity = missingPPG * (avgPts / avgMin)
        # > 15 means a significant scorer is out
        # Only apply to higher scoring players
        if predicted >= 15:
            injOpp = float(features["injury_opportunity"].iloc[0])
            
            if injOpp > 15.0:
                return True

        return False

    
    # Public entry point


    def run(self, startDate=None, endDate=None, edgeThresh=DEFAULT_EDGE_THRESH,
            underEdgeThresh=DEFAULT_UNDER_EDGE_THRESH,
            selectionMode=DEFAULT_SELECTION_MODE, betBudget=DEFAULT_BET_BUDGET,
            budgetTolerance=DEFAULT_BET_BUDGET_TOLERANCE,
            marketProbShrink=DEFAULT_MARKET_PROB_SHRINK,
            underCalibrationMode="hybrid",
            bankroll=DEFAULT_BANKROLL):
        self.calibrator.meta["under_calibration_mode"] = str(underCalibrationMode).lower().strip()
        
        # Load data from db

        conn = sqlite3.connect(str(self.dbPath))
        props = _loadProps(conn, startDate, endDate)

        if props.empty:
            raise ValueError("No props found for the requested timeframe")

        actuals = _loadActuals(conn)
        playerMap = _loadPlayerMap(conn)
        oppMap = _loadOppMap(conn)
        scheduleMap = _loadScheduleMap(conn)
        caches = preloadCaches(conn)

        conn.close()

        print(
            f"[BacktestEngine] {len(props)} props loaded \n"
            f"edge thresh={edgeThresh:.0%} | under edge thresh={underEdgeThresh:.0%} "
            f"| mode={selectionMode} | budget={betBudget}±{budgetTolerance} "
            f"| shrink={marketProbShrink:.0%} | underCal={underCalibrationMode} "
            f"| bankroll=${bankroll:.0f}"
        )
        print(
            f"[BacktestEngine] prop date range: "
            f"{props['game_date'].min()} -> {props['game_date'].max()}"
        )

        # Main loop

        records = []
        skips = SkipCounters()

        for _, prop in props.iterrows():
            record, skips = self._evaluateProp(
                prop = prop,
                actuals = actuals,
                playerMap = playerMap,
                oppMap = oppMap,
                scheduleMap = scheduleMap,
                caches = caches,
                skips = skips,
                marketProbShrink = marketProbShrink,
            )
            if record is not None:
                records.append(record)

        resultsDF = pd.DataFrame([vars(r) for r in records])
        if resultsDF.empty:
            Reporter.skipBreakdown(skips)
            Reporter.backtestSummary(resultsDF, bankroll, bankroll)
            return resultsDF

        if selectionMode == "rank":
            resultsDF = self._applyRankSelection(resultsDF, betBudget)
        else:
            resultsDF = self._applyThresholdSelection(
                resultsDF, edgeThresh=edgeThresh, underEdgeThresh=underEdgeThresh
            )
        currentBank = float(resultsDF["bankroll"].iloc[-1]) if not resultsDF.empty else bankroll
        resultsDF = self._scorePnlAndBankroll(resultsDF, bankroll)

        # Report

        Reporter.skipBreakdown(skips)
        Reporter.backtestSummary(resultsDF, bankroll, currentBank)

        return resultsDF


    # Single prop evaulation


    def _evaluateProp(self, prop, actuals, playerMap, oppMap, scheduleMap, caches,
                      skips, marketProbShrink):
        nameNorm = _normalize(prop.player_name)
        propDate = prop.game_date
        date = propDate

        # Player lookup
        if nameNorm not in playerMap:
            skips.noPlayerMatch += 1
            return None, skips
   
        # Get player ID off name
        playerID = playerMap.get(nameNorm)
        if playerID is None:
            skips.noPlayerMatch += 1
            return None, skips

        # Game context
        ctx = oppMap.get((playerID, date))
        if ctx is None:
            skips.noOppMatch += 1
            skips.noOppMatchByMonth[str(propDate)[:7]] += 1
            return None, skips

        # Actuals
        actualPts = actuals.get((playerID, date))
        if actualPts is None:
            skips.noActuals += 1
            return None, skips

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
            return None, skips


        # Line sanity filter
        avgPts = float(features["avgPts10"].iloc[0])
        if prop.line < MIN_LINE or abs(prop.line - avgPts) > MAX_LINE_DIFF:
            skips.noLine += 1
            return None, skips

        
        # Prediction
        predicted = self.points.predict(features)


        # Filter 2. Cut out high usage guards in the 15-18 and 18-22 range as 
        # model can't find a actual good edge here
        pos = float(features["pos"].iloc[0])
        if 15 <= predicted < 22 and pos <= 2.0:
            skips.noLine += 1
            return None, skips
       
        # Cut out all predicted lines below 12 as no edge
        if predicted < 12:
            skips.noLine += 1
            return None, skips

        # Probailites
        overProb = self.calibrator.probOver(predicted, prop.line)
        underProb = self.calibrator.probUnder(predicted, prop.line)

        fairOver, fairUnder = _removeVig(prop.over_odds, prop.under_odds)
        
        overProbAdj = ((1.0 - marketProbShrink) * overProb) + (marketProbShrink * fairOver)
        underShrink = float(marketProbShrink) + float(
            self.calibrator.underExtraShrink(predicted, prop.line)
        )
        underShrink = float(np.clip(underShrink, 0.0, 0.45))
        underProbAdj = ((1.0 - underShrink) * underProb) + (underShrink * fairUnder)
        if (
            UNDER22_MARKET_ANCHOR_ENABLED
            and predicted >= UNDER_MIN_PREDICTED_POINTS
            and prop.line >= UNDER_MIN_PREDICTED_POINTS
        ):
            underProbAdj = (0.90 * underProbAdj) + (0.10 * fairUnder)

        overEdge = overProbAdj - fairOver
        underEdge = underProbAdj - fairUnder
      
        overDisagreement = predicted - prop.line
        underDisagreement = prop.line - predicted

        canBetOver = overDisagreement >= OVER_MIN_DISAGREEMENT
        canBetUnder = underDisagreement >= UNDER_MIN_DISAGREEMENT

        underReliabilityPenalty = float(self.calibrator.underExtraShrink(predicted, prop.line))
        underBreakeven = float(_impliedProb(prop.under_odds))
        underOddsCost = max(0.0, underBreakeven - UNDER22_BREAKEVEN_BASE)
        underEvMargin = underProbAdj - underBreakeven
        underBucketAdj = _under22BreakevenBucketAdjust(underBreakeven)
        underQualityAdj = (
            (UNDER22_SCORE_W_EV_MARGIN * underEvMargin)
            - (UNDER22_SCORE_W_ODDS_COST * underOddsCost)
            - (UNDER22_SCORE_W_RELIABILITY * underReliabilityPenalty)
            + underBucketAdj
        ) if UNDER22_USE_EV_MARGIN else underBucketAdj
        underSelectionEdge = underEdge + underQualityAdj

        if canBetOver and (not canBetUnder or overEdge >= underSelectionEdge):
            edge = overEdge
            betSide = "over"
            betOdds = prop.over_odds
            myProb = overProb
            fair = fairOver
            myProbAdj = overProbAdj
            disagreement = overDisagreement
            evMargin = overProbAdj - float(_impliedProb(prop.over_odds))
            oddsCost = max(0.0, float(_impliedProb(prop.over_odds)) - UNDER22_BREAKEVEN_BASE)
            kellyFrac = _kellyFractional(edge, fair, betOdds)
        elif canBetUnder:
            edge = underSelectionEdge
            betSide = "under"
            betOdds = prop.under_odds
            myProb = underProb
            fair = fairUnder
            myProbAdj = underProbAdj
            disagreement = underDisagreement
            evMargin = underEvMargin
            oddsCost = underOddsCost
            kellyFrac = _kellyFractional(edge, fair, betOdds)
        else:
            edge = max(overEdge, underSelectionEdge)
            betSide = "over" if overEdge >= underSelectionEdge else "under"
            betOdds = prop.over_odds if betSide == "over" else prop.under_odds
            myProb = overProb if betSide == "over" else underProb
            fair = fairOver if betSide == "over" else fairUnder
            myProbAdj = overProbAdj if betSide == "over" else underProbAdj
            disagreement = overDisagreement if betSide == "over" else underDisagreement
            if betSide == "under":
                evMargin = underEvMargin
                oddsCost = underOddsCost
            else:
                overBreakeven = float(_impliedProb(prop.over_odds))
                evMargin = overProbAdj - overBreakeven
                oddsCost = max(0.0, overBreakeven - UNDER22_BREAKEVEN_BASE)
            kellyFrac = 0.0

        # Side-specific regime filters.
        # These are based on stable backtest underperformance pockets.
        if betSide == "under" and predicted < UNDER_MIN_PREDICTED_POINTS:
            skips.noLine += 1
            return None, skips

        overBlockLow, overBlockHigh = OVER_BLOCK_PREDICTED_RANGE
        if betSide == "over" and overBlockLow <= predicted < overBlockHigh:
            skips.noLine += 1
            return None, skips


        # Filter 1. Profitable edge cap
        if edge > self.edgeCap:
            skips.noLine += 1
            return None, skips

        breakevenProb = float(_impliedProb(betOdds))
        payoutMult = float(_payoutMultiplier(betOdds))
        evPerDollar = float((myProbAdj * payoutMult) - (1.0 - myProbAdj))
        confidence = abs(myProbAdj - 0.5)
        disagreementNorm = min(abs(disagreement), 8.0) / 8.0
        reliabilityPenalty = float(self.calibrator.underExtraShrink(predicted, prop.line))

        if betSide == "under" and predicted >= UNDER_MIN_PREDICTED_POINTS:
            edgeNorm = float(np.clip(edge / 0.15, 0.0, 1.0))
            evMarginNorm = float(np.clip((evMargin + 0.06) / 0.14, 0.0, 1.0))
            confNorm = float(np.clip(confidence / 0.25, 0.0, 1.0))
            oddsCostNorm = float(np.clip(oddsCost / 0.10, 0.0, 1.0))
            reliabilityNorm = float(np.clip(reliabilityPenalty / 0.25, 0.0, 1.0))
            bucketAdjNorm = float(np.clip((underBucketAdj + 0.06) / 0.07, 0.0, 1.0))
            under22Score = float(
                (UNDER22_SCORE_W_EDGE * edgeNorm)
                + (UNDER22_SCORE_W_EV_MARGIN * evMarginNorm)
                + (UNDER22_SCORE_W_CONFIDENCE * confNorm)
                + (0.10 * bucketAdjNorm)
                - (UNDER22_SCORE_W_ODDS_COST * oddsCostNorm)
                - (UNDER22_SCORE_W_RELIABILITY * reliabilityNorm)
            )
            if UNDER22_SCORE_MODE == "edge":
                score = (0.70 * edge) + (0.20 * confidence) + (0.10 * disagreementNorm)
            elif UNDER22_SCORE_MODE == "ev":
                score = (0.70 * evMargin) + (0.20 * confidence) + (0.10 * disagreementNorm)
            else:
                score = (0.40 * edge) + (0.30 * evMargin) + (0.30 * under22Score)
        else:
            under22Score = 0.0
            score = (0.55 * edge) + (0.35 * confidence) + (0.10 * disagreementNorm)

        return BetRecord(
                date = propDate,
                player = prop.player_name,
                line = prop.line,
                predicted = round(predicted, 1),
                actual = actualPts,
                myProb = round(myProbAdj, 3),
                bookProb = round(fair, 3),
                edge = round(edge, 3),
                bet = False,
                stake = 0.0,
                pnl = 0.0,
                bankroll = 0.0,
                betSide = betSide,
                betOdds = round(betOdds, 3),
                score = round(score, 5),
                evPerDollar = round(evPerDollar, 5),
                evMargin = round(evMargin, 5),
                breakevenProb = round(breakevenProb, 5),
                oddsCost = round(oddsCost, 5),
                under22Score = round(under22Score, 5),
                kellyFrac = round(kellyFrac, 6),
            ), skips

    def _applyThresholdSelection(self, df, edgeThresh, underEdgeThresh):
        out = df.copy()
        sideThresh = np.where(out["betSide"] == "under", underEdgeThresh, edgeThresh)
        out["bet"] = out["edge"].to_numpy(dtype=float) > sideThresh
        return out

    def _applyRankSelection(self, df, betBudget):
        out = df.copy()
        out["bet"] = False
        eligible = out[(out["edge"] > 0.0) & (out["myProb"] > out["bookProb"])].copy()
        if eligible.empty:
            return out
        chooseN = min(max(int(betBudget), 0), len(eligible))
        topIdx = eligible.sort_values("score", ascending=False).head(chooseN).index
        out.loc[topIdx, "bet"] = True
        return out

    def _scorePnlAndBankroll(self, df, startingBank):
        out = df.copy()
    
        currentBank = float(startingBank)
        stakes = []
        pnls = []
        banks = []

        for _, row in out.iterrows():
            if not row["bet"]:
                stakes.append(0.0)
                pnls.append(0.0)
                banks.append(currentBank)
                continue

            # Kelly stake from the fraction stored at evaluation time
            stake = round(row["kellyFrac"] * currentBank, 2)
            stake = round(min(stake, currentBank * 0.05), 2)  # 5% hard cap
            stake = max(stake, 1.0)                            # $1 floor

            won = (
                (row["betSide"] == "over"  and row["actual"] > row["line"]) or
                (row["betSide"] == "under" and row["actual"] < row["line"])
            )
            pnl = stake * _payoutMultiplier(row["betOdds"]) if won else -stake
            currentBank += pnl

            stakes.append(stake)
            pnls.append(round(pnl, 2))
            banks.append(round(currentBank, 2))

        out["stake"] = stakes
        out["pnl"] = pnls
        out["bankroll"] = banks
        return out

