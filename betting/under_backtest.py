import sqlite3
from pathlib import Path

import pandas as pd

from config import (
    DB_PATH, DEFAULT_EDGE_THRESH, MAX_BET_EDGE, DEFAULT_BANKROLL, DEFAULT_KELLY_FRAC, FLAT_STAKE
)
from features.builder import buildFeatures
from features.cache import preloadCaches
from metrics.reporter import Reporter
from betting.filters import UnderFilterSet
from betting.backtest import (
    BetRecord, SkipCounters,
    _impliedProb, _removeVig, _payoutMultiplier, _kellyFractional,
    _normalize, _loadProps, _loadActuals, _loadPlayerMap,
    _loadOppMap, _loadScheduleMap,
)


class UnderBacktestEngine:
    """
    Backtests under bets using the same PointsBundle predictions as the over
    engine, but with a separate UnderCalibrator and UnderFilterSet.

    Kept deliberately parallel to BacktestEngine so the two can be developed
    and debugged independently. They share all DB-loading helpers and the
    BetRecord / SkipCounters data structures, but diverge at the probability,
    edge, filter, and outcome calculation steps.
    """

    def __init__(self, pointsBundle, minutesBundle, underCalibrator,
                 dbPath=DB_PATH, filterSet=None):
        self.points     = pointsBundle
        self.minutes    = minutesBundle
        self.calibrator = underCalibrator        # UnderCalibrator instance
        self.dbPath     = Path(dbPath)
        self.edgeCap    = MAX_BET_EDGE
        self.filterSet  = filterSet if filterSet is not None else UnderFilterSet.baseline()

    def run(self, startDate=None, endDate=None,
            edgeThresh=DEFAULT_EDGE_THRESH, bankroll=DEFAULT_BANKROLL):
        # Load data from db
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
            f"[UnderBacktestEngine] {len(props)} props loaded\n"
            f"edge thresh={edgeThresh} | max edge={self.edgeCap:.0%} | "
            f"bankroll=${bankroll:.0f}"
        )
        print(
            f"[UnderBacktestEngine] prop date range: "
            f"{props['game_date'].min()} -> {props['game_date'].max()}"
        )

        # Main loop
        records = []
        currentBank = bankroll
        skips = SkipCounters()

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

        # Report
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
        wins      = int((bets["pnl"] > 0).sum())
        losses    = int((bets["pnl"] < 0).sum())
        totalPnl  = float(bets["pnl"].sum())
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

    # Single prop evaluation

    def _evaluateProp(self, prop, actuals, playerMap, oppMap, scheduleMap,
                      caches, edgeThresh, currentBank, skips):
        nameNorm = _normalize(prop.player_name)
        date     = prop.game_date

        # Player lookup
        playerID = playerMap.get(nameNorm)
        if playerID is None:
            skips.noPlayerMatch += 1
            return None, currentBank, skips

        # Game context
        ctx = oppMap.get((playerID, date))
        if ctx is None:
            skips.noOppMatch += 1
            skips.noOppMatchByMonth[str(date)[:7]] += 1
            return None, currentBank, skips

        # Actuals
        actualPts = actuals.get((playerID, date))
        if actualPts is None:
            skips.noActuals += 1
            return None, currentBank, skips

        # Feature building
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

        # Prediction (shared PointsBundle — same output as over engine)
        predicted = self.points.predict(features)

        # Under probabilities
        rawProb  = self.calibrator.rawProbUnder(predicted, prop.line)
        myProb   = self.calibrator.probUnder(predicted, prop.line)
        _, fairUnder = _removeVig(prop.over_odds, prop.under_odds)
        edge = myProb - fairUnder

        # Filters
        pos      = float(features["pos"].iloc[0]) if "pos" in features.columns else None
        predDiff = round(prop.line - predicted, 2)
        passed, filterReason = self.filterSet.passes(
            predicted=predicted,
            propLine=prop.line,
            edge=edge,
            pos=pos,
            betOdds=prop.under_odds,
            predDiff=predDiff,
        )
        if not passed:
            skips.filteredOut += 1
            skips.filterReasons[filterReason] += 1
            return None, currentBank, skips

        # No-bet record
        if edge <= edgeThresh or edge > self.edgeCap:
            return BetRecord(
                date=date,
                player=prop.player_name,
                line=prop.line,
                predicted=round(predicted, 1),
                actual=actualPts,
                predDiff=round(prop.line - predicted, 2),  # positive = book above model
                rawProb=round(rawProb, 3),
                myProb=round(myProb, 3),
                bookProb=round(fairUnder, 3),
                edge=round(edge, 3),
                bet=False,
                stake=0.0,
                pnl=0.0,
                bankroll=round(currentBank, 2),
                betSide="under",
                betOdds=prop.under_odds,
                avgPts10=float(features["avgPts10"].iloc[0]),
                last1Pts=float(features["last1Pts"].iloc[0]),
            ), currentBank, skips

        # Bet sizing
        stake = _kellyFractional(edge, prop.under_odds) * currentBank
        stake = round(min(stake, currentBank * 0.10), 2)
        #stake = FLAT_STAKE

        won = actualPts < prop.line
        pnl = stake * _payoutMultiplier(prop.under_odds) if won else -stake
        currentBank += pnl

        return BetRecord(
            date=date,
            player=prop.player_name,
            line=prop.line,
            predicted=round(predicted, 1),
            actual=actualPts,
            predDiff=round(prop.line - predicted, 2),  # positive = book above model
            rawProb=round(rawProb, 3),
            myProb=round(myProb, 3),
            bookProb=round(fairUnder, 3),
            edge=round(edge, 3),
            bet=True,
            stake=round(stake, 2),
            pnl=round(pnl, 2),
            bankroll=round(currentBank, 2),
            betSide="under",
            betOdds=prop.under_odds,
            avgPts10=float(features["avgPts10"].iloc[0]),
            last1Pts=float(features["last1Pts"].iloc[0]),
        ), currentBank, skips
