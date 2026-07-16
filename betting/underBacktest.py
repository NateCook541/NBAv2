import sqlite3
from pathlib import Path

import pandas as pd

from config import (
    DB_PATH, DEFAULT_EDGE_THRESH, MAX_BET_EDGE, DEFAULT_BANKROLL,
    DEFAULT_UNDER_KELLY_FRAC, DEFAULT_DAILY_CAP, DEFAULT_MAX_STAKE_ABS, FLAT_STAKE,
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

    Daily selection: all props for a given date are scored first, then sorted
    by edge descending. Stakes are allocated greedily from the top until the
    daily exposure cap is exhausted. This ensures the highest edge bets are
    always funded before the cap is hit
    """

    def __init__(self, pointsBundle, minutesBundle, underCalibrator,
                 dbPath=DB_PATH, filterSet=None,
                 kellyFrac=DEFAULT_UNDER_KELLY_FRAC,
                 maxDailyExposure=DEFAULT_DAILY_CAP,
                 maxStakeAbs=DEFAULT_MAX_STAKE_ABS):
        self.points = pointsBundle
        self.minutes = minutesBundle
        self.calibrator = underCalibrator
        self.dbPath = Path(dbPath)
        self.edgeCap = MAX_BET_EDGE
        self.filterSet = filterSet if filterSet is not None else UnderFilterSet.baseline()
        self.kellyFrac = kellyFrac
        self.maxDailyExposure = maxDailyExposure
        self.maxStakeAbs = maxStakeAbs # fraction of starting bankroll, fixed for the run

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
        scheduleMap = _loadScheduleMap(conn)
        caches = preloadCaches(conn)
        conn.close()

        absStakeCap = bankroll * self.maxStakeAbs   # fixed dollar cap for the entire run

        print(
            f"[UnderBacktestEngine] {len(props)} props loaded\n"
            f"edge thresh={edgeThresh} | max edge={self.edgeCap:.0%} | "
            f"bankroll=${bankroll:.0f}"
        )
        print(
            f"[UnderBacktestEngine] prop date range: "
            f"{props['game_date'].min()} -> {props['game_date'].max()}"
        )
        print(
            f"[UnderBacktestEngine] kelly_frac={self.kellyFrac:.0%} | "
            f"max_daily_exposure={self.maxDailyExposure:.0%} | "
            f"max_stake=${absStakeCap:.2f} (={self.maxStakeAbs:.0%} of starting bank)"
        )

        records     = []
        currentBank = bankroll
        skips       = SkipCounters()

        # Two pass per day loop: score all -> sort by edge -> settle
        for date, dayProps in props.groupby("game_date", sort=True):
            # Pass 1: score every prop, collect candidates that pass filters + edge thresh
            candidates = []
            nobet = [] # passed filters but edge too low/high. Record as no-bet

            for _, prop in dayProps.iterrows():
                result = self._scoreProp(
                    prop=prop,
                    actuals=actuals,
                    playerMap=playerMap,
                    oppMap=oppMap,
                    caches=caches,
                    edgeThresh=edgeThresh,
                    skips=skips,
                )
                if result is None:
                    continue   # skipped (no match / no features) — skips already updated
                scored, skips = result
                if scored["bettable"]:
                    candidates.append(scored)
                else:
                    nobet.append(scored)

            # Pass 2: sort candidates by edge descending, allocate stakes greedily
            candidates.sort(key=lambda c: c["edge"], reverse=True)

            dailyCap    = currentBank * self.maxDailyExposure
            dailyStaked = 0.0

            for scored in candidates:
                rawStake = (
                    _kellyFractional(scored["edge"], scored["under_odds"], fraction=self.kellyFrac)
                    * currentBank
                )
                rawStake = min(rawStake, absStakeCap)   # hard cap vs starting bankroll

                remaining = dailyCap - dailyStaked
                if remaining <= 0.0:
                    # Daily cap hit — record as no-bet and move on
                    nobet.append(scored)
                    continue

                stake = round(min(rawStake, remaining), 2)
                dailyStaked += stake

                won = scored["actual"] < scored["line"]
                pnl = stake * _payoutMultiplier(scored["under_odds"]) if won else -stake
                currentBank += pnl

                records.append(BetRecord(
                    date=scored["date"],
                    player=scored["player"],
                    line=scored["line"],
                    predicted=scored["predicted"],
                    actual=scored["actual"],
                    predDiff=scored["predDiff"],
                    rawProb=scored["rawProb"],
                    myProb=scored["myProb"],
                    bookProb=scored["bookProb"],
                    edge=scored["edge"],
                    bet=True,
                    stake=round(stake, 2),
                    pnl=round(pnl, 2),
                    bankroll=round(currentBank, 2),
                    betSide="under",
                    betOdds=scored["under_odds"],
                    avgPts10=scored["avgPts10"],
                    last1Pts=scored["last1Pts"],
                ))

            # Emit no-bet records (below edge thresh or daily cap overflow)
            for scored in nobet:
                records.append(BetRecord(
                    date=scored["date"],
                    player=scored["player"],
                    line=scored["line"],
                    predicted=scored["predicted"],
                    actual=scored["actual"],
                    predDiff=scored["predDiff"],
                    rawProb=scored["rawProb"],
                    myProb=scored["myProb"],
                    bookProb=scored["bookProb"],
                    edge=scored["edge"],
                    bet=False,
                    stake=0.0,
                    pnl=0.0,
                    bankroll=round(currentBank, 2),
                    betSide="under",
                    betOdds=scored["under_odds"],
                    avgPts10=scored["avgPts10"],
                    last1Pts=scored["last1Pts"],
                ))

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
        wins = int((bets["pnl"] > 0).sum())
        losses = int((bets["pnl"] < 0).sum())
        totalPnl = float(bets["pnl"].sum())
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

    def scoreAllProps(self, startDate, endDate, edgeThresh, bankroll):
        """
        Score all props for the period without settling any bets
        Returns (candidatesByDate, skips, absStakeCap) where candidatesByDate
        is {date: [scored_dict, ...]} containing only bettable candidates.
        Used by the combined backtest to merge over/under candidates before
        conflict resolution and shared bankroll settlement.
        """
        conn = sqlite3.connect(str(self.dbPath))
        props = _loadProps(conn, startDate, endDate)
        actuals = _loadActuals(conn)
        playerMap = _loadPlayerMap(conn)
        oppMap = _loadOppMap(conn)
        caches = preloadCaches(conn)
        conn.close()

        absStakeCap = bankroll * self.maxStakeAbs
        candidatesByDate = {}
        skips = SkipCounters()

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

        return candidatesByDate, skips, absStakeCap

    # Pass 1: score a single prop — no bankroll mutation

    def _scoreProp(self, prop, actuals, playerMap, oppMap, caches, edgeThresh, skips):
        """
        Evaluate a prop through lookups, feature building, and calibration.
        Returns a scored dict (bettable=True/False) or None if the prop must
        be skipped entirely (no player match, no features, etc)

        Does not mutate currentBank — that happens in run()
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

        rawProb = self.calibrator.rawProbUnder(predicted, prop.line)
        myProb = self.calibrator.probUnder(predicted, prop.line)
        _, fairUnder = _removeVig(prop.over_odds, prop.under_odds)
        edge = myProb - fairUnder

        pos = float(features["pos"].iloc[0]) if "pos" in features.columns else None
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
            return None

        bettable = edgeThresh < edge <= self.edgeCap

        scored = {
            "date":       date,
            "player":     prop.player_name,
            "player_id":  playerID,
            "line":       prop.line,
            "predicted":  round(predicted, 1),
            "actual":     actualPts,
            "predDiff":   predDiff,
            "rawProb":    round(rawProb, 3),
            "myProb":     round(myProb, 3),
            "bookProb":   round(fairUnder, 3),
            "edge":       round(edge, 3),
            "under_odds": prop.under_odds,
            "avgPts10":   float(features["avgPts10"].iloc[0]),
            "last1Pts":   float(features["last1Pts"].iloc[0]),
            "bettable":   bettable,
        }
        return scored, skips

