"""
Kalshi paper trader (points ladders).

decidePaperBets: snapshot open KXNBAPTS markets, score every rung of every player tipping in the decision
    window with the production model + market shrink, and record one taker and one maker paper bet per
    player-game. The first decision for a player-game stands, later runs only pick up new players.
settlePaperBets: for games that have tipped, pull 1-minute candles from decision to tip to get the
    closing quote (CLV) and decide whether each resting maker order would have filled, then grade
    against Kalshi's result.
paperReport: running results, taker vs maker.
fitShrinkWeight: fit the model-vs-market weight on backtest rungs (CSV) or settled paper rungs.
"""

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from betting.backtest import (
    KALSHI_NAME_ALIASES, KalshiBacktestEngine, _loadPlayerMap, _loadScheduleMap, _normalize,
    _sidePayout, applyMarketShrink, fitMarketShrink,
)
from betting.calibrator import Calibrator
from betting.kalshiCollector import (
    _fmtUtc, _getCandles, _parseUtc, _toFloat, settleKalshiProps, snapshotKalshiProps,
)
from data.dbManager import DBManager
from features.builder import buildFeatures
from features.cache import preloadCaches
from models.minutes import MinutesBundle
from models.points import PointsBundle

SHRINK_PATH = Path("models/kalshi_shrink.json")
SERIES = "KXNBAPTS"

# Training logs store team rest as days since the team's previous game that season, 20 for an opener
OPENER_REST_DAYS = 20


# HELPERS


def _loadShrink(path=SHRINK_PATH):
    if not Path(path).exists():
        raise RuntimeError(f"No shrink weight at {path}, run --kalshi-fit-shrink first")
    return json.loads(Path(path).read_text())


def _loadPlayerTeams(conn):
    df = pd.read_sql_query("SELECT player_id, team_id FROM Players WHERE is_active = 1", conn)
    return dict(zip(df.player_id, df.team_id))


def _teamRestDays(conn, teamID, date):
    """Days since the team's previous game this season, matching how scrapeLogs fills rest_days"""
    row = conn.execute(
        """
        SELECT MAX(game_date) FROM Games
        WHERE (home_team_id = ? OR away_team_id = ?) AND game_date < ?
          AND season = (SELECT season FROM Games WHERE (home_team_id = ? OR away_team_id = ?) AND game_date = ?)
        """,
        (teamID, teamID, date, teamID, teamID, date),
    ).fetchone()
    if not row or not row[0]:
        return OPENER_REST_DAYS
    return (datetime.strptime(date, "%Y-%m-%d") - datetime.strptime(row[0], "%Y-%m-%d")).days


def _candleQuote(block, key):
    if not block:
        return None
    return _toFloat(block.get(f"{key}_dollars", block.get(key)))


# DECIDE


def decidePaperBets(dbPath="NBA.db", minLeadHours=1.0, maxLeadHours=24.0, evThresh=0.03, minPrice=0.20,
                    maxPrice=0.80, makerMaxSpread=0.10, stake=10.0, dryRun=False):
    now = datetime.now(timezone.utc)
    decisionTs = _fmtUtc(now)

    marketRows, snapRows = snapshotKalshiProps(dbPath, "decision", series=[SERIES], dryRun=dryRun)
    if not snapRows:
        print("[paper] no open points markets")
        return pd.DataFrame()

    quotes = pd.DataFrame(marketRows).merge(pd.DataFrame(snapRows), on="market_ticker")
    tip = pd.to_datetime(quotes["est_tip_utc"]).dt.tz_localize("UTC")
    leadHours = (tip - now).dt.total_seconds() / 3600
    quotes["lead_hours"] = leadHours.round(2)
    quotes = quotes[leadHours.between(minLeadHours, maxLeadHours)]
    if quotes.empty:
        print(f"[paper] no markets tipping in {minLeadHours}-{maxLeadHours}h")
        return pd.DataFrame()

    db = DBManager(dbPath)
    decided = db.getKalshiPaperDecided(sorted(quotes["game_date"].unique()))

    points = PointsBundle.loadIfExists()
    minutes = MinutesBundle.loadIfExists()
    calibrator = Calibrator.loadIfExists()
    if points is None or minutes is None or calibrator is None:
        raise RuntimeError("Production points / minutes model or calibrator missing, run --train")
    shrink = _loadShrink()
    trainEnd = points.meta.get("calibration_end_date") or points.meta.get("train_last_date")

    conn = sqlite3.connect(str(dbPath))
    caches = preloadCaches(conn)
    playerMap = _loadPlayerMap(conn)
    for kalshiName, dbName in KALSHI_NAME_ALIASES.items():
        if _normalize(dbName) in playerMap:
            playerMap[_normalize(kalshiName)] = playerMap[_normalize(dbName)]
    playerTeams = _loadPlayerTeams(conn)
    scheduleMap = _loadScheduleMap(conn)

    rungRows = []
    misses = {"decided": 0, "noPlayer": 0, "noTeam": 0, "noGame": 0, "noFeatures": 0}
    missNames = set()
    for (date, name), ladder in quotes.groupby(["game_date", "player_name"], sort=True):
        playerID = playerMap.get(_normalize(name))
        if playerID is None:
            misses["noPlayer"] += 1
            missNames.add(name)
            continue
        if (date, playerID) in decided:
            misses["decided"] += 1
            continue
        teamID = playerTeams.get(playerID)
        if teamID is None:
            misses["noTeam"] += 1
            continue
        sched = scheduleMap.get((date, int(teamID)))
        if sched is None:
            # Usually a trade the Players table hasn't caught up with yet
            misses["noGame"] += 1
            missNames.add(f"{name} (no game for team {teamID})")
            continue

        features = buildFeatures(
            playerID=playerID,
            date=date,
            teamID=sched["team_id"],
            oppTeamID=sched["opp_team_id"],
            cache=caches.playerLogCache,
            posCache=caches.posCache,
            teamCache=caches.teamCache,
            statusDF=caches.statusDF,
            oppPosCache=caches.oppPosCache,
            teamGameTotals=caches.teamGameTotals,
            minutesModel=minutes,
            currentIsHome=sched["is_home"],
            currentRestDays=_teamRestDays(conn, int(teamID), date),
        )
        if features is None:
            misses["noFeatures"] += 1
            continue
        predicted = points.predict(features)

        for q in ladder.itertuples(index=False):
            twoSided = pd.notna(q.yes_bid) and pd.notna(q.yes_ask) and 0 < q.yes_bid and q.yes_ask < 1
            rungRows.append({
                "market_ticker": q.market_ticker, "decision_ts": decisionTs, "date": date, "game_date": date,
                "player": name, "player_name": name, "player_id": playerID, "strike": q.strike,
                "predicted": round(predicted, 2), "p_yes": calibrator.probOver(predicted, q.strike),
                "yes_bid": q.yes_bid, "yes_ask": q.yes_ask,
                "mid": (q.yes_bid + q.yes_ask) / 2 if twoSided else None,
                "est_tip_utc": q.est_tip_utc, "lead_hours": q.lead_hours,
            })
    conn.close()

    print(f"[paper] {len(rungRows)} rungs scored | skips: {misses}")
    if missNames:
        print(f"[paper] unmatched: {', '.join(sorted(missNames)[:20])}")
    if not rungRows:
        return pd.DataFrame()

    rungs = applyMarketShrink(pd.DataFrame(rungRows), shrink["weight"])
    rungs["p_model"] = rungs["p_yes"]
    rungs["shrink_weight"] = shrink["weight"]
    rungs["model_train_end"] = trainEnd

    picks = []
    for orderType, maxSpread in (("taker", None), ("maker", makerMaxSpread)):
        bets = KalshiBacktestEngine.selectBets(
            rungs, evThresh=evThresh, minPrice=minPrice, maxPrice=maxPrice, maxSpread=maxSpread,
            stake=stake, orderType=orderType,
        )
        if bets.empty:
            continue
        bets["order_type"] = orderType
        # Taker fills now at the quote, maker waits for settlePaperBets to check the candles
        bets["fill_status"] = "filled" if orderType == "taker" else "pending"
        picks.append(bets)
    picks = pd.concat(picks, ignore_index=True) if picks else pd.DataFrame()

    if not picks.empty:
        show = picks.sort_values(["order_type", "ev"], ascending=[False, False])
        print(show[["order_type", "game_date", "player", "strike", "side", "yes_bid", "yes_ask", "price",
                    "p_bet", "ev", "lead_hours"]].round(3).to_string(index=False))
    print(f"[paper] {len(picks)} paper bets "
          f"({(picks['order_type'] == 'taker').sum() if len(picks) else 0} taker), shrink weight {shrink['weight']:.2f}")

    if dryRun:
        print("[dry run] nothing written")
        return picks

    rungCols = ["market_ticker", "decision_ts", "game_date", "player_id", "player_name", "strike", "predicted",
                "p_model", "shrink_weight", "p_bet", "yes_bid", "yes_ask", "model_train_end"]
    db.insertKalshiPaperRungs(rungs[rungCols].replace({np.nan: None}).to_dict("records"))
    if not picks.empty:
        betCols = ["game_date", "player_id", "player_name", "market_ticker", "strike", "order_type", "side",
                   "price", "fee", "p_bet", "ev", "stake", "contracts", "yes_bid", "yes_ask", "decision_ts",
                   "est_tip_utc", "lead_hours", "fill_status"]
        picks = picks.rename(columns={"staked": "stake"})
        db.insertKalshiPaperBets(picks[betCols].replace({np.nan: None}).to_dict("records"))
    return picks


# SETTLE


def _evaluateWindow(bet, candles):
    """
    Closing quote, and for a maker order whether it would have filled between decision and tip.
    "filled": the market traded or quoted through our limit, so a resting order there was surely hit.
    "touched": it only reached our price; whether we'd have filled depends on queue position.
    """
    out = {"bet_id": bet.bet_id}
    # Kalshi only writes a candle when something changes, so no candles means the quote sat still
    if candles:
        last = candles[-1]
        out["close_bid"] = _candleQuote(last.get("yes_bid"), "close")
        out["close_ask"] = _candleQuote(last.get("yes_ask"), "close")
    else:
        out["close_bid"], out["close_ask"] = bet.yes_bid, bet.yes_ask
    if out["close_bid"] is not None and out["close_ask"] is not None and 0 < out["close_bid"] < out["close_ask"] < 1:
        mid = (out["close_bid"] + out["close_ask"]) / 2
        out["clv"] = (mid if bet.side == "yes" else 1 - mid) - bet.price

    if bet.order_type != "maker":
        return out

    # Our YES bid at L is hit when a seller comes to L. Our NO bid at L is a YES offer at 1 - L,
    # lifted when a YES buyer comes up to it
    status, fillTs = "unfilled", None
    yesLevel = bet.price if bet.side == "yes" else 1 - bet.price
    for c in candles:
        ts = _fmtUtc(datetime.fromtimestamp(c["end_period_ts"], timezone.utc))
        price = c.get("price") or {}
        traded = _toFloat(c.get("volume_fp", c.get("volume"))) or 0
        if bet.side == "yes":
            reach = [v for v in (_candleQuote(c.get("yes_ask"), "low"),
                                 _candleQuote(price, "low") if traded else None) if v is not None]
            through = any(v < yesLevel - 1e-9 for v in reach)
            touch = any(v <= yesLevel + 1e-9 for v in reach)
        else:
            reach = [v for v in (_candleQuote(c.get("yes_bid"), "high"),
                                 _candleQuote(price, "high") if traded else None) if v is not None]
            through = any(v > yesLevel + 1e-9 for v in reach)
            touch = any(v >= yesLevel - 1e-9 for v in reach)
        if through:
            status, fillTs = "filled", ts
            break
        if touch and status == "unfilled":
            status, fillTs = "touched", ts
    out["fill_status"], out["fill_ts"] = status, fillTs
    return out


def settlePaperBets(dbPath="NBA.db", dryRun=False):
    # Results first so bets on yesterday's games can be graded in the same run
    settleKalshiProps(dbPath, dryRun=dryRun)

    conn = sqlite3.connect(str(dbPath))
    bets = pd.read_sql_query(
        """
        SELECT b.*, m.result AS mkt_result, m.settlement_value
        FROM KalshiPaperBets b JOIN KalshiMarkets m ON m.market_ticker = b.market_ticker
        WHERE b.pnl IS NULL
        """,
        conn,
    )
    conn.close()
    now = datetime.now(timezone.utc)
    bets = bets[pd.to_datetime(bets["est_tip_utc"]).dt.tz_localize("UTC") <= now]
    if bets.empty:
        print("[paper] nothing to settle")
        return []

    updates = []
    for bet in bets.itertuples(index=False):
        row = {"bet_id": bet.bet_id}
        if bet.clv is None or pd.isna(bet.clv) or bet.fill_status == "pending":
            startTs = int(_parseUtc(bet.decision_ts.replace(" ", "T") + "Z").timestamp())
            endTs = int(_parseUtc(bet.est_tip_utc.replace(" ", "T") + "Z").timestamp())
            try:
                candles = _getCandles(SERIES, bet.market_ticker, startTs, endTs, isHistorical=False, periodMin=1)
            except requests.HTTPError:
                # Settled markets move to the archive after a while, which only has hourly candles
                candles = _getCandles(SERIES, bet.market_ticker, startTs, endTs, isHistorical=True)
            row.update(_evaluateWindow(bet, candles))

        fillStatus = row.get("fill_status", bet.fill_status)
        if bet.mkt_result in ("yes", "no", "scalar"):
            payout = _sidePayout(bet.side, bet.mkt_result, bet.settlement_value)
            row["result"] = bet.mkt_result
            row["payout"] = payout
            # pnl is what the bet would have made if filled; unfilled makers are kept apart in the report
            row["pnl"] = bet.contracts * (payout - bet.price - bet.fee)
        if fillStatus is not None:
            row["fill_status"] = fillStatus
        updates.append(row)

    graded = sum("pnl" in u for u in updates)
    print(f"[paper] {len(updates)} bets checked, {graded} graded")
    if dryRun:
        for u in updates[:10]:
            print("  ", u)
        return updates
    DBManager(dbPath).updateKalshiPaperBets(updates)
    return updates


# REPORT


def paperReport(dbPath="NBA.db", startDate=None, endDate=None):
    conn = sqlite3.connect(str(dbPath))
    bets = pd.read_sql_query("SELECT * FROM KalshiPaperBets", conn)
    rungs = pd.read_sql_query(
        """
        SELECT r.*, m.result FROM KalshiPaperRungs r JOIN KalshiMarkets m ON m.market_ticker = r.market_ticker
        """,
        conn,
    )
    conn.close()
    if startDate:
        bets, rungs = bets[bets["game_date"] >= startDate], rungs[rungs["game_date"] >= startDate]
    if endDate:
        bets, rungs = bets[bets["game_date"] <= endDate], rungs[rungs["game_date"] <= endDate]

    print(f"\n{'=' * 60}\nKALSHI PAPER TRADING {bets['game_date'].min()} -> {bets['game_date'].max()}\n{'=' * 60}")
    if bets.empty:
        print("No paper bets yet")
        return

    def _line(label, b):
        graded = b[b["pnl"].notna()]
        if graded.empty:
            print(f"  {label:<26} {len(b):>4} bets, none graded yet")
            return
        ret = graded["pnl"] / graded["stake"]
        se = ret.std() / np.sqrt(len(ret)) if len(ret) > 1 else np.nan
        won = graded[graded["result"] != "scalar"].eval("payout > price")
        clv = graded["clv"].dropna()
        clvText = f" | CLV {clv.mean() * 100:+.2f}c ({(clv > 0).mean():.0%} beat)" if len(clv) else ""
        print(f"  {label:<26} {len(graded):>4} graded | win {won.mean():.1%} (pred {graded['p_bet'].mean():.1%})"
              f" | ROI {ret.mean():+.2%} ±{1.96 * se:.2%} | PnL ${graded['pnl'].sum():+.2f}{clvText}")

    taker = bets[bets["order_type"] == "taker"]
    maker = bets[bets["order_type"] == "maker"]
    print("\nTaker (buy at the ask)")
    _line("all", taker)

    print("\nMaker (resting limit)")
    done = maker[maker["fill_status"] != "pending"]
    if not done.empty:
        rates = done["fill_status"].value_counts(normalize=True)
        print(f"  {len(done)} resolved orders: filled {rates.get('filled', 0):.0%} | "
              f"touched only {rates.get('touched', 0):.0%} | unfilled {rates.get('unfilled', 0):.0%}")
    _line("filled (conservative)", maker[maker["fill_status"] == "filled"])
    _line("filled + touched", maker[maker["fill_status"].isin(["filled", "touched"])])
    # Adverse selection check: if unfilled orders would have won more, fills are the bad ones
    _line("unfilled (would-have)", maker[maker["fill_status"] == "unfilled"])

    print(f"\nBy lead hours (taker)")
    if not taker.empty:
        t = taker[taker["pnl"].notna()].copy()
        if not t.empty:
            t["lead"] = pd.cut(t["lead_hours"], [0, 3, 6, 9, 12, 24])
            g = t.groupby("lead", observed=True)
            print(pd.DataFrame({"bets": g.size(), "roi%": g["pnl"].sum() / g["stake"].sum() * 100,
                                "clv_c": g["clv"].mean() * 100}).round(2).to_string())

    graded = rungs[rungs["result"].isin(["yes", "no"]) & rungs["yes_bid"].notna() & rungs["yes_ask"].notna()]
    graded = graded[(graded["yes_bid"] > 0) & (graded["yes_ask"] < 1)]
    if not graded.empty:
        y = (graded["result"] == "yes").astype(float).values
        mid = (graded["yes_bid"] + graded["yes_ask"]) / 2

        def _ll(p):
            p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
            return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

        print(f"\nLog loss on {len(graded)} settled two-sided rungs: model {_ll(graded['p_model']):.4f} | "
              f"market mid {_ll(mid):.4f} | shrunk {_ll(graded['p_bet']):.4f}")
    print(f"\nProgress: {taker['pnl'].notna().sum()} graded taker bets (target ~200)")


# SHRINK FIT


def fitShrinkWeight(dbPath="NBA.db", csvPath=None, sinceDate=None, savePath=SHRINK_PATH):
    """
    Fit the model weight on out-of-sample rungs: a walk-forward backtest rungs CSV, or (no csvPath)
    the paper trader's own settled rungs. Saves it for decidePaperBets.
    """
    if csvPath:
        rungs = pd.read_csv(csvPath, low_memory=False)
        source = str(csvPath)
    else:
        conn = sqlite3.connect(str(dbPath))
        rungs = pd.read_sql_query(
            """
            SELECT r.game_date AS date, r.p_model AS p_yes, r.yes_bid, r.yes_ask, m.result
            FROM KalshiPaperRungs r JOIN KalshiMarkets m ON m.market_ticker = r.market_ticker
            WHERE m.result IN ('yes', 'no') AND r.yes_bid > 0 AND r.yes_ask < 1
            """,
            conn,
        )
        conn.close()
        rungs["mid"] = (rungs["yes_bid"] + rungs["yes_ask"]) / 2
        source = "KalshiPaperRungs"
    if sinceDate:
        rungs = rungs[rungs["date"] >= sinceDate]

    weight = fitMarketShrink(rungs)
    if weight is None:
        print(f"[shrink] not enough settled rungs in {source} ({len(rungs)}), keeping the current weight")
        return None
    out = {
        "weight": weight,
        "source": source,
        "since": sinceDate or str(rungs["date"].min()),
        "through": str(rungs["date"].max()),
        "rungs": int(len(rungs)),
        "fitted_at": _fmtUtc(datetime.now(timezone.utc)),
    }
    Path(savePath).write_text(json.dumps(out, indent=2))
    print(f"[shrink] weight {weight:.2f} from {out['rungs']} rungs ({source}) -> {savePath}")
    return out
