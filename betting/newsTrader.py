"""
News trader (shadow, no money): trade teammates' points ladders on official injury report news.

Study 2026-11-14 (Oct 21 - Nov 13, 1-minute Kalshi candles vs every 15-minute report):
  - when a 12+ ppg player is ruled out, his teammates' ladders only finish repricing ~2h after the report,
    buying their YES at the ask at report time made +2.6c/contract (±3.3, 61 events)
  - while a 12+ ppg player sits Questionable/Doubtful (63% end up out), teammates' YES made +1.6c at the ask
Two strategies, both buy teammates' YES at the current ask:
  out_news:     a key player's status turned Out in a report at most NEWS_MAX_AGE_MIN old
  questionable: a key player is Questionable/Doubtful and his game tips within NEWS_Q_WINDOW_HOURS
Each news event is recorded once per teammate rung (UNIQUE event_key + market_ticker).

runNewsTrader: cron, every few minutes through the afternoon
settleNewsBets: grade, closing quote and the mid 15 / 60 minutes after entry (how fast the market caught up)
newsReport: results by strategy, CIs clustered by news event
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

from betting.backtest import KALSHI_NAME_ALIASES, _kalshiFee, _normalize, _sidePayout
from betting.kalshiCollector import _fmtUtc, _getCandles, _parseUtc, _toFloat, snapshotKalshiProps
from config import KEY_PLAYER_PPG, NEWS_MAX_AGE_MIN, NEWS_Q_WINDOW_HOURS, PROTECT_STATUSES
from data.dbManager import DBManager, connect
from data.injuryReports import ET, kalshiTeamCode, pollInjuryReports

SERIES = "KXNBAPTS"
MIN_ASK, MAX_ASK = 0.05, 0.95


def _ppgLast10(conn, playerIDs, beforeDate):
    """Last-10 points per game entering a date, for the players given"""
    ids = sorted({int(p) for p in playerIDs if pd.notna(p)})
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    logs = pd.read_sql_query(
        f"""
        SELECT l.player_id, g.game_date, l.points FROM Player_game_logs l JOIN Games g ON l.game_id = g.game_id
        WHERE l.player_id IN ({marks}) AND g.game_date < ?
        """,
        conn, params=[*ids, beforeDate],
    ).sort_values("game_date")
    avg = logs.groupby("player_id")["points"].apply(lambda s: s.tail(10).mean() if len(s) >= 3 else np.nan)
    return avg.dropna().to_dict()


def _detectEvents(reports, ppg, nowUtc):
    """
    reports: today's InjuryReports rows. Returns event dicts.
    Out changes compare each report to the previous one that listed the same team for the same game,
    so a player's first listing of the day isn't mistaken for news
    """
    events = []
    reports = reports[reports["player_id"].notna()].copy()
    reports["player_id"] = reports["player_id"].astype(int)
    for (gameDate, team), g in reports.groupby(["game_date", "team_abbr"]):
        stamps = sorted(g["report_et"].unique())
        snaps = {s: dict(zip(gs["player_id"], gs["status"])) for s, gs in g.groupby("report_et")}
        names = dict(zip(g["player_id"], g["player_name"]))
        utc = dict(zip(g["report_et"], g["report_utc"]))
        latest = snaps[stamps[-1]]

        for prev, cur in zip(stamps, stamps[1:]):
            age = (nowUtc - _parseUtc(utc[cur].replace(" ", "T") + "Z")).total_seconds() / 60
            if age > NEWS_MAX_AGE_MIN:
                continue
            for pid, status in snaps[cur].items():
                before = snaps[prev].get(pid, "not listed")
                if status == "Out" and before != "Out" and ppg.get(pid, 0) >= KEY_PLAYER_PPG:
                    events.append({"strategy": "out_news", "event_key": f"out|{gameDate}|{pid}",
                                   "game_date": gameDate, "team_abbr": team, "news_player": names[pid],
                                   "news_player_id": pid, "news_ppg": ppg[pid], "prev_status": before,
                                   "new_status": status, "report_et": cur, "report_utc": utc[cur]})

        for pid, status in latest.items():
            if status in PROTECT_STATUSES and ppg.get(pid, 0) >= KEY_PLAYER_PPG:
                events.append({"strategy": "questionable", "event_key": f"q|{gameDate}|{pid}",
                               "game_date": gameDate, "team_abbr": team, "news_player": names[pid],
                               "news_player_id": pid, "news_ppg": ppg[pid], "prev_status": None,
                               "new_status": status, "report_et": stamps[-1], "report_utc": utc[stamps[-1]]})
    return events


def runNewsTrader(dbPath="NBA.db", stake=10.0, dryRun=False):
    now = datetime.now(timezone.utc)
    pollInjuryReports(dbPath, dryRun=dryRun)

    todayEt = datetime.now(ET).strftime("%Y-%m-%d")
    conn = connect(dbPath)
    reports = pd.read_sql_query("SELECT * FROM InjuryReports WHERE game_date = ?", conn, params=[todayEt])
    if reports.empty:
        conn.close()
        print(f"[news] no reports stored for {todayEt}")
        return []
    ppg = _ppgLast10(conn, reports["player_id"].dropna().unique(), todayEt)
    done = {r[0] for r in conn.execute("SELECT DISTINCT event_key FROM KalshiNewsBets WHERE game_date = ?", (todayEt,))}
    conn.close()

    events = [e for e in _detectEvents(reports, ppg, now) if e["event_key"] not in done]
    if not events:
        print(f"[news] {todayEt}: no new key-player news (newest report {reports['report_et'].max()} ET)")
        return []

    marketRows, snapRows = snapshotKalshiProps(dbPath, "news", series=[SERIES], dryRun=dryRun)
    if not snapRows:
        print("[news] no open points markets")
        return []
    quotes = pd.DataFrame(marketRows).merge(pd.DataFrame(snapRows), on="market_ticker")
    quotes = quotes[quotes["game_date"] == todayEt].copy()
    quotes["team_code"] = quotes["market_ticker"].str.split("-").str[2].str[:3]
    aliases = {_normalize(k): _normalize(v) for k, v in KALSHI_NAME_ALIASES.items()}
    quotes["name_norm"] = quotes["player_name"].map(lambda n: aliases.get(_normalize(n), _normalize(n)))
    quotes["tip"] = pd.to_datetime(quotes["est_tip_utc"]).dt.tz_localize("UTC")

    rows = []
    for e in events:
        game = quotes[(quotes["team_code"] == kalshiTeamCode(e["team_abbr"])) & (quotes["tip"] > now)]
        if game.empty:
            continue
        if e["strategy"] == "questionable" and (game["tip"].min() - now) > timedelta(hours=NEWS_Q_WINDOW_HOURS):
            continue
        mates = game[(game["name_norm"] != _normalize(e["news_player"]))
                     & game["yes_ask"].between(MIN_ASK, MAX_ASK) & (game["yes_bid"] > 0)]
        lag = (now - _parseUtc(e["report_utc"].replace(" ", "T") + "Z")).total_seconds() / 60
        for q in mates.itertuples(index=False):
            fee = float(_kalshiFee(q.yes_ask, "taker"))
            rows.append({**{k: e[k] for k in ("strategy", "event_key", "game_date", "team_abbr", "news_player",
                                              "news_player_id", "news_ppg", "prev_status", "new_status", "report_et")},
                         "detected_ts": _fmtUtc(now), "lag_min": round(lag, 1), "market_ticker": q.market_ticker,
                         "player_name": q.player_name, "strike": q.strike, "side": "yes", "price": q.yes_ask,
                         "fee": fee, "stake": stake, "contracts": stake / (q.yes_ask + fee),
                         "yes_bid": q.yes_bid, "yes_ask": q.yes_ask, "est_tip_utc": q.est_tip_utc})

    for e in events:
        n = sum(r["event_key"] == e["event_key"] for r in rows)
        if n:
            print(f"[news] {e['strategy']}: {e['news_player']} ({e['team_abbr']}, {e['news_ppg']:.1f} ppg) "
                  f"{e['prev_status'] or ''} -> {e['new_status']} @ {e['report_et']} ET, {n} teammate rungs")
    if dryRun or not rows:
        print(f"[news] {len(rows)} shadow bets" + (" [dry run] nothing written" if dryRun else ""))
        return rows
    DBManager(dbPath).insertKalshiNewsBets(rows)
    return rows


# SETTLE


def _quoteAt(candles, ts):
    """YES mid from the last candle at or before a unix time (candles only exist when something changed)"""
    last = None
    for c in candles:
        if c["end_period_ts"] > ts:
            break
        last = c
    if last is None:
        return None, None
    bid = _toFloat((last.get("yes_bid") or {}).get("close_dollars"))
    ask = _toFloat((last.get("yes_ask") or {}).get("close_dollars"))
    return bid, ask


def settleNewsBets(dbPath="NBA.db", dryRun=False):
    conn = connect(dbPath)
    bets = pd.read_sql_query(
        """
        SELECT b.*, m.result AS mkt_result, m.settlement_value
        FROM KalshiNewsBets b JOIN KalshiMarkets m ON m.market_ticker = b.market_ticker
        WHERE b.pnl IS NULL
        """,
        conn,
    )
    conn.close()
    now = datetime.now(timezone.utc)
    bets = bets[pd.to_datetime(bets["est_tip_utc"]).dt.tz_localize("UTC") <= now]
    if bets.empty:
        print("[news] nothing to settle")
        return []

    updates, cache = [], {}
    for bet in bets.itertuples(index=False):
        row = {"bet_id": bet.bet_id}
        if bet.clv is None or pd.isna(bet.clv):
            start = int(_parseUtc(bet.detected_ts.replace(" ", "T") + "Z").timestamp())
            end = int(_parseUtc(bet.est_tip_utc.replace(" ", "T") + "Z").timestamp())
            key = (bet.market_ticker, start)
            if key not in cache:
                try:
                    cache[key] = _getCandles(SERIES, bet.market_ticker, start - 60, end, isHistorical=False, periodMin=1)
                except requests.HTTPError:
                    cache[key] = _getCandles(SERIES, bet.market_ticker, start - 3600, end, isHistorical=True)
            candles = cache[key]
            for label, offset in (("mid_15", 15), ("mid_60", 60)):
                if start + offset * 60 <= end:
                    b, a = _quoteAt(candles, start + offset * 60)
                    row[label] = (b + a) / 2 if b is not None and a is not None else None
            b, a = _quoteAt(candles, end)
            row["close_bid"], row["close_ask"] = (b, a) if b is not None else (bet.yes_bid, bet.yes_ask)
            if row["close_bid"] is not None and row["close_ask"] is not None and 0 < row["close_bid"] < row["close_ask"] < 1:
                row["clv"] = (row["close_bid"] + row["close_ask"]) / 2 - bet.price
        if bet.mkt_result in ("yes", "no", "scalar"):
            payout = _sidePayout(bet.side, bet.mkt_result, bet.settlement_value)
            row.update(result=bet.mkt_result, payout=payout, pnl=bet.contracts * (payout - bet.price - bet.fee))
        updates.append(row)

    print(f"[news] {len(updates)} news bets checked, {sum('pnl' in u for u in updates)} graded")
    if dryRun:
        for u in updates[:10]:
            print("  ", u)
        return updates
    DBManager(dbPath).updateKalshiPaperBets(updates, table="KalshiNewsBets")
    return updates


# REPORT


def _clustered(values, groups):
    d = pd.DataFrame({"v": values, "g": groups}).dropna()
    if d.empty:
        return np.nan, np.nan
    s = d.groupby("g")["v"].agg(["sum", "size"])
    n, k = s["size"].sum(), len(s)
    mean = s["sum"].sum() / n
    if k < 2:
        return mean, np.nan
    return mean, 1.96 * np.sqrt(k / (k - 1) * ((s["sum"] - mean * s["size"]) ** 2).sum()) / n


def newsReport(dbPath="NBA.db", startDate=None, endDate=None):
    conn = connect(dbPath)
    bets = pd.read_sql_query("SELECT * FROM KalshiNewsBets", conn)
    conn.close()
    if startDate:
        bets = bets[bets["game_date"] >= startDate]
    if endDate:
        bets = bets[bets["game_date"] <= endDate]
    print("\nNews trader (shadow): buy teammates' YES at the ask on key-player injury news")
    if bets.empty:
        print("  no news bets yet")
        return
    for strategy in ("out_news", "questionable"):
        b = bets[bets["strategy"] == strategy]
        if b.empty:
            print(f"  {strategy:<13} no bets yet")
            continue
        g = b[b["pnl"].notna()]
        line = f"  {strategy:<13} {b['event_key'].nunique():>3} events {len(b):>4} rungs"
        if g.empty:
            print(line + ", none graded yet")
            continue
        edge, ci = _clustered(g["payout"] - g["price"], g["event_key"])
        roi, roiCi = _clustered(g["pnl"] / g["stake"], g["event_key"])
        move15 = (g["mid_15"] - (g["yes_bid"] + g["yes_ask"]) / 2).mean()
        move60 = (g["mid_60"] - (g["yes_bid"] + g["yes_ask"]) / 2).mean()
        print(line + f" | {len(g)} graded | edge {edge * 100:+.2f}c ±{ci * 100:.1f} | ROI {roi:+.1%} ±{roiCi:.1%}"
              f" | CLV {g['clv'].mean() * 100:+.2f}c | mid move +15m {move15 * 100:+.2f}c +60m {move60 * 100:+.2f}c"
              f" | seen {b['lag_min'].median():.0f} min after stamp")
