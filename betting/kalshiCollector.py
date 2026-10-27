import time
import threading
import requests
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from data.dbManager import DBManager

# Kalshi public market data. No api key needed for reads, only for trading
BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

# Player prop ladders ("X: 20+ points"). Collect more than points so the data exists if we model them later
PROP_SERIES = ["KXNBAPTS", "KXNBAREB", "KXNBAAST", "KXNBA3PT", "KXNBAPRA"]

# Kalshi doesn't expose tip time directly. occurrence_datetime is tip + 3h
# (checked against ESPN tip times on 2026-10-22/23 games, exact on all of them).
# Older archived markets have no occurrence_datetime, but expected_expiration_time is also tip + 3h there
TIP_OFFSET = timedelta(hours=3)

# Backfill only keeps candles this far before tip. Markets open ~2 days early, but the
# early prices don't matter for a pregame decision and they double the row count
CANDLE_LOOKBACK = timedelta(hours=24)


# API HELPERS


# Shared throttle so the backfill's worker threads stay under Kalshi's public read limit
# (measured 2026-10-22: 4/s clean, 6/s starts getting 429s)
MAX_REQ_PER_SEC = 4.5
_throttleLock = threading.Lock()
_lastRequest = [0.0]

def _throttle():
    with _throttleLock:
        wait = _lastRequest[0] + 1 / MAX_REQ_PER_SEC - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _lastRequest[0] = time.monotonic()

# GET with a simple backoff on rate limits / transient errors
def _get(path, params=None, retries=6):
    for attempt in range(retries):
        _throttle()
        resp = requests.get(f"{BASE_URL}{path}", params=params, timeout=30)
        if resp.status_code == 429 or resp.status_code >= 500:
            wait = 2 ** attempt
            print(f"Kalshi {resp.status_code} on {path}, retrying in {wait}s")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()

# All open events for a series with their markets nested in. Follows the cursor until done
def _getOpenEvents(seriesTicker):
    events, cursor = [], None
    while True:
        params = {"series_ticker": seriesTicker, "status": "open", "limit": 200, "with_nested_markets": "true"}
        if cursor:
            params["cursor"] = cursor
        data = _get("/events", params)
        events.extend(data.get("events", []))
        cursor = data.get("cursor")
        if not cursor:
            return events

# Every settled market for a series, from both the historical archive and the live endpoint
# Returns a list of (market, isHistorical)
def _getSettledMarkets(seriesTicker):
    out = []
    for path, isHist, extra in (("/historical/markets", True, {}), ("/markets", False, {"status": "settled"})):
        cursor = None
        while True:
            params = {"series_ticker": seriesTicker, "limit": 1000, **extra}
            if cursor:
                params["cursor"] = cursor
            data = _get(path, params)
            markets = data.get("markets", [])
            out.extend((m, isHist) for m in markets)
            cursor = data.get("cursor")
            if not cursor or not markets:
                break
    return out

# Candles for one market between two unix timestamps (hourly by default, the live endpoint also does 1-minute)
def _getCandles(seriesTicker, ticker, startTs, endTs, isHistorical, periodMin=60):
    if isHistorical:
        path = f"/historical/markets/{ticker}/candlesticks"
    else:
        path = f"/series/{seriesTicker}/markets/{ticker}/candlesticks"
    params = {"start_ts": startTs, "end_ts": endTs, "period_interval": periodMin}
    return _get(path, params).get("candlesticks", [])

# All markets for one event. Settled markets get moved to the historical endpoint after a while, so fall back to it
def _getEventMarkets(eventTicker):
    markets = _get("/markets", {"event_ticker": eventTicker, "limit": 1000}).get("markets", [])
    if not markets:
        markets = _get("/historical/markets", {"event_ticker": eventTicker, "limit": 1000}).get("markets", [])
    return markets


# PARSING


def _toFloat(val):
    if val in (None, ""):
        return None
    try:
        return float(val)
    except ValueError:
        return None

def _parseUtc(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))

def _fmtUtc(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S")

# "KXNBAPTS-26OCT23NOPUTA" -> "2026-10-23" (US local game day, same convention as Games.game_date)
def _parseGameDate(eventTicker):
    code = eventTicker.split("-")[1][:7]
    return datetime.strptime(code, "%y%b%d").strftime("%Y-%m-%d")

def _estTip(market):
    occ = market.get("occurrence_datetime") or market.get("expected_expiration_time")
    return _parseUtc(occ) - TIP_OFFSET if occ else None

def _marketRow(market, seriesTicker):
    tip = _estTip(market)
    return {
        "market_ticker": market["ticker"],
        "event_ticker": market["event_ticker"],
        "series_ticker": seriesTicker,
        "game_date": _parseGameDate(market["event_ticker"]),
        "est_tip_utc": _fmtUtc(tip) if tip else None,
        "kalshi_player_id": (market.get("custom_strike") or {}).get("basketball_player"),
        # yes_sub_title is always "Zion Williamson: 20+" (older titles say "X records 20+ points")
        "player_name": (market.get("yes_sub_title") or market["title"]).split(":")[0].strip(),
        # floor_strike is 19.5 for "20+", yes wins if stat > strike
        "strike": market["floor_strike"],
    }

def _snapshotRow(market, snapshotType, snapshotTs):
    return {
        "market_ticker": market["ticker"],
        "snapshot_ts": snapshotTs,
        "snapshot_type": snapshotType,
        "yes_bid": _toFloat(market.get("yes_bid_dollars")),
        "yes_ask": _toFloat(market.get("yes_ask_dollars")),
        "yes_bid_size": _toFloat(market.get("yes_bid_size_fp")),
        "yes_ask_size": _toFloat(market.get("yes_ask_size_fp")),
        "last_price": _toFloat(market.get("last_price_dollars")),
        "volume": _toFloat(market.get("volume_fp")),
        "open_interest": _toFloat(market.get("open_interest_fp")),
    }


# Live and historical candles name their fields differently ("close_dollars" vs "close", "volume_fp" vs "volume")
def _candleVal(block, key):
    if not block:
        return None
    return _toFloat(block.get(f"{key}_dollars", block.get(key)))

def _candleRow(ticker, candle):
    price = candle.get("price") or {}
    return {
        "market_ticker": ticker,
        "end_ts": _fmtUtc(datetime.fromtimestamp(candle["end_period_ts"], timezone.utc)),
        "yes_bid": _candleVal(candle.get("yes_bid"), "close"),
        "yes_ask": _candleVal(candle.get("yes_ask"), "close"),
        "price_close": _candleVal(price, "close"),
        "price_mean": _candleVal(price, "mean"),
        "volume": _toFloat(candle.get("volume_fp", candle.get("volume"))),
        "open_interest": _toFloat(candle.get("open_interest_fp", candle.get("open_interest"))),
    }

def _resultRow(market):
    return {
        "market_ticker": market["ticker"],
        "result": market["result"],
        # Live settled markets give the actual stat, archived ones just say "points" (toFloat -> None)
        "actual_value": _toFloat(market.get("expiration_value")),
        "settlement_value": _toFloat(market.get("settlement_value_dollars")),
        "settled_ts": market.get("settlement_ts"),
    }


# PUBLIC


def snapshotKalshiProps(dbPath, snapshotType, windowMin=None, series=PROP_SERIES, dryRun=False):
    """
    Snapshot current prices for every pregame prop market.
    snapshotType: open / pregame / close (just a label stored on the row)
    windowMin: only markets whose game tips within the next N minutes (used by the close cron).
    Markets for games that already tipped are skipped, Kalshi keeps them trading in-game.
    """
    now = datetime.now(timezone.utc)
    snapshotTs = _fmtUtc(now)
    cutoff = now + timedelta(minutes=windowMin) if windowMin else None

    marketRows, snapRows = [], []
    for seriesTicker in series:
        events = _getOpenEvents(seriesTicker)
        kept = 0
        for event in events:
            for market in event.get("markets", []):
                if market.get("status") != "active" or market.get("floor_strike") is None:
                    continue
                tip = _estTip(market)
                if tip is None or tip <= now:
                    continue
                if cutoff and tip > cutoff:
                    continue
                marketRows.append(_marketRow(market, seriesTicker))
                snapRows.append(_snapshotRow(market, snapshotType, snapshotTs))
                kept += 1
        print(f"{seriesTicker}: {len(events)} open events, {kept} pregame markets")

    if dryRun:
        print(f"[dry run] would write {len(marketRows)} markets / {len(snapRows)} snapshots ({snapshotType})")
        for row in snapRows[:5]:
            print("  ", row)
        return marketRows, snapRows

    if snapRows:
        db = DBManager(dbPath)
        db.upsertKalshiMarkets(marketRows)
        db.upsertKalshiSnapshots(snapRows)
    else:
        print(f"No pregame markets to snapshot ({snapshotType})")
    return marketRows, snapRows


def settleKalshiProps(dbPath, throughDate=None, dryRun=False):
    """
    Fill result / actual stat for every market we've tracked whose game is on or before throughDate
    (default yesterday, US Eastern). Safe to rerun, only touches markets that are still unsettled.
    result is yes / no, or "scalar" when a player never took the court (settles at a fair price).
    """
    if throughDate is None:
        throughDate = (datetime.now(ZoneInfo("America/New_York")) - timedelta(days=1)).strftime("%Y-%m-%d")

    db = DBManager(dbPath)
    eventTickers = db.getUnsettledKalshiEvents(throughDate)
    print(f"{len(eventTickers)} kalshi events to settle through {throughDate}")

    rows, pending = [], 0
    for eventTicker in eventTickers:
        for market in _getEventMarkets(eventTicker):
            if market.get("result") not in ("yes", "no", "scalar"):
                pending += 1
                continue
            rows.append(_resultRow(market))

    if pending:
        print(f"{pending} markets not settled yet, will retry next run")
    if dryRun:
        print(f"[dry run] would settle {len(rows)} markets")
        for row in rows[:5]:
            print("  ", row)
        return rows

    if rows:
        db.updateKalshiResults(rows)
    return rows


def backfillKalshiArchive(dbPath, series=("KXNBAPTS",), startDate=None, endDate=None,
                          workers=4, chunkSize=500, dryRun=False):
    """
    Pull every settled prop market (archive + live) with its result, plus hourly candles
    from market open to tip. Resumable: markets that already have candles are skipped,
    and candles get written every chunkSize markets so a crash loses little.
    """
    from concurrent.futures import ThreadPoolExecutor

    db = DBManager(dbPath)
    haveCandles = set() if dryRun else db.getKalshiTickersWithCandles()

    for seriesTicker in series:
        settled = _getSettledMarkets(seriesTicker)
        work = []
        for market, isHist in settled:
            if market.get("result") not in ("yes", "no", "scalar") or market.get("floor_strike") is None:
                continue
            gameDate = _parseGameDate(market["event_ticker"])
            if (startDate and gameDate < startDate) or (endDate and gameDate > endDate):
                continue
            work.append((market, isHist))

        print(f"{seriesTicker}: {len(settled)} settled markets, {len(work)} in range")
        if dryRun:
            for market, isHist in work[:3]:
                tip = _estTip(market)
                candles = _getCandles(seriesTicker, market["ticker"], int(_parseUtc(market["open_time"]).timestamp()),
                                      int(tip.timestamp()), isHist)
                print("  ", _marketRow(market, seriesTicker), _resultRow(market))
                print(f"     {len(candles)} candles, last:", _candleRow(market["ticker"], candles[-1]) if candles else None)
            continue

        # Market info + results first, it's cheap (already in memory)
        db.upsertKalshiMarkets([_marketRow(m, seriesTicker) for m, _ in work])
        db.updateKalshiResults([_resultRow(m) for m, _ in work])

        # Rungs Kalshi adds after tip have no pregame price, nothing to fetch
        todo = [(m, h) for m, h in work if m["ticker"] not in haveCandles and _estTip(m)
                and _parseUtc(m["open_time"]) < _estTip(m)]
        print(f"{seriesTicker}: fetching candles for {len(todo)} markets ({len(work) - len(todo)} already done)")

        def fetch(item):
            market, isHist = item
            tip = _estTip(market)
            startTs = int(max(_parseUtc(market["open_time"]), tip - CANDLE_LOOKBACK).timestamp())
            endTs = int(tip.timestamp())
            try:
                candles = _getCandles(seriesTicker, market["ticker"], startTs, endTs, isHist)
            except requests.RequestException as e:
                print(f"  candle fetch failed for {market['ticker']}: {e}")
                return []
            return [_candleRow(market["ticker"], c) for c in candles]

        started = time.time()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i in range(0, len(todo), chunkSize):
                chunk = todo[i:i + chunkSize]
                rows = [r for batch in pool.map(fetch, chunk) for r in batch]
                if rows:
                    db.upsertKalshiCandles(rows)
                done = i + len(chunk)
                print(f"  {done}/{len(todo)} markets | {time.time() - started:.0f}s elapsed")
