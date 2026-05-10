import os
import requests
import sqlite3
import unicodedata
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from data.dbManager import DBManager

# Load api key from .env file
load_dotenv()

API_KEY = os.getenv("ODDS_API_KEY")
BASE_URL = "https://api.the-odds-api.com/v4"

# FIXME: Mabye wrap this in a class?

# ODDS API HELPERS

# Fetch the event id for all NBA games on a given date
def _getHistoricalEvents(date):
    url = f"{BASE_URL}/historical/sports/basketball_nba/events"
    resp = requests.get(url, params={
        "apiKey": API_KEY,
        "date": f"{date}T12:00:00Z", # Noon UTC time
    })
    resp.raise_for_status()
    remaining = resp.headers.get("x-requests-remaining", "?")
    print(f"Events fetched | requests remaining: {remaining}")
    return resp.json().get("data", [])

# Get player points props for a single event from the historical endpoint
def _getHistoricalProps(eventID, date):
    url = f"{BASE_URL}/historical/sports/basketball_nba/events/{eventID}/odds"
    resp = requests.get(url, params={
        "apiKey": API_KEY,
        "date": f"{date}T12:00:00Z",
        "regions": "us",
        "markets": "player_points",
        "oddsFormat": "american",
        "bookmakers": "draftkings", # For now we just use draftKings to limit the amount of credits used
    })
    resp.raise_for_status()
    remaining = resp.headers.get("x-requests-remaining", "?")
    print(f"Props fetched for {eventID} | requests remaining: {remaining}")
    return resp.json().get("data", [])


# PARSING

# Flatted the nested Odds API response into a list of flat dict for upsertProps
def _deriveGameDate(eventData, fallbackDate):
    """
    Use the event's commence_time instead of the requested pull date.
    This avoids systematic date drift when the API date window returns
    events that spill across neighboring calendar days.
    """
    commence = eventData.get("commence_time")
    if not commence:
        return fallbackDate

    try:
        dtUtc = datetime.fromisoformat(commence.replace("Z", "+00:00"))
        # NBA reference dates in this project are US-local game days.
        dtEt = dtUtc.astimezone(ZoneInfo("America/New_York"))
        return dtEt.strftime("%Y-%m-%d")
    except Exception:
        return fallbackDate


def _parseProps(eventData, fallbackDate):
    rows = []
    fetchedAt = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    gameDate = _deriveGameDate(eventData, fallbackDate)

    bookmakers = eventData.get("bookmakers", [])
    for book in bookmakers:
        bookName = book["key"]
        for market in book.get("markets", []):
            if market["key"] != "player_points":
                continue

            outcomes = market.get("outcomes", [])
            players = {}

            for outcome in outcomes:
                name = outcome["description"] # Player name
                side = outcome["name"] # Over or Under
                price = outcome["price"] # America odds
                point = outcome.get("point") # The line

                if name not in players:
                    players[name] = {"line": point, "over_odds": None, "under_odds": None}

                if side == "Over":
                    players[name]["over_odds"] = price
                    players[name]["line"] = point
                else:
                    players[name]["under_odds"] = price

            for playerName, props in players.items():
                rows.append({
                    "game_date": gameDate,
                    "player_name": playerName,
                    "line": props["line"],
                    "over_odds": props["over_odds"],
                    "under_odds": props["under_odds"],
                    "bookmaker": bookName,
                    "fetched_at": fetchedAt,
                })
    
    return rows


def _normalizeName(name):
    base = "".join(
        c for c in unicodedata.normalize("NFD", name or "")
        if unicodedata.category(c) != "Mn"
    ).lower().strip()
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


# Get each date string between start and end (inclusive)
def _datRange(startDate, endDate):
    current = datetime.strptime(startDate, "%Y-%m-%d") 
    end = datetime.strptime(endDate, "%Y-%m-%d")
    while current <= end:
        yield current.strftime("%Y-%m-%d")
        current += timedelta(days=1)


# PUBLIC ENTRY POINT


# Pull player point props for all games between start and end date ane store them in the props table in the db
def pullHistoricalProps(startDate, endDate, dbPath="NBA.db", dryRun=False):
    db = DBManager(dbPath)
    db.initSchema()

    totalRows = 0

    if dryRun:
        dates = list(_datRange(startDate, endDate))
        print(f"Dry run: {len(dates)} dates")
        print(f"Estimated requests: ~{len(dates)} (events) + ~{len(dates) * 6} (props)")
        print(f"Estimated total: ~{len(dates) * 7} requests")
        return

    for date in _datRange(startDate, endDate):
        print(f"\nDate: {date}")
        events = _getHistoricalEvents(date)
    
        if not events:
            print("No games found, skipping")
            continue

        dayRows = []
        for event in events:
            eventId = event["id"]
            eventData = _getHistoricalProps(eventId, date)
            rows = _parseProps(eventData, date)
            dayRows.extend(rows)

        if dayRows:
            db.upsertProps(dayRows)
            totalRows += len(dayRows)
            print(f"Stored {len(dayRows)} props for {date}")
        else:
            print(f"No props found for {date}")

    print(f"\nDone. Total props stored: {totalRows}")


def repairHistoricalPropDates(dbPath="NBA.db", dryRun=True):
    """
    Repair existing props that were stored with the wrong date by shifting only
    when there is exactly one unambiguous adjacent-day player log match.

    - Exact matches are left untouched.
    - If both -1 and +1 have a log (possible B2B ambiguity), row is skipped.
    - This keeps runtime backtest strict (no ±1 betting-time matching).
    """
    conn = sqlite3.connect(dbPath)
    cur = conn.cursor()

    players = cur.execute("SELECT player_id, name FROM Players").fetchall()
    playerMap = {_normalizeName(name): int(pid) for pid, name in players}

    logRows = cur.execute("""
        SELECT pgl.player_id, g.game_date
        FROM Player_game_logs pgl
        JOIN Games g ON pgl.game_id = g.game_id
    """).fetchall()
    datesByPlayer = {}
    for pid, gameDate in logRows:
        datesByPlayer.setdefault(int(pid), set()).add(gameDate)

    props = cur.execute("""
        SELECT prop_id, game_date, player_name
        FROM Props
        WHERE over_odds IS NOT NULL AND under_odds IS NOT NULL
        ORDER BY game_date, prop_id
    """).fetchall()

    stats = {
        "total": 0,
        "exact_ok": 0,
        "no_player": 0,
        "no_candidate": 0,
        "ambiguous_b2b": 0,
        "updated": 0,
        "update_ignored_conflict": 0,
    }
    offsetStats = {-1: 0, 1: 0}
    updates = []

    for propID, gameDate, playerName in props:
        stats["total"] += 1
        pid = playerMap.get(_normalizeName(playerName))
        if pid is None:
            stats["no_player"] += 1
            continue

        playerDates = datesByPlayer.get(pid, set())
        if gameDate in playerDates:
            stats["exact_ok"] += 1
            continue

        dt = datetime.strptime(gameDate, "%Y-%m-%d")
        minus1 = (dt - timedelta(days=1)).strftime("%Y-%m-%d")
        plus1 = (dt + timedelta(days=1)).strftime("%Y-%m-%d")
        candidates = []
        if minus1 in playerDates:
            candidates.append((-1, minus1))
        if plus1 in playerDates:
            candidates.append((1, plus1))

        if len(candidates) == 0:
            stats["no_candidate"] += 1
            continue
        if len(candidates) > 1:
            stats["ambiguous_b2b"] += 1
            continue

        offset, targetDate = candidates[0]
        offsetStats[offset] += 1
        updates.append((targetDate, int(propID)))

    print("\n[repair-prop-dates] Summary")
    for key, value in stats.items():
        print(f"{key}: {value}")
    print(f"candidate shifts -1 day: {offsetStats[-1]}")
    print(f"candidate shifts +1 day: {offsetStats[1]}")

    if dryRun:
        print("[repair-prop-dates] Dry run only. No rows updated.")
        conn.close()
        return

    for targetDate, propID in updates:
        before = conn.total_changes
        cur.execute(
            "UPDATE OR IGNORE Props SET game_date = ? WHERE prop_id = ?",
            (targetDate, propID),
        )
        if conn.total_changes > before:
            stats["updated"] += 1
        else:
            stats["update_ignored_conflict"] += 1

    conn.commit()
    conn.close()

    print("[repair-prop-dates] Applied updates")
    print(f"updated: {stats['updated']}")
    print(f"ignored_conflict: {stats['update_ignored_conflict']}")
