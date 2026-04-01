import os
import requests
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
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
def _parseProps(eventData, gameDate):
    rows = []
    fetchedAt = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

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

