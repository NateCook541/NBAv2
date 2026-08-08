"""
totalsCollector.py

Pulls historical game TOTALS (over/under) lines from The Odds API and stores them
in the Odds_archive table — the same table the totals model reads through
features/cache._buildOddsCache. This closes the gap the box-score/line-disjoint
memory describes: the SBR archive ends 2023-01-16, so 2023-10-onward games in the
DB have no market line. This backfills those from the odds API's historical
endpoint so the totals model's best feature (market_total_close) exists on the
recent, box-score-having population.

Mirrors betting/oddsCollector.py (which does the same for player_points props),
but:
  - market is "totals" (one line per GAME) instead of "player_points",
  - rows are keyed on (game_date, home_team_id, away_team_id) and upserted into
    Odds_archive via DBManager.upsertOddsArchive,
  - team names from the API are resolved to our team_ids.

CREDIT SAFETY
-------------
The historical odds endpoint bills 10 credits per (market x region) per event
call; the historical events call bills 1 credit per timestamp. We fetch one
market (totals), one region (us), one bookmaker (draftkings) -> 10 credits per
game + 1 per date for the events lookup.

`pullHistoricalTotals(..., dryRun=True)` estimates the cost WITHOUT spending a
single credit, by counting the real games already scheduled in the local Games
table for the requested window. Use it first, every time — messing this step up
by looping the wrong window burns credits that don't come back.
"""
import os
import sqlite3
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

from data.dbManager import DBManager
from config import TEAM_MAP

# Load api key from .env file
load_dotenv()

API_KEY = os.getenv("ODDS_API_KEY")
BASE_URL = "https://api.the-odds-api.com/v4"

# One market, one region, one bookmaker keeps the credit cost predictable.
REGIONS = "us"
MARKETS = "totals"
BOOKMAKER = "draftkings"

# Historical odds billing: 10 credits per (market x region) per event odds call.
CREDITS_PER_EVENT_ODDS = 10 * len(MARKETS.split(",")) * len(REGIONS.split(","))


# TEAM NAME RESOLUTION


# The Odds API returns full franchise names ("Los Angeles Lakers"). Map those to
# the DB abbreviations in config.TEAM_MAP -> team_id. Includes the relocated /
# renamed franchises the API may still surface for older seasons, mapped to the
# modern franchise id (matches data/oddsArchiveScraper's alias handling).
ODDS_NAME_TO_ABBR = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BRK",
    "Charlotte Hornets": "CHO",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC",
    "LA Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHO",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
    # Relocated / renamed aliases -> modern franchise id
    "Charlotte Bobcats": "CHO",
    "New Jersey Nets": "BRK",
    "New Orleans Hornets": "NOP",
    "Seattle SuperSonics": "OKC",
}

# Names the API returned that we could not resolve. Reported, never silently
# dropped — an unmapped team means a missing game, not a bad line (that was the
# Clippers bug in the archive scraper).
_unmappedNames = set()


def _teamId(name):
    abbr = ODDS_NAME_TO_ABBR.get((name or "").strip())
    if abbr is None:
        _unmappedNames.add(name)
        return None
    return TEAM_MAP.get(abbr)


# ODDS API HELPERS


def _remaining(resp):
    """Account credits left, per the API's x-requests-remaining header (ground
    truth). None if the header is absent/unparseable."""
    try:
        return int(resp.headers.get("x-requests-remaining"))
    except (TypeError, ValueError):
        return None


# Fetch the event id + teams for all NBA games at a given historical timestamp.
# Costs 1 credit per call. Returns (events, creditsRemaining).
def _getHistoricalEvents(date):
    url = f"{BASE_URL}/historical/sports/basketball_nba/events"
    resp = requests.get(url, params={
        "apiKey": API_KEY,
        "date": f"{date}T12:00:00Z",  # Noon UTC — mid-day snapshot for the slate
    })
    resp.raise_for_status()
    remaining = _remaining(resp)
    print(f"Events fetched for {date} | requests remaining: {remaining}")
    return resp.json().get("data", []), remaining


# Get the totals line for a single event from the historical endpoint.
# Costs CREDITS_PER_EVENT_ODDS credits per call. Returns (oddsData, creditsRemaining).
def _getHistoricalTotals(eventID, date):
    url = f"{BASE_URL}/historical/sports/basketball_nba/events/{eventID}/odds"
    resp = requests.get(url, params={
        "apiKey": API_KEY,
        "date": f"{date}T12:00:00Z",
        "regions": REGIONS,
        "markets": MARKETS,
        "oddsFormat": "american",
        "bookmakers": BOOKMAKER,
    })
    resp.raise_for_status()
    remaining = _remaining(resp)
    print(f"  Totals fetched for {eventID} | requests remaining: {remaining}")
    return resp.json().get("data", {}), remaining


# PARSING


def _deriveGameDate(eventData, fallbackDate):
    """US-local (ET) game day from the event commence_time, so the key matches
    the Games/Results tables (which store ET calendar days). Mirrors
    oddsCollector._deriveGameDate."""
    commence = eventData.get("commence_time")
    if not commence:
        return fallbackDate
    try:
        dtUtc = datetime.fromisoformat(commence.replace("Z", "+00:00"))
        dtEt = dtUtc.astimezone(ZoneInfo("America/New_York"))
        return dtEt.strftime("%Y-%m-%d")
    except Exception:
        return fallbackDate


def _parseTotals(eventData, fallbackDate):
    """Flatten one historical event's totals market into a single Odds_archive
    row (or None if it can't be resolved). The totals market lists an Over and an
    Under outcome that share one `point` (the line); we store that as
    total_close. No opening line is available from a single historical snapshot,
    so total_open is left NULL — line_minus_naive (which uses the close) still
    works; only open_close_move stays NaN for these rows."""
    homeName = eventData.get("home_team")
    awayName = eventData.get("away_team")
    homeId = _teamId(homeName)
    awayId = _teamId(awayName)
    if homeId is None or awayId is None:
        return None

    gameDate = _deriveGameDate(eventData, fallbackDate)

    line = None
    for book in eventData.get("bookmakers", []):
        for market in book.get("markets", []):
            if market.get("key") != "totals":
                continue
            for outcome in market.get("outcomes", []):
                # Over and Under carry the same point; take the first seen.
                if outcome.get("point") is not None:
                    line = float(outcome["point"])
                    break
            if line is not None:
                break
        if line is not None:
            break

    if line is None:
        return None

    return {
        "game_date": gameDate,
        "home_team_id": homeId,
        "away_team_id": awayId,
        "season": None,          # unknown from the odds feed; not used by the model
        "home_score": None,
        "away_score": None,
        "total_open": None,      # single snapshot: no separate opening line
        "total_close": line,
        "actual_total": None,
    }


# DRY-RUN ESTIMATION (SPENDS NO CREDITS)


def _datesWithGames(startDate, endDate, dbPath):
    """Distinct game dates in [start, end] straight from the local Games table,
    plus each date's game count. Zero API calls — this is what makes the dry run
    an exact credit estimate instead of a games/day guess."""
    conn = sqlite3.connect(str(dbPath))
    rows = conn.execute(
        """SELECT game_date, COUNT(*) AS n
           FROM Games
           WHERE game_date >= ? AND game_date <= ?
           GROUP BY game_date
           ORDER BY game_date""",
        (startDate, endDate),
    ).fetchall()
    conn.close()
    return rows


def _estimateCredits(startDate, endDate, dbPath):
    dateRows = _datesWithGames(startDate, endDate, dbPath)
    nDates = len(dateRows)
    nGames = sum(n for _, n in dateRows)
    eventsCost = nDates                          # 1 credit / date
    oddsCost = nGames * CREDITS_PER_EVENT_ODDS   # 10 credits / game
    return {
        "dates": nDates,
        "games": nGames,
        "events_cost": eventsCost,
        "odds_cost": oddsCost,
        "total_cost": eventsCost + oddsCost,
    }


# VERIFY MODE (spends the bare minimum: 1 events lookup + 1 game's odds)


def _pickVerifyDate(dbPath):
    """The earliest post-archive date that has games AND final results, so the
    verify join can prove the full chain (line -> Games -> Results) end to end.
    The odds archive ends 2023-01-16, so anything from the 2023-24 opener on is
    both a real gap the pull must fill and fully backtestable."""
    conn = sqlite3.connect(str(dbPath))
    row = conn.execute(
        """SELECT g.game_date
           FROM Games g JOIN Results r ON r.game_id = g.game_id
           WHERE g.game_date >= '2023-10-01'
           GROUP BY g.game_date
           ORDER BY g.game_date
           LIMIT 1"""
    ).fetchone()
    conn.close()
    return row[0] if row else None


def verifyPull(dbPath="NBA.db", date=None):
    """Cheap end-to-end proof that the collector fetches, parses, stores, and
    (most importantly) JOINS correctly — for ~11 credits, before committing to a
    multi-thousand-credit pull.

    Spends exactly: 1 events lookup + 1 game's odds = 1 + CREDITS_PER_EVENT_ODDS.
    Pulls only the FIRST event of one date, stores that single row, reads it back,
    and runs the exact (game_date, home, away) join the totals backtest uses
    (Odds_archive -> Games -> Results). Prints every step and a PASS/FAIL verdict.

    Idempotent: upsert is INSERT OR REPLACE, so re-running just overwrites the one
    verify row. It writes a real row into Odds_archive — that's the point (it
    proves storage), and a later full pull will simply re-write it.
    """
    if date is None:
        date = _pickVerifyDate(dbPath)
        if date is None:
            print("VERIFY FAILED: no post-2023-10 game with results to verify against.")
            return {"ok": False, "reason": "no candidate date"}

    print("=" * 64)
    print(f"VERIFY MODE — 1 events lookup + 1 game's odds "
          f"(~{1 + CREDITS_PER_EVENT_ODDS} credits)")
    print(f"Verify date: {date}")
    print("=" * 64)

    # 1. Fetch events for the date (1 credit)
    events, _ = _getHistoricalEvents(date)
    if not events:
        print(f"VERIFY FAILED: API returned no events for {date}. "
              f"Try another date via verifyPull(date=...).")
        return {"ok": False, "reason": "no events from API"}

    # The events endpoint returns a WINDOW spilling into neighboring days, so pick
    # an event whose real (ET) game date is actually `date` — same boundary filter
    # the full pull uses. Otherwise verify might store a game for the wrong day.
    sameDay = [e for e in events if _deriveGameDate(e, date) == date]
    if not sameDay:
        print(f"VERIFY FAILED: {len(events)} events returned but none fall on {date} "
              f"(all spilled to neighboring days). Try another date.")
        return {"ok": False, "reason": "no same-day event"}
    print(f"Step 1 OK: {len(events)} events in window, {len(sameDay)} on {date}; "
          f"using the first on-date one.")

    # 2. Fetch odds for exactly ONE on-date event (CREDITS_PER_EVENT_ODDS credits)
    firstEvent = sameDay[0]
    eventData, _ = _getHistoricalTotals(firstEvent["id"], date)

    # 3. Parse
    row = _parseTotals(eventData, date)
    print("\nStep 2/3 — parsed row:")
    print(f"  raw API teams : {eventData.get('away_team')} @ {eventData.get('home_team')}")
    print(f"  commence_time : {eventData.get('commence_time')}")
    if row is None:
        print("VERIFY FAILED: could not parse a totals line from the event "
              "(unmapped team or no totals market on this book/date).")
        if _unmappedNames:
            print(f"  unmapped names: {sorted(_unmappedNames)}")
        return {"ok": False, "reason": "parse returned None", "event": eventData}
    print(f"  parsed        : date={row['game_date']} "
          f"home_id={row['home_team_id']} away_id={row['away_team_id']} "
          f"total_close={row['total_close']}")

    # Sanity on the value itself — a real NBA total sits ~180-270.
    if not (150.0 <= float(row["total_close"]) <= 300.0):
        print(f"VERIFY WARNING: total_close={row['total_close']} is outside the "
              f"sane 150-300 range — parsing may be picking the wrong field.")

    # 4. Store (idempotent)
    db = DBManager(dbPath)
    db.initSchema()
    db.upsertOddsArchive([row])
    print("\nStep 4 OK: upserted the single row into Odds_archive.")

    # 5. Read it back AND run the exact backtest join (line -> Games -> Results)
    conn = sqlite3.connect(str(dbPath))
    back = conn.execute(
        """SELECT oa.total_close, g.game_id, (r.home_score + r.away_score) AS actual_total
           FROM Odds_archive oa
           JOIN Games g   ON oa.game_date = g.game_date
                         AND oa.home_team_id = g.home_team_id
                         AND oa.away_team_id = g.away_team_id
           JOIN Results r ON r.game_id = g.game_id
           WHERE oa.game_date = ? AND oa.home_team_id = ? AND oa.away_team_id = ?""",
        (row["game_date"], row["home_team_id"], row["away_team_id"]),
    ).fetchone()
    conn.close()

    print("\nStep 5 — backtest join (Odds_archive -> Games -> Results):")
    if back is None:
        print("VERIFY FAILED: the stored line did NOT join to a game+result.")
        print("  -> the line is stored but UNUSABLE by the backtest. Most likely a")
        print("     team-id or game_date (ET vs UTC) mismatch against Games.")
        print(f"     stored key: date={row['game_date']} home={row['home_team_id']} "
              f"away={row['away_team_id']}")
        return {"ok": False, "reason": "row does not join to Games/Results", "row": row}

    storedLine, gameId, actualTotal = back
    print(f"  JOINED -> game_id={gameId}  line={storedLine}  actual_total={actualTotal}")
    print(f"  bet check: a bet would settle {'OVER' if actualTotal > storedLine else 'UNDER'} "
          f"(actual {actualTotal} vs line {storedLine})")

    print("\n" + "=" * 64)
    print("VERIFY PASSED — fetch, parse, store, and the backtest join all work.")
    print("Safe to run the full pull. Do the --totals-dry-run first to confirm the")
    print("credit budget, then pull without --totals-dry-run.")
    print("=" * 64)
    return {"ok": True, "row": row, "game_id": gameId, "actual_total": actualTotal}


# PUBLIC ENTRY POINT


def pullHistoricalTotals(startDate, endDate, dbPath="NBA.db", dryRun=False,
                         maxCredits=None):
    """Pull DraftKings game totals for every NBA game between startDate and
    endDate (inclusive, YYYY-MM-DD) and upsert them into Odds_archive.

    Dry run (STRONGLY recommended first): prints the exact credit cost computed
    from the local schedule and returns without touching the API. Use it to
    confirm the window and budget before spending anything.

    maxCredits: hard budget cap. The pull walks dates in order and stops before
    any date whose FULL slate would push spend past the cap — so it never leaves
    a half-pulled date (which would give the backtest an incomplete slate). With
    a cap set, dryRun reports how many dates/games the budget buys instead of the
    whole-window cost. Remaining dates can be pulled later by re-running with a
    startDate past where it stopped.
    """
    est = _estimateCredits(startDate, endDate, dbPath)

    print(f"Window {startDate} -> {endDate}")
    print(f"  {est['dates']} game-dates, {est['games']} games (from local Games table)")
    print(f"  Full-window cost: {est['total_cost']} credits "
          f"({est['events_cost']} events + {est['odds_cost']} odds)")

    # With a budget cap, figure out exactly how far it reaches by walking the
    # per-date slate sizes and stopping before the first date that won't fit
    # whole. This is pure local arithmetic — no API calls, so it's the same in
    # dry run and live.
    plannedDates = _datesWithGames(startDate, endDate, dbPath)
    if maxCredits is not None:
        spent = 0
        fitDates, fitGames, stopDate = 0, 0, None
        for d, n in plannedDates:
            dateCost = 1 + n * CREDITS_PER_EVENT_ODDS   # events lookup + whole slate
            if spent + dateCost > maxCredits:
                stopDate = d
                break
            spent += dateCost
            fitDates += 1
            fitGames += n
        allowedDates = {d for d, _ in plannedDates[:fitDates]}
        print(f"  BUDGET CAP: {maxCredits} credits")
        print(f"  -> fits {fitDates} dates, {fitGames} games, spends {spent} credits")
        if stopDate is not None:
            print(f"  -> stops before {stopDate}; resume later with "
                  f"--pull-totals {stopDate} {endDate}")
        else:
            print(f"  -> the whole window fits under the cap")
    else:
        allowedDates = {d for d, _ in plannedDates}

    if dryRun:
        print("Dry run — no API calls made, no credits spent.")
        return est

    if not allowedDates:
        print("Nothing to pull (empty window, or the cap is below one date's cost).")
        return est

    db = DBManager(dbPath)
    db.initSchema()

    # Balance floor: refuse to start a new date unless the account has enough
    # credits left for the events lookup PLUS that whole slate's odds calls, so we
    # never strand a half-pulled date on an out-of-credits 401 (Bug 2). Read from
    # the API's own x-requests-remaining header, not a from-zero counter.
    scheduledByDate = dict(_datesWithGames(startDate, endDate, dbPath))

    totalRows = 0
    creditsSpent = 0
    creditsRemaining = None   # populated from the first real response
    stoppedForBalance = False

    for date in _dateRange(startDate, endDate, dbPath):
        if date not in allowedDates:
            continue

        # Before spending on this date, make sure the account can cover its whole
        # slate. Cost = 1 events lookup + (scheduled games * odds cost). We only
        # know the real balance after the first call, so the first date always
        # proceeds; every date after is gated on the live header.
        needed = 1 + scheduledByDate.get(date, 0) * CREDITS_PER_EVENT_ODDS
        if creditsRemaining is not None and creditsRemaining < needed:
            print(f"STOP: account has {creditsRemaining} credits, need {needed} for "
                  f"{date}'s slate. Halting before a partial date.")
            print(f"  Resume later with --pull-totals {date} {endDate}")
            stoppedForBalance = True
            break

        events, creditsRemaining = _getHistoricalEvents(date)
        creditsSpent += 1
        if not events:
            continue

        dayRows = []
        for event in events:
            # Bug 1 fix: the historical events endpoint returns a WINDOW of games
            # around the timestamp, spilling into neighboring days. Skip any event
            # whose real (ET) game date isn't the date we're processing BEFORE the
            # paid odds call — otherwise we pay 10 credits for a game that belongs
            # to another date (and re-pay for it when the loop reaches that date).
            if _deriveGameDate(event, date) != date:
                continue

            eventID = event["id"]
            eventData, creditsRemaining = _getHistoricalTotals(eventID, date)
            creditsSpent += CREDITS_PER_EVENT_ODDS
            row = _parseTotals(eventData, date)
            if row is not None:
                dayRows.append(row)

        if dayRows:
            db.upsertOddsArchive(dayRows)
            totalRows += len(dayRows)
            print(f"Stored {len(dayRows)} totals lines for {date} "
                  f"(spent {creditsSpent} | remaining {creditsRemaining})")

    print(f"\nDone. Total totals lines stored: {totalRows}  |  credits spent: {creditsSpent}"
          f"  |  remaining: {creditsRemaining}")
    if stoppedForBalance:
        print("Stopped early on low balance — see the resume command above.")
    if _unmappedNames:
        print(f"WARNING: {len(_unmappedNames)} unmapped team name(s), "
              f"games dropped: {sorted(_unmappedNames)}")
    return {"stored": totalRows, "creditsSpent": creditsSpent,
            "creditsRemaining": creditsRemaining, "stoppedForBalance": stoppedForBalance,
            "unmapped": sorted(_unmappedNames)}


def _dateRange(startDate, endDate, dbPath="NBA.db"):
    """Only yields dates that actually have games locally, so we never spend an
    events-lookup credit on an off day."""
    current = datetime.strptime(startDate, "%Y-%m-%d")
    end = datetime.strptime(endDate, "%Y-%m-%d")
    haveGames = {d for d, _ in _datesWithGames(startDate, endDate, dbPath)}
    while current <= end:
        d = current.strftime("%Y-%m-%d")
        if d in haveGames:
            yield d
        current += timedelta(days=1)
