"""
Official NBA injury reports, every 15 minutes.

The NBA posts a PDF report at a fixed URL every 15 minutes on game days (stamped in ET), and every past
report stays online. nbainjuries fetches + parses them (tabula, so it needs Java). Reports are stored
whole in InjuryReports (kalshi.db, the VPS owns it, the laptop's NBA.db push would wipe anything
written into NBA.db there), one row per player per report, so any moment's status can be rebuilt.

pollInjuryReports:  fetch reports newer than the last stored one (cron / decide / news trader)
backfillInjuryReports: fetch a date range (laptop, for studies and training)
latestStatus: each player's status in the newest report at or before a time, per game
"""

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from data.dbManager import DBManager, connect

ET = ZoneInfo("America/New_York")

# NBA report team names -> the basketball-reference abbreviations the Teams table uses
NBA_REPORT_TEAM_MAP = {
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
    "LA Clippers": "LAC",
    "Los Angeles Clippers": "LAC",   # NBA reports use both forms
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
}

# Kalshi market tickers use these team codes, the rest match the Teams table
KALSHI_TEAM_CODES = {"BRK": "BKN", "CHO": "CHA", "PHO": "PHX"}

# 15-minute reports started 2025-12-22 09:00 ET, before that the NBA posted hourly
FIFTEEN_MIN_START = datetime(2025, 12, 22, 9, 0)


def kalshiTeamCode(abbr):
    return KALSHI_TEAM_CODES.get(abbr, abbr)


def _flipName(name):
    """Report names are "Last, First" """
    parts = [p.strip() for p in str(name or "").split(",", 1)]
    return f"{parts[1]} {parts[0]}" if len(parts) == 2 else str(name or "")


def _slotStep(slot):
    return 15 if slot >= FIFTEEN_MIN_START else 60


def _floorSlot(dtEt):
    """Latest report slot at or before a naive ET datetime"""
    step = _slotStep(dtEt)
    return dtEt.replace(minute=dtEt.minute - dtEt.minute % step, second=0, microsecond=0)


def fetchReport(slotEt):
    """Parsed report for a naive ET slot, or None if the NBA didn't post one (403/404) or it failed to parse"""
    import jpype.config
    from nbainjuries import injury
    # jpype tears the JVM down at interpreter exit, which deadlocks with Java 25 + jpype 1.7 (VPS) and leaves
    # the process hanging forever after its work is done. Skip the teardown, the OS reclaims the JVM on exit
    jpype.config.destroy_jvm = False
    try:
        df = injury.get_reportdata(slotEt, return_df=True)
    except Exception as e:
        if "403" not in str(e) and "404" not in str(e):
            print(f"[injury] {slotEt:%Y-%m-%d %H:%M} ET: {type(e).__name__}: {str(e)[:120]}")
        return None
    return df if df is not None and len(df) else None


_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


def _stripSuffix(key):
    return " ".join(t for t in key.replace(",", " ").split() if t not in _SUFFIXES)


class _PlayerMatcher:
    """Report name -> player_id: exact normalized name first, then ignoring Jr./II/III (reports and
    basketball-reference disagree on suffixes, e.g. "GG Jackson" vs "GG Jackson II")"""

    def __init__(self, playerMap):
        from betting.backtest import _normalize
        self.norm, self.exact = _normalize, playerMap
        loose = {}
        for k, pid in playerMap.items():
            loose.setdefault(_stripSuffix(k), set()).add(pid)
        self.loose = {k: next(iter(v)) for k, v in loose.items() if len(v) == 1}

    def get(self, name):
        key = self.norm(name)
        return self.exact.get(key) or self.loose.get(_stripSuffix(key))


def _lookups(conn):
    from betting.backtest import _loadPlayerMap
    teams = pd.read_sql_query("SELECT DISTINCT team_id, name FROM Teams", conn)
    return _PlayerMatcher(_loadPlayerMap(conn)), dict(zip(teams["name"], teams["team_id"]))


def _toRows(df, slotEt, fetchedUtc, playerMap, teamMap):
    reportUtc = slotEt.replace(tzinfo=ET).astimezone(timezone.utc)
    rows = []
    for rec in df.to_dict("records"):
        name, teamName = rec.get("Player Name"), rec.get("Team")
        abbr = NBA_REPORT_TEAM_MAP.get(teamName)
        try:
            gameDate = datetime.strptime(str(rec.get("Game Date")), "%m/%d/%Y").strftime("%Y-%m-%d")
        except ValueError:
            continue
        # Teams that haven't submitted yet come through as a row with no player
        if not name or pd.isna(name) or str(name).strip().lower() in ("nan", "") or not abbr:
            continue
        flipped = _flipName(name)
        teamID = teamMap.get(abbr)
        rows.append({
            "report_et": f"{slotEt:%Y-%m-%d %H:%M}",
            "report_utc": f"{reportUtc:%Y-%m-%d %H:%M:%S}",
            "fetched_utc": fetchedUtc,
            "game_date": gameDate,
            "team_id": int(teamID) if teamID is not None else None,
            "team_abbr": abbr,
            "player_name": flipped,
            "player_id": playerMap.get(flipped),
            "status": rec.get("Current Status"),
            "reason": rec.get("Reason"),
        })
    return rows


def _storeSlots(dbPath, slots, dryRun=False, workers=1):
    """Fetch and store the given ET slots (oldest first). Returns the slots that had a report.
    workers > 1 fetches/parses in parallel (backfills), rows are still written in slot order"""
    from concurrent.futures import ThreadPoolExecutor
    conn = connect(dbPath)
    playerMap, teamMap = _lookups(conn)
    conn.close()
    db = DBManager(dbPath)
    found = []
    with ThreadPoolExecutor(max(1, workers)) as pool:
        fetched = pool.map(lambda s: (s, fetchReport(s)), slots)
        for slot, df in fetched:
            if df is None:
                continue
            rows = _toRows(df, slot, f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}", playerMap, teamMap)
            unmatched = sorted({r["player_name"] for r in rows if r["player_id"] is None})
            print(f"[injury] report {slot:%Y-%m-%d %H:%M} ET: {len(rows)} rows"
                  + (f", {len(unmatched)} unmatched names (e.g. {', '.join(unmatched[:3])})" if unmatched else ""))
            if not dryRun and rows:
                db.insertInjuryReports(rows)
            found.append(slot)
    return found


def pollInjuryReports(dbPath="NBA.db", lookbackMin=90, dryRun=False):
    """
    Fetch every report newer than the newest stored one, looking back at most lookbackMin.
    The PDF goes up a few minutes after its stamp, so a slot that isn't posted yet is just retried next run.
    Returns the new report slots (naive ET datetimes)
    """
    nowEt = datetime.now(ET).replace(tzinfo=None)
    conn = connect(dbPath)
    last = conn.execute("SELECT MAX(report_et) FROM InjuryReports").fetchone()[0]
    conn.close()
    newest = _floorSlot(nowEt)
    oldest = max(newest - timedelta(minutes=lookbackMin),
                 datetime.strptime(last, "%Y-%m-%d %H:%M") + timedelta(minutes=1) if last else datetime.min)
    slots, s = [], newest
    while s >= oldest:
        slots.append(s)
        s -= timedelta(minutes=_slotStep(s))
    found = _storeSlots(dbPath, sorted(slots), dryRun=dryRun)
    if not found:
        print(f"[injury] no new report (newest stored {last} ET)")
    return found


def backfillInjuryReports(dbPath, startDate, endDate=None, firstHour=8, lastHour=23, workers=6, dryRun=False):
    """Every report slot between firstHour and lastHour ET for each date in the range. Skips stored slots"""
    endDate = endDate or startDate
    conn = connect(dbPath)
    have = {r[0] for r in conn.execute("SELECT DISTINCT report_et FROM InjuryReports")}
    conn.close()
    slots = []
    day = datetime.strptime(startDate, "%Y-%m-%d")
    while day <= datetime.strptime(endDate, "%Y-%m-%d"):
        s = day.replace(hour=firstHour)
        while s <= day.replace(hour=lastHour, minute=45):
            if f"{s:%Y-%m-%d %H:%M}" not in have:
                slots.append(s)
            s += timedelta(minutes=_slotStep(s))
        day += timedelta(days=1)
    print(f"[injury] backfill {startDate} -> {endDate}: {len(slots)} slots to try")
    found = _storeSlots(dbPath, slots, dryRun=dryRun, workers=workers)
    print(f"[injury] backfill done, {len(found)} reports")
    return found


def latestStatus(conn, gameDates, asOfUtc=None):
    """
    For each (game_date, team) the newest report at or before asOfUtc that lists the team for that game,
    one row per player in it. Players missing from that report are not listed (healthy)
    """
    asOf = f"{(asOfUtc or datetime.now(timezone.utc)):%Y-%m-%d %H:%M:%S}"
    dates = sorted(set(gameDates))
    if not dates:
        return pd.DataFrame()
    marks = ",".join("?" * len(dates))
    df = pd.read_sql_query(
        f"""
        SELECT * FROM InjuryReports
        WHERE game_date IN ({marks}) AND report_utc <= ?
        """,
        conn, params=[*dates, asOf],
    )
    if df.empty:
        return df
    newest = df.groupby(["game_date", "team_abbr"])["report_et"].transform("max")
    return df[df["report_et"] == newest].reset_index(drop=True)
