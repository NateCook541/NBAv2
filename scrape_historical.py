"""
Resumable historical totals-data scrape for seasons 2015..2023.

Scrapes ONLY what the totals model (ResultsBundle) needs:
  Games  (B-Ref season index)  -> Games table + output/games.json
  Results(B-Ref season index)  -> Results table
  Ratings(nba.com per game)    -> Team_game_ratings table   <-- the slow part

Deliberately SKIPS player logs and injury/status data: the model's 12 real
RESULTS_FEATURES contain no player/injury feature (they're pruned), so those
scrapes would be wasted hours.

Resumability:
  - Games/Results per season are cheap and idempotent (INSERT OR REPLACE).
  - games.json is EXTENDED (existing 2024-2026 preserved, historical merged in).
  - Ratings are scraped season-by-season, and within each season we SKIP any
    game_id already present in Team_game_ratings. So a crash loses at most the
    in-progress season; re-running the script resumes where it stopped.
"""

import json
import sqlite3
import sys
import time

sys.path.insert(0, "/home/cookn1/NBAv2")

from data.scrapperEngine import ScrapeEngine
from data.dbManager import DBManager

DB = "/home/cookn1/NBAv2/NBA.db"
GAMES_JSON = "/home/cookn1/NBAv2/output/games.json"
TEAMS_MAP = "/home/cookn1/NBAv2/output/teams_map.json"
SEASONS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]  # 2015-16 .. 2022-23

db = DBManager(DB)
db.initSchema()

# Build the engine WITHOUT Selenium (scrapeGames/Results/Ratings use requests).
# __new__ skips __init__, so we set the attributes those methods need by hand:
# teamMap + fullNameConversion (the modern name dict covers 2016-2023 cleanly —
# Charlotte/New Orleans/Brooklyn renames all predate 2016).
engine = ScrapeEngine.__new__(ScrapeEngine)
engine.db = db
engine.teamMap = json.load(open(TEAMS_MAP))
engine.fullNameConversion = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BRK",
    "Charlotte Hornets": "CHO", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "LA Clippers": "LAC", "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM", "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHO",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR", "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}

# ---- Step 1: Games + Results per season (fast: ~1 B-Ref page each) ----------

def loadGamesJson():
    try:
        return json.load(open(GAMES_JSON))
    except (FileNotFoundError, json.JSONDecodeError):
        return []

existing = loadGamesJson()
byId = {g["game_id"]: g for g in existing}
print(f"[hist] games.json starts with {len(byId)} games")

for season in SEASONS:
    print(f"\n[hist] === season {season}: Games + Results ===", flush=True)
    try:
        games = engine.scrapeGames(season=season)
        if games:
            db.upsertGames(games)
            for g in games:
                byId[g["game_id"]] = g
    except Exception as e:
        print(f"[hist] scrapeGames({season}) FAILED: {e}", flush=True)
    try:
        results = engine.scrapeResults(season=season)
        if results:
            db.upsertResults(results)
    except Exception as e:
        print(f"[hist] scrapeResults({season}) FAILED: {e}", flush=True)

# Persist the merged games.json (existing 2024-26 + historical) — the
# authoritative game list. NOTE: unlike the first version, the ratings step
# below does NOT touch games.json (it scrapes via the engine's internal methods
# directly), so a mid-run kill can no longer truncate this file.
merged = list(byId.values())
merged.sort(key=lambda g: (g["game_date"], g["game_id"]))
json.dump(merged, open(GAMES_JSON, "w"), indent=2)
print(f"\n[hist] games.json now has {len(merged)} games", flush=True)

# ---- Step 2: Ratings per season, skipping already-scraped games -------------
#
# We drive the scrape ourselves (scoreboard index per date -> advanced ratings
# per game) instead of engine.scrapeTeamGameRatings, so we can: skip already-
# rated games, upsert incrementally (crash-safe), and FAIL FAST if nba.com is
# blocking this IP rather than grinding through 210 x 30s timeouts per season.

FAIL_FAST_THRESHOLD = 8   # consecutive scoreboard failures => nba.com is blocked

def ratedGameIds():
    conn = sqlite3.connect(DB)
    ids = {r[0] for r in conn.execute("SELECT DISTINCT game_id FROM Team_game_ratings")}
    conn.close()
    return ids

def scrapeSeasonRatings(season):
    """Returns True on completion, False if we bailed due to an nba.com block."""
    already = ratedGameIds()
    todo = [g for g in merged
            if g.get("season") == season and g["game_id"] not in already]
    print(f"\n[hist] === season {season}: Ratings — "
          f"{len(todo)} games to scrape (rest already done) ===", flush=True)
    if not todo:
        return True

    byDate = {}
    for g in todo:
        byDate.setdefault(g["game_date"], []).append(g)

    consecutiveFails = 0
    scraped = 0
    for i, (gameDate, dateGames) in enumerate(sorted(byDate.items())):
        index = engine._scoreboardGameIndex(gameDate)
        if not index:
            consecutiveFails += 1
            if consecutiveFails >= FAIL_FAST_THRESHOLD:
                print(f"[hist] ABORT season {season}: {consecutiveFails} consecutive "
                      f"scoreboard failures — stats.nba.com is blocking this IP. "
                      f"Re-run later / on a reachable machine; already-scraped games "
                      f"are saved and will be skipped.", flush=True)
                return False
            continue
        consecutiveFails = 0

        rows = []
        for g in dateGames:
            nbaID = index.get((g["home_team_id"], g["away_team_id"]))
            if nbaID is None:
                continue
            rows.extend(engine._fetchAdvancedRatings(nbaID, g["game_id"]))

        if rows:
            db.upsertTeamGameRatings(rows)   # incremental upsert => crash-safe
            scraped += len(rows)
        if (i + 1) % 20 == 0:
            print(f"  season {season}: {i + 1}/{len(byDate)} dates, "
                  f"{scraped} rating rows", flush=True)

    print(f"[hist] season {season}: DONE, {scraped} rating rows", flush=True)
    return True

for season in SEASONS:
    ok = scrapeSeasonRatings(season)
    if not ok:
        print("[hist] Stopping the ratings loop — nba.com unreachable.", flush=True)
        break

# ---- Verification -----------------------------------------------------------

conn = sqlite3.connect(DB)
c = conn.cursor()
print("\n[verify] Games/Results/Ratings coverage per season:")
c.execute("""SELECT g.season,
                    COUNT(DISTINCT g.game_id),
                    COUNT(DISTINCT r.game_id),
                    COUNT(DISTINCT tgr.game_id)
             FROM Games g
             LEFT JOIN Results r ON r.game_id = g.game_id
             LEFT JOIN Team_game_ratings tgr ON tgr.game_id = g.game_id
             GROUP BY g.season ORDER BY g.season""")
for s, ng, nr, nt in c.fetchall():
    print(f"    {s}: games {ng}  results {nr}  ratings {nt}")
conn.close()
print("\n[hist] DONE", flush=True)
