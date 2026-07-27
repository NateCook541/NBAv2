"""
Ratings-only historical scrape (Team_game_ratings) for seasons already present
in the Games table. Games + Results are assumed DONE (see scrape_historical.py);
this script does NOT touch them.

Scrapes per-game off/def rtg + pace from stats.nba.com. Resumable + crash-safe:
  - Skips every game already in Team_game_ratings (so a re-run resumes).
  - Upserts incrementally per date.
  - On a run of consecutive nba.com timeouts it SKIPS THE REST OF THAT SEASON
    and moves to the next season, rather than killing the whole run — a
    transient block at one season's playoff tail shouldn't abort everything.

Config: edit SEASONS / FAIL_FAST_THRESHOLD below.
  SEASONS defaults to 2017..2023. To also finish 2016's missing playoff games,
  add 2016 back to the list (already-rated games are skipped, so it's cheap).
"""

import json
import sqlite3
import sys

sys.path.insert(0, "/home/cookn1/NBAv2")

from data.scrapperEngine import ScrapeEngine
from data.dbManager import DBManager

DB = "/home/cookn1/NBAv2/NBA.db"
TEAMS_MAP = "/home/cookn1/NBAv2/output/teams_map.json"

# Seasons to scrape ratings for. Per user: start from 2017 (2016 mostly done).
# Add 2016 here if you want its ~336 missing playoff games too.
SEASONS = [2017, 2018, 2019, 2020, 2021, 2022, 2023]

# Consecutive scoreboard failures before we give up on the CURRENT season and
# move to the next (was an abort-everything before; now it's per-season).
FAIL_FAST_THRESHOLD = 12

db = DBManager(DB)

# Engine without Selenium — ratings scraping uses requests only.
engine = ScrapeEngine.__new__(ScrapeEngine)
engine.db = db
engine.teamMap = json.load(open(TEAMS_MAP))
# fullNameConversion isn't used by the ratings path, but set it to be safe.
engine.fullNameConversion = {}


def gamesForSeason(season):
    """Played games for a season, straight from the Games table (not games.json
    — the DB is authoritative and can't be truncated by a kill)."""
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT g.game_id, g.game_date, g.home_team_id, g.away_team_id
           FROM Games g
           JOIN Results r ON r.game_id = g.game_id
           WHERE g.season = ?
           ORDER BY g.game_date, g.game_id""",
        (season,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def ratedGameIds():
    conn = sqlite3.connect(DB)
    ids = {r[0] for r in conn.execute("SELECT DISTINCT game_id FROM Team_game_ratings")}
    conn.close()
    return ids


def scrapeSeasonRatings(season):
    already = ratedGameIds()
    todo = [g for g in gamesForSeason(season) if g["game_id"] not in already]
    print(f"\n[ratings] === season {season}: "
          f"{len(todo)} games to scrape (rest already done) ===", flush=True)
    if not todo:
        return

    byDate = {}
    for g in todo:
        byDate.setdefault(g["game_date"], []).append(g)

    consecutiveFails = 0
    scraped = 0
    dates = sorted(byDate.items())
    for i, (gameDate, dateGames) in enumerate(dates):
        index = engine._scoreboardGameIndex(gameDate)
        if not index:
            consecutiveFails += 1
            if consecutiveFails >= FAIL_FAST_THRESHOLD:
                print(f"[ratings] SKIP rest of season {season}: "
                      f"{consecutiveFails} consecutive scoreboard failures "
                      f"(nba.com throttling). {scraped} rows saved this season; "
                      f"moving to next season. Re-run later to fill the gap.",
                      flush=True)
                return
            continue
        consecutiveFails = 0

        rows = []
        for g in dateGames:
            nbaID = index.get((g["home_team_id"], g["away_team_id"]))
            if nbaID is None:
                continue
            rows.extend(engine._fetchAdvancedRatings(nbaID, g["game_id"]))

        if rows:
            db.upsertTeamGameRatings(rows)   # incremental => crash-safe
            scraped += len(rows)
        if (i + 1) % 20 == 0:
            print(f"  season {season}: {i + 1}/{len(dates)} dates, "
                  f"{scraped} rating rows", flush=True)

    print(f"[ratings] season {season}: DONE, {scraped} rating rows", flush=True)


for season in SEASONS:
    scrapeSeasonRatings(season)

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
             WHERE g.season BETWEEN 2016 AND 2026
             GROUP BY g.season ORDER BY g.season""")
for s, ng, nr, nt in c.fetchall():
    gap = ng - nt
    flag = "  <-- incomplete" if gap > 0 and s <= 2023 else ""
    print(f"    {s}: games {ng}  results {nr}  ratings {nt}{flag}")
conn.close()
print("\n[ratings] DONE", flush=True)
