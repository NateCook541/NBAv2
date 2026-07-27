"""
Scrapes historical NBA opening/closing game totals (and spreads) from the
sportsbookreviewsonline.com odds archive (seasons 2007-08 .. 2022-23) and maps
them onto our DB team_ids for join to the Games table.

Why this exists: our own DB starts at the 2023-24 season, so these archives do
NOT overlap it — they are used to (a) train a market-aware totals model on ~16
historical seasons and (b) backtest calibrated P(over) against REAL closing
lines. The archive has no game_id, so rows key on (game_date, home_team_id,
away_team_id) for the join to Games.

Source layout (one HTML table per season page):
    Date  Rot  VH  Team  1st 2nd 3rd 4th  Final  Open  Close  ML  2H
Each game is TWO consecutive rows, visitor (V) then home (H). The Open/Close
columns are overloaded: within each column one row of the pair carries the
game TOTAL (a value >= 100) and the other carries the SPREAD (< 100). The total
open and total close can even land on DIFFERENT rows of the pair, so we pick,
per column independently, the value >= 100 as the total.
"""

import time
import requests
from bs4 import BeautifulSoup

# Politeness — this is a small static site, but be a good citizen.
_REQUEST_SLEEP = 3
_BASE = "https://www.sportsbookreviewsonline.com/scoresoddsarchives"

# A real browser UA — the site 404s the default python/WebFetch agent.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

# Season pages available on the archive, oldest -> newest. The label (e.g.
# "2022-23") is also the season's two calendar years, which we need to turn the
# MMDD dates into real ISO dates.
SEASON_SLUGS = [
    "2007-08", "2008-09", "2009-10", "2010-11", "2011-12", "2012-13",
    "2013-14", "2014-15", "2015-16", "2016-17", "2017-18", "2018-19",
    "2019-20", "2020-21", "2021-22", "2022-23",
]

# Archive team string -> our DB Teams.name abbreviation. Covers the 30 MODERN
# franchises (all that appear 2022-23). Older seasons add relocated/renamed
# aliases below.
ARCHIVE_TO_ABBR = {
    "Atlanta": "ATL", "Boston": "BOS", "Brooklyn": "BRK", "Charlotte": "CHO",
    "Chicago": "CHI", "Cleveland": "CLE", "Dallas": "DAL", "Denver": "DEN",
    "Detroit": "DET", "GoldenState": "GSW", "Houston": "HOU", "Indiana": "IND",
    "LAClippers": "LAC", "LALakers": "LAL", "Memphis": "MEM", "Miami": "MIA",
    "Milwaukee": "MIL", "Minnesota": "MIN", "NewOrleans": "NOP", "NewYork": "NYK",
    "OklahomaCity": "OKC", "Orlando": "ORL", "Philadelphia": "PHI",
    "Phoenix": "PHO", "Portland": "POR", "Sacramento": "SAC", "SanAntonio": "SAS",
    "Toronto": "TOR", "Utah": "UTA", "Washington": "WAS",
}

# Relocations / renames that appear in the older archive seasons. These map to
# the SAME modern franchise team_id (the DB has no separate id for the old name).
#   Seattle SuperSonics  -> OKC Thunder  (moved 2008-09)
#   New Jersey Nets      -> Brooklyn Nets (moved 2012-13)
#   New Orleans Hornets  -> Pelicans      (renamed 2013-14)
#   Charlotte Bobcats    -> Hornets       (renamed 2014-15)
# The New Orleans/Oklahoma City Hornets Katrina-era split (2005-07) is before
# our 2007-08 start, so it is not needed here.
# Keys here are ALREADY whitespace/dot/slash-normalised (see _resolveTeamId).
ARCHIVE_HISTORICAL_ALIASES = {
    "Seattle": "OKC",
    "NewJersey": "BRK",
    # New Orleans appears as several strings across the Katrina-era shuffle; the
    # normalised "NewOrleansOklaCity" covers "NewOrleans/Okla.City" -> NOP.
    "NewOrleansOkla": "NOP",
    "NewOrleansOklaCity": "NOP",
    "Oklahoma": "OKC",       # bare "Oklahoma" fallback -> OKC
}

# NOTE on "Oklahoma": in 2007-08 the source used "Seattle"; from 2008-09 it is
# "OklahomaCity". A bare "Oklahoma" is not expected, but we map it to OKC to be
# safe. Any name we cannot resolve is reported (never silently dropped) — that
# was the bug that lost the Clippers before.


class OddsArchiveScraper:
    """Fetches and parses the SBR NBA odds archive into per-game total rows."""

    def __init__(self, abbrToTeamId):
        # abbrToTeamId: {DB abbreviation -> team_id}, from the Teams table.
        self.abbrToTeamId = abbrToTeamId
        self.unmappedNames = set()

    # Networking

    def _get(self, url):
        time.sleep(_REQUEST_SLEEP)
        resp = requests.get(url, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.text

    def seasonURL(self, slug):
        return f"{_BASE}/nba-odds-{slug}/"

    # Parsing helpers

    @staticmethod
    def _num(x):
        """Parse a cell to float, or None for 'pk', '', 'NL', etc."""
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _pickTotal(a, b):
        """Given the two paired-row values in one column (Open or Close), return
        the game TOTAL: the value >= 100. The other value is the spread."""
        cands = [x for x in (a, b) if x is not None and x >= 100]
        if len(cands) == 1:
            return cands[0]
        # Both >=100 (rare) or neither: fall back to the larger, or None.
        if a is not None and b is not None:
            return max(a, b)
        return a if a is not None else b

    def _toISODate(self, mmdd, startYear):
        """Archive date is MMDD with a stripped leading zero on the month:
        '1018' -> Oct 18 (month 10-12 => season's first year), '101' -> Jan 1
        (month 1-9 => season's second year). Returns 'YYYY-MM-DD' or None."""
        s = str(mmdd).strip()
        if not s.isdigit():
            return None
        if len(s) == 4:          # MMDD, month is 10/11/12
            month, day = int(s[:2]), int(s[2:])
        elif len(s) == 3:        # MDD, month is 1-9
            month, day = int(s[:1]), int(s[1:])
        else:
            return None
        if not (1 <= month <= 12 and 1 <= day <= 31):
            return None
        year = startYear if month >= 8 else startYear + 1
        return f"{year:04d}-{month:02d}-{day:02d}"

    def _resolveTeamId(self, name):
        # Names are inconsistently spaced across (and even within) seasons —
        # e.g. both "Golden State" and "GoldenState" appear in 2021-22. Normalise
        # by stripping all internal whitespace/dots/slashes before lookup so
        # every spacing variant resolves to the same alias.
        key = name.replace(" ", "").replace(".", "").replace("/", "")
        abbr = ARCHIVE_TO_ABBR.get(key) or ARCHIVE_HISTORICAL_ALIASES.get(key)
        if abbr is None:
            self.unmappedNames.add(name)
            return None
        teamId = self.abbrToTeamId.get(abbr)
        if teamId is None:
            self.unmappedNames.add(name)
        return teamId

    def parseSeason(self, html, slug):
        """Parse one season page's HTML into a list of game-total dicts.

        Returns rows shaped for DBManager.upsertOddsArchive plus diagnostics.
        """
        startYear = int(slug[:4])
        soup = BeautifulSoup(html, "html.parser")
        tables = soup.find_all("table")
        if not tables:
            return [], {"pairs": 0, "bad": 0, "unmapped": 0, "filtered": 0}
        table = max(tables, key=lambda t: len(t.find_all("tr")))

        rows = [
            [c.get_text(strip=True) for c in tr.find_all("td")]
            for tr in table.find_all("tr")
        ]
        # Drop the header row(s) — any row that isn't a V/H data row.
        data = [r for r in rows if len(r) >= 13 and r[2] in ("V", "H")]

        games, bad, filtered = [], 0, 0
        i = 0
        while i < len(data) - 1:
            v, h = data[i], data[i + 1]
            # Expect a V then H pair; if not aligned, skip one and resync.
            if v[2] != "V" or h[2] != "H":
                bad += 1
                i += 1
                continue
            i += 2

            date = self._toISODate(v[0], startYear)
            awayId = self._resolveTeamId(v[3])
            homeId = self._resolveTeamId(h[3])

            totalOpen = self._pickTotal(self._num(v[9]), self._num(h[9]))
            totalClose = self._pickTotal(self._num(v[10]), self._num(h[10]))

            vFinal, hFinal = self._num(v[8]), self._num(h[8])
            actualTotal = (
                vFinal + hFinal if (vFinal is not None and hFinal is not None)
                else None
            )

            if date is None or awayId is None or homeId is None:
                bad += 1
                continue

            # Sanity filter: a real NBA total is ~180-260. Reject the ~0.3% of
            # source typos (e.g. a total mistyped as '47.5'); fall back where we
            # can, else drop the offending value.
            if totalOpen is not None and totalOpen < 150:
                totalOpen = None
            if totalClose is not None and totalClose < 150:
                totalClose = None
            if totalOpen is None and totalClose is None:
                filtered += 1
                continue

            games.append({
                "season": startYear + 1,   # DB season = ending calendar year
                "game_date": date,
                "home_team_id": homeId,
                "away_team_id": awayId,
                "home_score": int(hFinal) if hFinal is not None else None,
                "away_score": int(vFinal) if vFinal is not None else None,
                "total_open": totalOpen,
                "total_close": totalClose,
                "actual_total": int(actualTotal) if actualTotal is not None else None,
            })

        diagnostics = {
            "pairs": len(games) + bad + filtered,
            "games": len(games),
            "bad": bad,
            "filtered": filtered,
        }
        return games, diagnostics

    def scrapeSeason(self, slug):
        html = self._get(self.seasonURL(slug))
        return self.parseSeason(html, slug)

    def scrapeAll(self, slugs=None):
        """Scrape every season, returning (allRows, perSeasonDiagnostics)."""
        slugs = slugs or SEASON_SLUGS
        allRows, perSeason = [], {}
        for slug in slugs:
            rows, diag = self.scrapeSeason(slug)
            allRows.extend(rows)
            perSeason[slug] = diag
            print(
                f"[OddsArchive] {slug}: {diag['games']} games "
                f"(bad {diag['bad']}, filtered {diag['filtered']})"
            )
        if self.unmappedNames:
            print(f"[OddsArchive] UNMAPPED names (need aliases): "
                  f"{sorted(self.unmappedNames)}")
        return allRows, perSeason
