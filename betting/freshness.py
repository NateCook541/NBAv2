"""
Data freshness check for live scoring

Before scoring a live slate the DB must be current with last nights 
game logs loaded and today's schedule present

checkFreshness() is read-only and returns a FreshnessReport with a binary ok flag
The live workflow gates on report.ok
"""

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class FreshnessReport:
    ok: bool = True
    warnings: list = field(default_factory=list)
    blockers: list = field(default_factory=list)
    # raw values for logging / tests
    maxLogDate: str = None
    scheduleCount: int = 0
    maxStatusDate: str = None
    maxTeamsDate: str = None

    def summary(self):
        lines = ["=" * 60, "Data-freshness check", "=" * 60]
        lines.append(f"max log date : {self.maxLogDate}")
        lines.append(f"games today : {self.scheduleCount}")
        lines.append(f"max status date: {self.maxStatusDate}")
        lines.append(f"max teams date : {self.maxTeamsDate}")
        lines.append("-" * 60)
        for b in self.blockers:
            lines.append(f"BLOCKER : {b}")
        for w in self.warnings:
            lines.append(f"WARN : {w}")
        if not self.blockers and not self.warnings:
            lines.append("All checks clean.")
        lines.append("-" * 60)
        lines.append(f"VERDICT: {'OK' if self.ok else 'NOT OK — do not score live'}")
        lines.append("=" * 60)
        return "\n".join(lines)


def _yesterday(date):
    return (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")


def _scalar(conn, query, params=()):
    cur = conn.execute(query, params)
    row = cur.fetchone()
    return row[0] if row else None


def checkFreshness(dbPath, date, quiet=False):
    """
    Verify the db is fresh enough to score the slate on given date
    
    Hard blockers (set ok=False):
      1. Last night's logs not loaded.
      2. No schedule rows for today

    Warnings (ok stays True):
      3. Injury Status stale — day-before report missing (This might need to be moved due to importance of injurys)
      4. Teams (off/def rtg, pace) stale
    """
    conn = sqlite3.connect(str(dbPath))
    try:
        report = FreshnessReport()
        yesterday = _yesterday(date)

        # 1. Last night's logs loaded
        report.maxLogDate = _scalar(
            conn,
            """
            SELECT MAX(g.game_date)
            FROM Player_game_logs pgl
            JOIN Games g ON pgl.game_id = g.game_id
            """,
        )
        if report.maxLogDate is None or report.maxLogDate < yesterday:
            report.ok = False
            report.blockers.append(
                f"Last night's logs not loaded (max log date {report.maxLogDate} "
                f"< {yesterday}). Rolling features (avgPts10, last1Pts) will be "
                f"stale — run --scrape (without a small --num-games cap)."
            )

        # 2. Today's schedule present
        report.scheduleCount = _scalar(
            conn, "SELECT COUNT(*) FROM Games WHERE game_date = ?", (date,)
        ) or 0
        if report.scheduleCount == 0:
            report.ok = False
            report.blockers.append(
                f"No schedule rows for {date}. Cannot source opp/home/rest pregame "
                f"— run --scrape to ingest the published schedule."
            )

        # 3. Injury Status - the day-before row is the one features read
        report.maxStatusDate = _scalar(conn, "SELECT MAX(scrape_date) FROM Status")
        if report.maxStatusDate is None or report.maxStatusDate < yesterday:
            report.warnings.append(
                f"Injury Status stale (max scrape_date {report.maxStatusDate} "
                f"< {yesterday}). Features read scrape_date == day-before; injury "
                f"context will be empty. Scrape after the ~17:30 ET report."
            )

        # 4. Teams freshness (off/def rtg, pace filtered < date)
        report.maxTeamsDate = _scalar(conn, "SELECT MAX(date) FROM Teams")
        if report.maxTeamsDate is None or report.maxTeamsDate < yesterday:
            report.warnings.append(
                f"Teams table stale (max date {report.maxTeamsDate} < {yesterday}). "
                f"opp_def_rtg / pace features may lag."
            )

        if not quiet:
            print(report.summary())

        return report
    finally:
        conn.close()

