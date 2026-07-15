import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# Defines the db schema.
# Creates all the tables and defines vars + pk
# If they already exist then just skipped over
dbSchema = {
    "Players": """
        CREATE TABLE IF NOT EXISTS Players (
            player_id   INTEGER PRIMARY KEY,
            name        TEXT    NOT NULL,
            team_id     INTEGER,
            position    TEXT,
            is_active   BOOLEAN NOT NULL DEFAULT 1
        )
    """,
    "Teams": """
        CREATE TABLE IF NOT EXISTS Teams (
            team_id  INTEGER,
            name     TEXT    NOT NULL,
            off_rtg  REAL,
            def_rtg  REAL,
            pace     REAL,
            date     TEXT    NOT NULL,
            PRIMARY KEY (team_id, date)
        )
    """,
    "Games": """
        CREATE TABLE IF NOT EXISTS Games (
            game_id      TEXT    PRIMARY KEY,
            game_date    TEXT    NOT NULL,
            home_team_id INTEGER NOT NULL,
            away_team_id INTEGER NOT NULL,
            season       INTEGER NOT NULL
        )
    """,
    "Player_game_logs": """
        CREATE TABLE IF NOT EXISTS Player_game_logs (
            log_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id  INTEGER NOT NULL,
            game_id    TEXT    NOT NULL,
            minutes    REAL,
            points     INTEGER,
            rebounds   INTEGER,
            assists    INTEGER,
            steals     INTEGER,
            blocks     INTEGER,
            turnovers  INTEGER,
            fg_pct     REAL,
            is_starter BOOLEAN,
            is_home    BOOLEAN,
            rest_days  INTEGER
        )
    """,
    "Status": """
        CREATE TABLE IF NOT EXISTS Status (
            player_id     INTEGER NOT NULL,
            team_id       INTEGER NOT NULL,
            game_id       TEXT,
            scrape_date   TEXT NOT NULL,
            report_time   TEXT,
            status        TEXT,
            reason        TEXT,
            comment       TEXT,
            PRIMARY KEY (player_id, game_id, scrape_date)
        )
    """,
    "Props": """
        CREATE TABLE IF NOT EXISTS Props (
            prop_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            game_date     TEXT    NOT NULL,
            player_name   TEXT    NOT NULL,
            line          REAL    NOT NULL,
            over_odds     INTEGER,
            under_odds    INTEGER,
            bookmaker     TEXT,
            fetched_at    TEXT    NOT NULL
        )
    """,
    # Live prop snapshots for CLV tracking. Separate from Props so an 'open'
    # (decision-time) and 'close' (near-tip) snapshot of the same prop coexist.
    "PropSnapshots": """
        CREATE TABLE IF NOT EXISTS PropSnapshots (
            snapshot_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            game_date     TEXT    NOT NULL,
            player_name   TEXT    NOT NULL,
            line          REAL    NOT NULL,
            over_odds     INTEGER,
            under_odds    INTEGER,
            bookmaker     TEXT,
            snapshot_type TEXT    NOT NULL,
            fetched_at    TEXT    NOT NULL
        )
    """,
    # Candidate bets recorded at decision time (open line) with CLV filled in
    # after the close snapshot, and actuals filled in (deferred) after games.
    "CLVLedger": """
        CREATE TABLE IF NOT EXISTS CLVLedger (
            bet_id          INTEGER PRIMARY KEY AUTOINCREMENT,
            game_date       TEXT    NOT NULL,
            player_name     TEXT    NOT NULL,
            player_id       INTEGER,
            side            TEXT    NOT NULL,
            open_line       REAL    NOT NULL,
            open_side_odds  INTEGER,
            predicted       REAL,
            pred_diff       REAL,
            my_prob         REAL,
            fair_open       REAL,
            edge            REAL,
            recorded_at     TEXT    NOT NULL,
            close_line      REAL,
            close_side_odds INTEGER,
            fair_close      REAL,
            clv_prob        REAL,
            clv_points      REAL,
            beat_close      INTEGER,
            actual_points   INTEGER,
            won             INTEGER
        )
    """,
}

extraIndexes = [
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_player_game ON Player_game_logs (player_id, game_id)",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_prop_unique ON Props (game_date, player_name, line, bookmaker)",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_player_name ON Players (name)",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_snapshot_unique ON PropSnapshots (game_date, player_name, line, bookmaker, snapshot_type)",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_clv_unique ON CLVLedger (game_date, player_name, side)",
]

# A small SQLite wrapper for the scrapped NBA data
class DBManager:
    def __init__(self, dbPath="NBA.db"):
        self.dbPath = dbPath

    # Creates a connect that will auto commit or roll back
    # Uses a decorator to write a connect with a with block
    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.dbPath)
        # Makes the rows returned appear as actual rows not like tuples that are hard to read
        conn.row_factory = sqlite3.Row
        try:
            # Yield here is important as it gives the control over to the with block that wil be called with this
            # So thr code in the with block will run then when that finishes this will get control back and commit the changes
            yield conn
            conn.commit()
        except Exception:
            # If a error does happen in the with block a rollback is happened to prevent errors from going into the db
            # and a error is raised
            conn.rollback()
            raise
        finally:
            # This will alsways run and just closes the db to prevent memory leakage
            conn.close()
    
    # Creates all tables + indexes if they don't already exist
    def initSchema(self):
        with self._connect() as conn:
            cur = conn.cursor()
            for ddl in dbSchema.values():
                cur.execute(ddl)
            for inx in extraIndexes:
                cur.execute(inx)
        print(f"DB Schema created in {self.dbPath}")

    # Bulk runs all the upsert methods
    def _bulk_execute(self, conn, sql, rows):
        cur = conn.cursor()
        cur.executemany(sql, rows)
        return cur.rowcount

    
    # UPSERT METHODS

    # TEAMS

    # Upserts data into the teams entity in the db
    def upsertTeams(self, data):
        # SQL query using slite lite upsert, meaning if an pk already exists just replace it or if not insert it.
        # Uses parameterized querys becuase teach told me 2. (Prevent SQL injection)
        sql = """
            INSERT OR REPLACE INTO Teams (team_id, name, off_rtg, def_rtg, pace, date)
            VALUES (:team_id, :name, :off_rtg, :def_rtg, :pace, :date)
            """

        # Opens a connect with the db and executes the query for each item in data
        with self._connect() as conn:
            conn.cursor().executemany(sql, data)

        print(f"Upserted {len(data)} team records")

    # PLAYERS

    def upsertPlayers(self, data):
        sql = """
            INSERT OR REPLACE INTO Players (player_id, name, team_id, position, is_active)
            VALUES (:player_id, :name, :team_id, :position, :is_active)
        """

        with self._connect() as conn:
            conn.cursor().executemany(sql, data)

        print(f"Upserted {len(data)} player records")

    # GAMES
    
    def upsertGames(self, data):
        sql = """
            INSERT OR REPLACE INTO Games (game_id, game_date, home_team_id, away_team_id, season)
            VALUES (:game_id, :game_date, :home_team_id, :away_team_id, :season)
        """

        with self._connect() as conn:
            conn.cursor().executemany(sql, data)

        print(f"Upserted {len(data)} games records")

    # LOGS

    def upsertLogs(self, data):
        sql = """
            INSERT OR IGNORE INTO Player_game_logs
                (player_id, game_id, minutes, points, rebounds, assists,
                 steals, blocks, turnovers, fg_pct, is_starter, is_home, rest_days)
            VALUES
                (:player_id, :game_id, :minutes, :points, :rebounds, :assists,
                 :steals, :blocks, :turnovers, :fg_pct, :is_starter, :is_home, :rest_days)
        """

        with self._connect() as conn:
            conn.cursor().executemany(sql, data)

        print(f"Upserted {len(data)} logs records")

    # STATUS

    def upsertStatus(self, data):
        sql = """
            INSERT OR IGNORE INTO Status
                (player_id, team_id, game_id, scrape_date, report_time, status, reason, comment)
            VALUES
                (:player_id, :team_id, :game_id, :scrape_date, :report_time, :status, :reason, :comment)
        """

        with self._connect() as conn:
            conn.cursor().executemany(sql, data)

        print(f"Upserted {len(data)} status records")

    def getLastStatusDate(self):
        with self._connect() as conn:
            res = conn.execute("SELECT MAX(scrape_date) FROM Status").fetchone()
        return res[0] if res and res[0] else None
    
    # PROPS

    def upsertProps(self, data):
        sql = """
            INSERT OR IGNORE INTO Props
                (game_date, player_name, line, over_odds, under_odds, bookmaker, fetched_at)
            VALUES
                (:game_date, :player_name, :line, :over_odds, :under_odds, :bookmaker, :fetched_at)
        """
        
        with self._connect() as conn:
            conn.cursor().executemany(sql, data)

        print(f"Upserted {len(data)} prop records")

    # PROP SNAPSHOTS (live CLV tracking)

    def upsertPropSnapshots(self, data):
        sql = """
            INSERT OR IGNORE INTO PropSnapshots
                (game_date, player_name, line, over_odds, under_odds,
                 bookmaker, snapshot_type, fetched_at)
            VALUES
                (:game_date, :player_name, :line, :over_odds, :under_odds,
                 :bookmaker, :snapshot_type, :fetched_at)
        """

        with self._connect() as conn:
            conn.cursor().executemany(sql, data)

        print(f"Upserted {len(data)} prop snapshot records")

    # CLV LEDGER

    def insertCLVCandidates(self, data):
        """Insert decision-time candidate bets (open line/odds). Idempotent per
        (game_date, player_name, side)."""
        sql = """
            INSERT OR IGNORE INTO CLVLedger
                (game_date, player_name, player_id, side, open_line,
                 open_side_odds, predicted, pred_diff, my_prob, fair_open, edge, recorded_at)
            VALUES
                (:game_date, :player_name, :player_id, :side, :open_line,
                 :open_side_odds, :predicted, :pred_diff, :my_prob, :fair_open, :edge, :recorded_at)
        """

        with self._connect() as conn:
            conn.cursor().executemany(sql, data)

        print(f"Inserted {len(data)} CLV candidate records")

    def updateCLVClose(self, data):
        """Fill close line/odds + CLV metrics for existing ledger rows."""
        sql = """
            UPDATE CLVLedger
            SET close_line = :close_line,
                close_side_odds = :close_side_odds,
                fair_close = :fair_close,
                clv_prob = :clv_prob,
                clv_points = :clv_points,
                beat_close = :beat_close
            WHERE game_date = :game_date
              AND player_name = :player_name
              AND side = :side
        """

        with self._connect() as conn:
            conn.cursor().executemany(sql, data)

        print(f"Updated {len(data)} CLV rows with close values")

    def updateCLVSettle(self, data):
        """Deferred: fill actual_points / won after games complete."""
        sql = """
            UPDATE CLVLedger
            SET actual_points = :actual_points,
                won = :won
            WHERE game_date = :game_date
              AND player_name = :player_name
              AND side = :side
        """

        with self._connect() as conn:
            conn.cursor().executemany(sql, data)

        print(f"Settled {len(data)} CLV rows")

