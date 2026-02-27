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
            status_log_id INTEGER,
            player_id     INTEGER NOT NULL,
            team_id       INTEGER NOT NULL,
            game_id       TEXT,
            scrape_date   TEXT    NOT NULL,
            status        TEXT,
            return_date   TEXT,
            comment       TEXT,
            PRIMARY KEY (status_log_id, scrape_date, game_id)
        )
    """,
    "Results": """
        CREATE TABLE IF NOT EXISTS Results (
            game_id      TEXT    PRIMARY KEY,
            home_team_id INTEGER NOT NULL,
            away_team_id INTEGER NOT NULL,
            home_score   INTEGER NOT NULL,
            away_score   INTEGER NOT NULL,
            winner_id    INTEGER NOT NULL
        )
    """,
}

# FIXME: Understand...
extraIndexes = [
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_player_game ON Player_game_logs (player_id, game_id)",
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
            conn.sursor().executemany(sql, data)

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
                (status_log_id, player_id, team_id, game_id, scrape_date, status, return_date, comment)
            VALUES
                (:status_log_id, :player_id, :team_id, :game_id, :scrape_date, :status, :return_date, :comment)
        """

        with self._connect() as conn:
            conn.cursor().executemany(sql, data)

        print(f"Upserted {len(data)} status records")

    # RESULTS

    def upsertResults(self, data):
        sql = """
            INSERT OR REPLACE INTO Results
                (game_id, home_team_id, away_team_id, home_score, away_score, winner_id)
            VALUES
                (:game_id, :home_team_id, :away_team_id, :home_score, :away_score, :winner_id)
        """

        with self._connect() as conn:
            conn.cursor().executemany(sql, data)

        print(f"Upserted {len(data)} results records")

