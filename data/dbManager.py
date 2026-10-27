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
    "Results": """
        CREATE TABLE IF NOT EXISTS Results (
            game_id TEXT PRIMARY KEY,
            home_team_id INTEGER,
            away_team_id INTEGER,
            home_score INTEGER,
            away_score INTEGER,
            winner_id INTEGER
        )
    """,
    "Odds_archive": """
        CREATE TABLE IF NOT EXISTS Odds_archive (
            game_date     TEXT    NOT NULL,
            home_team_id  INTEGER NOT NULL,
            away_team_id  INTEGER NOT NULL,
            season        INTEGER,
            home_score    INTEGER,
            away_score    INTEGER,
            total_open    REAL,
            total_close   REAL,
            actual_total  INTEGER,
            PRIMARY KEY (game_date, home_team_id, away_team_id)
        )
    """,
    "Team_game_ratings": """
        CREATE TABLE IF NOT EXISTS Team_game_ratings (
            game_id TEXT,
            team_id INTEGER,
            off_rtg REAL,
            def_rtg REAL,
            pace REAL,
            possessions REAL,
            PRIMARY KEY (game_id, team_id)
        )
    """,
    "KalshiMarkets": """
        CREATE TABLE IF NOT EXISTS KalshiMarkets (
            market_ticker     TEXT    PRIMARY KEY,
            event_ticker      TEXT    NOT NULL,
            series_ticker     TEXT    NOT NULL,
            game_date         TEXT    NOT NULL,
            est_tip_utc       TEXT,
            kalshi_player_id  TEXT,
            player_name       TEXT    NOT NULL,
            strike            REAL    NOT NULL,
            result            TEXT,
            actual_value      REAL,
            settlement_value  REAL,
            settled_ts        TEXT
        )
    """,
    "KalshiSnapshots": """
        CREATE TABLE IF NOT EXISTS KalshiSnapshots (
            market_ticker  TEXT    NOT NULL,
            snapshot_ts    TEXT    NOT NULL,
            snapshot_type  TEXT    NOT NULL,
            yes_bid        REAL,
            yes_ask        REAL,
            yes_bid_size   REAL,
            yes_ask_size   REAL,
            last_price     REAL,
            volume         REAL,
            open_interest  REAL,
            PRIMARY KEY (market_ticker, snapshot_ts)
        )
    """,
    "KalshiCandles": """
        CREATE TABLE IF NOT EXISTS KalshiCandles (
            market_ticker  TEXT    NOT NULL,
            end_ts         TEXT    NOT NULL,
            yes_bid        REAL,
            yes_ask        REAL,
            price_close    REAL,
            price_mean     REAL,
            volume         REAL,
            open_interest  REAL,
            PRIMARY KEY (market_ticker, end_ts)
        )
    """,
    # Every rung the paper trader scored, kept so the market shrink can be refit on live data
    "KalshiPaperRungs": """
        CREATE TABLE IF NOT EXISTS KalshiPaperRungs (
            market_ticker   TEXT    NOT NULL,
            decision_ts     TEXT    NOT NULL,
            game_date       TEXT    NOT NULL,
            player_id       INTEGER NOT NULL,
            player_name     TEXT,
            strike          REAL,
            predicted       REAL,
            p_model         REAL,
            shrink_weight   REAL,
            p_bet           REAL,
            yes_bid         REAL,
            yes_ask         REAL,
            model_train_end TEXT,
            PRIMARY KEY (market_ticker, decision_ts)
        )
    """,
    # Paper bets. One taker (buys at the ask now) and one maker (resting limit, filled or not
    # by tip) per player-game, so the two can be compared on the same decisions
    "KalshiPaperBets": """
        CREATE TABLE IF NOT EXISTS KalshiPaperBets (
            bet_id          INTEGER PRIMARY KEY AUTOINCREMENT,
            game_date       TEXT    NOT NULL,
            player_id       INTEGER NOT NULL,
            player_name     TEXT,
            market_ticker   TEXT    NOT NULL,
            strike          REAL,
            order_type      TEXT    NOT NULL,
            side            TEXT    NOT NULL,
            price           REAL    NOT NULL,
            fee             REAL,
            p_bet           REAL,
            ev              REAL,
            stake           REAL,
            contracts       REAL,
            yes_bid         REAL,
            yes_ask         REAL,
            decision_ts     TEXT    NOT NULL,
            est_tip_utc     TEXT,
            lead_hours      REAL,
            fill_status     TEXT,
            fill_ts         TEXT,
            close_bid       REAL,
            close_ask       REAL,
            clv             REAL,
            result          TEXT,
            payout          REAL,
            pnl             REAL,
            UNIQUE (game_date, player_id, order_type)
        )
    """,
}

extraIndexes = [
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_player_game ON Player_game_logs (player_id, game_id)",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_prop_unique ON Props (game_date, player_name, line, bookmaker)",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_player_name ON Players (name)",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_snapshot_unique ON PropSnapshots (game_date, player_name, line, bookmaker, snapshot_type)",
    "CREATE INDEX IF NOT EXISTS idx_kalshi_game_date ON KalshiMarkets (game_date)",
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

    # PROP SNAPSHOTS

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


    # KALSHI

    # Market info is mostly static, so on conflict only refresh the tip time (games get rescheduled)
    # Result fields are left alone here, they get filled by updateKalshiResults
    def upsertKalshiMarkets(self, data):
        sql = """
            INSERT INTO KalshiMarkets
                (market_ticker, event_ticker, series_ticker, game_date, est_tip_utc,
                 kalshi_player_id, player_name, strike)
            VALUES
                (:market_ticker, :event_ticker, :series_ticker, :game_date, :est_tip_utc,
                 :kalshi_player_id, :player_name, :strike)
            ON CONFLICT(market_ticker) DO UPDATE SET est_tip_utc = excluded.est_tip_utc
        """

        with self._connect() as conn:
            conn.cursor().executemany(sql, data)

        print(f"Upserted {len(data)} kalshi market records")

    def upsertKalshiSnapshots(self, data):
        sql = """
            INSERT OR IGNORE INTO KalshiSnapshots
                (market_ticker, snapshot_ts, snapshot_type, yes_bid, yes_ask,
                 yes_bid_size, yes_ask_size, last_price, volume, open_interest)
            VALUES
                (:market_ticker, :snapshot_ts, :snapshot_type, :yes_bid, :yes_ask,
                 :yes_bid_size, :yes_ask_size, :last_price, :volume, :open_interest)
        """

        with self._connect() as conn:
            conn.cursor().executemany(sql, data)

        print(f"Upserted {len(data)} kalshi snapshot records")

    def updateKalshiResults(self, data):
        sql = """
            UPDATE KalshiMarkets
            SET result = :result,
                actual_value = :actual_value,
                settlement_value = :settlement_value,
                settled_ts = :settled_ts
            WHERE market_ticker = :market_ticker
        """

        with self._connect() as conn:
            conn.cursor().executemany(sql, data)

        print(f"Settled {len(data)} kalshi market records")

    # Hourly price history from the archive backfill. yes_bid / yes_ask are the quote at candle close
    def upsertKalshiCandles(self, data):
        sql = """
            INSERT OR IGNORE INTO KalshiCandles
                (market_ticker, end_ts, yes_bid, yes_ask, price_close, price_mean, volume, open_interest)
            VALUES
                (:market_ticker, :end_ts, :yes_bid, :yes_ask, :price_close, :price_mean, :volume, :open_interest)
        """

        with self._connect() as conn:
            conn.cursor().executemany(sql, data)

        print(f"Upserted {len(data)} kalshi candle records")

    def getKalshiTickersWithCandles(self):
        with self._connect() as conn:
            rows = conn.execute("SELECT DISTINCT market_ticker FROM KalshiCandles").fetchall()
        return {r[0] for r in rows}

    def insertKalshiPaperRungs(self, data):
        sql = """
            INSERT OR IGNORE INTO KalshiPaperRungs
                (market_ticker, decision_ts, game_date, player_id, player_name, strike, predicted,
                 p_model, shrink_weight, p_bet, yes_bid, yes_ask, model_train_end)
            VALUES
                (:market_ticker, :decision_ts, :game_date, :player_id, :player_name, :strike, :predicted,
                 :p_model, :shrink_weight, :p_bet, :yes_bid, :yes_ask, :model_train_end)
        """

        with self._connect() as conn:
            conn.cursor().executemany(sql, data)

        print(f"Inserted {len(data)} kalshi paper rungs")

    # Ignore on conflict: the first decision for a player-game stands, reruns can't overwrite it
    def insertKalshiPaperBets(self, data):
        sql = """
            INSERT OR IGNORE INTO KalshiPaperBets
                (game_date, player_id, player_name, market_ticker, strike, order_type, side, price, fee,
                 p_bet, ev, stake, contracts, yes_bid, yes_ask, decision_ts, est_tip_utc, lead_hours, fill_status)
            VALUES
                (:game_date, :player_id, :player_name, :market_ticker, :strike, :order_type, :side, :price, :fee,
                 :p_bet, :ev, :stake, :contracts, :yes_bid, :yes_ask, :decision_ts, :est_tip_utc, :lead_hours,
                 :fill_status)
        """

        with self._connect() as conn:
            conn.cursor().executemany(sql, data)

        print(f"Inserted {len(data)} kalshi paper bets")

    def updateKalshiPaperBets(self, data):
        """data: dicts with bet_id plus any of the fill / close / settlement columns"""
        cols = ["fill_status", "fill_ts", "close_bid", "close_ask", "clv", "result", "payout", "pnl"]
        with self._connect() as conn:
            for row in data:
                sets = [c for c in cols if c in row]
                conn.execute(
                    f"UPDATE KalshiPaperBets SET {', '.join(f'{c} = :{c}' for c in sets)} WHERE bet_id = :bet_id",
                    row,
                )

        print(f"Updated {len(data)} kalshi paper bets")

    # (game_date, player_id) pairs the paper trader already decided on
    def getKalshiPaperDecided(self, gameDates):
        marks = ",".join("?" * len(gameDates))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT DISTINCT game_date, player_id FROM KalshiPaperRungs WHERE game_date IN ({marks})",
                list(gameDates),
            ).fetchall()
        return {(r[0], r[1]) for r in rows}

    # Event tickers that still have unsettled markets for games on or before a date
    def getUnsettledKalshiEvents(self, throughDate):
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT event_ticker FROM KalshiMarkets WHERE result IS NULL AND game_date <= ?",
                (throughDate,),
            ).fetchall()
        return [r[0] for r in rows]

    def upsertResults(self, data):
        sql = """
            INSERT OR REPLACE INTO Results (game_id, home_team_id, away_team_id, home_score, away_score, winner_id)
            VALUES (:game_id, :home_team_id, :away_team_id, :home_score, :away_score, :winner_id)
            """

        with self._connect() as conn:
            conn.cursor().executemany(sql, data)

        print(f"Upserted {len(data)} result records")


    def upsertOddsArchive(self, data):
        sql = """
            INSERT OR REPLACE INTO Odds_archive
                (game_date, home_team_id, away_team_id, season,
                 home_score, away_score, total_open, total_close, actual_total)
            VALUES
                (:game_date, :home_team_id, :away_team_id, :season,
                 :home_score, :away_score, :total_open, :total_close, :actual_total)
            """

        with self._connect() as conn:
            conn.cursor().executemany(sql, data)

        print(f"Upserted {len(data)} odds archive records")


    def upsertTeamGameRatings(self, data):
        sql = """
            INSERT OR REPLACE INTO Team_game_ratings
                (game_id, team_id, off_rtg, def_rtg, pace, possessions)
            VALUES (:game_id, :team_id, :off_rtg, :def_rtg, :pace, :possessions)
            """

        with self._connect() as conn:
            conn.cursor().executemany(sql, data)

        print(f"Upserted {len(data)} team game rating records")

