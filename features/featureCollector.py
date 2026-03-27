import sqlite3
import pandas as pd
from datetime import datetime, timedelta

#FIXME: add docstrings later

# Column order the model was trained on
featureOrder = [
    "avgPts10", "avgMin10", "avgFG10", "avgPPM10",
    "formPts5", "formMin5", "minStd10",
    "missing_ppg_injury", "starters_out_count", "injury_opportunity",
    "player_status_flag", "player_is_questionable",
    "opp_def_rtg", "opp_pace",
    "is_home", "rest_days",
]

# PRIVATE HELPERS

# Get a players stats for the last 20 games
def _rollingStats(playerID, date, conn):
    # Query the db for players rolling stats and create a pandas df to return of results
    query = """
        SELECT pgl.points, pgl.minutes, pgl.fg_pct, pgl.is_home, pgl.rest_days
        FROM Player_game_logs pgl
        JOIN Games g ON pgl.game_id = g.game_id
        WHERE pgl.player_id = ? AND g.game_date < ?
        ORDER BY g.game_date DESC
        LIMIT 20
    """
    df = pd.read_sql_query(query, conn, params=[playerID, date])
    
    return df if not df.empty else None


def _injuryContext(teamID, date, conn):
    
    # Reports are filled out the day before the game is actually played so we check both day before as well as game date
    dayBefore = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")

    # Identify starts by querying for players who started >= 50% of games in 100 game window
    startersQuery = """
        WITH RecentStarts AS (
            SELECT player_id, AVG(is_starter) AS start_rate
            FROM Player_game_logs
            WHERE game_id IN (
                SELECT game_id FROM Games
                WHERE game_date < ?
                ORDER BY game_date DESC
                LIMIT 100
            )
            GROUP BY player_id
            HAVING start_rate >= 0.5
        )
        SELECT COUNT(*) AS starters_out
        FROM Status
        WHERE team_id = ?
          AND scrape_date IN (?, ?)
          AND status IN ('Out', 'Doubtful')
          AND player_id IN (SELECT player_id FROM RecentStarts)
    """
    startersDF = pd.read_sql_query(startersQuery, conn, params=[date, teamID, dayBefore, date])

    # Get the missing points from injuryed players by getting avg points for all and summing
    missingPPGQuery = """
        SELECT SUM(avg_pts) AS total_missing
        FROM (
            SELECT player_id, AVG(points) AS avg_pts
            FROM Player_game_logs
            GROUP BY player_id
        ) p_avg
        WHERE player_id IN (
            SELECT player_id FROM Status
            WHERE team_id = ?
              AND scrape_date IN (?, ?)
              AND status IN ('Out', 'Doubtful')
        )
    """
    missingDF = pd.read_sql_query(missingPPGQuery, conn, params=[teamID, dayBefore, date])

    startersOut = int(startersDF["starters_out"].iloc[0]) if not startersDF.empty else 0
    missingPPG = float(missingDF["total_missing"].iloc[0] or 0.0) if not missingDF.empty else 0
    dataExists = 1 if startersOut > 0 or missingPPG > 0 else 0

    return {"missing_ppg": missingPPG, "starters_out": startersOut, "data_exists": dataExists}


def _oppContext(oppTeamID, date, conn):
    query = """
        SELECT def_rtg, pace
        FROM Teams
        WHERE team_id = ? AND date < ?
        ORDER BY date DESC
        LIMIT 1
    """

    df = pd.read_sql_query(query, conn, params=[oppTeamID, date])
    if df.empty:
        return {"def_rtg": 0.0, "pace": 0.0}
    return {"def_rtg": float(df["def_rtg"].iloc[0] or 0.0),
            "pace": float(df["pace"].iloc[0] or 0.0)}

# Used for player status feature which gathers the specfic players injury context (Important for players going into games listed as doutful)
def _playerStatusContext(playerID, date, conn):
    query = """
        SELECT status FROM Status
        WHERE player_id = ?
          AND scrape_date = ?
        LIMIT 1
    """
    # Check for the day before not day of
    dayBefore = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    df = pd.read_sql_query(query, conn, params=[playerID, dayBefore])

    if df.empty:
        return {"player_status_flag": 0, "player_is_questionable": 0}

    status = df["status"].iloc[0]
    return {
        # 1 if any flag at all (Out/Doubtful/Questionable) 0 if available or no record
        "player_status_flag":    0 if status in ("Available", None, "") else 1,
        # A flag specifically for questionable as these players play but for few mins or not as hard, in both cases less points are typically scored
        "player_is_questionable": 1 if status == "Questionable" else 0,
    }

# Builds the feature vector for training
def buildFeatures(playerID, date, teamID, oppTeamID, conn):
    rolling = _rollingStats(playerID, date, conn)
    if rolling is None or rolling.empty or rolling['points'].isna().all():
        return None

    baseline = rolling.head(10).mean()
    ewma = rolling.head(10).ewm(span=5).mean().iloc[-1]

    # Get injury (status) and oppnenet features
    injuryFeatures = _injuryContext(teamID, date, conn)
    oppFeatures = _oppContext(oppTeamID, date, conn)
    playerStatus = _playerStatusContext(playerID, date, conn)

    # Read the home and rest days stats
    isHome = int(rolling.iloc[0]["is_home"]) if "is_home" in rolling.columns else 0
    restDays = int(rolling.iloc[0]["rest_days"]) if "rest_days" in rolling.columns else 0

    avgMin = baseline["minutes"] if baseline["minutes"] > 0 else 1

    features = pd.DataFrame([{
        "avgPts10":              baseline["points"],
        "avgMin10":              baseline["minutes"],
        "avgFG10":               baseline["fg_pct"],
        "avgPPM10":              baseline["points"] / avgMin,
        "formPts5":              ewma["points"],
        "formMin5":              ewma["minutes"],
        "minStd10":              float(rolling.head(10)["minutes"].std() or 0.0),
        "missing_ppg_injury":    injuryFeatures["missing_ppg"],
        "starters_out_count":    injuryFeatures["starters_out"],
        "injury_opportunity":    injuryFeatures["missing_ppg"] * (baseline["points"] / avgMin),
        "player_status_flag":    playerStatus["player_status_flag"],
        "player_is_questionable":playerStatus["player_is_questionable"],
        "opp_def_rtg":           oppFeatures["def_rtg"],
        "opp_pace":              oppFeatures["pace"],
        "is_home":               isHome,
        "rest_days":             restDays,
    }])

    # Make sure the model revices order in which it was trained on
    return features[featureOrder]

