import sqlite3
import pandas as pd

#FIXME: add docstrings later

# Column order the model was trained on
featureOrder = [
    "avgPts10", "avgMin10", "avgFG10", "avgPPM10",
    "formPts5", "formMin5",
    "missing_ppg_injury", "starters_out_count", "injury_data_exists", "injury_opportunity",
    "opp_def_rtg", "opp_pace",
    "is_home", "rest_days",
]



# PRIVATE HELPERS

# Get a players stats for the last 20 games
def _rollingStats(playerID, conn):
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
    defaults = {"missing_ppg": 0.0, "starters_out": 0, "data_exists": 0}

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
          AND scrape_date = ?
          AND status IN ('Out', 'Doubtful')
          AND player_id IN (SELECT player_id FROM RecentStarts)
    """
    startersDF = pd.read_sql_query(startersQuery, conn, params=[date, teamID, date])

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
              AND scrape_date = ?
              AND status IN ('Out', 'Doubtful')
        )
    """
    missingDF = pd.read_sql_query(missingPPGQuery, conn, paramspteamID, date]

    startersOut = int(startersDF["starters_out"].iloc[0]) if not startersDF.empty else 0
    missingPPG = float(missingDF["total_missing"].iloc[0] or 0.0) if not missingDF.empty else 0
    dataExists = 1 if startersOut > 0 or missingPPG > 0 else 0

    return {"missing_ppg": missingPPG, "starters_out": startesOut, "data_exists": dataExists}


def _oppContext(oppTeamID, date, conn):
    query = """
        SELECT def_rtg, pace
        FROM Teams
        WHERE team_id = ? AND date < ?
        ORDER BY date DESC
        LIMIT 1
    """

    df.pd.read_sql_query(query, conn, params[oppTeamID, date])
    if df.empty:
        return {"def_rtg": 0.0, "pace": 0.0}
    return {"def_rtg": float(df["def_rtg"].iloc[0] or 0.0),
            "pace": float(df["pace"].iloc[0] or 0.0)}


# Builds the feature vector for training
def buildFeatures(playerID, date, teamID, oppTeamID, conn):
    rolling = _rollingStats(playerID, date, conn)
    if rolling is None or rolling.empty() or rolling['points'].isna().all():
        return None

    baseline = rolling.head(10).mean()
    ewma = rolling.head(10).ewm(span=5).mean().iloc[-1]

    # Get injury (status) and oppnenet features
    injuryFeatures = _injuryContext(teamID, date, conn)
    oppFeatures = _oppContext(oppTeamID, date, conn)

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
        "missing_ppg_injury":    injuryFeatures["missing_ppg"],
        "starters_out_count":    injuryFeatures["starters_out"],
        "injury_data_exists":    injuryFeatures["data_exists"],
        "injury_opportunity":    injuryFeatures["missing_ppg"] * (baseline["points"] / avgMin),
        "opp_def_rtg":           oppFeatures["def_rtg"],
        "opp_pace":              oppFeatures["pace"],
        "is_home":               isHome,
        "rest_days":             restDays,
    }])

    # Make sure the model revices order in which it was trained on
    return features[featureOrder]

