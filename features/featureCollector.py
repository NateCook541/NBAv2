import sqlite3
import pandas as pd
from datetime import datetime, timedelta

#FIXME: add docstrings later

# Column order the model was trained on
featureOrder = [
    "avgPts10", "avgMin10", "avgFG10", "avgPPM10",
    "formPts5", "formMin5", "minStd10", "ptsStd10",
    "ptsMax10", "ptsMin10", "ptsMedian10", "over15_rate",
    "over20_rate", "over25_rate",
    "missing_ppg_injury", "starters_out_count", "injury_opportunity",
    "player_status_flag", "player_is_questionable",
    "opp_def_rtg", "opp_pace",
    "is_home", "rest_days",
    "pos", "usage_rate",
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

# Gets a players stats for the last 20 games but for each player adds it to a cache and then can be fetch mutiple times without having to rehit db a ton
def _rollingStatsCache(playerID, date, cache):
    if playerID not in cache:
        return None

    df = cache[playerID]

    # 20 games before this date
    past = df[df["game_date"] < date].tail(20)
    if past.empty:
        return None

    return past.drop(columns=["game_date"]).reset_index(drop=True)

def _injuryContext(statusDF, playerAvgCache, teamID, date):
    
    # Reports are filled out the day before the game is actually played so we check both day before as well as game date
    dayBefore = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")

    # Get how many starters are out
    injured = statusDF[
        (statusDF.team_id == teamID) &
        (statusDF.scrape_date.isin([date, dayBefore])) &
        (statusDF.status.isin(["Out", "Doubtful"]))
    ]
    startersOut = len(injured)

    # Get the missing points from injuryed players by getting avg points for all and summing
    missingPPG = sum(
            playerAvgCache.get(pid, 0)
            for pid in injured.player_id.values
    )

    return {"missing_ppg": missingPPG, "starters_out": startersOut}

def _oppContext(teamCache, oppTeamID, date):
    df = teamCache.get(oppTeamID)

    if df is None or df.empty:
        return {"def_rtg": 0.0, "pace": 0.0}

    past = df[df["date"] < date]
    if past.empty:
        return {"def_rtg": 0.0, "pace": 0.0}

    latest = past.iloc[-1]
    return {"def_rtg": latest.def_rtg, "pace": latest.pace}

# Used for player status feature which gathers the specfic players injury context (Important for players going into games listed as doutful)
def _playerStatusContext(statusDF, playerID, date): 
    # Check for the day before not day of
    dayBefore = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    df = statusDF[
            (statusDF.player_id == playerID) &
            (statusDF.scrape_date == dayBefore)
    ]

    if df.empty:
        return {"player_status_flag": 0, "player_is_questionable": 0}

    status = df["status"].iloc[0]
    return {
        # 1 if any flag at all (Out/Doubtful/Questionable) 0 if available or no record
        "player_status_flag":    0 if status in ("Available", None, "") else 1,
        # A flag specifically for questionable as these players play but for few mins or not as hard, in both cases less points are typically scored
        "player_is_questionable": 1 if status == "Questionable" else 0,
    }

# Gets the usage rate for a player to detemine their importance
def _getUsageRate(playerID, date, cache, teamGameTotals):
    if playerID not in cache:
        return 0.0
    
    playerGames = cache[playerID]
    recent = playerGames[playerGames["game_date"] < date].tail(10)
    
    if recent.empty:
        return 0.0
    
    shares = []
    for _, row in recent.iterrows():
        teamTotal = teamGameTotals.get((row["game_date"], row["is_home"]), 0)
        if teamTotal > 0:
            shares.append(row["points"] / teamTotal)
    
    return float(sum(shares) / len(shares)) if shares else 0.0 

# Builds the feature vector for training
def buildFeatures(playerID, date, teamID, oppTeamID, 
                  cache, posCache, teamCache, statusDF, playerAvgCache,
                  usageRateCache):
    # Gets player rolling stats depending on if cache is used or not
    rolling = _rollingStatsCache(playerID, date, cache)
    if rolling is None or rolling.empty:
        return None
    
    baseline = rolling.head(10).mean()
    ewma = rolling.head(10).ewm(span=5).mean().iloc[-1]

    # Last 10 games for features below (can add more exg last 5)
    last10 = rolling.head(10)

    # Volatility
    ptsStd10 = float(last10["points"].std() or 0.0)
    minStd10 = float(rolling.head(10)["minutes"].std() or 0.0)

    # Ceiling and floor
    ptsMax10 = float(last10["points"].max())
    ptsMin10 = float(last10["points"].min())

    # Consistency
    ptsMedian10 = float(last10["points"].median())

    # % of games over thresholds
    over15_rate = float((last10["points"] > 15).mean())
    over20_rate = float((last10["points"] > 20).mean())
    over25_rate = float((last10["points"] > 25).mean())

    # Get injury (status) and oppnenet features
    injuryFeatures = _injuryContext(statusDF, playerAvgCache, teamID, date)
    oppFeatures = _oppContext(teamCache, oppTeamID, date)
    playerStatus = _playerStatusContext(statusDF, playerID, date)

    # Read the home and rest days stats
    isHome = int(rolling.iloc[0]["is_home"]) if "is_home" in rolling.columns else 0
    restDays = int(rolling.iloc[0]["rest_days"]) if "rest_days" in rolling.columns else 0

    avgMin = baseline["minutes"] if baseline["minutes"] > 0 else 1

    # Get player position info 
    posMap = {"PG": 1, "SG": 2, "SF": 3, "PF": 4, "C": 5, "G": 1.5, "F": 3.5, "G-F": 2, "F-C": 4.5} 
    #FIXME: Currently just defualts to SF mabye change in future, mabye add a unlisted encodin
    pos = posMap.get(posCache.get(playerID), 3)

    # Get a players usage rate
    usageRate = _getUsageRate(playerID, date, cache, usageRateCache)
    
    # Full feature vertex
    features = pd.DataFrame([{
        "avgPts10":              baseline["points"],
        "avgMin10":              baseline["minutes"],
        "avgFG10":               baseline["fg_pct"],
        "avgPPM10":              baseline["points"] / avgMin,
        "formPts5":              ewma["points"],
        "formMin5":              ewma["minutes"],
        "minStd10":              minStd10,
        "ptsStd10":              ptsStd10,
        "ptsMax10":              ptsMax10,
        "ptsMin10":              ptsMin10,
        "ptsMedian10":           ptsMedian10,
        "over15_rate":           over15_rate,
        "over20_rate":           over20_rate,
        "over25_rate":           over25_rate,
        "missing_ppg_injury":    injuryFeatures["missing_ppg"],
        "starters_out_count":    injuryFeatures["starters_out"],
        "injury_opportunity":    injuryFeatures["missing_ppg"] * (baseline["points"] / avgMin),
        "player_status_flag":    playerStatus["player_status_flag"],
        "player_is_questionable":playerStatus["player_is_questionable"],
        "opp_def_rtg":           oppFeatures["def_rtg"],
        "opp_pace":              oppFeatures["pace"],
        "is_home":               isHome,
        "rest_days":             restDays,
        "pos":                   pos,
        "usage_rate":            usageRate,
    }])

    # Make sure the model recicves order in which it was trained on
    return features[featureOrder]

