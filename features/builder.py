import sqlite3
import pandas as pd
from datetime import datetime, timedelta

#FIXME: add docstrings later

# Column order the model was trained on
featureOrder = [
    # Avg stats for last 10 games
    "avgPts10", "avgMin10", "avgFG10", "avgPPM10",

    # Pts and mins recent games stats
    "last1Pts", "last3Pts", "last1Mins", "last3Mins",

    # Home and away pts stats
    "home_pts_avg", "away_pts_avg", "home_away_diff",
        
    # ewm stats and standard dev stats
    "formPts5", "formMin5", "minStd10", "ptsStd10", "over20_rate",

    # Simple trend stats
    "pts_trend", "min_trend", "usage_rate",

    # Injury stats
    "missing_ppg_injury", "starters_out_count", "injury_opportunity", "player_status_flag", "player_is_questionable",
    "missing_usage",

    # Opp stats
    "opp_def_rtg", "opp_pace", "opp_pts_allowed_to_pos",
    
    # Location and rest stats
    "is_home", "rest_days", "back_to_back", 

    # FIXME: Mabye add this later
    #"games_last_7_days",
    
    # Pos stats
    "pos", "pos_injury_opportunity",

    # Minutes prediction
    "mins_prediction",
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

    # 20 most recent games before this date, newest first.
    past = df[df["game_date"] < date].tail(20).sort_values("game_date", ascending=False)
    if past.empty:
        return None
    
    return past.drop(columns=["game_date"]).reset_index(drop=True)

def _playerAverageToDate(playerID, date, cache, window=20):
    if playerID not in cache:
        return 0.0

    past = cache[playerID][cache[playerID]["game_date"] < date].tail(window)
    if past.empty:
        return 0.0

    return float(past["points"].mean())

def _injuryContext(statusDF, playerLogCache, teamGameTotals, teamID, date):
    # Use only the most recent pregame report date to avoid same-day leakage.
    dayBefore = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")

    injured = statusDF[
        (statusDF.team_id == teamID) &
        (statusDF.scrape_date == dayBefore) &
        (statusDF.status.isin(["Out", "Doubtful"]))
    ]
    startersOut = len(injured)

    missingPPG = sum(
        _playerAverageToDate(pid, date, playerLogCache)
        for pid in injured.player_id.values
    )

    missingUsage = sum(
        _getUsageRate(pid, date, playerLogCache, teamGameTotals)
        for pid in injured.player_id.values
    )

    return {
            "missing_ppg": missingPPG, 
            "starters_out": startersOut,
            "missing_usage": missingUsage
    }

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
        teamTotal = teamGameTotals.get((row["game_id"], row["team_id"]), 0)
        if teamTotal > 0:
            shares.append(row["points"] / teamTotal)
    
    return float(sum(shares) / len(shares)) if shares else 0.0 

def _oppVsPosContext(oppPosCache, oppTeamID, playerPos, date):
    key = (oppTeamID, playerPos)
    df = oppPosCache.get(key)

    if df is None or df.empty:
        return 0.0

    past = df[df["game_date"] < date].tail(10)
    if past.empty:
        return 0.0

    return float(past["points"].mean())

# Gets how injurys affect specfic pos game stats
def _injuryOpportunityByPos(statusDF, oppPosCache, playerPos, teamID, oppTeamID, date):
    dayBefore = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")

    # Get injured players on opponent team
    injured = statusDF[
        (statusDF.team_id == oppTeamID) &
        (statusDF.scrape_date == dayBefore) &
        (statusDF.status.isin(["Out", "Doubtful"]))
    ]

    if injured.empty:
        return 0.0

    # Base pts the opps allow for this pos
    key = (oppTeamID, playerPos)
    df = oppPosCache.get(key)
    if df is None or df.empty:
        return 0.0
    
    past = df[df["game_date"] < date].tail(10)   
    if past.empty:
        return 0.0

    defendersMissing = len(injured)
    basePtsAllowed = float(past["points"].mean())

    return basePtsAllowed * (1 + 0.05 * defendersMissing)


# Builds the feature vector for training
def buildFeatures(playerID, date, teamID, oppTeamID, 
                  cache, posCache, teamCache, statusDF, 
                  oppPosCache, teamGameTotals, minutesModel=None,
                  currentIsHome=None, currentRestDays=None
):
    # Gets player rolling stats
    rolling = _rollingStatsCache(playerID, date, cache)
    if rolling is None or rolling.empty:
        return None
    
    baseline = rolling.head(10).mean(numeric_only=True)
    ewma = rolling.head(10).ewm(span=5).mean(numeric_only=True).iloc[-1]

    # Last 10 games for features below (can add more exg last 5)
    last10 = rolling.head(10)

    # Volatility
    ptsStd10 = float(last10["points"].std() or 0.0)
    minStd10 = float(rolling.head(10)["minutes"].std() or 0.0)

    # Home away split
    homePts = float(rolling[rolling["is_home"] == 1]["points"].mean()) if len(rolling[rolling["is_home"] == 1]) > 0 else baseline["points"]
    
    awayPts = float(rolling[rolling["is_home"] == 0]["points"].mean()) if len(rolling[rolling["is_home"] == 0]) > 0 else baseline["points"]

    homeAwayDiff = homePts - awayPts

    # Trend
    last5avg = float(rolling.head(5)["points"].mean())
    last10avg = float(baseline["points"])
    ptsTrend = last5avg - last10avg

    # Last 1 and 3 games pts/mins for more recent hard nums
    last1Pts = float(rolling.iloc[0]["points"]) if len(rolling) >= 1 else last10avg
    last3Pts = float(rolling.head(3)["points"].mean()) if len(rolling) >= 3 else last10avg
    last1Mins = float(rolling.iloc[0]["minutes"]) if len(rolling) >= 1 else float(baseline["minutes"])
    last3Mins = float(rolling.head(3)["minutes"].mean()) if len(rolling) >= 3 else float(baseline["minutes"])

    # % of games over thresholds
    over20_rate = float((last10["points"] > 20).mean())

    # Get injury (status) and oppnenet features
    injuryFeatures = _injuryContext(statusDF, cache, teamGameTotals, teamID, date)
    oppFeatures = _oppContext(teamCache, oppTeamID, date)
    playerStatus = _playerStatusContext(statusDF, playerID, date)

    # Read the actual target-game context when available. Fallback to latest prior row.
    if currentIsHome is None:
        isHome = int(rolling.iloc[0]["is_home"]) if "is_home" in rolling.columns else 0
    else:
        isHome = int(currentIsHome)

    if currentRestDays is None:
        restDays = int(rolling.iloc[0]["rest_days"]) if "rest_days" in rolling.columns else 0
    else:
        restDays = int(currentRestDays)

    avgMin = baseline["minutes"] if baseline["minutes"] > 0 else 1

    isB2B = 1 if restDays == 1 else 0

    minLast5 = float(rolling.head(5)["minutes"].mean())
    minTrend = minLast5 - float(baseline["minutes"])

    # Get player position info 
    posMap = {"PG": 1, "SG": 2, "SF": 3, "PF": 4, "C": 5, "G": 1.5, "F": 3.5, "G-F": 2, "F-C": 4.5} 
    #FIXME: Currently just defualts to SF mabye change in future, mabye add a unlisted encodin
    pos = posMap.get(posCache.get(playerID), 3)

    # Get a players usage rate
    usageRate = _getUsageRate(playerID, date, cache, teamGameTotals)
    
    # Def to a specfic pos
    posStr = posCache.get(playerID, "SF")
    oppPtsAllowedToPos = _oppVsPosContext(oppPosCache, oppTeamID, posStr, date)    

    # Injury opp by pos
    posInjuryOpp = _injuryOpportunityByPos(statusDF, oppPosCache, posStr, teamID, oppTeamID, date)

    # Minutes prediction
    minFeatures = pd.DataFrame([{
        "avgMin10": float(baseline["minutes"]),
        "minStd10": minStd10,
        "last3Mins": last3Mins,
        "last1Mins": last1Mins,
        "minTrend": minTrend,
        "isQuestionable": playerStatus["player_is_questionable"],
        "pos": pos,
    }])

    if minutesModel is not None:
        predictedMins = float(minutesModel.predict(minFeatures)[0])
    else:
        predictedMins = float(baseline["minutes"])

    # Full feature vertex
    features = pd.DataFrame([{
        "avgPts10":              baseline["points"],
        "avgMin10":              baseline["minutes"],
        "avgFG10":               baseline["fg_pct"],
        "avgPPM10":              baseline["points"] / avgMin,
        "last1Pts":              last1Pts,
        "last3Pts":              last3Pts,
        "last1Mins":             last1Mins,
        "last3Mins":             last3Mins,
        "home_pts_avg":          homePts,
        "away_pts_avg":          awayPts,
        "home_away_diff":        homeAwayDiff, 
        "formPts5":              ewma["points"],
        "formMin5":              ewma["minutes"],
        "minStd10":              minStd10,
        "ptsStd10":              ptsStd10,
        "over20_rate":           over20_rate,
        "pts_trend":             ptsTrend,
        "min_trend":             minTrend,
        "usage_rate":            usageRate,
        "missing_ppg_injury":    injuryFeatures["missing_ppg"],
        "starters_out_count":    injuryFeatures["starters_out"],
        "injury_opportunity":    injuryFeatures["missing_ppg"] * (baseline["points"] / avgMin),
        "player_status_flag":    playerStatus["player_status_flag"],
        "player_is_questionable":playerStatus["player_is_questionable"],
        "missing_usage":         injuryFeatures["missing_usage"],
        "opp_def_rtg":           oppFeatures["def_rtg"],
        "opp_pace":              oppFeatures["pace"],
        "opp_pts_allowed_to_pos":oppPtsAllowedToPos,
        "is_home":               isHome,
        "rest_days":             restDays,
        "back_to_back":          isB2B,
        "pos":                   pos,
        "pos_injury_opportunity":posInjuryOpp,
        "mins_prediction":       predictedMins,
    }])

    # Make sure the model recicves order in which it was trained on
    return features[featureOrder]
