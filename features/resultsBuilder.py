import sqlite3
import pandas as pd
from datetime import datetime, timedelta

featureOrder = [
    # Team scoring / pace core
    "team_pts_avg10", "opp_pts_avg10",
    "team_pts_allowed10", "opp_pts_allowed10",
    "team_pace10", "opp_pace10", "combined_pace_avg",

    # Efficiency
    "team_off_rtg10", "team_def_rtg10", "opp_off_rtg10", "opp_def_rtg10",
    "naive_total_projection", # (team_off_rtg + opp_def_rtg)/2 + (opp_off_rtg + team_def_rtg)/2 scaled by combined pace
    "off_def_mismatch_team", "off_def_mismatch_opp", # rating diff for both offense and defense

    # Form / volatility
    "formTotal5",
    "totalStd10", "paceStd10",
    "total_cv",
    "pts_trend_team", "pts_trend_opp",

    # Injury context
    "team_off_rtg_injury_impact", "opp_off_rtg_injury_impact", # Estimated ppg 100 lost from missing rotation players
    "team_def_rtg_injury_impact", "opp_def_rtg_injury_impact",
    "combined_injury_pace_effect",

    # Game flow / situational
    "h2h_avg_total", "h2h_n", "days_since_last_meeting",
    "blowout_risk", # Team quality gap

    # Rest / schedule
    "team_rest_days", "opp_rest_days", "rest_diff",
    "team_b2b", "opp_b2b", "both_b2b",
    "team_games_last7", "opp_games_last7"
]

# The subset the model actually trains on. The full featureOrder above is still built
# (it carries naive_total_projection, which the residual target needs, plus useful
# diagnostics), but the remaining ~26 columns showed |corr| < 0.05 with the game total
# and only added overfit — see the pruning experiment. Keep in sync with what carries signal.
RESULTS_FEATURES = [
    "naive_total_projection",
    "combined_pace_avg",
    "team_pace10", "opp_pace10",
    "team_pts_allowed10", "opp_pts_allowed10",
    "formTotal5",
    "team_pts_avg10", "opp_pts_avg10",
    "team_def_rtg10", "opp_def_rtg10",
    "days_since_last_meeting",
]


# PRIVATE HELPERS


# Rolling team game log
def _teamRollingCache(teamID, date, teamGameCache, window=10):
    if teamID not in teamGameCache:
        return None

    df = teamGameCache[teamID]
    past = df[df["date"] < date].tail(window).sort_values("date", ascending=False)
    if past.empty:
        return None

    return past.reset_index(drop=True)

def _teamBaseline(teamID, date, teamGameCache, window=10):
    rolling = _teamRollingCache(teamID, date, teamGameCache, window)
    if rolling is None or rolling.empty:
        return None
    return rolling

# EWMA form
def _teamForm(rolling, span=5, col="pts_scored"):
    if rolling is None or rolling.empty:
        return 0.0
    ewma = rolling.head(span * 2).ewm(span=span).mean(numeric_only=True)
    if col not in ewma.columns or ewma.empty:
        return 0.0
    return float(ewma.iloc[-1][col])

# Aggregate injury impact to a teams off/def
def _teamInjuryRatingImpact(statusDF, playerLogCache, teamGameTotals, teamID, date, side="offense"):
    dayBefore = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    injured = statusDF[
            (statusDF.team_id == teamID) &
            (statusDF.scrape_date == dayBefore) &
            (statusDF.status.isin(["Out", "Doubtful"]))
    ]

    if injured.empty:
        return 0.0

    impact = 0.0
    for pid in injured.player_id.values:
        if pid not in playerLogCache:
            continue
        pdf = playerLogCache[pid]
        # playerLogCache uses "game_date" (see cache.preloadCaches)
        past = pdf[pdf["game_date"] < date].tail(10)
        if past.empty:
            continue

        usageShares = []
        for _, row in past.iterrows():
            teamTotal = teamGameTotals.get((row["game_id"], row["team_id"]), 0)
            if teamTotal > 0:
                usageShares.append(row["points"] / teamTotal)
        usage = sum(usageShares) / len(usageShares) if usageShares else 0.0
        avgPts = float(past["points"].mean())
        # offense: lossing a scorer hurts team_off_rtg proportinal to usage*production
        # defense: lossing a two way/defensive player hurts team_def_rtg - approximate
        # with the same usage weighted signal since we don't have on/off def data here
        impact += avgPts * (1 + usage)
    
    return float(impact) if side == "offense" else float(impact) * 0.6

# Rolling pace volatility as a rough proxy for whether missing players change tempo
def _injuryPaceEffect(teamStartersOut, oppStartersOut):
    return float(teamStartersOut + oppStartersOut) * 0.15
 
def _h2hTotals(teamID, oppTeamID, date, h2hCache, window=6):
    key = tuple(sorted((teamID, oppTeamID)))
    df = h2hCache.get(key)
    if df is None or df.empty:
        return 0.0, 0, None
 
    past = df[df["date"] < date].tail(window)
    if past.empty:
        return 0.0, 0, None
 
    avgTotal = float(past["total_pts"].mean())
    n = len(past)
    lastDate = past.iloc[-1]["date"]
    return avgTotal, n, lastDate

def _daysSince(date, lastDate):
    if lastDate is None:
        return 999
    d1 = datetime.strptime(date, "%Y-%m-%d")
    d2 = datetime.strptime(lastDate, "%Y-%m-%d") if isinstance(lastDate, str) else lastDate
    return (d1 - d2).days 
 
def _blowoutRisk(teamOffRtg, teamDefRtg, oppOffRtg, oppDefRtg):
    # crude net-rating gap - bigger gap => more likely one team pulls away and
    # empties the bench, which suppresses total scoring late
    teamNet = teamOffRtg - teamDefRtg
    oppNet = oppOffRtg - oppDefRtg
    return abs(teamNet - oppNet)
 
def _restContext(teamGameCache, teamID, date):
    rolling = _teamRollingCache(teamID, date, teamGameCache, window=1)
    if rolling is None or rolling.empty:
        return 0, 0
    restDays = int(rolling.iloc[0]["rest_days"]) if "rest_days" in rolling.columns else 0
    b2b = 1 if restDays <= 1 else 0
    return restDays, b2b
 
def _gamesLast7(teamGameCache, teamID, date):
    if teamID not in teamGameCache:
        return 0
    df = teamGameCache[teamID]
    cutoff = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")
    return int(len(df[(df["date"] < date) & (df["date"] >= cutoff)]))
 
 
# Builds the feature vector for training a game totals model
def buildTotalsFeatures(gameID, date, teamID, oppTeamID,
                         teamGameCache, statusDF, playerLogCache,
                         teamGameTotals, h2hCache,
                         oddsDF=None):
 
    teamRolling = _teamBaseline(teamID, date, teamGameCache)
    oppRolling = _teamBaseline(oppTeamID, date, teamGameCache)
    if teamRolling is None or oppRolling is None:
        return None
 
    teamBaseline = teamRolling.mean(numeric_only=True)
    oppBaseline = oppRolling.mean(numeric_only=True)
 
    teamPtsAvg10 = float(teamBaseline["pts_scored"])
    oppPtsAvg10 = float(oppBaseline["pts_scored"])
    teamPtsAllowed10 = float(teamBaseline["pts_allowed"])
    oppPtsAllowed10 = float(oppBaseline["pts_allowed"])
    teamPace10 = float(teamBaseline["pace"])
    oppPace10 = float(oppBaseline["pace"])
    combinedPaceAvg = (teamPace10 + oppPace10) / 2
 
    teamOffRtg10 = float(teamBaseline["off_rtg"])
    teamDefRtg10 = float(teamBaseline["def_rtg"])
    oppOffRtg10 = float(oppBaseline["off_rtg"])
    oppDefRtg10 = float(oppBaseline["def_rtg"])
 
    # simple baseline total estimate, handed to the model as its own feature
    expectedTeamPts = ((teamOffRtg10 + oppDefRtg10) / 2) * (combinedPaceAvg / 100.0)
    expectedOppPts = ((oppOffRtg10 + teamDefRtg10) / 2) * (combinedPaceAvg / 100.0)
    naiveTotalProjection = expectedTeamPts + expectedOppPts
 
    offDefMismatchTeam = teamOffRtg10 - oppDefRtg10
    offDefMismatchOpp = oppOffRtg10 - teamDefRtg10
 
    # form / volatility
    formTotal5Team = _teamForm(teamRolling, span=5, col="pts_scored")
    formTotal5Opp = _teamForm(oppRolling, span=5, col="pts_scored")
    formTotal5 = formTotal5Team + formTotal5Opp
 
    combinedTotalsSeries = (
        teamRolling.head(10)["pts_scored"].reset_index(drop=True)
        + oppRolling.head(10)["pts_scored"].reset_index(drop=True)
    ) if len(teamRolling) >= 10 and len(oppRolling) >= 10 else pd.Series(dtype=float)
 
    totalStd10 = float(combinedTotalsSeries.std() or 0.0) if not combinedTotalsSeries.empty else 0.0
    paceStd10 = float(pd.concat([teamRolling.head(10)["pace"], oppRolling.head(10)["pace"]]).std() or 0.0)
    combinedAvgTotal = teamPtsAvg10 + oppPtsAvg10
    totalCV = totalStd10 / max(combinedAvgTotal, 1.0)
 
    if "total_line" in teamRolling.columns and not combinedTotalsSeries.empty:
        overRate10 = float((combinedTotalsSeries > teamRolling.head(10)["total_line"].reset_index(drop=True)).mean())
    else:
        overRate10 = 0.0
 
    ptsTrendTeam = float(teamRolling.head(5)["pts_scored"].mean()) - teamPtsAvg10
    ptsTrendOpp = float(oppRolling.head(5)["pts_scored"].mean()) - oppPtsAvg10
 
    # injury impact, both teams, both sides of the ball
    teamOffInjury = _teamInjuryRatingImpact(statusDF, playerLogCache, teamGameTotals, teamID, date, "offense")
    oppOffInjury = _teamInjuryRatingImpact(statusDF, playerLogCache, teamGameTotals, oppTeamID, date, "offense")
    teamDefInjury = _teamInjuryRatingImpact(statusDF, playerLogCache, teamGameTotals, teamID, date, "defense")
    oppDefInjury = _teamInjuryRatingImpact(statusDF, playerLogCache, teamGameTotals, oppTeamID, date, "defense")
 
    dayBefore = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    teamStartersOut = len(statusDF[
        (statusDF.team_id == teamID) & (statusDF.scrape_date == dayBefore) &
        (statusDF.status.isin(["Out", "Doubtful"]))
    ])
    oppStartersOut = len(statusDF[
        (statusDF.team_id == oppTeamID) & (statusDF.scrape_date == dayBefore) &
        (statusDF.status.isin(["Out", "Doubtful"]))
    ])
    combinedInjuryPaceEffect = _injuryPaceEffect(teamStartersOut, oppStartersOut)
 
    # matchup history
    h2hAvgTotal, h2hN, h2hLastDate = _h2hTotals(teamID, oppTeamID, date, h2hCache)
    daysSinceLastMeeting = _daysSince(date, h2hLastDate)
 
    blowoutRisk = _blowoutRisk(teamOffRtg10, teamDefRtg10, oppOffRtg10, oppDefRtg10)
 
    # rest / schedule
    teamRestDays, teamB2B = _restContext(teamGameCache, teamID, date)
    oppRestDays, oppB2B = _restContext(teamGameCache, oppTeamID, date)
    restDiff = teamRestDays - oppRestDays
    bothB2B = 1 if (teamB2B and oppB2B) else 0
    teamGamesLast7 = _gamesLast7(teamGameCache, teamID, date)
    oppGamesLast7 = _gamesLast7(teamGameCache, oppTeamID, date)
 
    features = pd.DataFrame([{
        "team_pts_avg10":         teamPtsAvg10,
        "opp_pts_avg10":          oppPtsAvg10,
        "team_pts_allowed10":     teamPtsAllowed10,
        "opp_pts_allowed10":      oppPtsAllowed10,
        "team_pace10":            teamPace10,
        "opp_pace10":             oppPace10,
        "combined_pace_avg":      combinedPaceAvg,
        "team_off_rtg10":         teamOffRtg10,
        "team_def_rtg10":         teamDefRtg10,
        "opp_off_rtg10":          oppOffRtg10,
        "opp_def_rtg10":          oppDefRtg10,
        "naive_total_projection": naiveTotalProjection,
        "off_def_mismatch_team":  offDefMismatchTeam,
        "off_def_mismatch_opp":   offDefMismatchOpp,
        "formTotal5":             formTotal5,
        "totalStd10":             totalStd10,
        "paceStd10":              paceStd10,
        "total_cv":               totalCV,
        "pts_trend_team":         ptsTrendTeam,
        "pts_trend_opp":          ptsTrendOpp,
        "team_off_rtg_injury_impact": teamOffInjury,
        "opp_off_rtg_injury_impact":  oppOffInjury,
        "team_def_rtg_injury_impact": teamDefInjury,
        "opp_def_rtg_injury_impact":  oppDefInjury,
        "combined_injury_pace_effect": combinedInjuryPaceEffect,
        "h2h_avg_total":          h2hAvgTotal,
        "h2h_n":                  float(h2hN),
        "days_since_last_meeting": float(daysSinceLastMeeting),
        "blowout_risk":           blowoutRisk,
        "team_rest_days":         teamRestDays,
        "opp_rest_days":          oppRestDays,
        "rest_diff":              restDiff,
        "team_b2b":               teamB2B,
        "opp_b2b":                oppB2B,
        "both_b2b":               bothB2B,
        "team_games_last7":       teamGamesLast7,
        "opp_games_last7":        oppGamesLast7,
    }])
 
    # Make sure the model receives columns in the order it was trained on
    return features[featureOrder]

