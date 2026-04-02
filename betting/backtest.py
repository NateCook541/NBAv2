import sqlite3
import joblib
import numpy as np
import pandas as pd
import unicodeddata
from pathlib import Path
from scipy.stats import t as t_dist

from features.featureCollector import buildFeatures
# FIXME: When I move calibration stuff over I need to remember to fix this import
from models.evaluate import calibratedProbOver

# FIXME: Look into moving all this into a class?

# HELPERS

def _normalizeName(name):
    return "".join(
        c for c in unicodedata.normalize("NFD", name)
        if unicodedata.category(c) != "Mn"
    ).lower().strip()

# Converts the odds a number like this -100 to a percent like 52.6
def _impliedProb(americanOdds):
    if americanOdds < 0:
        return abs(americanOdds) / (abs(americanOdds) + 100)
    return 100 / (americanOdds + 100)

# Removes the built in vig from the sportsbook
def _removeVig(overOdds, underOdds):
    overProb = _impliedProb(overOdds)
    underProb = _impliedProb(underOdds)

    total = overProb + underProb
    return overProb / total, underProb / total

# Returns the net profit per 1$ staked
def _payoutMultiplier(americanOdds):
    if americanOdds > 0:
        return americanOdds / 100
    return 100 / abs(americanOdds)


# We are using a quarter kelly stake for the bankroll as reccommended
# Kelly stake determines the optimal bet size based on the found edge and the current odds
def _kellyFractional(edge, americanOdds, fraction=0.25):
    b = _payoutMultiplier(americanOdds)
    q = 1 - (edge + _impliedProb(americanOdds))
    p = edge + _impliedProb(americanOdds)
    kelly = (b * p - q) / b
    
    return max(0.0, kelly * fraction)


# DATA LOADING


def _loadProps(conn, startDate=None, endDate=None):
    query = """
        SELECT p.prop_id, p.game_date, p.player_name, p.line,
               p.over_odds, p.under_odds
        FROM Props p
        WHERE p.over_odds IS NOT NULL
          AND p.under_odds IS NOT NULL
    """
    params = []
    if startDate:
        query += " AND p.game_date >= ?"
        params.append(startDate)
    if endDate:
        query += " AND p.game_date >= ?"
        params.append(endDate)


    query += " ORDER BY p.game_date"
    return pd.read_sql_query(query, conn, params=params)


# Loads the players actual points keyed by normalized_game and game_date
def _loadActuals(conn):
    df = pd.read_sql_query("""
         SELECT p.name, g.game_date, pgl.points
        FROM Player_game_logs pgl
        JOIN Games   g ON pgl.game_id   = g.game_id
        JOIN Players p ON pgl.player_id = p.player_id
    """, conn)
    df["name_norm"] = df["name"].apply(_normalizeName)
    return df.set_index(["name_norm", "game_date"])["points"].to_dict()


# Load player map (player name to player id and team id)
def _loadPlayerMap(conn):
    df = pd.read_sql_query("SELECT player_id, name, team_id FROM Players", conn)
    df["name_norm"] = df["name"].apply(_normalizeName)
    return df.set_index("name_norm")[["player_id", "team_id"]].to_dict("index")


# Create a map for team_id and game_date to opp_team_id
def _loadOppMap(conn):
    df = pd.read_sql_query("""
        SELECT game_date, home_team_id, away_team_id FROM Games
    """, conn)

    result = {}
    for _, in df.iterrows():
        result[(row.home_team_id, row.game_date)] = row.away_team_id
        result[(row.away_team_id, row.game_date)] = row.home_team_id
    return result


# MAIN BACKTEST

# This is the main function that will backtest off the data from the api stored in the db
# The dates let you set a timeframe but default to none currently due to size of data
# edge thr
def runBacktest(dbPath = "NBA.db", startDate=None, endDate=None, edgeThresh=0.03,
                bankroll=1000, kellyFrac=0.25, df=3)







    














