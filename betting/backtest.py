import sqlite3
import joblib
import numpy as np
import pandas as pd
import unicodedata
from pathlib import Path
from scipy.stats import t as t_dist

from features.featureCollector import buildFeatures
from models.train import preloadCaches, trainModel, trainMinutes
from betting.cailbrator import cailbratedProbOver

# FIXME: Look into moving all this into a class?

# HELPERS

def _printSummary(df, startingBank, finalBank, skipped):
    bets = df[df["bet"] == True]

    print(f"\n{'='*50}")
    print("BACKTEST SUMARY") 
    print(f"\n{'='*50}")
    print(f"Props evaluated: {len(df)}")
    print(f"Skipped: {skipped}")
    print(f"Bets placed: {len(bets)}")
    
    if len(bets) == 0:
        print("No bets placed")
        return

    wins = (bets["pnl"] > 0).sum()
    losses = (bets["pnl"] < 0).sum()
    winRate = wins / len(bets)
    totalPnl = bets["pnl"].sum()
    roi = totalPnl / bets["stake"].sum()

    print(f"Win/Loss: {wins}W / {losses}L ({winRate:.1%})")
    print(f"Total R&L: ${totalPnl:.2f}")
    print(f"ROI: {roi:.1%}")
    print(f"Starting bankroll: ${startingBank:.2f}")
    print(f"Final bankroll: ${finalBank:.2f}") 
    print(f"Return: {((finalBank - startingBank) / startingBank):.1%}")

    print(f"\nTop 5 bets by edge:")
    print(bets.sort_values("edge", ascending=False)[
        ["date", "player", "line", "predicted", "actual", "edge", "pnl"]
    ].head(5).to_string(index=False))
    
    print(f"\nWorst 5 bets by P&L:")
    print(bets.sort_values("pnl")[
        ["date", "player", "line", "predicted", "actual", "edge", "pnl"]
    ].head(5).to_string(index=False))

    bets = bets.copy()
    bets["pred_bucket"] = pd.cut(bets["predicted"], bins=[0,15,20,25,30,99], 
                              labels=["<15","15-20","20-25","25-30","30+"])

    print("\nWin rate by predicted score:")
    print(bets.groupby("pred_bucket", observed=True).agg(
        bets=("pnl","count"),
        win_rate=("pnl", lambda x: (x>0).mean()),
        avg_edge=("edge","mean"),
        total_pnl=("pnl","sum")
    ).to_string())

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
        query += " AND p.game_date <= ?"
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
    df = df.sort_values("player_id").drop_duplicates(subset="name_norm", keep="last")
    return df.set_index("name_norm")[["player_id", "team_id"]].to_dict("index")


# Create a opp map keyed by player_id and game_date (this handles players getting traded midseason
def _loadOppMap(conn):
    df = pd.read_sql_query("""
        SELECT pgl.player_id, g.game_date,
               g.home_team_id, g.away_team_id, pgl.is_home, pgl.rest_days
        FROM Player_game_logs pgl
        JOIN Games g ON pgl.game_id = g.game_id
    """, conn)

    result = {}
    for _, row in df.iterrows():
        if row.is_home:
            teamID = row.home_team_id
            oppTeamID = row.away_team_id
        else:
            teamID = row.away_team_id
            oppTeamID = row.home_team_id

        result[(int(row.player_id), row.game_date)] = {
            "team_id": teamID,
            "opp_team_id": oppTeamID,
            "is_home": int(row.is_home),
            "rest_days": int(row.rest_days) if pd.notna(row.rest_days) else 0,
        }

    return result


def _loadSavedModelMeta():
    metaPath = Path("models/nba_model_meta.joblib")
    if not metaPath.exists():
        return None
    return joblib.load(metaPath)


def _loadSavedMinutesMeta():
    metaPath = Path("models/nba_minutes_model_meta.joblib")
    if not metaPath.exists():
        return None
    return joblib.load(metaPath)


def _bundleIsBacktestSafe(modelMeta, minutesMeta, calBundle, backtestStartDate):
    if modelMeta is None or minutesMeta is None or calBundle is None:
        return False

    trainEndDate = modelMeta.get("train_end_date")
    minutesTrainEndDate = minutesMeta.get("train_end_date")
    calEndDate = calBundle.get("calibration_end_date")

    if not trainEndDate or not minutesTrainEndDate or not calEndDate:
        return False

    return (
        trainEndDate <= backtestStartDate and
        minutesTrainEndDate <= backtestStartDate and
        calEndDate < backtestStartDate
    )


# MAIN BACKTEST

# This is the main function that will backtest off the data from the api stored in the db
# The dates let you set a timeframe but default to none currently due to size of data
def runBacktest(dbPath = "NBA.db", startDate=None, endDate=None, edgeThresh=0.03,
                bankroll=1000, kellyFrac=0.25, tdf=3):
    
    # Load model + cailbrator
    modelPath = Path("models/nba_model.joblib")
    cailbratorPath = Path("models/nba_calibrator.joblib")

    conn = sqlite3.connect(dbPath)

    try:
        props = _loadProps(conn, startDate, endDate)
        if props.empty:
            raise ValueError("No props found for the requested backtest window")

        backtestStartDate = str(props["game_date"].min())
        actuals = _loadActuals(conn)
        playerMap = _loadPlayerMap(conn)
        oppMap = _loadOppMap(conn)

        playerLogCache, posCache, teamCache, statusDF, oppPosCache, teamGameTotals = preloadCaches(conn) 
    
    finally:
        conn.close()

    # Load all models and cailbrator
    modelMeta = _loadSavedModelMeta()
    minutesMeta = _loadSavedMinutesMeta()
    bundle = joblib.load(cailbratorPath) if cailbratorPath.exists() else None
    
    # If the current models are safe (trained before backtest range) then use them, and if not then retrain with correct date to prevent data leakage
    useSavedBundle = modelPath.exists() and _bundleIsBacktestSafe(modelMeta, minutesMeta, bundle, backtestStartDate)
    if useSavedBundle:
        model = joblib.load(modelPath)
        minutesModelPath = Path("models/nba_minutes_model.joblib")
        minutesModel = joblib.load(minutesModelPath) if minutesModelPath.exists() else None

    else:
        print(
            f"Saved model bundle is not leakage-safe for backtest starting {backtestStartDate}. "
            f"Training a fresh model using data before {backtestStartDate}."
        )
        model, bundle = trainModel (
            save=False,
            metrics=False,
            dbPath=dbPath,
            train_end_date=backtestStartDate,
        )
        minutesModel = trainMinutes (
            save=False,
            dbPath=dbPath,
            endDate=backtestStartDate,
        )

    sigma = bundle.get("sigma", bundle["residualStd"])
    df = bundle.get("df", 5)

    print(f"Loaded {len(props)} props | edge threshold: {edgeThresh:.0%} | bankroll: ${bankroll:.0f}")

    results = []
    currentBank = bankroll
    skipped = 0

    # Skipped vars to let us know what is skipping
    noPlayerMatch = 0
    noOppMatch = 0
    noActuals = 0
    noFeatures = 0
    noLine = 0

    for _, prop in props.iterrows():
        nameNorm = _normalizeName(prop.player_name)
        date = prop.game_date

        if nameNorm not in playerMap:
            noPlayerMatch += 1
            continue

        playerInfo = playerMap[nameNorm]
        playerID = playerInfo["player_id"]

        gameContext = oppMap.get((playerID, date))
    
        if gameContext is None:
            noOppMatch += 1
            continue
    
        teamID = gameContext["team_id"]
        oppTeamID = gameContext["opp_team_id"]

        # Get actual points from logs
        actualPts = actuals.get((nameNorm, date))
        if actualPts is None:
            noActuals += 1
            continue

        # Build features and predict
        features = buildFeatures(
            playerID=playerID,
            date=date,
            teamID=teamID,
            oppTeamID=oppTeamID,
            cache=playerLogCache,
            posCache=posCache,
            teamCache=teamCache,
            statusDF=statusDF,
            oppPosCache=oppPosCache,
            teamGameTotals=teamGameTotals,
            minutesModel=minutesModel,
            currentIsHome=gameContext["is_home"],
            currentRestDays=gameContext["rest_days"],
        )

        if features is None:
            noFeatures += 1
            continue

        # FIXME: Look into removing this later and see effects
        # Filter out lines that are 10+ pts from players last 10 avg and filter out lines that are less than 10
        avgPts = features["avgPts10"].iloc[0]
        if prop.line < 10 or abs(prop.line - avgPts) > 7:
            noLine += 1
            continue

        predicted = float(model.predict(features)[0])
       
        # Only bet where model has demonstrated signal
        if predicted >= 15:
            noLine += 1
            continue

        # Calibrated prob
        myProb = cailbratedProbOver(predicted, prop.line, sigma, bundle, df=df)

        # Fair prob (no vig)
        fairOverProb, _ = _removeVig(prop.over_odds, prop.under_odds)

        edge = myProb - fairOverProb
        
        # Only Bet if edge is greater than treshhold defined in parameters
        if edge <= edgeThresh:
            results.append({
                "date": date,
                "player": prop.player_name,
                "line": prop.line,
                "predicted": round(predicted, 1),
                "actual": actualPts,
                "my_prob": round(myProb, 3),
                "book_prob": round(fairOverProb, 3),
                "edge": round(edge, 3),
                "bet": False,
                "stake": 0.0,
                "pnl": 0.0,
                "bankroll": round(currentBank, 2),
            })
            continue

        # Size the bet with kelly formula
        #stake = _kellyFractional(edge, prop.over_odds, fraction=kellyFrac) * currentBank
        #stake = round(min(stake, currentBank * 0.10), 2) # Hard cap at 10% of current bankroll

        # Hardcode current stake for testing as its not good enough to run kelly as it will just add noise
        stake = 10

        won = actualPts > prop.line
        pnl = stake * _payoutMultiplier(prop.over_odds) if won else -stake
        currentBank += pnl

        results.append({
                "date": date,
                "player": prop.player_name,
                "line": prop.line,
                "predicted": round(predicted, 1),
                "actual": actualPts,
                "my_prob": round(myProb, 3),
                "book_prob": round(fairOverProb, 3),
                "edge": round(edge, 3),
                "bet": True,
                "stake": round(stake, 2),
                "pnl": round(pnl, 2),
                "bankroll": round(currentBank, 2),
            })

    resultsDF = pd.DataFrame(results)
        
    # Display skip vars
    print(f"\nSkip breakdown")
    print(f"No player match {noPlayerMatch}")
    print(f"No opp match {noOppMatch}")
    print(f"No actuals {noActuals}")
    print(f"No features {noFeatures}")

    print(f"\nLines skip breakdown")
    print(f"Lines skipped {noLine}")

    _printSummary(resultsDF, bankroll, currentBank, skipped)
    return resultsDF
