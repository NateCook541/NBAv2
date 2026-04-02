import argparse
import json
import subprocess
import joblib
from pathlib import Path

from data.scrapperEngine import ScrapeEngine
from data.dbManager import DBManager

from models.train import preloadCaches, generateTrainingData
from models.evaluate import evaluateModel

from betting.oddsCollector import pullHistoricalProps
from betting.backtest import runBacktest

TeamMap = {
    "DEN": 1,  "OKC": 2,  "HOU": 3,  "NYK": 4,  "MIA": 5,
    "SAS": 6,  "UTA": 7,  "MIN": 8,  "LAL": 9,  "DET": 10,
    "POR": 11, "CLE": 12, "CHI": 13, "ORL": 14, "ATL": 15,
    "PHI": 16, "BOS": 17, "CHO": 18, "TOR": 19, "NOP": 20,
    "MEM": 21, "PHO": 22, "GSW": 23, "MIL": 24, "DAL": 25,
    "WAS": 26, "SAC": 27, "LAC": 28, "IND": 29, "BRK": 30,
}

# LOL
def _doubleCheckTeamMap(outputDir="output"):
    path = Path(outputDir) / "teams_map.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w") as f:
            json.dump(TeamMap, f, indent=2)
        print(f"Created {path}")

def scrape(dbPath='NBA.db', outputDir="output", numLogGames=None, backfillFrom=None):
    _doubleCheckTeamMap(outputDir)
    db = DBManager(dbPath)
    db.initSchema()

    engine = ScrapeEngine(db=dbPath, headless=True)
    

    try:
        print("\n--------Scraping--------")
        teams = engine.scrapeTeams()

        players = engine.scrapePlayers()
        with open(f"{outputDir}/players.json", "w") as f:
            json.dump(players, f, indent=2)

        games = engine.scrapeGames()
        with open(f"{outputDir}/games.json", "w") as f:
            json.dump(games, f, indent=2)

        logs = engine.scrapeLogs(numGames=numLogGames)
        
        if backfillFrom:
            status = engine.scrapeStatusRange(backfillFrom)
        else:
            status = engine.scrapeAutoFill()
        
    finally:
        engine.close()

    print("\n--------Loading Items into DB--------")

    db.upsertTeams(teams)
    db.upsertPlayers(players)
    db.upsertGames(games)
    db.upsertLogs(logs)
    db.upsertStatus(status)

    print("\n--------DB Updated Complete--------")

def scrapeHistorical(seasons, dbPath="NBA.db", outputDir="output"):
    # Setup DB and Scrapper Engine
    _doubleCheckTeamMap(outputDir)
    db = DBManager(dbPath)
    db.initSchema()
    engine = ScrapeEngine(db=dbPath, headless=True)

    try:
        print(f"\n-------- Historical Scrape: {seasons} --------")
        
        # Teams
        teams = engine.scrapeTeamsHistorical(seasons)
        db.upsertTeams(teams)

        # Players
        players = engine.scrapePlayersHistorical(seasons)
        with open(f"{outputDir}/players.json", "w")  as f:
            json.dump(players, f, indent=2)
        db.upsertPlayers(players)

        # Games and logs
        allGames = []
        for season in seasons:
            games = engine.scrapeGames(season=season)
            allGames.extend(games)
        with open(f"{outputDir}/games.json", "w") as f:
            json.dump(allGames, f, indent=2)
        db.upsertGames(allGames)

        for season in seasons:
            logs = engine.scrapeLogsHistorical(season=season)
            db.upsertLogs(logs)

        # Status
        for season in seasons:
            startDate = f"{season - 1}-10-01"
            endDate = f"{season}-06-30"
            status = engine.scrapeStatusRange(startDate, endDate)
            db.upsertStatus(status)
    
    finally:
        engine.close()


def retrainModel(metrics=True):
    from models.train import trainModel
    print("\n--------Training Model--------")
    trainModel(save=True, metrics=metrics)
    print("--------Training complete--------")

def evaluateCurrentModel(dbPath="NBA.db"):
    modelPath = Path("models/nba_model.joblib")
    if not modelPath.exists():
        print("No saved model found")
        return

    model = joblib.load(modelPath)
    print("\n-------- Evaluating Saved Model --------")

    X, y = generateTrainingData()

    splitIdx = int(len(X) * 0.8)
    XTest = X.iloc[splitIdx:]
    yTest = y.iloc[splitIdx:]

    evaluateModel(model, XTest, yTest)
    print("-------- Evaluation Complete --------")


# ENTRY POINT

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NBA prediction pipeline")
    
    # Scrape args
    parser.add_argument("--scrape", action="store_true",
                        help="Scrape data and store in DB")
    parser.add_argument("--backfill-from", type=str, default=None, metavar="YYYY-MM-DD",
                        help="Backfill status data from this date instead of auto-detecting. "
                             "Example: --backfill-from 2023-10-01")
    parser.add_argument("--num-games",  type=int, default=None,
                        help="Limit log scraping to N games (debug only)")
    
    # Historical scrapping
    parser.add_argument("--historical-seasons", type=int,
                        nargs="+", default=None, metavar="SEASON",
                        help="Scrape historical data for given season exg --historical-seasons 2024 2025")

    # Train args
    parser.add_argument("--train", action="store_true",
                        help="Train model")
    parser.add_argument("--metrics", action="store_true",
                    help="Show training metrics")

    # Evalute the current model with out retrain
    parser.add_argument("--evaluate", action="store_true",
                        help="Show metrics for current saved model without having to retrain")

    # Props args
    parser.add_argument("--pull-props", nargs=2,
                        metavar=("START_DATE", "END_DATE"),
                        help="Pull historical props exg --pull-props 2025-02-01 2025-02-28")

    # Backtesint args
    parser.add_argument("--backtest", action="store_true",
                        help="Run backtest against stored props")
    parser.add_argument("--edge-thresh", type=float, default=0.03,
                        help="Minium edge to place a bet (default: 0.03)")
    parser.add_argument("--bankroll", type=float, default=1000.0,
                        help="Starting bankroll in dollars (default: 1000)")

    # Shared args
    parser.add_argument("--db",  default="NBA.db",  help="SQLite DB path")
    parser.add_argument("--out", default="output",  help="JSON output dir")
    
    args = parser.parse_args()

    if args.train:
        retrainModel(metrics=args.metrics)
    if args.evaluate:
        evaluateCurrentModel(dbPath=args.db)
    if args.scrape:
        scrape(dbPath=args.db, outputDir=args.out, numLogGames=args.num_games, backfillFrom=args.backfill_from)
    if args.historical_seasons:
        scrapeHistorical(args.historical_seasons, dbPath=args.db, outputDir=args.out)
    if args.pull_props:
        pullHistoricalProps(args.pull_props[0], args.pull_props[1], dbPath=args.db)
    if args.backtest:
        runBacktest(dbPath=args.db, edgeThresh=args.edge_thresh, bankroll=args.bankroll)


    if not args.train and not args.scrape and not args.historical_seasons and not args.evaluate and not args.pull_props and not args.backtest:
        parser.print_help()

# :steam_smile

