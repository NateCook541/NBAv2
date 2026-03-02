import argparse
import json
from pathlib import Path

from data.scrapperEngine import ScrapeEngine
from data.dbManager import DBManager

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

def scrape(dbPath='NBA.db', outputDir="output", numLogGames=None):
    _doubleCheckTeamMap(outputDir)
    db = DBManager(dbPath)
    db.initSchema()

    engine = ScrapeEngine(db=dbPath, headless=True)

    try:
        print("\n--------Scraping--------")
        teams = engine.scrapeTeams()
        players = engine.scrapePlayers()
        games = engine.scrapeGames()
        logs = engine.scrapeLogs(numGames=numLogGames)
        status = engine.scrapeStatus()
        results = engine.scrapeResults()
    finally:
        engine.close()

    print("\n--------Loading Items into DB--------")
    db.upsertTeams(teams)
    db.upsertPlayers(players)
    db.upsertGames(games)
    db.upsertLogs(logs)
    db.upsertStatus(status)
    db.upsertResults(results)

    print("\n--------DB Updated Complete--------")


def retrainModel():
    from models.train import trainModel
    print("\n--------Training Model--------")
    trainModel(save=True)
    print("--------Training complete--------")


# ENTRY POINT


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NBA prediction pipeline")
    parser.add_argument("--train",      action="store_true",
                        help="Refresh data then retrain model")
    parser.add_argument("--train-only", action="store_true",
                        help="Retrain without scraping")
    parser.add_argument("--num-games",  type=int, default=None,
                        help="Limit log scraping to N games (debug only)")
    parser.add_argument("--db",  default="NBA.db",  help="SQLite DB path")
    parser.add_argument("--out", default="output",  help="JSON output dir")
    args = parser.parse_args()

    if args.train_only:
        retrainModel()
    elif args.train:
        scrape(dbPath=args.db, outputDir=args.out, numLogGames=args.num_games)
        retrainModel()
    else:
        scrape(dbPath=args.db, outputDir=args.out, numLogGames=args.num_games)

# :steam_smile

