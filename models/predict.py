import sqlite3
import unicodedata
import joblib
import pandas as pd
from pathlib import Path

from features.featureCollecter import buildFeatures

#FIXME: Add docstrings bum

# HELPER FUNCTIONS

def _normalizeName(name):
    return "".join(
            c for c in unicodedata.normalize("NFD", name)
            if unicodedata.category(c) != "Mn"
        ).lower().strip()

# Loads the trained model from the disk. Raises FileNotFound if no model trained/found
def _loadModel():
    modelPath = Path(__file__).parent / "nba_model.joblib"
    if not modelPath.exists():
        raise FileNotFoundError(
                "Non trained model found"
        )

        return joblib.load(modelPath)


# PUBLIC API


def predict(playerName, gameDate):
    model = _loadModel()
    conn = sqlite3.connect('NBA.db')

    try:
        cleanName = _normalizeName(playerName)
        playerRow = pd.read_sql_query(
            "SELECT player_id, name, team_id FROM Players WHERE is_active = 1", conn
        )
        playerRow["norm"] = playerRow["name"].apply(_normalizeName)
        match = playerRow[playerRow["norm"] == cleanName]

        if match.empty:
            return None

        playerID = int(match.iloc[0]["player_id"])
        teamID = int(match.iloc[0]["team_id"])
        fullName = match.iloc[0]["name"]

        gameRow = pd.read_sql_query(
                """
            SELECT game_id, home_team_id, away_team_id
            FROM Games
            WHERE (home_team_id = ? OR away_team_id = ?) AND game_date = ?
            """,
            conn,
            params=[teamID, teamID, gameDate],
        )

        if gameRow.empty:
            return None

        gameID = gameRow.iloc[0]["game_id"]
        homeTeam = int(gameRow.iloc[0]["home_team_id"])
        awayTeam = int(gameRow.iloc[0]["away_team_id"])
        oppTeamID = awayTeam if homeTeam == teamID else homeTeam

        # Get injury status
        injuryRow = pd.read_sql_query(
                "SELECT status FROM Status WHERE player_id = ? AND game_id = ?",
            conn,
            params=[playerID, gameID],
        )
        injuryStatus = injuryRow.iloc[0]["status"] if not injuryRow.empty else None

        # If player is out just return default stats
        if injuryStatus == "Out":
            return {
                "player":           fullName,
                "date":             gameDate,
                "team_id":          teamID,
                "opp_team_id":      oppTeamID,
                "predicted_points": 0.0,
                "injury_status":    "Out",
            }
        
        # If not build features and run the prediction
        features = buildFeatures(playerID, gameDate, teamID, oppTeamID, conn)
        if features is None:
            return None

        predicted = float(model.predict(features)[0])

        return {
                "player": fullName,
                "date": gameDate,
                "team_id": teamID,
                "opp_team_id": oppTeamID,
                "predicted_points": round(predicted, 1),
                "injury_status": injuryStatus,
        }

    finally:
        conn.close()


def predictTeamRoster(teamID, gameDate):
    conn = sqlite3.connect('NBA.db')
    roster = pd.read_sql_query(
            "SELECT name FROM Players WHERE team_id = ? AND is_active = 1",
        conn,
        params=[teamID],
    )
    conn.close()

    results = []
    for name in roster["name"]:
        result = predict(name, gameDate)
        if result:
            results.append(result)

    return sorted(results, key=lambda r: r["predicted_points"], reverse=True)


# CLI ENTRY POINT

if __name__ == '__main__':
    import sys

    if len(sys.argv) != 3:
        print("python -m models.predict (player name), (date)")
        sys.exit(1)

    result = predict(sys.argv[1], sys.argv[2])
    if result is None:
        print("Player or game not found in db")
    else:
        status = (f"[{result['injury_status']}]" if result["injury_status"] else ""
        print(f"{result['player']}{status}: {result['predicted_points']} pts on {result['date']}")
















