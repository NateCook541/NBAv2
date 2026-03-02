from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware


from models.predict import predict, predictTeamRoster

app = FastAPI(
        title = "NBA Point Predictor",
        description = "Predicts player points in NBA",
        version = "1.0.0",
)

app.add_middleware(
        CORSMiddleware,
        allow_origins = ["*"],
        allow_methods = ["GET"],
        allow_headers = ["*"],
)

@app.get("/health")
def health():
    """ Checks to make sure the app is running """
    return {"status": "ok"}


@app.get("/predict")
def getPrediction(
        playerName: str = Query(..., description="Player full name"),
        gameDate:   str = Query(..., description="Game date in YYYY-MM-DD"),
):
    try:
        result = predict(playerName, gameDate)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error - {e}")
        
    if result is None:
        raise HTTPException(
                status_code=404,
                detail=f"No game found for {playerName} on {gameDate}"
        )

    return result

@app.get("/predict/team")
def getTeamPrediction(
        teamID: int = Query(..., description="Team id of team (1-30)"),
        gameDate:   str = Query(..., description="Game date in YYYY-MM-DD"),
):
    try:
        results = predictTeamRoster(teamID, gameDate)
    except FileNotFoundError as e:
        raise HTTPException(status=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status=500, detail=f"Prediction error - {e}")

    if result is None:
        raise HTTPException(
                status_code=404,
                detail=f"No game found for {teamID} on {gameDate}"
        )

    return results

