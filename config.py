from pathlib import Path

# Paths
DB_PATH = Path("NBA.db")
MODELS_DIR = Path("models")
FEATURE_CACHE = MODELS_DIR / "feature_cache.parquet"
MODEL_PATH = MODELS_DIR / "nba_model.joblib"
MINUTES_PATH = MODELS_DIR / "nba_minutes_model.joblib"
CALIBRATOR_PATH = MODELS_DIR / "nba_calibrator.joblib"

# Scraping

BREF_BASE        = "https://www.basketball-reference.com"
ODDS_BASE_URL    = "https://api.the-odds-api.com/v4"
BREF_SLEEP_SECS = 4
CURRENT_SEASON = 2026

TEAM_MAP = {
    "DEN": 1,  "OKC": 2,  "HOU": 3,  "NYK": 4,  "MIA": 5,
    "SAS": 6,  "UTA": 7,  "MIN": 8,  "LAL": 9,  "DET": 10,
    "POR": 11, "CLE": 12, "CHI": 13, "ORL": 14, "ATL": 15,
    "PHI": 16, "BOS": 17, "CHO": 18, "TOR": 19, "NOP": 20,
    "MEM": 21, "PHO": 22, "GSW": 23, "MIL": 24, "DAL": 25,
    "WAS": 26, "SAC": 27, "LAC": 28, "IND": 29, "BRK": 30,
}

# Training

# Minimum minutes for a player to be included in trainin
MIN_MINUTES_TRAIN = 5

# Makes the calibrator only sees players who get consistent time and points
MIN_AVGPTS_CAL = 5
MIN_AVGMIN_CAL = 10

# Chronological holdout fraction for the train / calibration split
HOLDOUT_RATIO = 0.20

# XGBoost hyperparamters

POINTS_MODEL_PARAMS = {
    "n_estimators":    400,
    "max_depth":       6,
    "learning_rate":   0.05,
    "subsample":       0.8,
    "colsample_bytree":0.8,
    "min_child_weight":5,
    "n_jobs":         -1,
    "objective":      "reg:squarederror",
    "random_state":    42,
}
 
MINUTES_MODEL_PARAMS = {
    "n_estimators":    300,
    "max_depth":       4,
    "learning_rate":   0.05,
    "subsample":       0.8,
    "min_child_weight":5,
    "n_jobs":         -1,
    "random_state":    42,
}

# Calibrator

CAL_FIT_LINES   = list(range(10, 40, 3))
PLATT_FIT_LINES = list(range(10, 46, 3))

SIGMA_BOUNDS = (3.0, 25.0)
DF_BOUNDS    = (2.1, 30.0)

# Betting

DEFAULT_EDGE_THRESH = 0.03
DEFAULT_BANKROLL = 1000
DEFAULT_KELLY_FRACT = 0.25
FLAT_STAKE = 10

# Used to filter out really far out line preds as this is likely injury effects that scraper didn't catch
MIN_LINE = 5
MAX_LINE_DIFF = 10

