from pathlib import Path

# Paths
DB_PATH = Path("NBA.db")
MODELS_DIR = Path("models")

FEATURE_CACHE_PATH = MODELS_DIR / "feature_cache.parquet"
MODEL_PATH = MODELS_DIR / "nba_model.joblib"
MINUTES_PATH = MODELS_DIR / "nba_minutes_model.joblib"
CALIBRATOR_PATH = MODELS_DIR / "nba_calibrator.joblib"

MINUTES_META_PATH = MODELS_DIR / "nba_minutes_model_meta.joblib"
MODEL_META_PATH = MODELS_DIR / "nba_model_meta.joblib"

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
    "max_depth":       5,
    "learning_rate":   0.05,
    "subsample":       0.8,
    "colsample_bytree":0.8,
    "min_child_weight":7,
    "n_jobs":         -1,
    "objective":      "reg:absoluteerror",
    "random_state":    42,
}

# Points target behavior
# residual: model learns points - avgPts10, then adds avgPts10 back at inference
# absolute: model learns raw points directly
POINTS_TARGET_MODE = "residual"

# Optional prediction clipping around player recent mean/stdev.
# Set to <= 0 to disable clipping.
PREDICTION_CLIP_K = 2.5
PREDICTION_CLIP_CANDIDATES = [0.0, 2.5, 3.5]
CLIP_LOWER_K_CANDIDATES = [0.0, 2.5]
CLIP_UPPER_K_CANDIDATES = [2.5, 3.5, 4.5]
CLIP_TAIL_PENALTY = 0.15

# Walk-forward evaluation defaults
WALKFORWARD_SPLITS = 6
WALKFORWARD_MIN_TRAIN_ROWS = 16000
WALKFORWARD_MAX_NEG_DELTA = -0.10
WALKFORWARD_ALERT_DELTA = -0.05
WF_TUNE_RATIO = 0.15
USE_WF_RECENCY_WEIGHTS = True
ENABLE_BASELINE_BLEND = True
ENABLE_REGIME_ALPHA = True
ALPHA_MIN = 0.55
ALPHA_MAX = 1.0
CHAOS_Q_MINSTD = 0.75
CHAOS_Q_INJURY = 0.75
TUNE_SCORE_LAMBDA_P95 = 0.08
TUNE_SCORE_LAMBDA_NEG_DELTA = 0.75
TUNE_SCORE_LAMBDA_NEG_DELTA_SQ = 1.25
TUNE_MIN_DELTA_FOR_MODEL = 0.00
ALPHA_CHAOS_MAX_WHEN_WEAK = 0.60
ALPHA_SHRINK_TO_BASELINE = 0.20
ALPHA_MAX_WHEN_NO_CLIP = 0.65

# Recency weighting for points training rows
USE_RECENCY_WEIGHTS = True
RECENCY_WEIGHT_MIN = 0.8
RECENCY_WEIGHT_MAX = 1.2
 
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
DEFAULT_UNDER_CALIBRATION_MODE = "hybrid"
DEFAULT_UNDER_HIGH_CUTOFF = 22.0
DEFAULT_UNDER_RELIABILITY_SHRINK = 0.10

# Betting

DEFAULT_EDGE_THRESH = 0.03
DEFAULT_UNDER_EDGE_THRESH = 0.10
DEFAULT_SELECTION_MODE = "threshold"
DEFAULT_BET_BUDGET = 1000
DEFAULT_BET_BUDGET_TOLERANCE = 50
DEFAULT_MARKET_PROB_SHRINK = 0.15
DEFAULT_BANKROLL = 1000
DEFAULT_KELLY_FRAC = 0.25
FLAT_STAKE = 10
UNDER22_SCORE_MODE = "hybrid"  # edge | ev | hybrid
# 22+ under quality model (config-only; no CLI knobs)
UNDER22_USE_EV_MARGIN = True
UNDER22_SCORE_W_EV_MARGIN = 0.40
UNDER22_SCORE_W_CONFIDENCE = 0.15
UNDER22_SCORE_W_ODDS_COST = 0.20
UNDER22_SCORE_W_RELIABILITY = 0.25
UNDER22_SCORE_W_EDGE = 0.20
# Reference break-even probability where odds are considered "neutral cost".
UNDER22_BREAKEVEN_BASE = 0.52
# Soft price-mix controls (applied only to 22+ under selection edge).
# Negative values penalize a breakeven bucket, positive values boost it.
UNDER22_BREAKEVEN_SOFT_ADJUST = {
    "le_52": -0.060,   # <=52%
    "52_54": -0.025,   # 52-54%
    "54_56":  0.010,   # 54-56%
    "gt_56": -0.010,   # >56%
}
UNDER22_MARKET_ANCHOR_ENABLED = True

# Used to filter out really far out line preds as this is likely injury effects that scraper didn't catch
MIN_LINE = 5
MAX_LINE_DIFF = 10

UNDER_MIN_DISAGREEMENT = 5.0
OVER_MIN_DISAGREEMENT  = 2.0

# Side/prediction regime filters from backtest diagnostics
# Unders below this predicted points level have been structurally weak.
UNDER_MIN_PREDICTED_POINTS = 22.0
# Overs in this band have been structurally weak.
OVER_BLOCK_PREDICTED_RANGE = (18.0, 22.0)
