from pathlib import Path

# Paths
DB_PATH = Path("NBA.db")
MODELS_DIR = Path("models")

FEATURE_CACHE_PATH = MODELS_DIR / "feature_cache.parquet"
MODEL_PATH = MODELS_DIR / "nba_model.joblib"
MINUTES_PATH = MODELS_DIR / "nba_minutes_model.joblib"
CALIBRATOR_PATH = MODELS_DIR / "nba_calibrator.joblib"
UNDER_CALIBRATOR_PATH = MODELS_DIR / "nba_under_calibrator.joblib"

MINUTES_META_PATH = MODELS_DIR / "nba_minutes_model_meta.joblib"
MODEL_META_PATH = MODELS_DIR / "nba_model_meta.joblib"

RESULTS_MODEL_PATH = MODELS_DIR / "nba_results_model.joblib"
RESULTS_META_PATH = MODELS_DIR / "nba_results_model_meta.joblib"

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
MIN_AVGPTS_CAL = 12
MIN_AVGMIN_CAL = 15

# Chronological holdout fraction for the train / calibration split
HOLDOUT_RATIO = 0.20

# XGBoost hyperparamters

POINTS_MODEL_PARAMS = {
    "n_estimators":    400,
    "max_depth":       4,
    "learning_rate":   0.05,
    "subsample":       0.8,
    "colsample_bytree":0.8,
    "min_child_weight":10,
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

# Game totals (results) model. Fewer rows than the player model (one row per game),
# so a shallow, heavily-regularized tree — the useful signal is thin (see
# resultsBuilder.RESULTS_FEATURES) and deeper trees just overfit noise.
RESULTS_MODEL_PARAMS = {
    "n_estimators":    200,
    "max_depth":       3,
    "learning_rate":   0.05,
    "subsample":       0.8,
    "colsample_bytree":0.8,
    "min_child_weight":20,
    "reg_lambda":      3.0,
    "n_jobs":         -1,
    "objective":      "reg:absoluteerror",
    "random_state":    42,
}

# Totals target behavior (mirrors POINTS_TARGET_MODE):
# residual: model learns total - naive_total_projection, then adds it back at inference
# absolute: model learns the raw game total directly
RESULTS_TARGET_MODE = "residual"

BIAS_SHRINK_K = 500

# Calibrator (player points)

CAL_FIT_LINES   = list(range(10, 40, 3))
PLATT_FIT_LINES = list(range(10, 46, 3))

SIGMA_BOUNDS = (3.0, 25.0)
DF_BOUNDS    = (2.1, 30.0)

PLATT_DAMPING = 0.5

# Totals (game over/under) calibrator
#
# The results model predicts in a narrow ~7-point band while real book totals sit
# right next to the prediction, so this calibrator fits P(actual > line) on lines
# OFFSET from each prediction (pred + offset) rather than on fixed absolute totals —
# that concentrates the fit where every real over/under decision actually lives.
RESULTS_CAL_PATH = MODELS_DIR / "nba_results_calibrator.joblib"

# Offsets (points) applied around each prediction to build the calibration lines.
RESULTS_CAL_OFFSETS = list(range(-12, 13, 2))    # pred-12 .. pred+12

# Totals residual std is ~19 (MAE 15.25), so sigma lives far higher than the points
# model. Bounds and initial-guess grid are scaled accordingly.
RESULTS_SIGMA_BOUNDS = (10.0, 30.0)
RESULTS_PLATT_DAMPING = 0.5

# Betting

DEFAULT_EDGE_THRESH = 0.05
MAX_BET_EDGE = 0.15
DEFAULT_BANKROLL = 1000
DEFAULT_KELLY_FRAC = 0.25         # over Kelly fraction
DEFAULT_UNDER_KELLY_FRAC = 0.20   # under Kelly fraction (lower — reduces variance at high bankrolls)
DEFAULT_DAILY_CAP = 0.15          # max fraction of bankroll to stake across all bets on one date
DEFAULT_MAX_STAKE_ABS = 0.05      # max single bet as fraction of STARTING bankroll (prevents stake explosion at high bankrolls)
FLAT_STAKE = 10

# Used to filter out really far out line preds as this is likely injury effects that scraper didn't catch
MIN_LINE = 10
MAX_LINE_DIFF = 15

OVER_MIN_DISAGREEMENT  = 2.0
