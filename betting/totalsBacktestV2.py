"""
totalsBacktestV2.py

Three experimental changes over totalsBacktest.py, run on the same 2016-2023
real-line data ([[odds-archive]]) so results are directly comparable to the
baseline:

  #1  RESIDUAL-VS-LINE TARGET. The baseline model learns `total - naive_total_projection`
      and treats the line as just another feature, so it spends capacity re-deriving
      the line. Here the target is `total - betLine` — the model explicitly predicts
      the MARKET'S ERROR, which is the only quantity that generates edge. At inference
      the prediction of the total = betLine + model_output.

  #3  NO CLOSE-LINE LEAKAGE IN THE OPEN-LINE TEST. The baseline open-line run trains a
      model that sees `market_total_close`, `line_minus_naive`, `open_close_move` — all
      of which embed the CLOSING number — then bets the OPEN. That's look-ahead: at open
      time you don't know the close. For useLine="open" we restrict the model to an
      open-only feature set (OPEN_FEATURES) so the edge, if any, is real. The
      suspiciously-clean 90-95% CLV in the baseline is the thing this checks.

  #2  SUBSET CONCENTRATION. Instead of one blended win rate over 40-64% of games, we
      bucket bets by |pred - betLine| (how hard the model disagrees with the market)
      and report WR/ROI per bucket. If the edge lives in the high-disagreement tail,
      that's a far more realistic strategy than betting half the slate.

Everything else (fresh retrain on the train slice, chronological split, -110 vig
assumption, calibrator fit on the inner train tail) matches the baseline exactly.
"""

import sqlite3
import numpy as np
import pandas as pd

from config import DB_PATH, RESULTS_MODEL_PARAMS
from features.cache import preloadCaches
from features.resultsBuilder import buildTotalsFeatures
from betting.resultsCalibrator import ResultsCalibrator
from xgboost import XGBRegressor

TOTALS_VIG_ODDS = -110

# Feature set for the CLOSE-line strategy: close-derived features are legitimate
# because you bet after the close is known. Mirrors baseline RESULTS_FEATURES.
CLOSE_FEATURES = [
    "naive_total_projection",
    "combined_pace_avg",
    "team_pace10", "opp_pace10",
    "team_pts_allowed10", "opp_pts_allowed10",
    "formTotal5",
    "team_pts_avg10", "opp_pts_avg10",
    "team_def_rtg10", "opp_def_rtg10",
    "days_since_last_meeting",
    "market_total_close", "line_minus_naive", "open_close_move",
]

# Feature set for the OPEN-line strategy: ONLY features knowable at open time. No
# market_total_close, no line_minus_naive, no open_close_move — those embed the close.
OPEN_FEATURES = [
    "naive_total_projection",
    "combined_pace_avg",
    "team_pace10", "opp_pace10",
    "team_pts_allowed10", "opp_pts_allowed10",
    "formTotal5",
    "team_pts_avg10", "opp_pts_avg10",
    "team_def_rtg10", "opp_def_rtg10",
    "days_since_last_meeting",
    "market_total_open", "open_minus_naive",
]


def _impliedProb(usOdds):
    if usOdds < 0:
        return abs(usOdds) / (abs(usOdds) + 100)
    return 100 / (usOdds + 100)


def _payoutMultiplier(usOdds):
    return usOdds / 100 if usOdds > 0 else 100 / abs(usOdds)


def _fairProbPerSide(vigOdds=TOTALS_VIG_ODDS):
    over = _impliedProb(vigOdds)
    under = _impliedProb(vigOdds)
    return over / (over + under)  # 0.5 for symmetric -110


def _buildRows(caches, conn, seasons=(2016, 2023)):
    """One feature row per game with ratings + a closing line (open is a subset of
    those), with market open/close and the actual total, in date order."""
    q = f"""
        SELECT g.game_id, g.game_date, g.home_team_id, g.away_team_id, g.season,
               (r.home_score + r.away_score) AS total_pts,
               oa.total_open, oa.total_close
        FROM Games g
        JOIN Results r ON r.game_id = g.game_id
        JOIN Odds_archive oa
             ON oa.game_date = g.game_date
            AND oa.home_team_id = g.home_team_id
            AND oa.away_team_id = g.away_team_id
        WHERE oa.total_close IS NOT NULL
          AND g.season BETWEEN {seasons[0]} AND {seasons[1]}
        ORDER BY g.game_date, g.game_id
    """
    games = pd.read_sql_query(q, conn)

    rows, keep = [], []
    for g in games.itertuples(index=False):
        f = buildTotalsFeatures(
            g.game_id, g.game_date, int(g.home_team_id), int(g.away_team_id),
            caches.teamGameCache, caches.statusDF, caches.playerLogCache,
            caches.teamGameTotals, caches.h2hCache, oddsCache=caches.oddsCache,
        )
        if f is None:
            continue
        rows.append(f)
        keep.append(g)
    X = pd.concat(rows, ignore_index=True)
    meta = pd.DataFrame(keep)
    return X, meta


def _lineArray(meta, useLine):
    col = "total_close" if useLine == "close" else "total_open"
    return meta[col].to_numpy(float)


def _trainOnSlice(X, meta, trainIdx, features, useLine):
    """CHANGE #1: fresh residual-VS-LINE XGB on the train slice, then a calibrator on
    the inner last-20% of train. Target = total - betLine (the market's error)."""
    line = _lineArray(meta, useLine)
    y = meta["total_pts"].to_numpy(float)

    # Only train on rows where the bet line exists (open can be sparser than close).
    valid = np.isfinite(line[trainIdx])
    fitIdx = trainIdx[valid]

    Xtr = X.iloc[fitIdx]
    yres = y[fitIdx] - line[fitIdx]

    model = XGBRegressor(**RESULTS_MODEL_PARAMS)
    model.fit(Xtr[features], yres)

    # Calibrator fit on the last 20% of the (valid) train slice. It works in TOTAL
    # space, so reconstruct pred_total = betLine + residual_pred.
    cutoff = int(len(fitIdx) * 0.8)
    calIdx = fitIdx[cutoff:]
    calPred = model.predict(X.iloc[calIdx][features]) + line[calIdx]
    calAct = y[calIdx]
    calibrator = ResultsCalibrator.fit(predictions=calPred, actuals=calAct, savePath=None)
    return model, calibrator


def run(edgeThresh=0.03, seasons=(2016, 2023), trainFrac=0.7, useLine="close",
        flatStake=1.0, dbPath=DB_PATH, caches=None, conn=None):
    """
    edgeThresh : min (myProb - fairProb) to place a bet.
    useLine    : 'close' (uses CLOSE_FEATURES) or 'open' (uses OPEN_FEATURES, no leak).
    caches/conn: optional preloaded caches + open connection (so a driver can reuse
                 one expensive feature build across configs).
    """
    ownConn = conn is None
    if ownConn:
        conn = sqlite3.connect(str(dbPath))
        caches = preloadCaches(conn)
    if caches is None:
        caches = preloadCaches(conn)

    X, meta = _buildRows(caches, conn, seasons)
    if ownConn:
        conn.close()

    features = CLOSE_FEATURES if useLine == "close" else OPEN_FEATURES

    n = len(X)
    split = int(n * trainFrac)
    trainIdx = np.arange(split)
    betIdx = np.arange(split, n)
    print(f"[totalsBTv2] {n} usable games | train {len(trainIdx)} | bet {len(betIdx)} "
          f"| line={useLine} | features={'CLOSE' if useLine=='close' else 'OPEN(no-leak)'} "
          f"| target=residual-vs-{useLine} | edge>={edgeThresh:.1%}")

    model, calibrator = _trainOnSlice(X, meta, trainIdx, features, useLine)

    line = _lineArray(meta, useLine)
    preds = model.predict(X.iloc[betIdx][features]) + line[betIdx]

    fair = _fairProbPerSide()
    payout = _payoutMultiplier(TOTALS_VIG_ODDS)
    edgeCap = float(calibrator.profitableEdgeCap)

    bets = []
    for k, i in enumerate(betIdx):
        row = meta.iloc[i]
        pred = float(preds[k])
        openLine = row["total_open"]
        closeLine = row["total_close"]
        betLine = closeLine if useLine == "close" else openLine
        if betLine is None or not np.isfinite(betLine):
            continue
        actual = float(row["total_pts"])

        pOver = calibrator.probOver(pred, betLine)
        pUnder = 1.0 - pOver

        overEdge = pOver - fair
        underEdge = pUnder - fair
        if overEdge >= underEdge:
            side, myProb, edge = "over", pOver, overEdge
        else:
            side, myProb, edge = "under", pUnder, underEdge

        if not (edgeThresh <= edge <= edgeCap):
            continue

        if actual == betLine:
            pnl, result = 0.0, "push"
        elif (side == "over" and actual > betLine) or (side == "under" and actual < betLine):
            pnl, result = flatStake * payout, "win"
        else:
            pnl, result = -flatStake, "loss"

        clvBeat = None
        if openLine is not None and np.isfinite(openLine) and useLine == "open" \
                and closeLine is not None and np.isfinite(closeLine):
            move = closeLine - openLine
            clvBeat = (move > 0) if side == "over" else (move < 0)

        bets.append({
            "date": row["game_date"], "season": int(row["season"]),
            "side": side, "line": float(betLine), "pred": round(pred, 2),
            "disagree": abs(pred - float(betLine)),
            "my_prob": round(myProb, 3), "edge": round(edge, 3),
            "actual": actual, "result": result, "pnl": pnl,
            "clv_beat": clvBeat,
        })

    df = pd.DataFrame(bets)
    _summarize(df, betIdx, flatStake, useLine)
    _subsetConcentration(df, flatStake)
    return df


def _summarize(df, betIdx, flatStake, useLine):
    print("\n" + "=" * 60)
    if df.empty:
        print("[totalsBTv2] NO bets cleared the edge threshold.")
        print("=" * 60)
        return
    nb = len(df)
    wins = (df["result"] == "win").sum()
    losses = (df["result"] == "loss").sum()
    pushes = (df["result"] == "push").sum()
    decided = wins + losses
    winRate = wins / decided if decided else 0.0
    pnl = df["pnl"].sum()
    staked = flatStake * decided
    roi = pnl / staked if staked else 0.0

    print(f"[totalsBTv2] RESULTS ({useLine} line, residual-vs-{useLine} target)")
    print(f"  games offered : {len(betIdx)}")
    print(f"  bets placed   : {nb}  ({nb / len(betIdx):.1%} of games)")
    print(f"  record        : {wins}-{losses}-{pushes}  (W-L-Push)")
    print(f"  win rate      : {winRate:.3%}   (break-even @ -110 = 52.381%)")
    print(f"  flat P&L      : {pnl:+.2f} u   on {staked:.0f}u staked")
    print(f"  ROI           : {roi:+.2%}")
    if df["clv_beat"].notna().any():
        clv = df["clv_beat"].dropna()
        print(f"  CLV (beat close): {clv.mean():.1%}  (close moved our way; >50% = good)")
    print("  by season:")
    for s, g in df.groupby("season"):
        d = (g["result"] != "push").sum()
        w = (g["result"] == "win").sum()
        print(f"     {s}: {len(g):>4} bets  {w/d if d else 0:.1%} WR  {g['pnl'].sum():+.1f}u")
    print("=" * 60)


def _subsetConcentration(df, flatStake, nBuckets=4):
    """CHANGE #2: does the edge concentrate where the model disagrees most with the
    market? Bucket bets by |pred - line| (quartiles) and report each bucket."""
    if df.empty or len(df) < nBuckets * 5:
        return
    print("\n[totalsBTv2] EDGE CONCENTRATION by |pred - line| (quartiles):")
    try:
        df = df.copy()
        df["bucket"] = pd.qcut(df["disagree"], nBuckets, labels=False, duplicates="drop")
    except ValueError:
        return
    print(f"  {'bucket':<8}{'disagree':<16}{'n':>5}{'WR':>9}{'ROI':>9}{'P&L':>9}")
    for b, g in df.groupby("bucket"):
        d = (g["result"] != "push").sum()
        w = (g["result"] == "win").sum()
        wr = w / d if d else 0.0
        pnl = g["pnl"].sum()
        staked = flatStake * d
        roi = pnl / staked if staked else 0.0
        lo, hi = g["disagree"].min(), g["disagree"].max()
        rng = f"{lo:.1f}-{hi:.1f}"
        print(f"  Q{int(b)+1:<7}{rng:<16}{len(g):>5}{wr:>8.1%}{roi:>8.1%}{pnl:>+8.1f}u")
    print("  (break-even WR @ -110 = 52.4%)")


if __name__ == "__main__":
    import sys
    thr = float(sys.argv[1]) if len(sys.argv) > 1 else 0.03
    line = sys.argv[2] if len(sys.argv) > 2 else "close"
    run(edgeThresh=thr, useLine=line)
