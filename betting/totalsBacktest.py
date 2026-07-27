"""
totalsBacktest.py

The real edge test for the game-totals product: on 2016-2023 games that have a
REAL closing line (Odds_archive), does our calibrated P(over/under) at the book's
line beat the vig-adjusted book-implied probability often enough to profit?

Design / honest caveats:
  * TRAIN/TEST SPLIT is chronological WITHIN 2016-2023. We retrain a fresh model
    + calibrator on the train slice ONLY, then bet the held-out tail — so nothing
    the model saw leaks into the bets. (We do NOT reuse the production model,
    which was trained through 2026 and would leak.)
  * The archive has total_open/total_close but NOT the over/under juice. Totals
    are almost universally priced ~-110/-110, so we ASSUME that (TOTALS_VIG_ODDS).
    Fair prob after de-vig = 0.5 each side; break-even win rate = 52.38%.
  * Bet line = the CLOSING line by default (also reports the OPENING line).
  * CLV proxy: since we have open AND close, we report how often the close moved
    TOWARD our bet side after we'd have bet the open — real closing-line value.

Not a substitute for live odds-API validation, but it answers the core question
on 20k+ real historical lines at $0 risk.
"""

import sqlite3
import numpy as np
import pandas as pd

from config import DB_PATH, RESULTS_MODEL_PARAMS
from features.cache import preloadCaches
from features.resultsBuilder import buildTotalsFeatures, RESULTS_FEATURES
from betting.resultsCalibrator import ResultsCalibrator
from xgboost import XGBRegressor

# Totals are ~always -110/-110. De-vigged fair prob per side = 0.5; a -110 bet
# needs to win 110/210 = 52.38% to break even.
TOTALS_VIG_ODDS = -110


def _impliedProb(usOdds):
    if usOdds < 0:
        return abs(usOdds) / (abs(usOdds) + 100)
    return 100 / (usOdds + 100)


def _payoutMultiplier(usOdds):
    return usOdds / 100 if usOdds > 0 else 100 / abs(usOdds)


def _fairProbPerSide(vigOdds=TOTALS_VIG_ODDS):
    # Symmetric -110/-110 -> de-vigged 0.5 each side.
    over = _impliedProb(vigOdds)
    under = _impliedProb(vigOdds)
    return over / (over + under)  # = 0.5


def _buildRows(caches, conn, seasons=(2016, 2023)):
    """One feature row per game that has ratings + a closing line, with the
    market open/close and the actual total, in date order."""
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


def _trainOnSlice(X, meta, trainIdx):
    """Fit a fresh residual XGB model on the train slice, then fit a calibrator on
    an inner held-out tail of the TRAIN slice (so the calibrator also never sees
    the bet games)."""
    base = X["naive_total_projection"].to_numpy(float)
    y = meta["total_pts"].to_numpy(float)

    Xtr = X.iloc[trainIdx]
    yres = y[trainIdx] - base[trainIdx]

    model = XGBRegressor(**RESULTS_MODEL_PARAMS)
    model.fit(Xtr[RESULTS_FEATURES], yres)

    # Calibrator fit on the last 20% of the TRAIN slice.
    cutoff = int(len(trainIdx) * 0.8)
    calIdx = trainIdx[cutoff:]
    calPred = model.predict(X.iloc[calIdx][RESULTS_FEATURES]) + base[calIdx]
    calAct = y[calIdx]
    calibrator = ResultsCalibrator.fit(predictions=calPred, actuals=calAct, savePath=None)
    return model, calibrator


def run(edgeThresh=0.03, seasons=(2016, 2023), trainFrac=0.7, useLine="close",
        flatStake=1.0, dbPath=DB_PATH):
    """
    edgeThresh : min (myProb - fairProb) to place a bet.
    useLine    : 'close' (default) or 'open' — which line we bet into.
    flatStake  : units per bet (flat betting; ROI = pnl / total staked).
    """
    conn = sqlite3.connect(str(dbPath))
    caches = preloadCaches(conn)
    X, meta = _buildRows(caches, conn, seasons)
    conn.close()

    n = len(X)
    split = int(n * trainFrac)
    trainIdx = np.arange(split)
    betIdx = np.arange(split, n)
    print(f"[totalsBT] {n} usable games | train {len(trainIdx)} | bet {len(betIdx)} "
          f"| line={useLine} | edge>={edgeThresh:.1%} | vig={TOTALS_VIG_ODDS}")

    model, calibrator = _trainOnSlice(X, meta, trainIdx)

    base = X["naive_total_projection"].to_numpy(float)
    preds = model.predict(X.iloc[betIdx][RESULTS_FEATURES]) + base[betIdx]

    fair = _fairProbPerSide()             # 0.5 for symmetric -110
    payout = _payoutMultiplier(TOTALS_VIG_ODDS)   # ~0.909
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

        # Pick the side we have positive edge on.
        overEdge = pOver - fair
        underEdge = pUnder - fair
        if overEdge >= underEdge:
            side, myProb, edge = "over", pOver, overEdge
        else:
            side, myProb, edge = "under", pUnder, underEdge

        if not (edgeThresh <= edge <= edgeCap):
            continue

        # Settle vs actual total (push if exactly on the line).
        if actual == betLine:
            pnl, result = 0.0, "push"
        elif (side == "over" and actual > betLine) or (side == "under" and actual < betLine):
            pnl, result = flatStake * payout, "win"
        else:
            pnl, result = -flatStake, "loss"

        # CLV proxy: did the close move toward our side vs the open?
        clvBeat = None
        if openLine is not None and np.isfinite(openLine) and useLine == "open":
            move = closeLine - openLine
            clvBeat = (move > 0) if side == "over" else (move < 0)

        bets.append({
            "date": row["game_date"], "season": int(row["season"]),
            "side": side, "line": float(betLine), "pred": round(pred, 1),
            "my_prob": round(myProb, 3), "edge": round(edge, 3),
            "actual": actual, "result": result, "pnl": pnl,
            "clv_beat": clvBeat,
        })

    return _summarize(pd.DataFrame(bets), betIdx, flatStake, useLine)


def _summarize(df, betIdx, flatStake, useLine):
    print("\n" + "=" * 60)
    if df.empty:
        print("[totalsBT] NO bets cleared the edge threshold.")
        print("=" * 60)
        return df

    nb = len(df)
    wins = (df["result"] == "win").sum()
    losses = (df["result"] == "loss").sum()
    pushes = (df["result"] == "push").sum()
    decided = wins + losses
    winRate = wins / decided if decided else 0.0
    pnl = df["pnl"].sum()
    staked = flatStake * decided          # pushes stake nothing net
    roi = pnl / staked if staked else 0.0

    print(f"[totalsBT] RESULTS ({useLine} line)")
    print(f"  games offered : {len(betIdx)}")
    print(f"  bets placed   : {nb}  ({nb / len(betIdx):.1%} of games)")
    print(f"  record        : {wins}-{losses}-{pushes}  (W-L-Push)")
    print(f"  win rate      : {winRate:.3%}   (break-even @ -110 = 52.381%)")
    print(f"  flat P&L      : {pnl:+.2f} u   on {staked:.0f}u staked")
    print(f"  ROI           : {roi:+.2%}")
    if df["clv_beat"].notna().any():
        clv = df["clv_beat"].dropna()
        print(f"  CLV (beat close): {clv.mean():.1%}  (close moved our way; >50% = good)")
    # Per-season sanity
    print("  by season:")
    for s, g in df.groupby("season"):
        d = (g["result"] != "push").sum()
        w = (g["result"] == "win").sum()
        print(f"     {s}: {len(g):>4} bets  {w/d if d else 0:.1%} WR  {g['pnl'].sum():+.1f}u")
    print("=" * 60)
    return df


if __name__ == "__main__":
    import sys
    thr = float(sys.argv[1]) if len(sys.argv) > 1 else 0.03
    line = sys.argv[2] if len(sys.argv) > 2 else "close"
    run(edgeThresh=thr, useLine=line)
