"""
Hub emitter for the NBA points props model

Runs todays live scoring and writes a single JSON file describing the bets
this model wants, into the hub inbox. The hub reads that file, it never
imports this repo. Everything below is a thin translation layer over
betting.liveScorer.scoreLiveDay 
"""

from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from config import DB_PATH, DEFAULT_EDGE_THRESH

MODEL_NAME = "nba_points_props"
SPORT = "nba"
PROP_TYPE = "points"
SCHEMA_VERSION = 1

# The hub inbox path
DEFAULT_INBOX = Path(os.environ.get("HUB_INBOX", Path.home() / "betting" / "inbox"))

# All bets bet at flat stake currently
STAKE_UNITS = 1.0


def _gitSha():
    """Short SHA of this repo or None outside a git checkout"""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _gitDirty():
    """
    True if tracked files differ from HEAD, the emitted bets then came
    from code that is not the committed SHA, which matters when attributing
    in season performance to a model version
    """
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=Path(__file__).parent,
            capture_output=True, text=True, timeout=10,
        )
        return bool(out.stdout.strip()) if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _modelVersion(points, overCal, underCal):
    """
    Version stamp carried on every bet, so a bet is permanently attributable
    to the exact model state that produced it. trainedThrough comes from the
    points bundle's own metadata, not from the run date
    """
    meta = dict(getattr(points, "meta", {}) or {})
    overMeta = dict(getattr(overCal, "meta", {}) or {})
    underMeta = dict(getattr(underCal, "meta", {}) or {})
    return {
        "gitSha": _gitSha(),
        "gitDirty": _gitDirty(),
        "trainedThrough": meta.get("train_last_date"),
        "calibratedThrough": overMeta.get("calibration_end_date"),
        "pointsBundleVersion": meta.get("points_bundle_version"),
        "overCalibratorVersion": overMeta.get("calibrator_algorithm_version"),
        "underCalibratorVersion": underMeta.get("under_calibrator_algorithm_version"),
        "targetMode": meta.get("target_mode"),
    }


def _betKey(gameID, playerID, side, line, slateDate):
    """
    Deterministic bet id, so re running the emitter is idempotent at the hub
    (INSERT OR IGNORE). Includes line on purpose: the same player at a moved
    line is a genuinely different bet. Excludes odds and timestamps, which
    drift between runs without changing the bet
    """
    raw = "|".join([
        MODEL_NAME, str(gameID), str(playerID), PROP_TYPE,
        str(side), f"{float(line):.1f}", str(slateDate),
    ])
    return "sha1:" + hashlib.sha1(raw.encode()).hexdigest()[:16]


def _toBet(c, slateDate):
    """Translate one scoreLiveDay candidate into the hub bet shape"""
    return {
        "betKey": _betKey(c["gameID"], c["playerID"], c["side"], c["line"], slateDate),
        "sport": SPORT,
        "market": "player_prop",
        "propType": PROP_TYPE,
        "gameRef": {
            "nativeId": c["gameID"],
            "gameDate": c["date"],
            "homeTeam": c["teamAbbr"] if c["isHome"] else c["oppAbbr"],
            "awayTeam": c["oppAbbr"] if c["isHome"] else c["teamAbbr"],
            # PropSnapshots carries no tip-off time, so the hub derives its
            # closing-line capture window from gameDate for now.
            "commenceTime": None,
        },
        "subject": {
            "name": c["player"],
            "nativeId": str(c["playerID"]),
            "team": c["teamAbbr"],
        },
        "selection": c["side"],
        "line": float(c["line"]),
        "odds": int(c["sideOdds"]),
        "book": c.get("bookmaker") or "draftkings",
        "stake": STAKE_UNITS,
        "modelProb": float(c["myProb"]),
        "fairProb": float(c["bookProb"]),
        "edge": float(c["edge"]),
        "predictedValue": float(c["predicted"]),
        "snapshotTime": None,   # set from the snapshot row if it gains a timestamp
        "reason": c.get("filterReason"),
        "metadata": {
            "rawProb": c.get("rawProb"),
            "predDiff": c.get("predDiff"),
            "avgPts10": c.get("avgPts10"),
            "isHome": c.get("isHome"),
            "oppTeam": c.get("oppAbbr"),
        },
    }


def buildPayload(date, dbPath=DB_PATH, edgeThresh=DEFAULT_EDGE_THRESH):
    """
    Score the slate and return the hub envelope

    Never raises for an empty slate as that is a legitimate no_slate result
    A genuine failure (missing bundles, unreadable db) propagates, and the
    caller writes a failed envelope so the hub can see the model broke
    rather than silently reporting nothing
    """
    from betting.liveScorer import scoreLiveDay
    from models.points import PointsBundle
    from betting.calibrator import Calibrator
    from betting.underCalibrator import UnderCalibrator

    candidates = scoreLiveDay(
        dbPath=dbPath, date=date,
        overThresh=edgeThresh, underThresh=edgeThresh,
        record=False, 
    )

    points = PointsBundle.loadIfExists()
    overCal = Calibrator.loadIfExists()
    underCal = UnderCalibrator.loadIfExists()

    bets, dropped = [], 0
    for c in candidates:
        # A candidate without a resolved game cannot be graded or correlated
        # by the hub, so drop it loudly rather than emit a half identified bet.
        if not c.get("gameID") or c.get("playerID") is None:
            dropped += 1
            continue
        bets.append(_toBet(c, date))

    status = "ok" if bets else "no_slate"
    if dropped:
        status = "degraded"

    return {
        "schemaVersion": SCHEMA_VERSION,
        "modelName": MODEL_NAME,
        "modelVersion": _modelVersion(points, overCal, underCal),
        "sport": SPORT,
        "slateDate": date,
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": status,
        "candidatesConsidered": len(candidates),
        "droppedUnresolved": dropped,
        "edgeThreshold": edgeThresh,
        "bets": bets,
    }


def writePayload(payload, outDir):
    """Write atomically (.tmp -> rename) so the hub never reads a partial file"""
    dest = Path(outDir) / payload["slateDate"]
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{MODEL_NAME}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description="Emit NBA points prop bets for the hub.")
    ap.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"),
                    help="slate date YYYY-MM-DD (default: today)")
    ap.add_argument("--dbPath", default=str(DB_PATH))
    ap.add_argument("--outDir", default=str(DEFAULT_INBOX))
    ap.add_argument("--edgeThresh", type=float, default=DEFAULT_EDGE_THRESH)
    ap.add_argument("--stdout", action="store_true", help="print payload, write nothing")
    args = ap.parse_args(argv)

    # scoreLiveDay and the cache loader print progress to stdout. Under --stdout
    # that would corrupt the JSON, so send their chatter to stderr and keep
    # stdout parseable (`emit --stdout | jq`).
    noise = contextlib.redirect_stdout(sys.stderr) if args.stdout \
        else contextlib.nullcontext()

    try:
        with noise:
            payload = buildPayload(args.date, dbPath=args.dbPath,
                                   edgeThresh=args.edgeThresh)
    except Exception as e:
        payload = {
            "schemaVersion": SCHEMA_VERSION,
            "modelName": MODEL_NAME,
            "modelVersion": {"gitSha": _gitSha(), "gitDirty": _gitDirty()},
            "sport": SPORT,
            "slateDate": args.date,
            "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": "failed",
            "error": f"{type(e).__name__}: {e}",
            "bets": [],
        }
        if not args.stdout:
            writePayload(payload, args.outDir)
        print(f"[emit] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    if args.stdout:
        print(json.dumps(payload, indent=2))
        return 0

    path = writePayload(payload, args.outDir)
    print(f"[emit] {payload['status']}: {len(payload['bets'])} bet(s) -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

