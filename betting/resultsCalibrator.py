"""
resultsCalibrator.py

Turns the game-totals point prediction (models/results.py, ~15.25 MAE) into a
calibrated P(total > line) for over/under betting.

Mirrors betting/calibrator.py's public interface (fit / probOver / rawProbOver /
save / load / loadIfExists / calibrationCheck / isSafeFor) so the betting and
backtest layers can treat it the same way, but the internals are tuned for game
totals rather than player points:

  * Lines are fit as OFFSETS around each prediction (pred + offset), not on fixed
    absolute totals. The model predicts in a narrow ~7pt band while book totals sit
    right next to the prediction, so this is where every real bet actually lives.
  * Sigma lives near ~19 (totals residual std) instead of ~8, with wider bounds.
  * A single global Platt scaler — unlike the points model there are no meaningful
    "low/mid/high total" tiers (every game total sits in a narrow band), so tiering
    would just split thin data into noise.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

from config import (
    RESULTS_CAL_PATH, RESULTS_CAL_OFFSETS, RESULTS_SIGMA_BOUNDS,
    RESULTS_PLATT_DAMPING,
)

RESULTS_CALIBRATOR_ALGORITHM_VERSION = 1


# Core probability helpers


def _estimateResidualStd(predictions, actuals):
    residuals = np.asarray(actuals) - np.asarray(predictions)
    q75, q25 = np.percentile(residuals, [75, 25])
    iqrStd = (q75 - q25) / 1.349
    plainStd = float(np.std(residuals))
    return max(iqrStd, plainStd)


def _fitSplitSigma(predictions, actuals):
    """
    Split-normal sigma: separate spread below (left) and above (right) the
    prediction, so mild skew in the totals distribution is captured. Totals
    residuals are near-symmetric so these come out close, but keeping the split
    matches the points calibrator and costs nothing.
    """
    residuals = np.asarray(actuals) - np.asarray(predictions)
    left = residuals[residuals <= 0]
    right = residuals[residuals > 0]

    sigmaLeft = float(np.sqrt(np.mean(left ** 2))) if len(left) > 10 else 19.0
    sigmaRight = float(np.sqrt(np.mean(right ** 2))) if len(right) > 10 else 19.0

    sigmaLeft = float(np.clip(sigmaLeft, *RESULTS_SIGMA_BOUNDS))
    sigmaRight = float(np.clip(sigmaRight, *RESULTS_SIGMA_BOUNDS))

    print(f"[ResultsCal] split-normal: sigma_left={sigmaLeft:.3f}, sigma_right={sigmaRight:.3f}")
    print(f"[ResultsCal] left tail n={len(left)}, right tail n={len(right)}")
    return sigmaLeft, sigmaRight


def _probOverSplitNormal(predicted, line, sigmaLeft, sigmaRight):
    # P(actual > line) under a split-normal centred on the prediction.
    delta = line - predicted
    if delta <= 0:
        return float(norm.sf(delta / sigmaLeft))
    return float(norm.sf(delta / sigmaRight))


def _buildPlattDataOffsets(predictions, actuals, offsets, sigmaLeft, sigmaRight):
    """
    Build (rawProb, hit) pairs across offset lines around each prediction.

    For every offset o, the line for game i is pred_i + o, and the hit is
    (actual_i > pred_i + o). This trains the Platt scaler on exactly the
    prediction-relative geometry the model will see at bet time.
    """
    predictions = np.asarray(predictions, dtype=float)
    actuals = np.asarray(actuals, dtype=float)
    rawProbsAll, hitsAll = [], []
    for o in offsets:
        lines = predictions + o
        raw = np.array([
            _probOverSplitNormal(p, ln, sigmaLeft, sigmaRight)
            for p, ln in zip(predictions, lines)
        ])
        hit = (actuals > lines).astype(float)
        rawProbsAll.append(raw)
        hitsAll.append(hit)
    return np.concatenate(rawProbsAll), np.concatenate(hitsAll)


class ResultsCalibrator:
    def __init__(self, platt, sigmaLeft, sigmaRight, residualStd, meta, plattDamping=None):
        self.platt = platt                # single global LogisticRegression
        self.sigmaLeft = sigmaLeft
        self.sigmaRight = sigmaRight
        self.residualStd = residualStd
        self.meta = meta
        self.plattDamping = RESULTS_PLATT_DAMPING if plattDamping is None else plattDamping


    # Core probability methods


    def probOver(self, predicted, line):
        """Calibrated P(total > line) given the model's point prediction."""
        raw = _probOverSplitNormal(predicted, line, self.sigmaLeft, self.sigmaRight)
        calibrated = float(self.platt.predict_proba([[raw]])[0, 1])
        return raw + self.plattDamping * (calibrated - raw)

    def probUnder(self, predicted, line):
        return 1.0 - self.probOver(predicted, line)

    def rawProbOver(self, predicted, line):
        return _probOverSplitNormal(predicted, line, self.sigmaLeft, self.sigmaRight)


    # Convenience


    @property
    def profitableEdgeCap(self):
        return float(self.meta.get("profitable_edge_cap", 0.15))


    # Diagnostics


    def printExamples(self, predMean=228.0, offsets=None):
        if offsets is None:
            offsets = [-10, -6, -3, 0, 3, 6, 10]
        print(f"\n[ResultsCal] sigma_left={self.sigmaLeft:.3f}  sigma_right={self.sigmaRight:.3f}")
        print(f"[ResultsCal] examples around pred={predMean:.1f}")
        print(f"  {'Line':>7}  {'Offset':>7}  {'Raw':>8}  {'Calibrated':>12}")
        for o in offsets:
            line = predMean + o
            raw = self.rawProbOver(predMean, line)
            cal = self.probOver(predMean, line)
            print(f"  {line:>7.1f}  {o:>+7}  {raw:>8.3f}  {cal:>12.3f}")

    def calibrationCheck(self, predictions, actuals, offsets=None):
        """
        Reliability table: bucket calibrated probs and compare to realised hit
        rate, across offset lines around each prediction.
        """
        if offsets is None:
            offsets = RESULTS_CAL_OFFSETS
        predictions = np.asarray(
            predictions.values if hasattr(predictions, "values") else predictions,
            dtype=float,
        )
        actuals = np.asarray(
            actuals.values if hasattr(actuals, "values") else actuals,
            dtype=float,
        )

        # Collect (calibrated prob, hit) across all games and offsets.
        probsAll, hitsAll = [], []
        for o in offsets:
            lines = predictions + o
            probs = np.array([self.probOver(p, ln) for p, ln in zip(predictions, lines)])
            hits = (actuals > lines).astype(float)
            probsAll.append(probs)
            hitsAll.append(hits)
        probs = np.concatenate(probsAll)
        hits = np.concatenate(hitsAll)

        bins = np.linspace(0, 1, 11)
        results = []
        for i in range(len(bins) - 1):
            mask = (probs >= bins[i]) & (probs < bins[i + 1])
            if mask.sum() < 20:
                continue
            results.append({
                "prob_bin": f"{bins[i]:.1f}-{bins[i+1]:.1f}",
                "predicted prob": float(probs[mask].mean()),
                "actual rate": float(hits[mask].mean()),
                "n": int(mask.sum()),
            })
        return pd.DataFrame(results)

    def brierScore(self, predictions, actuals, offsets=None):
        """Mean Brier score of calibrated P(over) across offset lines. Lower is better."""
        if offsets is None:
            offsets = RESULTS_CAL_OFFSETS
        predictions = np.asarray(
            predictions.values if hasattr(predictions, "values") else predictions,
            dtype=float,
        )
        actuals = np.asarray(
            actuals.values if hasattr(actuals, "values") else actuals,
            dtype=float,
        )
        probsAll, hitsAll = [], []
        for o in offsets:
            lines = predictions + o
            probs = np.array([self.probOver(p, ln) for p, ln in zip(predictions, lines)])
            hits = (actuals > lines).astype(float)
            probsAll.append(probs)
            hitsAll.append(hits)
        probs = np.concatenate(probsAll)
        hits = np.concatenate(hitsAll)
        return float(np.mean((probs - hits) ** 2))


    # Persistence


    def save(self, path=RESULTS_CAL_PATH):
        bundle = {
            "platt": self.platt,
            "sigma left": self.sigmaLeft,
            "sigma right": self.sigmaRight,
            "residual std": self.residualStd,
            "platt damping": self.plattDamping,
            **self.meta,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, path)
        print(f"[ResultsCal] Saved -> {path}")

    @classmethod
    def load(cls, path=RESULTS_CAL_PATH):
        if not Path(path).exists():
            raise FileNotFoundError(f"No results calibrator at {path}")
        bundle = joblib.load(path)
        return cls(
            platt=bundle["platt"],
            sigmaLeft=bundle["sigma left"],
            sigmaRight=bundle["sigma right"],
            residualStd=bundle.get("residual std"),
            meta={
                k: v for k, v in bundle.items()
                if k not in ("platt", "sigma left", "sigma right", "residual std", "platt damping")
            },
            plattDamping=bundle.get("platt damping"),
        )

    @classmethod
    def loadIfExists(cls, path=RESULTS_CAL_PATH):
        if not Path(path).exists():
            return None
        return cls.load(path)


    # Fitting


    @classmethod
    def fit(cls, predictions, actuals, savePath=None, metadata=None):
        """
        Fit a totals calibrator from held-out predictions and actual totals.

        1. Estimate residual std.
        2. Fit split-normal sigmas.
        3. Build (rawProb, hit) pairs on offset lines around each prediction.
        4. Fit a single global Platt scaler.
        5. Report compression / set a profitable edge cap.
        6. Optional save.
        """
        predictions = np.asarray(
            predictions.values if hasattr(predictions, "values") else predictions,
            dtype=float,
        )
        actuals = np.asarray(
            actuals.values if hasattr(actuals, "values") else actuals,
            dtype=float,
        )

        # 1. Residual std
        residualStd = _estimateResidualStd(predictions, actuals)
        meanPred, meanAct = float(predictions.mean()), float(actuals.mean())
        print(f"[ResultsCal] n={len(predictions)}  mean pred={meanPred:.1f}  "
              f"mean actual={meanAct:.1f}  residual std={residualStd:.3f}")

        # 2. Split-normal sigmas
        sigmaLeft, sigmaRight = _fitSplitSigma(predictions, actuals)

        # 3. Platt data on offset lines
        rawProbs, hits = _buildPlattDataOffsets(
            predictions, actuals, RESULTS_CAL_OFFSETS, sigmaLeft, sigmaRight
        )
        print(f"[ResultsCal] Platt training pairs: {len(rawProbs)} "
              f"(across {len(RESULTS_CAL_OFFSETS)} offset lines)")

        # 4. Single global Platt scaler
        platt = LogisticRegression(solver="lbfgs", max_iter=2000)
        platt.fit(rawProbs.reshape(-1, 1), hits)

        # 5. Compression check + edge cap
        calAt55 = float(platt.predict_proba([[0.55]])[0, 1])
        calAt60 = float(platt.predict_proba([[0.60]])[0, 1])
        calAt70 = float(platt.predict_proba([[0.70]])[0, 1])
        span = calAt70 - calAt55
        print(f"[ResultsCal] Platt output: @55%={calAt55:.3f}, @60%={calAt60:.3f}, @70%={calAt70:.3f}")
        print(f"[ResultsCal] Compression 55->70 span={span:.3f}")

        if span < 0.05:
            profitableEdgeCap = 0.08
            print(f"[ResultsCal] Severe compression — edge cap {profitableEdgeCap:.0%}")
        elif span < 0.10:
            profitableEdgeCap = 0.10
            print(f"[ResultsCal] Moderate compression — edge cap {profitableEdgeCap:.0%}")
        else:
            profitableEdgeCap = 0.15
            print(f"[ResultsCal] Compression acceptable — edge cap {profitableEdgeCap:.0%}")

        instance = cls(
            platt=platt,
            sigmaLeft=sigmaLeft,
            sigmaRight=sigmaRight,
            residualStd=residualStd,
            meta={
                **(metadata or {}),
                "results_calibrator_algorithm_version": RESULTS_CALIBRATOR_ALGORITHM_VERSION,
                "profitable_edge_cap": profitableEdgeCap,
                "mean_pred": meanPred,
                "platt_55": calAt55,
                "platt_60": calAt60,
                "platt_70": calAt70,
            },
            plattDamping=RESULTS_PLATT_DAMPING,
        )
        instance.printExamples(predMean=meanPred)

        # 6. Optional save
        if savePath:
            instance.save(savePath)
        return instance


    # Backtest safety


    def isSafeFor(self, backtestStartDate):
        if int(self.meta.get("results_calibrator_algorithm_version", 0)) < RESULTS_CALIBRATOR_ALGORITHM_VERSION:
            return False
        end = self.meta.get("calibration_end_date", "")
        return bool(end) and str(end) < str(backtestStartDate)
