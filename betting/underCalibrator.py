import joblib
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression

from config import (
    UNDER_CALIBRATOR_PATH, PLATT_FIT_LINES, SIGMA_BOUNDS, PLATT_DAMPING
)
from betting.calibrator import (
    _fitSplitSigma, _probOverSplitNormal, _estimateResidualStd
)

UNDER_CALIBRATOR_ALGORITHM_VERSION = 1


def _buildUnderPlattData(predictions, actuals, lines, sigmaLeft, sigmaRight):
    """
    Builds (rawUnderProb, underHit) pairs across all given lines.
    rawUnderProb = 1 - P(actual > line)
    underHit = 1 if actual < line (strict under)
    """
    actuals = np.asarray(actuals)
    rawProbsAll, hitsAll = [], []
    for line in lines:
        raw = np.array([
            1.0 - _probOverSplitNormal(p, line, sigmaLeft, sigmaRight)
            for p in predictions
        ])
        hit = (actuals < line).astype(float)
        rawProbsAll.append(raw)
        hitsAll.append(hit)
    return np.concatenate(rawProbsAll), np.concatenate(hitsAll)


class UnderCalibrator:
    """
    Calibrates P(actual < line) for under bets

    Architecture mirrors the over calibrator but is fit and stored independently 

    The two calibrators share the same PointsBundle
    predictions and the same split normal distribution parameters, but
    their Platt layers are trained on opposite hit labels

    Only probUnder() should be called by the backtest layer
    """

    def __init__(self, platt, sigmaLeft, sigmaRight, residualStd, meta, plattDamping):
        self.platt = platt
        self.sigmaLeft = sigmaLeft
        self.sigmaRight = sigmaRight
        self.residualStd = residualStd
        self.meta = meta
        self.plattDamping = PLATT_DAMPING if plattDamping is None else plattDamping


    # Core probability methods


    def probUnder(self, predicted, line):
        """
        Returns calibrated P(actual < line)
        Only method the under backtest layer should call
        """
        raw = 1.0 - _probOverSplitNormal(predicted, line, self.sigmaLeft, self.sigmaRight)
        if predicted < 15.0:
            scaler = self.platt["low"]
        elif predicted < 22.0:
            scaler = self.platt["mid"]
        else:
            scaler = self.platt["high"]
        calibrated = float(scaler.predict_proba([[raw]])[0, 1])
        return raw + self.plattDamping * (calibrated - raw)

    def rawProbUnder(self, predicted, line):
        """Uncalibrated under probability (for debugging)"""
        return 1.0 - _probOverSplitNormal(predicted, line, self.sigmaLeft, self.sigmaRight)

    @property
    def profitableEdgeCap(self):
        return float(self.meta.get("profitable_edge_cap", 0.20))

    # Persistence

    def save(self, path=UNDER_CALIBRATOR_PATH):
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
        print(f"[UnderCalibrator] Saved -> {path}")

    @classmethod
    def load(cls, path=UNDER_CALIBRATOR_PATH):
        if not Path(path).exists():
            raise FileNotFoundError(f"No under calibrator at {path}")
        bundle = joblib.load(path)
        platt = bundle["platt"]
        if not isinstance(platt, dict):
            platt = {"low": platt, "mid": platt, "high": platt, "global": platt}
        return cls(
            platt=platt,
            sigmaLeft=bundle["sigma left"],
            sigmaRight=bundle["sigma right"],
            residualStd=bundle.get("residual std"),
            meta={k: v for k, v in bundle.items()
                  if k not in ("platt", "sigma left", "sigma right", "residual std")},
            plattDamping=bundle.get("platt damping"),
        )

    @classmethod
    def loadIfExists(cls, path=UNDER_CALIBRATOR_PATH):
        if not Path(path).exists():
            return None
        return cls.load(path)

    # --- Fitting ---

    @classmethod
    def fit(cls, predictions, actuals, savePath=None, metadata=None):
        """
        Fits the under calibrator from holdout predictions and actuals.

        Steps:
        1. Estimate residual std
        2. Print under hit rate diagnostics
        3. Fit split-normal sigmas (same as over — shared distribution)
        4. Build (rawUnderProb, underHit) Platt training data
        5. Fit tiered Platt scalers on under hits
        6. Compression check -> set profitable_edge_cap
        7. Optional save
        """
        predictions = np.asarray(predictions, dtype=float)
        actuals = np.asarray(
            actuals.values if hasattr(actuals, "values") else actuals, dtype=float
        )

        # 1. Residual std
        residualStd = _estimateResidualStd(predictions, actuals)
        print(f"[UnderCalibrator] residual std = {residualStd:.3f}")

        # 2. Under hit rate diagnostics
        meanPred = float(predictions.mean())
        meanAct  = float(actuals.mean())
        print(f"\n[UnderCalibrator] Under hit rates (mean pred={meanPred:.1f}, mean actual={meanAct:.1f}):")
        print(f"  {'Line':>6}  {'Actual under rate':>18}  {'Model < line':>14}")
        for line in [10, 15, 20, 25, 30]:
            actRate  = float(np.mean(actuals < line))
            predRate = float(np.mean(predictions < line))
            print(f"  {line:>6}  {actRate:>18.3f}  {predRate:>14.3f}")

        # 3. Fit split normal sigmas (same distribution as over calibrator)
        sigmaLeft, sigmaRight = _fitSplitSigma(predictions, actuals)

        # 4. Build Platt training data with under hit labels
        plattLines = [line for line in PLATT_FIT_LINES if 5 <= line <= 35]
        rawProbs, hits = _buildUnderPlattData(
            predictions, actuals, plattLines, sigmaLeft, sigmaRight
        )

        # 5. Tiered Platt scaling on under hits
        predArray = np.asarray(predictions, dtype=float)
        lowMask   = predArray < 15.0
        midMask   = (predArray >= 15.0) & (predArray < 22.0)
        highMask  = predArray >= 22.0

        print(
            f"[UnderCalibrator] Tiered Platt split: "
            f"low(n={lowMask.sum()}) mid(n={midMask.sum()}) high(n={highMask.sum()})"
        )

        def _fitTierPlatt(tierMask, tierLabel):
            if tierMask.sum() < 30:
                print(f"[UnderCalibrator] {tierLabel} tier too small ({tierMask.sum()}), using global fit")
                return None
            tierRawProbs, tierHits = _buildUnderPlattData(
                predictions[tierMask], actuals[tierMask], plattLines, sigmaLeft, sigmaRight
            )
            lr = LogisticRegression(solver="lbfgs", max_iter=2000)
            lr.fit(tierRawProbs.reshape(-1, 1), tierHits)
            cal60 = float(lr.predict_proba([[0.60]])[0, 1])
            cal80 = float(lr.predict_proba([[0.80]])[0, 1])
            print(f"[UnderCalibrator] {tierLabel} Platt: @60%={cal60:.3f}, @80%={cal80:.3f}")
            return lr

        plattLow    = _fitTierPlatt(lowMask,  "low (<15)")
        plattMid    = _fitTierPlatt(midMask,  "mid (15-22)")
        plattHigh   = _fitTierPlatt(highMask, "high (22+)")

        plattGlobal = LogisticRegression(solver="lbfgs", max_iter=2000)
        plattGlobal.fit(rawProbs.reshape(-1, 1), hits)

        platt = {
            "low":    plattLow    or plattGlobal,
            "mid":    plattMid    or plattGlobal,
            "high":   plattHigh   or plattGlobal,
            "global": plattGlobal,
        }

        # 6. Compression check
        calAt60 = float(platt["global"].predict_proba([[0.60]])[0, 1])
        calAt80 = float(platt["global"].predict_proba([[0.80]])[0, 1])
        calAt90 = float(platt["global"].predict_proba([[0.90]])[0, 1])
        highCompression = calAt90 - calAt60
        midCompression  = calAt80 - calAt60

        print(
            f"[UnderCalibrator] Platt output: "
            f"@60%={calAt60:.3f}, @80%={calAt80:.3f}, @90%={calAt90:.3f}"
        )
        print(
            f"[UnderCalibrator] Compression: "
            f"60→90 span={highCompression:.3f}, 60→80 span={midCompression:.3f}"
        )

        if highCompression < 0.10:
            profitableEdgeCap = 0.10
            print(f"[UnderCalibrator] Severe compression — capping profitable edge at {profitableEdgeCap:.0%}")
        elif midCompression < 0.08:
            profitableEdgeCap = 0.12
            print(f"[UnderCalibrator] Moderate compression — capping profitable edge at {profitableEdgeCap:.0%}")
        else:
            profitableEdgeCap = 0.18
            print(f"[UnderCalibrator] Compression acceptable — profitable edge cap={profitableEdgeCap:.0%}")

        profitableEdgeCap = min(profitableEdgeCap, 0.15)
        print(f"[UnderCalibrator] Final enforced edge cap={profitableEdgeCap:.0%}")

        instance = cls(
            platt=platt,
            sigmaLeft=sigmaLeft,
            sigmaRight=sigmaRight,
            residualStd=residualStd,
            meta={
                **(metadata or {}),
                "under_calibrator_algorithm_version": UNDER_CALIBRATOR_ALGORITHM_VERSION,
                "profitable_edge_cap": profitableEdgeCap,
                "mean_pred": meanPred,
                "platt_60": calAt60,
                "platt_80": calAt80,
                "platt_90": calAt90,
            },
            plattDamping=PLATT_DAMPING,
        )

        if savePath:
            instance.save(savePath)
        return instance

    # Backtest safety

    def isSafeFor(self, backtestStartDate):
        if int(self.meta.get("under_calibrator_algorithm_version", 0)) < UNDER_CALIBRATOR_ALGORITHM_VERSION:
            return False
        end = self.meta.get("calibration_end_date", "")
        return bool(end) and end < backtestStartDate
