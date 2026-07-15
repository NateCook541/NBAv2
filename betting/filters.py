from dataclasses import dataclass, field, asdict
from typing import Tuple

@dataclass
class FilterSet:
    """
    Container for bet acceptance filters applied inside _evaluateProp
    """

    name: str = "default"

    # Predicted score floor
    # Drops props where the model has very low confidence in the player
    # getting enough volume to matter. Disabled by default.
    minPredicted: float = 0.0

    # Minimum model disagreement with the book line (over side only)
    # Disabled by default.
    overMinDisagreement: float = 0.0

    # Edge donut hole filter
    # This excludes a middle band of edges while keeping both the low and high edge bands
    # Disabled by default
    donutLow: float = 0.0
    donutHigh: float = 0.0

    maxPredicted: float = 0.0

    # PG / PF high scorer filter
    # Set blockGuardHighScorer=True to enable
    blockGuardHighScorer: bool = False
    guardHighScorerMinPred: float = 18.0
    guardHighScorerMaxPred: float = 22.0

    # Mid range edge donut (15-18 predicted, 9-12% edge)
    # This is NOT a global donut it only applies when 15 <= predicted < 18
    # Disabled by default, set midRangeDonutLow/High to enable
    midRangeDonutLow: float = 0.0
    midRangeDonutHigh: float = 0.0

    # High range low edge donut (18-22 predicted, 7-9% edge)
    # In this bucket the model needs stronger conviction (10%+) to be reliable.
    # This is NOT a global donut it only applies when 18 <= predicted < 22
    # Disabled by default, set highRangeDonutLow/High to enable.
    highRangeDonutLow: float = 0.0
    highRangeDonutHigh: float = 0.0

    def passes(self, predicted: float, propLine: float, edge: float = None,
               pos: float = None) -> tuple[bool, str]:
        """
        Returns (passes: bool, reason: str).
        reason is an empty string when the prop passes.

        edge is optional so existing callers that don't pass it still work
        for the minPredicted / overMinDisagreement filters. The donut hole
        filter is a no-op if edge is not provided.
        """

        if self.minPredicted > 0 and predicted < self.minPredicted:
            return False, "minPredicted"

        if self.maxPredicted > 0 and predicted >= self.maxPredicted:
            return False, "maxPredicted"

        if self.overMinDisagreement > 0:
            if (predicted - propLine) < self.overMinDisagreement:
                return False, "overMinDisagreement"

        if edge is not None and self.donutHigh > self.donutLow:
            if self.donutLow <= edge < self.donutHigh:
                return False, "edgeDonutHole"

        if self.blockGuardHighScorer and pos is not None:
            if (pos in (1.0, 1.5, 4.0, 4.5) and
                    self.guardHighScorerMinPred <= predicted < self.guardHighScorerMaxPred):
                return False, "pgPfHighScorer"

        if edge is not None and self.midRangeDonutHigh > self.midRangeDonutLow:
            if 15.0 <= predicted < 18.0 and self.midRangeDonutLow <= edge < self.midRangeDonutHigh:
                return False, "midRangeEdgeDonut"

        if edge is not None and self.highRangeDonutHigh > self.highRangeDonutLow:
            if 18.0 <= predicted < 22.0 and self.highRangeDonutLow <= edge < self.highRangeDonutHigh:
                return False, "highRangeEdgeDonut"

        return True, ""


    # Serialisation helpers


    def asDict(self) -> dict:
        """
        Returns a flat dict of all filter parameters (excluding name).
        Used by the reporter to append filter config to summary rows.
        """
        d = asdict(self)
        d.pop("name", None)
        return d


    # Named constructors


    @classmethod
    def baseline(cls) -> "FilterSet":
        """
        Pass-through FilterSet.  Nothing is filtered beyond what the
        backtest engine already enforces (line sanity, feature availability
        etc).  Every other FilterSet should be compared against this.
        """
        return cls(name="baseline")

    @classmethod
    def edgeDonutHole(cls, low: float = 0.08, high: float = 0.11) -> "FilterSet":
        """
        Skips bets with edge in [low, high). Defaults to the 8-11% band
        identified as consistently weak across walk forward folds 
        """
        return cls(name="edgeDonutHole", donutLow=low, donutHigh=high)

    @classmethod
    def under22(cls):
        return cls(name="under22", maxPredicted=22.0)

    @classmethod
    def pgPfHighScorerBlock(cls) -> "FilterSet":
        """
        Blocks PG (pos=1/1.5) and PF (pos=4/4.5) props when the model
        predicts 18-22 pts. Both positions showed consistently negative ROI
        in this bucket across both seasons; SG/SF/C are unaffected.
        """
        return cls(name="pgPfHighScorerBlock", blockGuardHighScorer=True)

    @classmethod
    def production(cls) -> "FilterSet":
        """
        The tuned and validated OVER filter set used in the backtest/orchestrator
        and in live scoring
          - maxPredicted 22.0 (cap high scorer bets)
          - PG/PF 18-22 block (blockGuardHighScorer)
          - midRangeDonut 9-12% (15-18 pred + 9-12% edge)
          - highRangeDonut 7-9% (18-22 pred + 7-9% edge)
        """
        return cls(
            name="pgPfBlock_midDonut_highDonut",
            maxPredicted=22.0,
            blockGuardHighScorer=True,
            midRangeDonutLow=0.09,
            midRangeDonutHigh=0.12,
            highRangeDonutLow=0.07,
            highRangeDonutHigh=0.09,
        )


@dataclass
class UnderFilterSet:
    """
    Container for under bet acceptance filters applied inside UnderBacktestEngine

    Mirrors FilterSet but targets the under side. Differences:
    - underMinDisagreement: blocks if (line - predicted) < threshold (only
      bet under when the model thinks the player will fall well short of the line)
    - No blockGuardHighScorer as that was over specfic
    - Donut fields are bucket specific. Each pred range has its own donut
      so bad sub bands can be removed without affecting adjacent good bands
    """

    name: str = "under_default"

    # Predicted score floor / ceiling
    minPredicted: float = 0.0
    maxPredicted: float = 0.0

    # Minimum model disagreement with the book line (under side)
    # Disabled by default.
    underMinDisagreement: float = 0.0

    # Global edge floor. Blocks all bets below this edge regardless of predicted bucket 
    # The 5-7% band is a net loser across nearly all
    # pred buckets. Set donutLow=0.05, donutHigh=0.07 to raise effective
    # edge threshold to 7% without changing DEFAULT_EDGE_THRESH globally.
    # Disabled by default.
    donutLow: float = 0.0
    donutHigh: float = 0.0

    # Low scorer range donut (12-15 predicted, 7-11% edge)
    # Set midLowRangeDonutLow=0.07, midLowRangeDonutHigh=0.11 to enable.
    midLowRangeDonutLow: float = 0.0
    midLowRangeDonutHigh: float = 0.0

    # Mid range low edge donut (15-18 predicted, 5-7% edge)
    # Set midRangeLowEdgeDonutLow=0.05, midRangeLowEdgeDonutHigh=0.07 to enable.
    midRangeLowEdgeDonutLow: float = 0.0
    midRangeLowEdgeDonutHigh: float = 0.0

    # Mid range mid edge donut (15-18 predicted, 9-11% edge)
    # Set midRangeMidEdgeDonutLow=0.09, midRangeMidEdgeDonutHigh=0.11 to enable.
    midRangeMidEdgeDonutLow: float = 0.0
    midRangeMidEdgeDonutHigh: float = 0.0

    # Near even odds block
    # Under odds > maxUnderOdds (e.g. -105) are near even lines. 
    # Disabled by default (0.0 = no cap).
    maxUnderOdds: float = 0.0

    # High predDiff + low predicted block (predDiff 2.5-3.0, predicted < 18)
    # Set blockHighDiffLowPred=True to enable.
    blockHighDiffLowPred: bool = False
    highDiffLowPredMin: float = 2.5
    highDiffLowPredMax: float = 3.0
    highDiffLowPredCeiling: float = 18.0

    def passes(self, predicted, propLine, edge=None,
               pos=None, betOdds=None, predDiff=None):
        """
        Returns (passes: bool, reason: str)
        reason is an empty string when the prop passes
        """
        if self.minPredicted > 0 and predicted < self.minPredicted:
            return False, "minPredicted"

        if self.maxPredicted > 0 and predicted >= self.maxPredicted:
            return False, "maxPredicted"

        if self.underMinDisagreement > 0:
            if (propLine - predicted) < self.underMinDisagreement:
                return False, "underMinDisagreement"

        if edge is not None and self.donutHigh > self.donutLow:
            if self.donutLow <= edge < self.donutHigh:
                return False, "edgeDonutHole"

        if edge is not None and self.midLowRangeDonutHigh > self.midLowRangeDonutLow:
            if 12.0 <= predicted < 15.0 and self.midLowRangeDonutLow <= edge < self.midLowRangeDonutHigh:
                return False, "midLowRangeDonut"

        if edge is not None and self.midRangeLowEdgeDonutHigh > self.midRangeLowEdgeDonutLow:
            if 15.0 <= predicted < 18.0 and self.midRangeLowEdgeDonutLow <= edge < self.midRangeLowEdgeDonutHigh:
                return False, "midRangeLowEdgeDonut"

        if edge is not None and self.midRangeMidEdgeDonutHigh > self.midRangeMidEdgeDonutLow:
            if 15.0 <= predicted < 18.0 and self.midRangeMidEdgeDonutLow <= edge < self.midRangeMidEdgeDonutHigh:
                return False, "midRangeMidEdgeDonut"

        if self.maxUnderOdds != 0.0 and betOdds is not None:
            if betOdds > self.maxUnderOdds:
                return False, "nearEvenOdds"

        if self.blockHighDiffLowPred and predDiff is not None:
            if (self.highDiffLowPredMin <= predDiff < self.highDiffLowPredMax
                    and predicted < self.highDiffLowPredCeiling):
                return False, "highDiffLowPred"

        return True, ""

    def asDict(self) -> dict:
        d = asdict(self)
        d.pop("name", None)
        return d

    @classmethod
    def baseline(cls) -> "UnderFilterSet":
        """Pass through — no filters active. Every other UnderFilterSet should be compared against this."""
        return cls(name="under_baseline")

    @classmethod
    def production(cls) -> "UnderFilterSet":
        """
        Active production filter set derived from cross season flat stake analysis
        All filters confirmed consistent across both 2024-25 and 2025-26 seasons

        Edge/bucket filters:
          1. Global 5-7% edge block
          2. 12-15 pred / 7-11% edge
          3. 15-18 pred / 5-7% edge
          4. 15-18 pred / 9-11% edge

        Model disagreement filters:
          5. predDiff < 1.0
          6. predDiff 2.5-3.0 + predicted < 18
          7. Odds > -105
        """
        return cls(
            name="under_production",
            donutLow=0.05,
            donutHigh=0.07,
            midLowRangeDonutLow=0.07,
            midLowRangeDonutHigh=0.11,
            midRangeLowEdgeDonutLow=0.05,
            midRangeLowEdgeDonutHigh=0.07,
            midRangeMidEdgeDonutLow=0.09,
            midRangeMidEdgeDonutHigh=0.11,
            underMinDisagreement=1.0,
            maxUnderOdds=-105,
            blockHighDiffLowPred=True,
        )

