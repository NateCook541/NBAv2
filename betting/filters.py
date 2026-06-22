from dataclasses import dataclass, field, asdict
from typing import Tuple


@dataclass
class FilterSet:
    """
    Container for bet-acceptance filters applied inside _evaluateProp.

    Design intent
    -------------
    - Start with FilterSet.baseline() — every prop passes, giving a clean
      no-filter benchmark.
    - Add one filter at a time, compare against baseline in walk-forward
      results before keeping it.
    - Never tune filter thresholds on the same data you evaluate them on.

    Adding a new filter
    -------------------
    1. Add a field here with a sensible default that leaves it disabled.
    2. Add a branch in passes() that returns (False, "filterName") when it
       should block a bet.
    3. Add the field name to asDict() if it isn't already captured by
       dataclasses.asdict().
    """

    name: str = "default"

    # ------------------------------------------------------------------ #
    # Predicted score floor
    # Drops props where the model has very low confidence in the player
    # getting enough volume to matter.  Disabled by default.
    # ------------------------------------------------------------------ #
    minPredicted: float = 0.0

    # ------------------------------------------------------------------ #
    # Minimum model disagreement with the book line  (over side only)
    # e.g. overMinDisagreement=2.0 means only bet when
    # predicted >= line + 2.0
    # Disabled by default.
    # ------------------------------------------------------------------ #
    overMinDisagreement: float = 0.0

    # ------------------------------------------------------------------ #
    # Edge "donut hole"
    # Across all 5 walk-forward folds, the 8-11% edge band was negative
    # in 4/5 folds (n=91-142 per fold) while 5-8% and 11-15% were
    # positive in 4/5 folds each. This excludes a middle band of edges
    # while keeping both the low- and high-edge bands.
    # Disabled by default — set donutLow/donutHigh to enable.
    # e.g. donutLow=0.08, donutHigh=0.11 skips bets with
    # 0.08 <= edge < 0.11
    # ------------------------------------------------------------------ #
    donutLow: float = 0.0
    donutHigh: float = 0.0


    maxPredicted: float = 0.0

    # ------------------------------------------------------------------ #
    # PG / PF high-scorer filter
    # PG (pos=1) and PF (pos=4) in the 18-22 predicted range showed
    # consistently negative ROI across both seasons combined:
    #   PG 18-22: 36.9% WR / -$370 (n=122)
    #   PF 18-22: 44.9% WR / -$185 (n=118)
    # SG (52.9% / -$12), SF (58.2% / +$142), C (56.8% / +$67) are fine.
    # Set blockGuardHighScorer=True to enable.
    # ------------------------------------------------------------------ #
    blockGuardHighScorer: bool = False
    guardHighScorerMinPred: float = 18.0
    guardHighScorerMaxPred: float = 22.0

    # ------------------------------------------------------------------ #
    # Mid-range edge donut (15-18 predicted, 9-12% edge)
    # The 9-12% edge band within the 15-18 predicted bucket showed
    # consistently negative ROI across BOTH seasons:
    #   10-11%: 2024-25 -$54 (n=26), 2025-26 -$86 (n=39)
    #   11-12%: 2024-25 -$55 (n=15), 2025-26 -$43 (n=31)
    #    9-10%: 2024-25 -$76 (n=30), 2025-26 +$23 (n=37) — borderline but included
    # 12%+ is strongly positive in both seasons; <9% is mixed.
    # This is NOT a global donut — only applies when 15 <= predicted < 18.
    # Disabled by default — set midRangeDonutLow/High to enable.
    # ------------------------------------------------------------------ #
    midRangeDonutLow: float = 0.0
    midRangeDonutHigh: float = 0.0

    # ------------------------------------------------------------------ #
    # High-range low-edge donut (18-22 predicted, 7-9% edge)
    # The 7-9% edge band within the 18-22 predicted bucket showed
    # consistently negative ROI across BOTH seasons:
    #   7-8%: 2024-25 -$101 (n=38), 2025-26 -$185 (n=51) — combined -$286
    #   8-9%: 2024-25  -$59 (n=51), 2025-26 -$105 (n=46) — combined -$164
    # In this bucket the model needs stronger conviction (10%+) to be reliable.
    # This is NOT a global donut — only applies when 18 <= predicted < 22.
    # Disabled by default — set highRangeDonutLow/High to enable.
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # Serialisation helpers
    # ------------------------------------------------------------------ #

    def asDict(self) -> dict:
        """
        Returns a flat dict of all filter parameters (excluding name).
        Used by the reporter to append filter config to summary rows.
        """
        d = asdict(self)
        d.pop("name", None)
        return d

    # ------------------------------------------------------------------ #
    # Named constructors
    # ------------------------------------------------------------------ #

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
        identified as consistently weak across walk-forward folds.
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

