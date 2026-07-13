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


@dataclass
class UnderFilterSet:
    """
    Container for under-bet acceptance filters applied inside UnderBacktestEngine.

    Mirrors FilterSet but targets the under side. Key differences:
    - underMinDisagreement: blocks if (line - predicted) < threshold, i.e. only
      bet under when the model thinks the player will fall well short of the line.
    - No blockGuardHighScorer — that was over-specific. Add positional blocks
      only after under diagnostics reveal equivalent patterns.
    - Donut fields are bucket-specific — each pred range has its own donut
      so bad sub-bands can be removed without affecting adjacent good bands.

    Design intent mirrors FilterSet: start with baseline(), add one filter at a
    time, always compare against baseline in walk-forward before keeping it.

    Confirmed bad buckets (both seasons, flat stake analysis):
      - 15-18 pred / 5-7% edge:    42% WR, -$361 combined  -> midRangeLowEdgeDonutLow/High
      - 12-15 pred / 7-11% edge:   47-48% WR, -$320 combined -> midLowRangeDonutLow/High
      - global 5-7% edge:          net loser across most pred buckets -> donutLow/High
      - predDiff < 1.0:            29.8% WR, -$193 — both seasons, model/book agree, nothing to exploit
      - predDiff 2.5-3.0 + pred<18: 29.3% WR, -$196 — book knows something model doesn't
      - odds > -105:               48% WR, -$92 — near-even lines, both seasons
    Confirmed good buckets (keep unfiltered):
      - 18-22 pred / 13-15% edge: 65% WR, +$218 combined
      - 12-15 pred / 11-13% edge: 60% WR, +$149 combined
      - <12   pred / 11-13% edge: 56% WR, +$191 combined
    """

    name: str = "under_default"

    # ------------------------------------------------------------------ #
    # Predicted score floor / ceiling
    # ------------------------------------------------------------------ #
    minPredicted: float = 0.0
    maxPredicted: float = 0.0

    # ------------------------------------------------------------------ #
    # Minimum model disagreement with the book line (under side)
    # e.g. underMinDisagreement=2.0 means only bet under when
    # line >= predicted + 2.0  (book line is at least 2 pts above model)
    # Disabled by default.
    # ------------------------------------------------------------------ #
    underMinDisagreement: float = 0.0

    # ------------------------------------------------------------------ #
    # Global edge floor — blocks all bets below this edge regardless of
    # predicted bucket. The 5-7% band is a net loser across nearly all
    # pred buckets. Set donutLow=0.05, donutHigh=0.07 to raise effective
    # edge threshold to 7% without changing DEFAULT_EDGE_THRESH globally.
    # Disabled by default.
    # ------------------------------------------------------------------ #
    donutLow: float = 0.0
    donutHigh: float = 0.0

    # ------------------------------------------------------------------ #
    # Low-scorer range donut (12-15 predicted, 7-11% edge)
    # 12-15 / 7-9%:  47.7% WR, -$210 — bad in BOTH seasons
    # 12-15 / 9-11%: 48.4% WR, -$110 — bad in BOTH seasons
    # 12-15 / 11-13% is +$149 and must not be blocked.
    # Set midLowRangeDonutLow=0.07, midLowRangeDonutHigh=0.11 to enable.
    # ------------------------------------------------------------------ #
    midLowRangeDonutLow: float = 0.0
    midLowRangeDonutHigh: float = 0.0

    # ------------------------------------------------------------------ #
    # Mid-range low-edge donut (15-18 predicted, 5-7% edge)
    # 15-18 / 5-7%: 42.3% WR, -$361 — worst bucket, consistent both seasons
    # 15-18 / 7-9% is +$66 and must not be blocked.
    # Set midRangeLowEdgeDonutLow=0.05, midRangeLowEdgeDonutHigh=0.07 to enable.
    # ------------------------------------------------------------------ #
    midRangeLowEdgeDonutLow: float = 0.0
    midRangeLowEdgeDonutHigh: float = 0.0

    # ------------------------------------------------------------------ #
    # Mid-range mid-edge donut (15-18 predicted, 9-11% edge)
    # 15-18 / 9-11%: 48.6% WR, -$123 — bad in both seasons
    # 15-18 / 7-9% (+$66) and 15-18 / 11-13% (+$89) are both fine.
    # Set midRangeMidEdgeDonutLow=0.09, midRangeMidEdgeDonutHigh=0.11 to enable.
    # ------------------------------------------------------------------ #
    midRangeMidEdgeDonutLow: float = 0.0
    midRangeMidEdgeDonutHigh: float = 0.0

    # ------------------------------------------------------------------ #
    # Near-even odds block
    # Under odds > maxUnderOdds (e.g. -105) are near-even lines. The book
    # has minimal conviction on these and so does the model — 48% WR,
    # net negative in both seasons. Set maxUnderOdds=-105 to enable.
    # Disabled by default (0.0 = no cap).
    # ------------------------------------------------------------------ #
    maxUnderOdds: float = 0.0

    # ------------------------------------------------------------------ #
    # High predDiff + low predicted block (predDiff 2.5-3.0, predicted < 18)
    # When the book sets a line 2.5-3 pts above the model for a player
    # predicted to score under 18, the book is pricing in something the
    # model hasn't caught (lineup change, minutes spike, role shift).
    # Actual avg = 24.2 vs line avg = 23.3 in this bucket — players beat
    # the line. Both seasons bad: 25% WR (2024-25), 33% WR (2025-26).
    # Set blockHighDiffLowPred=True to enable.
    # ------------------------------------------------------------------ #
    blockHighDiffLowPred: bool = False
    highDiffLowPredMin: float = 2.5
    highDiffLowPredMax: float = 3.0
    highDiffLowPredCeiling: float = 18.0

    def passes(self, predicted: float, propLine: float, edge: float = None,
               pos: float = None, betOdds: float = None,
               predDiff: float = None) -> tuple[bool, str]:
        """
        Returns (passes: bool, reason: str).
        reason is an empty string when the prop passes.
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
        """Pass-through — no filters active. Every other UnderFilterSet should be compared against this."""
        return cls(name="under_baseline")

    @classmethod
    def production(cls) -> "UnderFilterSet":
        """
        Active production filter set derived from cross-season flat-stake analysis.
        All filters confirmed consistent across both 2024-25 and 2025-26 seasons.

        Edge/bucket filters:
          1. Global 5-7% edge block (net loser across almost all pred buckets)
          2. 12-15 pred / 7-11% edge (-$320, 47-48% WR, bad both seasons)
          3. 15-18 pred / 5-7% edge (-$361, 42% WR, worst bucket, bad both seasons)
          4. 15-18 pred / 9-11% edge (-$123, 48% WR, bad both seasons)

        Model disagreement filters:
          5. predDiff < 1.0 (-$193, 30% WR) — model and book agree, nothing to exploit
          6. predDiff 2.5-3.0 + predicted < 18 (-$196, 29% WR) — book pricing something model missed
          7. Odds > -105 (-$92, 48% WR) — near-even lines, both seasons net negative

        Good buckets unaffected: 18-22/13-15% (+$218), 12-15/11-13% (+$149), <12/11-13% (+$191)

        Simulation (10% thresh, flat stake):
          Before filters: 1,659 bets, 54.3% WR, +$351
          After  filters: 1,410 bets, 56.0% WR, +$661
          Removed:          249 bets, 44.6% WR, -$310
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

