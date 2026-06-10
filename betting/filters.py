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

    def passes(self, predicted: float, propLine: float) -> tuple[bool, str]:
        """
        Returns (passes: bool, reason: str).
        reason is an empty string when the prop passes.
        """

        if self.minPredicted > 0 and predicted < self.minPredicted:
            return False, "minPredicted"

        if self.overMinDisagreement > 0:
            if (predicted - propLine) < self.overMinDisagreement:
                return False, "overMinDisagreement"

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
