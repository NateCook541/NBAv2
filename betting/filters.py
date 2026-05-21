from dataclasses import dataclass, field
from typing import Optional, Tuple

@dataclass
class FilterSet:
    name: str = "default"
    
    # Predicted score floor
    minPredicted: float = 12.0

    # Position guard for mid range predicted scores
    midPosGuardEnable: bool = True
    midPosGuardRange: Tuple[float, float] = (15.0, 22.0)
    midPosGuardMaxPos: float = 2.0

    # Weak bucket guard
    weakBucketGuardEnabled: bool = False
    weakBucketRange: Tuple[float, float] = (18.0, 22.0)
    weakBucketMaxPos: float = 3.0
    
    # Minimum model disagreement with the line
    overMinDisagreement: float = 0.0

    # Reliability shrink
    reliabilityShrinkEnabled: bool = True

    # Quality gate
    qualityGateEnabled: bool = False
    qualityGateMaxEdge: float = 0.10
    qualityGateMinScore: float = 0.0


    def passess(self, predicted, pos, propLine):
        """
        Returns (passes: bool, reasons: str)
        reason is empty string if passes
        """
        if predicted < self.minPredicted:
            return False, "minPredicted"

        lo, hi = self.midPosGuardRange
        if (self.midPosGuardEnabled
                and lo <= predicted < hi
                and pos <= self.midPosGuardMaxPos):
            return False, "midPosGuard"

        lo, hi = self.weakBucketRange
        if (self.weakBucketGuardEnabled
                and lo <= predicted < hi
                and pos <= self.weakBucketMaxPos):
            return False, "weakBucketGuard"

        disagreement = predicted - propLine
        if self.overMinDisagreement > 0 and disagreement < self.overMinDisagreement:
            return False, "minDisagreement"

        return True, ""

