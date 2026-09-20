#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TearingPlan:
    tear_variable_indices: tuple[int, ...]
    residual_indices: tuple[int, ...]

    def __post_init__(self):
        if len(self.tear_variable_indices) != len(self.residual_indices):
            raise ValueError("Tearing variables and residuals must align.")


__all__ = ["TearingPlan"]
