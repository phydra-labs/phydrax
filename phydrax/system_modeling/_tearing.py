#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TearingPlan:
    tear_variable_indices: tuple[int, ...]
    residual_indices: tuple[int, ...]
    variable_count: int
    residual_count: int

    def __post_init__(self):
        if (
            isinstance(self.variable_count, bool)
            or not isinstance(self.variable_count, int)
            or self.variable_count <= 0
            or isinstance(self.residual_count, bool)
            or not isinstance(self.residual_count, int)
            or self.residual_count <= 0
        ):
            raise ValueError("Tearing dimensions must be positive integers.")
        if len(self.tear_variable_indices) != len(self.residual_indices):
            raise ValueError("Tearing variables and residuals must align.")
        if (
            len(set(self.tear_variable_indices)) != len(self.tear_variable_indices)
            or len(set(self.residual_indices)) != len(self.residual_indices)
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or index >= self.variable_count
                for index in self.tear_variable_indices
            )
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or index >= self.residual_count
                for index in self.residual_indices
            )
        ):
            raise ValueError("Tearing indices must be unique and within their domains.")


__all__ = ["TearingPlan"]
