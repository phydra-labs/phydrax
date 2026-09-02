#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias


DifferentiabilityPolicy: TypeAlias = Literal[
    "smooth_discrete",
    "branchwise",
    "smooth_surrogate",
    "unsupported",
]


def validate_differentiability_policy(
    value: DifferentiabilityPolicy | str, /
) -> DifferentiabilityPolicy:
    policy = str(value)
    if policy not in (
        "smooth_discrete",
        "branchwise",
        "smooth_surrogate",
        "unsupported",
    ):
        raise ValueError("Unknown conservation differentiability policy.")
    return policy


__all__ = ["DifferentiabilityPolicy", "validate_differentiability_policy"]
