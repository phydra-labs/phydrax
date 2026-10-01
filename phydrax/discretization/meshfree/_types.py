# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Explicit approximation, policy, and row-outcome vocabulary."""

from enum import IntEnum
from typing import Literal, TypeAlias


MeshfreeApproximation: TypeAlias = Literal["gmls", "phs-rbf-fd"]
StencilWeightKernel: TypeAlias = Literal["wendland-c2", "inverse-square"]
StencilAcceptance: TypeAlias = Literal["refuse", "mask"]


class MeshfreeRowStatus(IntEnum):
    VALID = 0
    UNDERSAMPLED = 1
    RANK_DEFICIENT = 2
    ILL_CONDITIONED = 3
    EXCESSIVE_AMPLIFICATION = 4
    MOMENT_FAILURE = 5
