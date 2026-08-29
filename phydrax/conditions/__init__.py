#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Scientific conditions independent of soft or hard numerical treatment."""

from . import (
    cfd,
    conservation,
    electromagnetics,
    free_boundary,
    solids,
    stochastic,
    thermal,
)
from ._base import (
    AbstractCondition,
    AbstractMomentCondition,
    AbstractResidualCondition,
    ConditionSupport,
    Moment,
    Observation,
    Residual,
)
from .boundary import Absorbing, ConditionValue, Dirichlet, Neumann, Robin
from .initial import Initial


__all__ = [
    "AbstractCondition",
    "AbstractMomentCondition",
    "AbstractResidualCondition",
    "Absorbing",
    "ConditionSupport",
    "ConditionValue",
    "cfd",
    "conservation",
    "Dirichlet",
    "Initial",
    "electromagnetics",
    "free_boundary",
    "Moment",
    "Neumann",
    "Observation",
    "Residual",
    "solids",
    "stochastic",
    "thermal",
    "Robin",
]
