#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Adaptive collocation policies for pointwise-capable sampling terms."""

from ._adaptive import (
    AbstractCollocationPolicy,
    CollocationPolicy,
    CollocationPopulation,
    PeriodicCollocation,
    PointwiseSamplingTerm,
    R3,
    RARD,
)
from ._control import (
    AdaptationBudget,
    COLLOCATION_POLICY_SUPPORT,
    collocation_policy_support,
    CollocationDefaults,
    CollocationPolicySupport,
    controlled_collocation,
    ControlledCollocationPolicy,
    ControlledCollocationPopulation,
    CoverageAnchors,
    PolicySupportTier,
    RECOMMENDED_COLLOCATION_DEFAULTS,
    RefreshGuard,
    RefreshSchedule,
    ResidualMonitor,
)
from ._coreset import CoresetCollocation, CoresetCollocationPolicy
from ._separable import (
    HierarchicalAxisCollocation,
    HierarchicalAxisPolicy,
    PeriodicSeparableCollocation,
    SeparableCollocationPolicy,
    SeparableCollocationPopulation,
)


__all__ = [
    "AbstractCollocationPolicy",
    "AdaptationBudget",
    "COLLOCATION_POLICY_SUPPORT",
    "CollocationDefaults",
    "CollocationPolicy",
    "CollocationPolicySupport",
    "CollocationPopulation",
    "ControlledCollocationPolicy",
    "ControlledCollocationPopulation",
    "CoverageAnchors",
    "CoresetCollocation",
    "CoresetCollocationPolicy",
    "HierarchicalAxisCollocation",
    "HierarchicalAxisPolicy",
    "PeriodicCollocation",
    "PeriodicSeparableCollocation",
    "PointwiseSamplingTerm",
    "PolicySupportTier",
    "R3",
    "RARD",
    "RECOMMENDED_COLLOCATION_DEFAULTS",
    "RefreshGuard",
    "RefreshSchedule",
    "ResidualMonitor",
    "SeparableCollocationPolicy",
    "SeparableCollocationPopulation",
    "collocation_policy_support",
    "controlled_collocation",
]
