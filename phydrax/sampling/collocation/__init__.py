#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Adaptive collocation policies for pointwise-capable sampling terms."""

from typing import TYPE_CHECKING

from ..._deferred_exports import deferred_exports


# Re-exports load on first use, so importing one owner never imports the
# whole package and its import cycles.
_DEFERRED_EXPORTS = {
    "AbstractCollocationPolicy": "._adaptive",
    "CollocationPolicy": "._adaptive",
    "CollocationPopulation": "._adaptive",
    "PeriodicCollocation": "._adaptive",
    "PointwiseSamplingTerm": "._adaptive",
    "R3": "._adaptive",
    "RARD": "._adaptive",
    "ResidualAttentionCollocation": "._attention",
    "ResidualAttentionPopulation": "._attention",
    "AdaptationBudget": "._control",
    "COLLOCATION_POLICY_SUPPORT": "._control",
    "collocation_policy_support": "._control",
    "CollocationDefaults": "._control",
    "CollocationPolicySupport": "._control",
    "controlled_collocation": "._control",
    "ControlledCollocationPolicy": "._control",
    "ControlledCollocationPopulation": "._control",
    "CoverageAnchors": "._control",
    "PolicySupportTier": "._control",
    "RECOMMENDED_COLLOCATION_DEFAULTS": "._control",
    "RefreshGuard": "._control",
    "RefreshSchedule": "._control",
    "ResidualMonitor": "._control",
    "CoresetCollocation": "._coreset",
    "CoresetCollocationPolicy": "._coreset",
    "NarrowBandCollocationPolicy": "._interface",
    "HierarchicalAxisCollocation": "._separable",
    "HierarchicalAxisPolicy": "._separable",
    "PeriodicSeparableCollocation": "._separable",
    "SeparableCollocationPolicy": "._separable",
    "SeparableCollocationPopulation": "._separable",
    "CausalTimeSlabSchedule": "._time_slab",
}


__getattr__, __dir__ = deferred_exports(__name__, globals(), _DEFERRED_EXPORTS)


if TYPE_CHECKING:
    from ._adaptive import (
        AbstractCollocationPolicy,
        CollocationPolicy,
        CollocationPopulation,
        PeriodicCollocation,
        PointwiseSamplingTerm,
        R3,
        RARD,
    )
    from ._attention import ResidualAttentionCollocation, ResidualAttentionPopulation
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
    from ._interface import NarrowBandCollocationPolicy
    from ._separable import (
        HierarchicalAxisCollocation,
        HierarchicalAxisPolicy,
        PeriodicSeparableCollocation,
        SeparableCollocationPolicy,
        SeparableCollocationPopulation,
    )
    from ._time_slab import CausalTimeSlabSchedule


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
    "CausalTimeSlabSchedule",
    "CoresetCollocation",
    "CoresetCollocationPolicy",
    "HierarchicalAxisCollocation",
    "HierarchicalAxisPolicy",
    "PeriodicCollocation",
    "PeriodicSeparableCollocation",
    "PointwiseSamplingTerm",
    "PolicySupportTier",
    "NarrowBandCollocationPolicy",
    "R3",
    "RARD",
    "RECOMMENDED_COLLOCATION_DEFAULTS",
    "ResidualAttentionCollocation",
    "ResidualAttentionPopulation",
    "RefreshGuard",
    "RefreshSchedule",
    "ResidualMonitor",
    "SeparableCollocationPolicy",
    "SeparableCollocationPopulation",
    "collocation_policy_support",
    "controlled_collocation",
]
