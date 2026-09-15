#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from ._assignment import AssignmentDecision, BipartiteAssignmentSpace, HungarianAssignment
from ._blackbox import (
    blackbox_solution,
    BlackboxInterpolation,
    BlackboxPullbackResult,
    estimate_blackbox_pullback,
)
from ._cardinality import (
    CardinalityDecision,
    CardinalitySpace,
    StableCardinalityOracle,
)
from ._explicit import ExhaustiveLinearOracle, ExplicitDecision, ExplicitDecisionSpace
from ._method import (
    AbstractBoundableLinearCombinatorialMethod,
    AbstractLinearCombinatorialMethod,
    CombinatorialPlan,
    plan_combinatorial,
    solve_combinatorial,
    solve_restricted_combinatorial,
)
from ._min_cost_flow import (
    CapacitatedFlowSpace,
    CycleCancelingMinCostFlow,
    FlowDecision,
)
from ._problem import AbstractCombinatorialSpace, LinearCombinatorialProblem
from ._restriction import (
    AbstractBoundableCombinatorialSpace,
    audit_combinatorial_restriction,
    BoundedCombinatorialExecution,
    CombinatorialFeatureRestriction,
)
from ._set_packing import (
    BranchAndBoundSetPacking,
    GreedySetPacking,
    SetPackingDecision,
    SetPackingSpace,
)
from ._shortest_path import DAGShortestPath, PathDecision, ShortestPathSpace
from ._types import (
    combinatorial_status_message,
    CombinatorialCertificate,
    CombinatorialCertification,
    CombinatorialFeasibility,
    CombinatorialMethodCapabilities,
    CombinatorialProvenance,
    CombinatorialResult,
    CombinatorialStatus,
)


__all__ = [
    "AbstractBoundableCombinatorialSpace",
    "AbstractBoundableLinearCombinatorialMethod",
    "AbstractCombinatorialSpace",
    "AbstractLinearCombinatorialMethod",
    "AssignmentDecision",
    "BipartiteAssignmentSpace",
    "BranchAndBoundSetPacking",
    "BlackboxInterpolation",
    "BoundedCombinatorialExecution",
    "BlackboxPullbackResult",
    "CapacitatedFlowSpace",
    "ExhaustiveLinearOracle",
    "ExplicitDecision",
    "ExplicitDecisionSpace",
    "CardinalityDecision",
    "CardinalitySpace",
    "CycleCancelingMinCostFlow",
    "StableCardinalityOracle",
    "FlowDecision",
    "GreedySetPacking",
    "HungarianAssignment",
    "DAGShortestPath",
    "PathDecision",
    "ShortestPathSpace",
    "SetPackingDecision",
    "SetPackingSpace",
    "CombinatorialCertificate",
    "CombinatorialCertification",
    "CombinatorialFeasibility",
    "CombinatorialMethodCapabilities",
    "CombinatorialFeatureRestriction",
    "CombinatorialPlan",
    "CombinatorialProvenance",
    "CombinatorialResult",
    "CombinatorialStatus",
    "LinearCombinatorialProblem",
    "audit_combinatorial_restriction",
    "blackbox_solution",
    "estimate_blackbox_pullback",
    "combinatorial_status_message",
    "plan_combinatorial",
    "solve_combinatorial",
    "solve_restricted_combinatorial",
]
