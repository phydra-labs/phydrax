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
    AbstractLinearCombinatorialMethod,
    CombinatorialPlan,
    plan_combinatorial,
    solve_combinatorial,
)
from ._problem import AbstractCombinatorialSpace, LinearCombinatorialProblem
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
    "AbstractCombinatorialSpace",
    "AbstractLinearCombinatorialMethod",
    "AssignmentDecision",
    "BipartiteAssignmentSpace",
    "BlackboxInterpolation",
    "BlackboxPullbackResult",
    "ExhaustiveLinearOracle",
    "ExplicitDecision",
    "ExplicitDecisionSpace",
    "CardinalityDecision",
    "CardinalitySpace",
    "StableCardinalityOracle",
    "HungarianAssignment",
    "DAGShortestPath",
    "PathDecision",
    "ShortestPathSpace",
    "CombinatorialCertificate",
    "CombinatorialCertification",
    "CombinatorialFeasibility",
    "CombinatorialMethodCapabilities",
    "CombinatorialPlan",
    "CombinatorialProvenance",
    "CombinatorialResult",
    "CombinatorialStatus",
    "LinearCombinatorialProblem",
    "blackbox_solution",
    "estimate_blackbox_pullback",
    "combinatorial_status_message",
    "plan_combinatorial",
    "solve_combinatorial",
]
