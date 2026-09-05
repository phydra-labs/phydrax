# Copyright © 2026 PHYDRA, Inc. All rights reserved.
import phydrax as phx


class _BoundedLeaves(phx.optim.AbstractBranchAndBoundProblem):
    sibling_bound: float

    def __init__(self, sibling_bound):
        self.sibling_bound = sibling_bound
        self.problem_id = "frontier-certification"

    def root(self):
        return "root"

    def node_id(self, node):
        return node

    def lower_bound(self, node):
        return {"root": 0.0, "candidate": 1.0, "sibling": self.sibling_bound}[node]

    def feasible(self, node):
        return True

    def complete(self, node):
        return node != "root"

    def objective(self, node):
        return 2.0 if node == "candidate" else self.sibling_bound

    def branch(self, node):
        return ("candidate", "sibling")


def test_dominated_frontier_proves_optimality_without_positive_gap_claim():
    result = phx.optim.branch_and_bound(_BoundedLeaves(2.0))
    assert result.successful
    assert result.objective == 2.0
    assert result.global_lower_bound == 2.0
    assert result.absolute_gap == 0.0


def test_positive_gap_retains_unresolved_competitor_and_is_not_exact():
    result = phx.optim.branch_and_bound(
        _BoundedLeaves(1.5),
        policy=phx.optim.BranchAndBoundPolicy(absolute_gap=0.5),
    )
    assert not result.successful
    assert result.status == phx.optim.BranchAndBoundStatus.GAP_REACHED
    assert result.objective == 2.0
    assert result.global_lower_bound == 1.5
    assert result.absolute_gap == 0.5
