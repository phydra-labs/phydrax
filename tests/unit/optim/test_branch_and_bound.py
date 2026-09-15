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

    def evaluate(self, node):
        bound = {"root": 0.0, "candidate": 1.0, "sibling": self.sibling_bound}[node]
        lower = phx.optim.BranchBoundEvidence(
            bound,
            certified=True,
            certificate_id=f"{node}:bound",
        )
        if node == "root":
            return phx.optim.BranchNodeEvaluation(lower_bound=lower)
        objective = 2.0 if node == "candidate" else self.sibling_bound
        return phx.optim.BranchNodeEvaluation(
            lower_bound=lower,
            candidate=phx.optim.BranchCandidate(
                node,
                objective,
                certificate_id=f"{node}:candidate",
            ),
            terminal=True,
        )

    def branch(self, node, evaluation):
        del node, evaluation
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


class _FailedSibling(phx.optim.AbstractBranchAndBoundProblem):
    problem_id = "failed-sibling"

    def __init__(self):
        self.problem_id = "failed-sibling"

    def root(self):
        return "root"

    def node_id(self, node):
        return node

    def evaluate(self, node):
        if node == "root":
            return phx.optim.BranchNodeEvaluation(
                lower_bound=phx.optim.BranchBoundEvidence(
                    0.0,
                    certified=True,
                    certificate_id="root-bound",
                )
            )
        if node == "candidate":
            return phx.optim.BranchNodeEvaluation(
                lower_bound=phx.optim.BranchBoundEvidence(
                    1.0,
                    certified=True,
                    certificate_id="candidate-bound",
                ),
                candidate=phx.optim.BranchCandidate(
                    node,
                    2.0,
                    certificate_id="candidate-audit",
                ),
                terminal=True,
            )
        return phx.optim.BranchNodeEvaluation(
            lower_bound=phx.optim.BranchBoundEvidence(
                1.5,
                certified=True,
                certificate_id="failed-bound",
            ),
            failure=phx.optim.BranchNodeFailure(
                "deliberate-failure",
                "The sibling could not be evaluated.",
            ),
        )

    def branch(self, node, evaluation):
        del node, evaluation
        return ("candidate", "failed")


class _NegativeInfiniteRoot(phx.optim.AbstractBranchAndBoundProblem):
    problem_id = "negative-infinite-root"

    def __init__(self):
        self.problem_id = "negative-infinite-root"

    def root(self):
        return "root"

    def node_id(self, node):
        return node

    def evaluate(self, node):
        return phx.optim.BranchNodeEvaluation(
            lower_bound=phx.optim.BranchBoundEvidence(
                float("-inf"),
                certified=True,
                certificate_id="universal-bound",
            ),
            candidate=phx.optim.BranchCandidate(
                node,
                1.0,
                certificate_id="finite-candidate",
            ),
            terminal=True,
        )

    def branch(self, node, evaluation):
        raise AssertionError((node, evaluation))


class _InfeasibleRoot(phx.optim.AbstractBranchAndBoundProblem):
    problem_id = "infeasible-root"

    def __init__(self):
        self.problem_id = "infeasible-root"

    def root(self):
        return "root"

    def node_id(self, node):
        return node

    def evaluate(self, node):
        del node
        return phx.optim.BranchNodeEvaluation.proven_infeasible("root-infeasibility")

    def branch(self, node, evaluation):
        raise AssertionError((node, evaluation))


def test_failed_node_prevents_global_success_without_discarding_incumbent():
    result = phx.optim.branch_and_bound(_FailedSibling())

    assert result.status == phx.optim.BranchAndBoundStatus.EVALUATION_FAILURE
    assert result.incumbent == "candidate"
    assert result.objective == 2.0
    assert not result.search_complete
    assert not result.successful


def test_negative_infinite_bound_is_not_infeasibility():
    result = phx.optim.branch_and_bound(_NegativeInfiniteRoot())

    assert result.status == phx.optim.BranchAndBoundStatus.OPTIMAL
    assert result.objective == 1.0
    assert result.successful


def test_only_certified_infeasible_root_reports_infeasible():
    result = phx.optim.branch_and_bound(_InfeasibleRoot())

    assert result.status == phx.optim.BranchAndBoundStatus.INFEASIBLE
    assert result.incumbent is None
    assert result.search_complete
    assert not result.successful
