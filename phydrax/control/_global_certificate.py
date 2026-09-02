#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable, Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..optim import (
    AbstractBranchAndBoundProblem,
    branch_and_bound,
    BranchAndBoundPolicy,
    BranchAndBoundStatus,
)
from ._continuous_certification import ContinuousPathConstraintCertificate


class ControlRelaxationBound(StrictModule, NonTrainableState):
    lower_bound: Array
    candidate: Array
    candidate_objective: Array
    feasibility_excluded: Array
    valid: Array
    primal_dual_gap: Array
    relaxation_id: str = eqx.field(static=True)


class AbstractControlRelaxation(StrictModule):
    """Certified lower-bound provider for one finite coefficient box."""

    relaxation_id: str = eqx.field(static=True)
    exact_convex: bool = eqx.field(static=True)

    @abc.abstractmethod
    def bound(self, lower: Array, upper: Array, /) -> ControlRelaxationBound:
        raise NotImplementedError


class ConvexTranscriptionRelaxation(AbstractControlRelaxation, NonTrainableState):
    """Caller-declared convex canonical solve with primal/dual gap evidence."""

    solver: Callable[[Array, Array], tuple[Array, Array, Array, Array, Array]]
    gap_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        solver: Callable[[Array, Array], tuple[Array, Array, Array, Array, Array]],
        /,
        *,
        gap_tolerance: float = 1.0e-8,
        relaxation_id: str,
    ):
        if not callable(solver):
            raise TypeError("convex relaxation solver must be callable.")
        tolerance = float(gap_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("gap_tolerance must be finite and nonnegative.")
        self.solver = solver
        self.gap_tolerance = tolerance
        self.relaxation_id = relaxation_id
        self.exact_convex = True

    def bound(self, lower: Array, upper: Array, /) -> ControlRelaxationBound:
        bound, candidate, objective, gap, valid = self.solver(lower, upper)
        bound_ = jnp.asarray(bound).reshape(())
        candidate_ = jnp.asarray(candidate)
        objective_ = jnp.asarray(objective).reshape(())
        gap_ = jnp.asarray(gap).reshape(())
        candidate_valid = jnp.asarray(candidate_.shape == lower.shape)
        if candidate_.shape == lower.shape:
            candidate_valid = (
                candidate_valid
                & jnp.all(jnp.isfinite(candidate_))
                & jnp.all(candidate_ >= lower)
                & jnp.all(candidate_ <= upper)
            )
        valid_ = (
            jnp.asarray(valid, dtype=bool).reshape(())
            & jnp.all(jnp.isfinite(lower))
            & jnp.all(jnp.isfinite(upper))
            & candidate_valid
            & jnp.isfinite(bound_)
            & jnp.isfinite(objective_)
            & jnp.isfinite(gap_)
            & (gap_ >= 0)
            & (gap_ <= self.gap_tolerance)
            & (bound_ <= objective_ + self.gap_tolerance)
            & (objective_ - bound_ <= gap_ + self.gap_tolerance)
        )
        return ControlRelaxationBound(
            bound_,
            candidate_,
            objective_,
            jnp.asarray(False),
            valid_,
            gap_,
            self.relaxation_id,
        )


class LipschitzBoxControlRelaxation(AbstractControlRelaxation, NonTrainableState):
    """Finite-box objective lower bound with a declared global Lipschitz constant."""

    objective: Callable[[Array], Array]
    lipschitz_constant: float = eqx.field(static=True)

    def __init__(
        self,
        objective: Callable[[Array], Array],
        lipschitz_constant: float,
        /,
        *,
        relaxation_id: str,
    ):
        if not callable(objective):
            raise TypeError("objective must be callable.")
        lipschitz = float(lipschitz_constant)
        if not np.isfinite(lipschitz) or lipschitz < 0:
            raise ValueError("lipschitz_constant must be finite and nonnegative.")
        self.objective = objective
        self.lipschitz_constant = lipschitz
        self.relaxation_id = relaxation_id
        self.exact_convex = False

    def bound(self, lower: Array, upper: Array, /) -> ControlRelaxationBound:
        center = 0.5 * (lower + upper)
        objective = jnp.asarray(self.objective(center)).reshape(())
        radius = 0.5 * jnp.sqrt(jnp.sum(jnp.square(upper - lower)))
        lower_bound = objective - self.lipschitz_constant * radius
        valid = (
            jnp.all(jnp.isfinite(lower))
            & jnp.all(jnp.isfinite(upper))
            & jnp.isfinite(objective)
            & (lower_bound <= objective)
        )
        return ControlRelaxationBound(
            lower_bound,
            center,
            objective,
            jnp.asarray(False),
            valid,
            jnp.asarray(jnp.nan),
            self.relaxation_id,
        )


class BoundedControlCertificatePlan(StrictModule, NonTrainableState):
    objective: Callable[[Array], Array]
    continuous_feasibility: Callable[[Array], ContinuousPathConstraintCertificate | Array]
    lower: Array
    upper: Array
    relaxation: AbstractControlRelaxation
    branch_policy: BranchAndBoundPolicy
    minimum_box_width: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        objective: Callable[[Array], Array],
        lower: ArrayLike,
        upper: ArrayLike,
        relaxation: AbstractControlRelaxation,
        branch_policy: BranchAndBoundPolicy,
        continuous_feasibility: Callable[
            [Array], ContinuousPathConstraintCertificate | Array
        ],
        /,
        *,
        minimum_box_width: float = 1.0e-6,
        problem_id: str,
    ):
        lower_ = jnp.asarray(lower)
        upper_ = jnp.asarray(upper, dtype=lower_.dtype)
        if lower_.ndim != 1 or lower_.shape != upper_.shape or lower_.size == 0:
            raise ValueError(
                "bounded control coefficients require matching nonempty vectors."
            )
        if (
            np.any(~np.isfinite(np.asarray(lower_)))
            or np.any(~np.isfinite(np.asarray(upper_)))
            or np.any(np.asarray(lower_) >= np.asarray(upper_))
        ):
            raise ValueError(
                "bounded control coefficient boxes must be finite and nonempty."
            )
        if not callable(objective) or not callable(continuous_feasibility):
            raise TypeError("objective and continuous_feasibility must be callable.")
        if not isinstance(relaxation, AbstractControlRelaxation):
            raise TypeError("relaxation must be an AbstractControlRelaxation.")
        if not isinstance(branch_policy, BranchAndBoundPolicy):
            raise TypeError("branch_policy must be a BranchAndBoundPolicy.")
        width = float(minimum_box_width)
        if not np.isfinite(width) or width <= 0:
            raise ValueError("minimum_box_width must be positive and finite.")
        self.objective = objective
        self.continuous_feasibility = continuous_feasibility
        self.lower = lower_
        self.upper = upper_
        self.relaxation = relaxation
        self.branch_policy = branch_policy
        self.minimum_box_width = width
        self.plan_id = canonical_fingerprint(
            {
                "kind": "bounded-control-certificate-plan",
                "problem": problem_id,
                "dimension": int(lower_.size),
                "relaxation": relaxation.relaxation_id,
                "minimum_box_width": width,
            }
        )


class _ControlBox(StrictModule, NonTrainableState):
    lower: Array
    upper: Array
    depth: int = eqx.field(static=True)
    node_id: str = eqx.field(static=True)


class _BoundedControlBranchProblem(AbstractBranchAndBoundProblem):
    plan: BoundedControlCertificatePlan
    problem_id: str = eqx.field(static=True)
    terminal_lower_bounds: list[float] = eqx.field(static=True)
    terminal_widths: list[float] = eqx.field(static=True)
    relaxation_validity: list[bool] = eqx.field(static=True)

    def __init__(self, plan: BoundedControlCertificatePlan, /):
        self.plan = plan
        self.problem_id = plan.plan_id
        self.terminal_lower_bounds = []
        self.terminal_widths = []
        self.relaxation_validity = []

    def root(self, /):
        return _ControlBox(self.plan.lower, self.plan.upper, 0, "root")

    def node_id(self, node: _ControlBox, /) -> str:
        return node.node_id

    def lower_bound(self, node: _ControlBox, /) -> float:
        result = self.plan.relaxation.bound(node.lower, node.upper)
        valid = bool(
            np.asarray(result.valid) & np.asarray(jnp.isfinite(result.lower_bound))
        )
        self.relaxation_validity.append(valid)
        if not valid or bool(np.asarray(result.feasibility_excluded)):
            return float("inf")
        return float(np.asarray(result.lower_bound))

    def feasible(self, node: _ControlBox, /) -> bool:
        result = self.plan.relaxation.bound(node.lower, node.upper)
        valid = bool(
            np.asarray(result.valid) & np.asarray(jnp.isfinite(result.lower_bound))
        )
        self.relaxation_validity.append(valid)
        return valid and not bool(np.asarray(result.feasibility_excluded))

    def complete(self, node: _ControlBox, /) -> bool:
        width = float(np.max(np.asarray(node.upper - node.lower)))
        complete = width <= self.plan.minimum_box_width
        if complete:
            result = self.plan.relaxation.bound(node.lower, node.upper)
            valid = bool(
                np.asarray(result.valid) & np.asarray(jnp.isfinite(result.lower_bound))
            )
            self.relaxation_validity.append(valid)
            if valid and not bool(np.asarray(result.feasibility_excluded)):
                self.terminal_lower_bounds.append(float(np.asarray(result.lower_bound)))
                self.terminal_widths.append(width)
        return complete

    def objective(self, node: _ControlBox, /) -> float:
        center = 0.5 * (node.lower + node.upper)
        feasible = self.plan.continuous_feasibility(center)
        if isinstance(feasible, ContinuousPathConstraintCertificate):
            valid = bool(np.asarray(jnp.all(feasible.certified)))
        else:
            valid = bool(np.asarray(jnp.all(jnp.asarray(feasible, dtype=bool))))
        return float(np.asarray(self.plan.objective(center))) if valid else float("inf")

    def branch(self, node: _ControlBox, /) -> Sequence[_ControlBox]:
        widths = np.asarray(node.upper - node.lower)
        axis = int(np.argmax(widths))
        midpoint = 0.5 * (node.lower[axis] + node.upper[axis])
        return (
            _ControlBox(
                node.lower,
                node.upper.at[axis].set(midpoint),
                node.depth + 1,
                f"{node.node_id}:0",
            ),
            _ControlBox(
                node.lower.at[axis].set(midpoint),
                node.upper,
                node.depth + 1,
                f"{node.node_id}:1",
            ),
        )


class BoundedControlOptimalityCertificate(StrictModule, NonTrainableState):
    incumbent: Array
    objective: Array
    global_lower_bound: Array
    absolute_gap: Array
    relative_gap: Array
    explored_nodes: Array
    pruned_nodes: Array
    frontier_size: Array
    domain_covered: Array
    continuous_feasible: Array
    relaxation_valid: Array
    exact: Array
    epsilon_global: Array
    status: Array
    plan_id: str = eqx.field(static=True)

    @property
    def certified(self) -> Array:
        return self.exact | self.epsilon_global


def _continuous_feasible(
    plan: BoundedControlCertificatePlan, candidate: Array, /
) -> Array:
    feasibility = plan.continuous_feasibility(candidate)
    return (
        jnp.all(feasibility.certified)
        if isinstance(feasibility, ContinuousPathConstraintCertificate)
        else jnp.all(jnp.asarray(feasibility, dtype=bool))
    )


def certify_bounded_control_optimum(
    plan: BoundedControlCertificatePlan,
    incumbent: ArrayLike | None = None,
    /,
) -> BoundedControlOptimalityCertificate:
    """Certify only complete finite-box coverage with valid lower bounds."""

    if not isinstance(plan, BoundedControlCertificatePlan):
        raise TypeError("plan must be a BoundedControlCertificatePlan.")
    root_bound = plan.relaxation.bound(plan.lower, plan.upper)
    root_relaxation_valid = (
        root_bound.valid
        & jnp.isfinite(root_bound.lower_bound)
        & (~root_bound.feasibility_excluded)
    )
    if isinstance(plan.relaxation, ConvexTranscriptionRelaxation):
        candidate = jnp.asarray(root_bound.candidate, dtype=plan.lower.dtype)
        candidate_shape_valid = candidate.shape == plan.lower.shape
        if candidate_shape_valid:
            candidate_objective = jnp.asarray(plan.objective(candidate)).reshape(())
            candidate_in_box = (
                jnp.all(jnp.isfinite(candidate))
                & jnp.all(candidate >= plan.lower)
                & jnp.all(candidate <= plan.upper)
            )
            continuous = _continuous_feasible(plan, candidate)
            tolerance = plan.relaxation.gap_tolerance
            objective_matches = jnp.isfinite(candidate_objective) & (
                jnp.abs(candidate_objective - root_bound.candidate_objective) <= tolerance
            )
            global_lower_bound = jnp.minimum(root_bound.lower_bound, candidate_objective)
            absolute_gap = candidate_objective - global_lower_bound
            exact = (
                root_relaxation_valid
                & candidate_in_box
                & continuous
                & objective_matches
                & jnp.isfinite(root_bound.primal_dual_gap)
                & (root_bound.primal_dual_gap >= 0.0)
                & (root_bound.primal_dual_gap <= tolerance)
                & (absolute_gap <= tolerance)
            )
            if bool(np.asarray(exact)):
                relative_gap = absolute_gap / jnp.maximum(
                    jnp.abs(candidate_objective), 1.0
                )
                return BoundedControlOptimalityCertificate(
                    candidate,
                    candidate_objective,
                    global_lower_bound,
                    absolute_gap,
                    relative_gap,
                    jnp.asarray(1, dtype=jnp.int32),
                    jnp.asarray(0, dtype=jnp.int32),
                    jnp.asarray(0, dtype=jnp.int32),
                    jnp.asarray(True),
                    continuous,
                    root_relaxation_valid,
                    exact,
                    jnp.asarray(False),
                    jnp.asarray(int(BranchAndBoundStatus.OPTIMAL), dtype=jnp.int32),
                    plan.plan_id,
                )

    supplied_candidate = None
    supplied_objective = jnp.asarray(jnp.inf)
    supplied_continuous = jnp.asarray(False)
    if incumbent is not None:
        candidate_ = jnp.asarray(incumbent)
        candidate_shape_valid = candidate_.shape == plan.lower.shape
        candidate_real = not jnp.issubdtype(candidate_.dtype, jnp.complexfloating)
        if candidate_shape_valid and candidate_real:
            candidate_in_box = (
                jnp.all(jnp.isfinite(candidate_))
                & jnp.all(candidate_ >= plan.lower)
                & jnp.all(candidate_ <= plan.upper)
            )
            if bool(np.asarray(candidate_in_box)):
                continuous_ = _continuous_feasible(plan, candidate_)
                if bool(np.asarray(continuous_)):
                    objective_ = jnp.asarray(plan.objective(candidate_)).reshape(())
                    if bool(np.asarray(jnp.isfinite(objective_))):
                        supplied_candidate = candidate_
                        supplied_objective = objective_
                        supplied_continuous = continuous_

    problem = _BoundedControlBranchProblem(plan)
    result = branch_and_bound(problem, policy=plan.branch_policy)
    candidate = None
    objective = jnp.asarray(jnp.inf, dtype=result.objective.dtype)
    continuous = jnp.asarray(False)
    if result.incumbent is not None and bool(np.asarray(jnp.isfinite(result.objective))):
        tree_candidate = 0.5 * (result.incumbent.lower + result.incumbent.upper)
        tree_continuous = _continuous_feasible(plan, tree_candidate)
        if bool(np.asarray(tree_continuous)):
            candidate = tree_candidate
            objective = result.objective
            continuous = tree_continuous
    if supplied_candidate is not None and (
        candidate is None or bool(np.asarray(supplied_objective < objective))
    ):
        candidate = supplied_candidate
        objective = supplied_objective
        continuous = supplied_continuous
    incumbent_available = candidate is not None
    if candidate is None:
        candidate = jnp.full(
            plan.lower.shape,
            jnp.nan,
            dtype=jnp.result_type(plan.lower.dtype, jnp.float32),
        )

    relaxation_valid = root_relaxation_valid & jnp.asarray(
        all(problem.relaxation_validity)
    )
    terminal_lower_bound = (
        min(problem.terminal_lower_bounds)
        if problem.terminal_lower_bounds
        else float("inf")
    )
    global_lower_bound = jnp.minimum(
        result.global_lower_bound,
        jnp.asarray(terminal_lower_bound, dtype=result.global_lower_bound.dtype),
    )
    global_lower_bound = jnp.minimum(global_lower_bound, objective)
    finite_gap = jnp.isfinite(objective) & jnp.isfinite(global_lower_bound)
    absolute_gap = jnp.where(
        finite_gap, objective - global_lower_bound, jnp.asarray(jnp.inf)
    )
    relative_gap = jnp.where(
        finite_gap,
        absolute_gap / jnp.maximum(jnp.abs(objective), 1.0),
        jnp.asarray(jnp.inf),
    )
    positive_width_terminal = any(width > 0.0 for width in problem.terminal_widths)
    exact = (
        (result.status == int(BranchAndBoundStatus.OPTIMAL))
        & (~jnp.asarray(positive_width_terminal))
        & continuous
        & relaxation_valid
        & jnp.asarray(incumbent_available)
        & (absolute_gap == 0.0)
    )
    epsilon = (
        continuous
        & relaxation_valid
        & jnp.asarray(incumbent_available)
        & jnp.isfinite(global_lower_bound)
        & (
            (absolute_gap <= plan.branch_policy.absolute_gap)
            | (relative_gap <= plan.branch_policy.relative_gap)
        )
        & (~exact)
    )
    covered = exact | epsilon
    unresolved_status = jnp.where(
        (
            (result.status == int(BranchAndBoundStatus.OPTIMAL))
            | (result.status == int(BranchAndBoundStatus.GAP_REACHED))
            | (
                (result.status == int(BranchAndBoundStatus.INFEASIBLE))
                & jnp.asarray(incumbent_available)
            )
        ),
        int(BranchAndBoundStatus.WORK_LIMIT),
        result.status,
    )
    reported_status = jnp.where(
        exact,
        int(BranchAndBoundStatus.OPTIMAL),
        jnp.where(
            epsilon,
            int(BranchAndBoundStatus.GAP_REACHED),
            unresolved_status,
        ),
    ).astype(jnp.int32)
    return BoundedControlOptimalityCertificate(
        candidate,
        objective,
        global_lower_bound,
        absolute_gap,
        relative_gap,
        result.explored_nodes,
        result.pruned_nodes,
        result.frontier_size,
        covered,
        continuous,
        relaxation_valid,
        exact,
        epsilon,
        reported_status,
        plan.plan_id,
    )


__all__ = [
    "AbstractControlRelaxation",
    "BoundedControlCertificatePlan",
    "BoundedControlOptimalityCertificate",
    "ControlRelaxationBound",
    "ConvexTranscriptionRelaxation",
    "LipschitzBoxControlRelaxation",
    "certify_bounded_control_optimum",
]
