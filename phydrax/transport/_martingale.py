#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._bounds import Bounds
from .._strict import StrictModule
from ..optim import (
    ConvexProgramResult,
    ConvexSolvePolicy,
    LinearProgram,
    solve_linear_program,
)
from ._problem import DiscreteTransportProblem


class MartingaleTransportStatus(IntEnum):
    """Terminal status of a finite martingale transport calculation."""

    OPTIMAL = 0
    CONVEX_ORDER_VIOLATION = 1
    OPTIMIZER_FAILED = 2
    INVALID_COUPLING = 3


class ConvexOrderEvidence(StrictModule):
    """Finite-support convex-order checks prior to a martingale solve.

    In one dimension the call-function inequalities on the combined atom set,
    together with equality of mass and mean, are a complete finite-support
    criterion.  In higher dimensions the mean check is only necessary; the
    martingale linear program remains the complete feasibility test.
    """

    source_mean: Array
    target_mean: Array
    mean_defect: Array
    strikes: Array
    call_value_gaps: Array
    minimum_call_value_gap: Array
    feasible: Array
    criterion_complete: bool = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)


class MartingaleCouplingEvidence(StrictModule):
    """Marginal, conditional-mean, entropy, and objective evidence."""

    source_marginal: Array
    target_marginal: Array
    conditional_target_mean: Array
    source_marginal_residual: Array
    target_marginal_residual: Array
    martingale_defect: Array
    minimum_coupling: Array
    primal_objective: Array
    dual_objective: Array
    primal_dual_gap: Array
    entropy: Array
    finite: Array
    valid: Array
    tolerance: float = eqx.field(static=True)


class MartingaleDualEvidence(StrictModule):
    """Audited semi-static dual of the finite martingale LP.

    The inequality is ``source + target + dynamic·(y-x) <= cost``.  Thus this
    is a subhedge dual for a minimization problem, not an automatically inferred
    superhedge or a classical-OT potential pair.
    """

    source_potential: Array
    target_potential: Array
    dynamic_holding: Array
    dual_objective: Array
    maximum_inequality_violation: Array
    stationarity_residual: Array
    orientation: int = eqx.field(static=True)
    valid: Array


class MartingaleRefinementEvidence(StrictModule):
    """Comparison of two independently solved finite martingale discretizations."""

    coarse_objective: Array
    refined_objective: Array
    objective_change: Array
    coarse_martingale_defect: Array
    refined_martingale_defect: Array
    coarse_marginal_residual: Array
    refined_marginal_residual: Array
    accepted: Array
    objective_tolerance: float = eqx.field(static=True)
    constraint_tolerance: float = eqx.field(static=True)
    refinement_id: str = eqx.field(static=True)


class MartingaleTransportProblem(StrictModule):
    """Balanced finite transport with explicit conditional-mean constraints.

    ``source_coordinates`` and ``target_coordinates`` are the quantities that
    must be martingales.  They need not be the features consumed by the ground
    cost, which prevents a feature embedding from silently changing the
    financial/probabilistic martingale constraint.
    """

    transport: DiscreteTransportProblem
    source_coordinates: Array
    target_coordinates: Array
    convex_order: ConvexOrderEvidence
    constraint_tolerance: float = eqx.field(static=True)
    constraint_kind: str = eqx.field(static=True, default="martingale")

    def __init__(
        self,
        transport: DiscreteTransportProblem,
        /,
        *,
        source_coordinates: ArrayLike | None = None,
        target_coordinates: ArrayLike | None = None,
        constraint_tolerance: float = 1e-7,
    ):
        if not isinstance(transport, DiscreteTransportProblem):
            raise TypeError("transport must be a DiscreteTransportProblem.")
        tolerance = float(constraint_tolerance)
        if not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("constraint_tolerance must be finite and nonnegative.")
        source = _martingale_coordinates(
            transport.source.points if source_coordinates is None else source_coordinates,
            transport.shape[0],
            name="source_coordinates",
        )
        target = _martingale_coordinates(
            transport.target.points if target_coordinates is None else target_coordinates,
            transport.shape[1],
            name="target_coordinates",
        )
        if source.shape[1] != target.shape[1]:
            raise ValueError("Source and target martingale coordinate sizes must agree.")
        self.transport = transport
        self.source_coordinates = source
        self.target_coordinates = target
        self.convex_order = convex_order_evidence(
            transport.source_weights,
            source,
            transport.target_weights,
            target,
            tolerance=tolerance,
        )
        self.constraint_tolerance = tolerance
        self.constraint_kind = "martingale"

    @property
    def shape(self) -> tuple[int, int]:
        return self.transport.shape

    @property
    def coordinate_size(self) -> int:
        return int(self.source_coordinates.shape[1])


class MartingaleTransportResult(StrictModule):
    """Finite martingale coupling with complete primal and dual diagnostics."""

    problem: MartingaleTransportProblem
    coupling: Array
    evidence: MartingaleCouplingEvidence
    dual: MartingaleDualEvidence
    optimizer_result: ConvexProgramResult | None
    status: Array
    independent_equality_rows: tuple[int, ...] = eqx.field(static=True)
    method: str = eqx.field(static=True, default="finite-martingale-linear-program")

    @property
    def successful(self) -> Array:
        return (
            self.status == int(MartingaleTransportStatus.OPTIMAL)
        ) & self.evidence.valid

    @property
    def converged(self) -> Array:
        return self.successful

    def source_marginal(self) -> Array:
        return self.evidence.source_marginal

    def target_marginal(self) -> Array:
        return self.evidence.target_marginal

    def conditional_mean(self) -> Array:
        return self.evidence.conditional_target_mean

    def dense_plan(self) -> Array:
        return self.coupling


def _martingale_coordinates(value: ArrayLike, atoms: int, /, *, name: str) -> Array:
    coordinates = jnp.asarray(value, dtype=float)
    if coordinates.ndim == 1:
        coordinates = coordinates[:, None]
    if coordinates.ndim != 2 or coordinates.shape[0] != atoms or coordinates.shape[1] < 1:
        raise ValueError(f"{name} must have shape (atom, coordinate).")
    if bool(jnp.any(~jnp.isfinite(coordinates))):
        raise ValueError(f"{name} must contain only finite values.")
    return coordinates


def convex_order_evidence(
    source_weights: ArrayLike,
    source_coordinates: ArrayLike,
    target_weights: ArrayLike,
    target_coordinates: ArrayLike,
    /,
    *,
    tolerance: float = 1e-7,
) -> ConvexOrderEvidence:
    """Check finite marginal mass/means and the complete scalar call criterion."""
    source = jnp.asarray(source_coordinates, dtype=float)
    target = jnp.asarray(target_coordinates, dtype=float)
    if source.ndim == 1:
        source = source[:, None]
    if target.ndim == 1:
        target = target[:, None]
    source_w = jnp.asarray(source_weights, dtype=float)
    target_w = jnp.asarray(target_weights, dtype=float)
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("Convex-order coordinates must be two matrices of equal width.")
    if source_w.shape != (source.shape[0],) or target_w.shape != (target.shape[0],):
        raise ValueError(
            "Convex-order weights must contain one value per coordinate row."
        )
    tolerance_ = float(tolerance)
    if not isfinite(tolerance_) or tolerance_ < 0.0:
        raise ValueError("tolerance must be finite and nonnegative.")
    if bool(
        jnp.any(~jnp.isfinite(source_w))
        | jnp.any(~jnp.isfinite(target_w))
        | jnp.any(source_w < 0.0)
        | jnp.any(target_w < 0.0)
    ):
        raise ValueError("Convex-order weights must be finite and nonnegative.")
    source_mass = jnp.sum(source_w)
    target_mass = jnp.sum(target_w)
    source_mean = jnp.sum(source_w[:, None] * source, axis=0)
    target_mean = jnp.sum(target_w[:, None] * target, axis=0)
    mean_defect = jnp.max(jnp.abs(source_mean - target_mean))
    complete = source.shape[1] == 1
    if complete:
        strikes = jnp.sort(jnp.concatenate((source[:, 0], target[:, 0])))
        source_calls = jnp.sum(
            source_w[:, None] * jnp.maximum(source[:, 0, None] - strikes[None, :], 0.0),
            axis=0,
        )
        target_calls = jnp.sum(
            target_w[:, None] * jnp.maximum(target[:, 0, None] - strikes[None, :], 0.0),
            axis=0,
        )
        gaps = target_calls - source_calls
        minimum_gap = jnp.min(gaps)
    else:
        strikes = jnp.zeros((0,), dtype=source.dtype)
        gaps = jnp.zeros((0,), dtype=source.dtype)
        minimum_gap = jnp.asarray(jnp.inf, dtype=source.dtype)
    mass_matches = jnp.abs(source_mass - target_mass) <= tolerance_
    means_match = mean_defect <= tolerance_
    calls_match = minimum_gap >= -tolerance_
    feasible = mass_matches & means_match & (calls_match if complete else True)
    return ConvexOrderEvidence(
        source_mean,
        target_mean,
        mean_defect,
        strikes,
        gaps,
        minimum_gap,
        feasible,
        complete,
        tolerance_,
    )


def _constraint_system(problem: MartingaleTransportProblem, /) -> tuple[Array, Array]:
    n, m = problem.shape
    d = problem.coordinate_size
    variables = n * m
    source_rows = jnp.repeat(jnp.eye(n), m, axis=1)
    target_rows = jnp.tile(jnp.eye(m), (1, n))
    increments = (
        problem.target_coordinates[None, :, :] - problem.source_coordinates[:, None, :]
    )
    martingale_rows = jnp.zeros((n, d, n, m), dtype=increments.dtype)
    row_indices = jnp.arange(n)
    martingale_rows = martingale_rows.at[row_indices, :, row_indices, :].set(
        jnp.moveaxis(increments, -1, 1)
    )
    matrix = jnp.concatenate(
        (source_rows, target_rows, martingale_rows.reshape((n * d, variables))),
        axis=0,
    )
    rhs = jnp.concatenate(
        (
            problem.transport.source_weights,
            problem.transport.target_weights,
            jnp.zeros((n * d,), dtype=increments.dtype),
        )
    )
    return matrix, rhs


def _independent_rows(matrix: Array, /) -> tuple[int, ...]:
    host = np.asarray(matrix, dtype=float)
    selected: list[int] = []
    rank = 0
    for index in range(host.shape[0]):
        candidate = host[selected + [index]]
        candidate_rank = int(np.linalg.matrix_rank(candidate))
        if candidate_rank > rank:
            selected.append(index)
            rank = candidate_rank
    return tuple(selected)


def audit_martingale_coupling(
    problem: MartingaleTransportProblem,
    coupling: ArrayLike,
    /,
    *,
    primal_dual_gap: ArrayLike = jnp.inf,
    dual_objective: ArrayLike = -jnp.inf,
) -> MartingaleCouplingEvidence:
    """Independently recompute every defining finite martingale constraint."""
    if not isinstance(problem, MartingaleTransportProblem):
        raise TypeError("problem must be a MartingaleTransportProblem.")
    plan = jnp.asarray(coupling, dtype=float)
    if plan.shape != problem.shape:
        raise ValueError(f"coupling must have shape {problem.shape}.")
    source_marginal = jnp.sum(plan, axis=1)
    target_marginal = jnp.sum(plan, axis=0)
    conditional_numerator = plan @ problem.target_coordinates
    source_weights = problem.transport.source_weights
    conditional_mean = jnp.where(
        source_weights[:, None] > 0.0,
        conditional_numerator
        / jnp.where(source_weights[:, None] > 0.0, source_weights[:, None], 1.0),
        problem.source_coordinates,
    )
    source_residual = jnp.max(jnp.abs(source_marginal - source_weights))
    target_residual = jnp.max(jnp.abs(target_marginal - problem.transport.target_weights))
    active = source_weights > 0.0
    conditional_defect = jnp.where(
        active[:, None],
        jnp.abs(conditional_mean - problem.source_coordinates),
        0.0,
    )
    martingale_defect = jnp.max(conditional_defect)
    minimum = jnp.min(plan)
    safe = jnp.where(plan > 0.0, plan, 1.0)
    entropy = -jnp.sum(jnp.where(plan > 0.0, plan * jnp.log(safe), 0.0))
    cost = problem.transport.cost_matrix()
    primal = jnp.sum(plan * cost)
    gap = jnp.asarray(primal_dual_gap)
    dual = jnp.asarray(dual_objective)
    finite = (
        jnp.all(jnp.isfinite(plan))
        & jnp.isfinite(primal)
        & jnp.isfinite(entropy)
        & jnp.isfinite(source_residual)
        & jnp.isfinite(target_residual)
        & jnp.isfinite(martingale_defect)
    )
    tolerance = problem.constraint_tolerance
    valid = (
        finite
        & (minimum >= -tolerance)
        & (source_residual <= tolerance)
        & (target_residual <= tolerance)
        & (martingale_defect <= tolerance)
        & problem.convex_order.feasible
    )
    return MartingaleCouplingEvidence(
        source_marginal,
        target_marginal,
        conditional_mean,
        source_residual,
        target_residual,
        martingale_defect,
        minimum,
        primal,
        dual,
        gap,
        entropy,
        finite,
        valid,
        tolerance,
    )


def _empty_dual(problem: MartingaleTransportProblem, /) -> MartingaleDualEvidence:
    n, m = problem.shape
    dtype = problem.source_coordinates.dtype
    return MartingaleDualEvidence(
        jnp.zeros((n,), dtype=dtype),
        jnp.zeros((m,), dtype=dtype),
        jnp.zeros((n, problem.coordinate_size), dtype=dtype),
        jnp.asarray(-jnp.inf, dtype=dtype),
        jnp.asarray(jnp.inf, dtype=dtype),
        jnp.asarray(jnp.inf, dtype=dtype),
        1,
        jnp.asarray(False),
    )


def _dual_evidence(
    problem: MartingaleTransportProblem,
    equality_dual: Array,
    stationarity_residual: Array,
    primal_objective: Array,
    /,
) -> MartingaleDualEvidence:
    n, m = problem.shape
    matrix, rhs = _constraint_system(problem)
    cost = problem.transport.cost_matrix().reshape((-1,))
    candidate = matrix.T @ equality_dual
    positive_violation = jnp.max(jnp.maximum(candidate - cost, 0.0))
    negative_violation = jnp.max(jnp.maximum(-candidate - cost, 0.0))
    positive_objective = jnp.sum(rhs * equality_dual)
    negative_objective = -positive_objective
    positive_score = positive_violation + jnp.abs(primal_objective - positive_objective)
    negative_score = negative_violation + jnp.abs(primal_objective - negative_objective)
    orientation = 1 if bool(positive_score <= negative_score) else -1
    dual = orientation * equality_dual
    payoff = orientation * candidate
    violation = jnp.max(jnp.maximum(payoff - cost, 0.0))
    objective = jnp.sum(rhs * dual)
    source = dual[:n]
    target = dual[n : n + m]
    dynamic = dual[n + m :].reshape((n, problem.coordinate_size))
    valid = (
        jnp.all(jnp.isfinite(dual))
        & jnp.isfinite(objective)
        & (violation <= problem.constraint_tolerance)
    )
    return MartingaleDualEvidence(
        source,
        target,
        dynamic,
        objective,
        violation,
        jnp.asarray(stationarity_residual),
        orientation,
        valid,
    )


def solve_martingale_transport(
    problem: MartingaleTransportProblem,
    /,
    *,
    policy: ConvexSolvePolicy | None = None,
) -> MartingaleTransportResult:
    """Solve the exact finite martingale LP; never accepts classical OT input."""
    if not isinstance(problem, MartingaleTransportProblem):
        raise TypeError(
            "solve_martingale_transport requires a MartingaleTransportProblem; "
            "a classical DiscreteTransportProblem has no martingale constraint."
        )
    n, m = problem.shape
    if problem.convex_order.criterion_complete and not bool(
        problem.convex_order.feasible
    ):
        coupling = jnp.zeros((n, m), dtype=problem.source_coordinates.dtype)
        evidence = audit_martingale_coupling(problem, coupling)
        return MartingaleTransportResult(
            problem,
            coupling,
            evidence,
            _empty_dual(problem),
            None,
            jnp.asarray(
                int(MartingaleTransportStatus.CONVEX_ORDER_VIOLATION), dtype=jnp.int32
            ),
            (),
            "finite-martingale-linear-program",
        )
    matrix, rhs = _constraint_system(problem)
    independent = _independent_rows(matrix)
    selected = jnp.asarray(independent, dtype=jnp.int32)
    program = LinearProgram(
        problem.transport.cost_matrix().reshape((-1,)),
        equality_matrix=matrix[selected],
        equality_rhs=rhs[selected],
        bounds=Bounds(0.0, jnp.inf),
        problem_id="finite-martingale-transport",
    )
    optimizer = solve_linear_program(program, policy=policy)
    coupling = optimizer.primal.reshape((n, m))
    expanded_dual = jnp.zeros((matrix.shape[0],), dtype=optimizer.equality_dual.dtype)
    expanded_dual = expanded_dual.at[selected].set(optimizer.equality_dual)
    primal_objective = audit_martingale_coupling(problem, coupling).primal_objective
    dual = _dual_evidence(
        problem,
        expanded_dual,
        optimizer.stationarity_residual,
        primal_objective,
    )
    gap = jnp.abs(primal_objective - dual.dual_objective)
    evidence = audit_martingale_coupling(
        problem,
        coupling,
        primal_dual_gap=gap,
        dual_objective=dual.dual_objective,
    )
    status = jnp.where(
        optimizer.successful
        & evidence.valid
        & dual.valid
        & (gap <= 10.0 * problem.constraint_tolerance),
        int(MartingaleTransportStatus.OPTIMAL),
        jnp.where(
            ~problem.convex_order.feasible,
            int(MartingaleTransportStatus.CONVEX_ORDER_VIOLATION),
            jnp.where(
                ~optimizer.successful,
                int(MartingaleTransportStatus.OPTIMIZER_FAILED),
                int(MartingaleTransportStatus.INVALID_COUPLING),
            ),
        ),
    ).astype(jnp.int32)
    return MartingaleTransportResult(
        problem,
        coupling,
        evidence,
        dual,
        optimizer,
        status,
        independent,
        "finite-martingale-linear-program",
    )


def martingale_refinement_evidence(
    coarse: MartingaleTransportResult,
    refined: MartingaleTransportResult,
    /,
    *,
    refinement_id: str,
    objective_tolerance: float,
    constraint_tolerance: float,
) -> MartingaleRefinementEvidence:
    """Compare two genuine martingale results without accepting classical OT plans."""
    if not isinstance(coarse, MartingaleTransportResult) or not isinstance(
        refined, MartingaleTransportResult
    ):
        raise TypeError("Refinement requires two MartingaleTransportResult objects.")
    identifier = str(refinement_id)
    objective_tolerance_ = float(objective_tolerance)
    constraint_tolerance_ = float(constraint_tolerance)
    if not identifier:
        raise ValueError("refinement_id must be nonempty.")
    if (
        not isfinite(objective_tolerance_)
        or objective_tolerance_ < 0.0
        or not isfinite(constraint_tolerance_)
        or constraint_tolerance_ < 0.0
    ):
        raise ValueError("Refinement tolerances must be finite and nonnegative.")
    if coarse.problem.coordinate_size != refined.problem.coordinate_size:
        raise ValueError("Refinement problems must use the same martingale dimension.")
    coarse_marginal = jnp.maximum(
        coarse.evidence.source_marginal_residual,
        coarse.evidence.target_marginal_residual,
    )
    refined_marginal = jnp.maximum(
        refined.evidence.source_marginal_residual,
        refined.evidence.target_marginal_residual,
    )
    change = jnp.abs(refined.evidence.primal_objective - coarse.evidence.primal_objective)
    accepted = (
        coarse.successful
        & refined.successful
        & (change <= objective_tolerance_)
        & (refined.evidence.martingale_defect <= constraint_tolerance_)
        & (refined_marginal <= constraint_tolerance_)
    )
    return MartingaleRefinementEvidence(
        coarse.evidence.primal_objective,
        refined.evidence.primal_objective,
        change,
        coarse.evidence.martingale_defect,
        refined.evidence.martingale_defect,
        coarse_marginal,
        refined_marginal,
        accepted,
        objective_tolerance_,
        constraint_tolerance_,
        identifier,
    )


__all__ = [
    "ConvexOrderEvidence",
    "MartingaleCouplingEvidence",
    "MartingaleDualEvidence",
    "MartingaleRefinementEvidence",
    "MartingaleTransportProblem",
    "MartingaleTransportResult",
    "MartingaleTransportStatus",
    "audit_martingale_coupling",
    "convex_order_evidence",
    "martingale_refinement_evidence",
    "solve_martingale_transport",
]
