#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxtyping import Array, ArrayLike

from phydrax.linalg import DenseLinearOperator

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ._blocks import PreparedScalarBlocks


class CrossingConePlan(StrictModule):
    """Finite sampled crossing equation lowered to a nonnegative polyhedral cone."""

    identity_vector: Array
    block_vectors: Array
    scaling_dimensions: Array
    maximum_iterations: int = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    maximum_matrix_entries: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        identity_vector: ArrayLike,
        block_vectors: ArrayLike,
        scaling_dimensions: ArrayLike,
        /,
        *,
        maximum_iterations: int = 4096,
        residual_tolerance: float = 1e-8,
        maximum_matrix_entries: int = 1_000_000,
    ):
        identity = np.asarray(identity_vector, dtype=float)
        blocks = np.asarray(block_vectors, dtype=float)
        dimensions = np.asarray(scaling_dimensions, dtype=float)
        iterations = int(maximum_iterations)
        tolerance = float(residual_tolerance)
        maximum = int(maximum_matrix_entries)
        if identity.ndim != 1 or identity.size == 0:
            raise ValueError("identity_vector must be one nonempty vector.")
        if blocks.ndim != 2 or blocks.shape[0] != identity.size:
            raise ValueError("block_vectors must have shape (constraints, operators).")
        if dimensions.shape != (blocks.shape[1],):
            raise ValueError("scaling_dimensions must label every block column.")
        if not np.all(np.isfinite(identity)) or not np.all(np.isfinite(blocks)):
            raise ValueError("Crossing vectors must be finite.")
        if not np.all(np.isfinite(dimensions)) or np.any(dimensions <= 0.0):
            raise ValueError("Scaling dimensions must be finite and positive.")
        if np.any(np.diff(dimensions) <= 0.0):
            raise ValueError("Scaling dimensions must be strictly increasing.")
        if iterations < 1 or tolerance < 0.0 or maximum < 1:
            raise ValueError("Crossing solver resources/tolerance are invalid.")
        if blocks.size > maximum:
            raise ValueError("Crossing matrix exceeds maximum_matrix_entries.")
        self.identity_vector = jnp.asarray(identity)
        self.block_vectors = jnp.asarray(blocks)
        self.scaling_dimensions = jnp.asarray(dimensions)
        self.maximum_iterations = iterations
        self.residual_tolerance = tolerance
        self.maximum_matrix_entries = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-sampled-crossing-cone-plan",
                "identity": array_tree_fingerprint(identity),
                "blocks": array_tree_fingerprint(blocks),
                "scaling_dimensions": array_tree_fingerprint(dimensions),
                "maximum_iterations": iterations,
                "residual_tolerance": tolerance,
                "maximum_matrix_entries": maximum,
            }
        )


class PreparedCrossingCone(StrictModule):
    plan: CrossingConePlan
    operator: DenseLinearOperator
    step_size: Array
    prepared_id: str = eqx.field(static=True)


class CrossingConicEvidence(StrictModule):
    coefficients: Array
    reconstructed_crossing: Array
    residual_norm: Array
    relative_residual: Array
    minimum_coefficient: Array
    stationarity_violation: Array
    status: Array
    converged: Array
    plan_id: str = eqx.field(static=True)
    status_meaning: tuple[str, ...] = eqx.field(static=True)
    claim: str = eqx.field(static=True)


class CrossingExclusionEvidence(StrictModule):
    functional: Array
    identity_evaluation: Array
    block_evaluations: Array
    minimum_block_evaluation: Array
    candidate_gap: Array
    excludes_candidate_gap: Array
    primal: CrossingConicEvidence
    plan_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


class KnownBoundEvidence(StrictModule):
    reference_label: str = eqx.field(static=True)
    published_upper_bound: float = eqx.field(static=True)
    candidate_gap: Array
    candidate_excluded: Array
    reproduces_or_strengthens: Array
    claim: str = eqx.field(static=True)


def assemble_scalar_crossing_cone(
    blocks: PreparedScalarBlocks,
    scaling_dimensions: Sequence[float],
    /,
    *,
    maximum_iterations: int = 4096,
    residual_tolerance: float = 1e-8,
    maximum_matrix_entries: int = 1_000_000,
) -> CrossingConePlan:
    if not isinstance(blocks, PreparedScalarBlocks):
        raise TypeError("blocks must be PreparedScalarBlocks.")
    dimensions = tuple(float(value) for value in scaling_dimensions)
    if not dimensions:
        raise ValueError("At least one non-identity scaling dimension is required.")
    if len(dimensions) * blocks.plan.cross_ratios.size > int(maximum_matrix_entries):
        raise ValueError("Crossing block evaluation exceeds maximum_matrix_entries.")
    columns = jnp.stack(
        tuple(blocks.crossing_vector(value) for value in dimensions), axis=1
    )
    return CrossingConePlan(
        blocks.identity_crossing_vector(),
        columns,
        dimensions,
        maximum_iterations=maximum_iterations,
        residual_tolerance=residual_tolerance,
        maximum_matrix_entries=maximum_matrix_entries,
    )


def prepare_crossing_cone(plan: CrossingConePlan, /) -> PreparedCrossingCone:
    if not isinstance(plan, CrossingConePlan):
        raise TypeError("plan must be CrossingConePlan.")
    squared_frobenius = float(np.sum(np.asarray(plan.block_vectors) ** 2))
    if squared_frobenius <= 0.0 or not np.isfinite(squared_frobenius):
        raise ValueError("Crossing block matrix must have nonzero finite norm.")
    operator = DenseLinearOperator(
        plan.block_vectors,
        operator_id=f"crossing-cone:{plan.plan_id}",
    )
    return PreparedCrossingCone(
        plan=plan,
        operator=operator,
        step_size=jnp.asarray(1.0 / squared_frobenius),
        prepared_id=canonical_fingerprint(
            {
                "kind": "prepared-finite-crossing-cone",
                "plan": plan.plan_id,
                "operator": operator.operator_id,
                "algorithm": "fixed-projected-gradient",
            }
        ),
    )


def solve_crossing_cone(prepared: PreparedCrossingCone, /) -> CrossingConicEvidence:
    """Compute a fixed-work nonnegative least-squares crossing certificate."""
    if not isinstance(prepared, PreparedCrossingCone):
        raise TypeError("prepared must be PreparedCrossingCone.")
    plan = prepared.plan
    coefficients = jnp.zeros(
        (plan.block_vectors.shape[1],), dtype=plan.block_vectors.dtype
    )

    def projected_step(_, current):
        residual = plan.identity_vector + prepared.operator.mv(current)
        gradient = prepared.operator.adjoint_mv(residual)
        return jnp.maximum(0.0, current - prepared.step_size * gradient)

    coefficients = lax.fori_loop(0, plan.maximum_iterations, projected_step, coefficients)
    residual = plan.identity_vector + prepared.operator.mv(coefficients)
    residual_norm = jnp.linalg.norm(residual)
    scale = jnp.maximum(1.0, jnp.linalg.norm(plan.identity_vector))
    relative = residual_norm / scale
    gradient = prepared.operator.adjoint_mv(residual)
    complementarity = jnp.where(coefficients > 0.0, jnp.abs(gradient), 0.0)
    inactive_violation = jnp.where(coefficients == 0.0, jnp.maximum(-gradient, 0.0), 0.0)
    stationarity = jnp.max(jnp.maximum(complementarity, inactive_violation))
    finite = (
        jnp.isfinite(relative)
        & jnp.isfinite(stationarity)
        & jnp.all(jnp.isfinite(coefficients))
    )
    converged = finite & (relative <= plan.residual_tolerance)
    status = jnp.where(converged, 0, jnp.where(finite, 1, 2)).astype(jnp.int32)
    return CrossingConicEvidence(
        coefficients=coefficients,
        reconstructed_crossing=-prepared.operator.mv(coefficients),
        residual_norm=residual_norm,
        relative_residual=relative,
        minimum_coefficient=jnp.min(coefficients),
        stationarity_violation=stationarity,
        status=status,
        converged=converged,
        plan_id=plan.plan_id,
        status_meaning=("crossing-satisfied", "positive-crossing-residual", "nonfinite"),
        claim="finite-grid-conic-reference-only",
    )


def exclude_scalar_gap(
    prepared: PreparedCrossingCone,
    candidate_gap: float,
    /,
) -> CrossingExclusionEvidence:
    """Produce and independently evaluate a separating crossing functional."""
    if not isinstance(prepared, PreparedCrossingCone):
        raise TypeError("prepared must be PreparedCrossingCone.")
    gap = float(candidate_gap)
    if gap <= 0.0 or not np.isfinite(gap):
        raise ValueError("candidate_gap must be finite and positive.")
    dimensions = np.asarray(prepared.plan.scaling_dimensions)
    selected = dimensions >= gap
    if not np.any(selected):
        raise ValueError("No sampled scaling dimensions lie at or above candidate_gap.")
    restricted_plan = CrossingConePlan(
        prepared.plan.identity_vector,
        np.asarray(prepared.plan.block_vectors)[:, selected],
        dimensions[selected],
        maximum_iterations=prepared.plan.maximum_iterations,
        residual_tolerance=prepared.plan.residual_tolerance,
        maximum_matrix_entries=prepared.plan.maximum_matrix_entries,
    )
    restricted = prepare_crossing_cone(restricted_plan)
    primal = solve_crossing_cone(restricted)
    residual = restricted_plan.identity_vector + restricted.operator.mv(
        primal.coefficients
    )
    identity_evaluation_raw = jnp.vdot(residual, restricted_plan.identity_vector).real
    safe = jnp.where(jnp.abs(identity_evaluation_raw) > 0.0, identity_evaluation_raw, 1.0)
    functional = residual / safe
    identity_evaluation = jnp.vdot(functional, restricted_plan.identity_vector).real
    block_evaluations = restricted.operator.adjoint_mv(functional).real
    minimum = jnp.min(block_evaluations)
    tolerance = restricted_plan.residual_tolerance
    excludes = (
        jnp.isfinite(identity_evaluation)
        & jnp.all(jnp.isfinite(block_evaluations))
        & (identity_evaluation > 0.0)
        & (jnp.abs(identity_evaluation - 1.0) <= 10.0 * tolerance)
        & (minimum >= -10.0 * tolerance)
        & (~primal.converged)
    )
    return CrossingExclusionEvidence(
        functional=functional,
        identity_evaluation=identity_evaluation,
        block_evaluations=block_evaluations,
        minimum_block_evaluation=minimum,
        candidate_gap=jnp.asarray(gap),
        excludes_candidate_gap=excludes,
        primal=primal,
        plan_id=restricted_plan.plan_id,
        claim="finite-sampled-spectrum-research-only-no-continuum-bootstrap-claim",
    )


def compare_known_gap_bound(
    exclusion: CrossingExclusionEvidence,
    published_upper_bound: float,
    reference_label: str,
    /,
) -> KnownBoundEvidence:
    if not isinstance(exclusion, CrossingExclusionEvidence):
        raise TypeError("exclusion must be CrossingExclusionEvidence.")
    bound = float(published_upper_bound)
    label = str(reference_label)
    if bound <= 0.0 or not np.isfinite(bound) or not label:
        raise ValueError("Known bound and reference label must be valid.")
    reproduces = exclusion.excludes_candidate_gap & (exclusion.candidate_gap <= bound)
    return KnownBoundEvidence(
        reference_label=label,
        published_upper_bound=bound,
        candidate_gap=exclusion.candidate_gap,
        candidate_excluded=exclusion.excludes_candidate_gap,
        reproduces_or_strengthens=reproduces,
        claim="comparison-to-user-declared-finite-reference-bound-only",
    )


__all__ = [
    "CrossingConePlan",
    "CrossingConicEvidence",
    "CrossingExclusionEvidence",
    "KnownBoundEvidence",
    "PreparedCrossingCone",
    "assemble_scalar_crossing_cone",
    "compare_known_gap_bound",
    "exclude_scalar_gap",
    "prepare_crossing_cone",
    "solve_crossing_cone",
]
