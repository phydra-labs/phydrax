#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

from .._bounds import Bounds
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import AbstractVectorSpace, LinearSubspace
from ..optim import ConvexProgramStatus, LinearProgram, solve_linear_program


class EstimatorGreedyPlan(StrictModule, NonTrainableState):
    maximum_rank: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, maximum_rank: int, /, *, tolerance: float):
        rank = int(maximum_rank)
        tolerance_ = float(tolerance)
        if rank <= 0 or not np.isfinite(tolerance_) or tolerance_ < 0.0:
            raise ValueError("Greedy rank and tolerance are invalid.")
        self.maximum_rank = rank
        self.tolerance = tolerance_
        self.plan_id = canonical_fingerprint(
            {"kind": "estimator-greedy-plan", "rank": rank, "tolerance": tolerance_}
        )


class GreedyBasisEvidence(StrictModule, NonTrainableState):
    selected_indices: Array
    maximum_estimates: Array
    stopped_by_tolerance: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    candidate_pool_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


def estimator_greedy_basis(
    space: AbstractVectorSpace,
    candidate_count: int,
    truth_snapshot: Callable[[int], PyTree[Array]],
    estimator: Callable[[LinearSubspace | None, int], float],
    plan: EstimatorGreedyPlan,
    /,
    *,
    candidate_pool_id: str,
) -> tuple[LinearSubspace, GreedyBasisEvidence]:
    """Deterministic host-side estimator greedy with physical orthogonalization."""
    if not isinstance(space, AbstractVectorSpace):
        raise TypeError("space must be AbstractVectorSpace.")
    if not isinstance(plan, EstimatorGreedyPlan):
        raise TypeError("plan must be EstimatorGreedyPlan.")
    count = int(candidate_count)
    pool = str(candidate_pool_id)
    if count <= 0 or not pool:
        raise ValueError("candidate_count and candidate_pool_id must be valid.")
    selected: list[int] = []
    estimates: list[float] = []
    basis_vectors: list[Array] = []
    subspace: LinearSubspace | None = None
    stopped = False
    for _ in range(min(plan.maximum_rank, count)):
        scores = np.asarray(
            [
                -np.inf if index in selected else float(estimator(subspace, index))
                for index in range(count)
            ]
        )
        if np.any(np.isnan(scores)):
            raise ValueError("Greedy estimator returned NaN.")
        index = int(np.argmax(scores))
        maximum = float(scores[index])
        if maximum <= plan.tolerance:
            stopped = True
            break
        vector = space.validate(truth_snapshot(index))
        residual = vector
        for coordinates in basis_vectors:
            basis_vector = space.unflatten(coordinates)
            coefficient = space.inner(basis_vector, residual)
            residual = jax.tree.map(
                lambda value, basis_value, coefficient=coefficient: (
                    value - coefficient * basis_value
                ),
                residual,
                basis_vector,
            )
        norm = jnp.sqrt(jnp.maximum(jnp.real(space.inner(residual, residual)), 0.0))
        norm_value = float(np.asarray(norm))
        if not np.isfinite(norm_value) or norm_value <= 64.0 * np.finfo(np.float64).eps:
            raise ValueError("Selected greedy snapshot is physically linearly dependent.")
        normalized = jax.tree.map(lambda value, norm=norm: value / norm, residual)
        basis_vectors.append(space.flatten(normalized))
        selected.append(index)
        estimates.append(maximum)
        basis = jnp.stack(tuple(basis_vectors), axis=1)
        subspace = LinearSubspace(
            space,
            basis,
            orthonormal=True,
            subspace_id=canonical_fingerprint(
                {
                    "kind": "estimator-greedy-subspace",
                    "pool": pool,
                    "plan": plan.plan_id,
                    "selected": selected,
                    "content": array_tree_fingerprint(basis)["sha256"],
                }
            ),
        )
    if subspace is None:
        raise ValueError("Greedy plan selected no basis vector.")
    evidence_id = canonical_fingerprint(
        {
            "kind": "greedy-basis-evidence",
            "plan": plan.plan_id,
            "pool": pool,
            "selected": selected,
            "estimates": estimates,
            "stopped": stopped,
        }
    )
    return subspace, GreedyBasisEvidence(
        jnp.asarray(selected, dtype=jnp.int32),
        jnp.asarray(estimates),
        stopped,
        plan.plan_id,
        pool,
        evidence_id,
    )


class SuccessiveConstraintArtifact(StrictModule, NonTrainableState):
    """Affine stability lower bound from declared SCM constraints."""

    component_lower: Array
    component_upper: Array
    sample_coefficients: Array
    sample_bounds: Array
    family_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    artifact_id: str = eqx.field(static=True)

    def __init__(
        self,
        component_lower: ArrayLike,
        component_upper: ArrayLike,
        sample_coefficients: ArrayLike,
        sample_bounds: ArrayLike,
        /,
        *,
        family_id: str,
        support_id: str,
    ):
        lower = jnp.asarray(component_lower)
        upper = jnp.asarray(component_upper)
        samples = jnp.asarray(sample_coefficients)
        bounds = jnp.asarray(sample_bounds)
        if lower.ndim != 1 or upper.shape != lower.shape:
            raise ValueError("SCM component bounds must be aligned vectors.")
        if (
            samples.ndim != 2
            or samples.shape[1] != lower.size
            or bounds.shape != (samples.shape[0],)
        ):
            raise ValueError("SCM sample constraints have invalid shape.")
        if np.any(np.asarray(lower) > np.asarray(upper)) or np.any(
            np.asarray(bounds) <= 0.0
        ):
            raise ValueError("SCM bounds are inconsistent or nonpositive.")
        family = str(family_id)
        support = str(support_id)
        if not family or not support:
            raise ValueError("SCM family and support IDs must be non-empty.")
        self.component_lower = lower
        self.component_upper = upper
        self.sample_coefficients = samples
        self.sample_bounds = bounds
        self.family_id = family
        self.support_id = support
        self.artifact_id = canonical_fingerprint(
            {
                "kind": "successive-constraint-artifact",
                "family": family,
                "support": support,
                "content": array_tree_fingerprint(
                    {"lower": lower, "upper": upper, "samples": samples, "bounds": bounds}
                )["sha256"],
            }
        )

    def lower_bound(self, coefficients: ArrayLike, /) -> Array:
        values = jnp.asarray(coefficients)
        if values.shape != self.component_lower.shape:
            raise ValueError("SCM query coefficients have invalid shape.")
        program = LinearProgram(
            values,
            inequality_matrix=-self.sample_coefficients,
            inequality_rhs=-self.sample_bounds,
            bounds=Bounds(self.component_lower, self.component_upper),
        )
        result = solve_linear_program(program)
        valid = result.status == ConvexProgramStatus.OPTIMAL
        objective = jnp.vdot(values, result.primal).real
        return jnp.where(valid & (objective > 0.0), objective, jnp.nan)


class PrimalDualOutputBound(StrictModule, NonTrainableState):
    corrected_output: Array
    absolute_error_bound: Array
    valid: Array
    output_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        primal_output: ArrayLike,
        primal_residual_on_dual: ArrayLike,
        primal_residual_norm: ArrayLike,
        dual_residual_norm: ArrayLike,
        stability_lower_bound: ArrayLike,
        /,
        *,
        output_id: str,
        evidence_ids: Sequence[str],
    ):
        output = jnp.asarray(primal_output)
        correction = jnp.asarray(primal_residual_on_dual)
        primal = jnp.asarray(primal_residual_norm)
        dual = jnp.asarray(dual_residual_norm)
        stability = jnp.asarray(stability_lower_bound)
        valid = (
            jnp.all(jnp.isfinite(output))
            & jnp.all(jnp.isfinite(correction))
            & jnp.all(jnp.isfinite(primal))
            & jnp.all(jnp.isfinite(dual))
            & jnp.all(jnp.isfinite(stability))
            & jnp.all(stability > 0.0)
        )
        identifier = str(output_id)
        evidence = tuple(str(value) for value in evidence_ids)
        if not identifier or not evidence or any(not value for value in evidence):
            raise ValueError("Output and evidence IDs must be non-empty.")
        self.corrected_output = output + correction
        self.absolute_error_bound = primal * dual / stability
        self.valid = valid
        self.output_id = identifier
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "primal-dual-output-bound",
                "output": identifier,
                "evidence": list(evidence),
            }
        )


__all__ = [
    "EstimatorGreedyPlan",
    "GreedyBasisEvidence",
    "PrimalDualOutputBound",
    "SuccessiveConstraintArtifact",
    "estimator_greedy_basis",
]
